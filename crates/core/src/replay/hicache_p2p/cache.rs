// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Page-level two-tier (GPU + host) radix cache for the HiCache/P2P topology runtime.
//!
//! One tree per worker. A node holds a run-length list of 64-token block ids. A node can be
//! on the device tier (`dev`), on the host tier (`host`), or both. Write-through: every node
//! inserted after a prefill is resident on both tiers. Device eviction is leaf-first LRU over
//! device nodes with no device children; an evicted node stays as a host-only node. Host
//! eviction removes host-only leaves, LRU by the last *local* use. Matching for a peer pull
//! (`match_peek`) never refreshes the source. Locks (in-flight requests) and leases (peer
//! pull sources) both pin a path from the root. A demoted node (source copy after a completed
//! pull) is evicted before every non-demoted node of the same tier; a later local match clears
//! the flag.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::FxHashMap;

/// Node index in the arena.
pub type NodeId = u32;
const NONE: NodeId = u32::MAX;
pub const ROOT: NodeId = 0;

#[derive(Debug, Clone)]
pub struct Node {
    /// Block runs `(first_block_id, count)`.
    pub runs: Vec<(u64, u32)>,
    pub nb: u32,
    pub ch: FxHashMap<u64, NodeId>,
    pub par: NodeId,
    pub dev: bool,
    pub host: bool,
    /// Last local use (virtual seconds).
    pub last: f64,
    /// In-flight request locks passing through this node.
    pub lock: u32,
    /// Peer-pull source leases passing through this node.
    pub lease: u32,
    /// Host locks (fix-3 queued-prefix guard, pull landing): block host eviction only.
    pub hlock: u32,
    /// Children resident on the device tier.
    pub ndev: u32,
    /// Demoted source copy: evicted first.
    pub demoted: bool,
    /// Time the node (or its oldest part) was first written to the host tier.
    pub host_t: f64,
    pub alive: bool,
}

impl Node {
    fn new(runs: Vec<(u64, u32)>, nb: u32, par: NodeId) -> Self {
        Self {
            runs,
            nb,
            ch: FxHashMap::default(),
            par,
            dev: false,
            host: false,
            last: 0.0,
            lock: 0,
            lease: 0,
            hlock: 0,
            ndev: 0,
            demoted: false,
            host_t: 0.0,
            alive: true,
        }
    }

    /// Pinned against device eviction (in-flight request locks; leases if lease_pins).
    fn pinned(&self) -> bool {
        self.lock > 0 || self.lease > 0
    }

    /// Pinned against host eviction.
    fn host_pinned(&self) -> bool {
        self.lock > 0 || self.lease > 0 || self.hlock > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapKey {
    class: u8,
    last: f64,
    seq: u64,
    node: NodeId,
}

impl Eq for HeapKey {}

impl Ord for HeapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; invert so the smallest (class, last, seq) pops first.
        other
            .class
            .cmp(&self.class)
            .then_with(|| other.last.total_cmp(&self.last))
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for HeapKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Which tiers the demote flag reorders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemoteScope {
    /// Demoted nodes are evicted first from the host tier only.
    Host,
    /// Demoted nodes are evicted first from both tiers.
    DeviceAndHost,
}

/// Counters for one tree.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub host_evicted_blocks: u64,
    pub host_evicted_nodes: u64,
    pub dev_evicted_blocks: u64,
    pub loaded_back_blocks: u64,
    pub host_inserted_blocks: u64,
    /// Sum over host-evicted blocks of (evict time - last local use), seconds x blocks.
    pub host_age_last_sum: f64,
    /// Sum over host-evicted blocks of (evict time - host write time), seconds x blocks.
    pub host_age_write_sum: f64,
    pub demote_flagged_nodes: u64,
    pub demote_flagged_blocks: u64,
    pub demote_evicted_nodes: u64,
    pub demote_evicted_blocks: u64,
    pub demote_cleared_nodes: u64,
    pub demote_cleared_blocks: u64,
    /// evict_host calls that ended still over capacity, and the summed shortfall (blocks).
    pub overcommit_events: u64,
    pub overcommit_blocks: u64,
    /// Local matches that end inside a node (partial node match) and the unmatched rest of that
    /// node (blocks); match-time splits done (split_on_match).
    pub partial_local_matches: u64,
    pub partial_rest_blocks: u64,
    pub match_splits: u64,
    /// Oracle O2 (perfect move): source blocks freed at transfer, and blocks only flagged because a
    /// node was pinned at that moment.
    pub moved_freed_blocks: u64,
    pub moved_flagged_blocks: u64,
}

/// Window-filtered host eviction samples: (evict time, blocks, age since last use, age since write).
#[derive(Debug, Clone, Copy)]
pub struct HostEviction {
    pub t: f64,
    pub blocks: u32,
    pub age_last: f64,
    pub age_write: f64,
    pub demoted: bool,
    /// Namespace of the evicted node's first block (block id >> 32 = task admission).
    pub ns: u64,
}

/// Host-pool census at one instant (tokens are blocks here; the caller scales).
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolCensus {
    pub host_used: u64,
    pub pinned_host: u64,
    pub device_resident: u64,
    pub demoted_host: u64,
}

/// Result of a prefix match.
#[derive(Debug, Clone, Default)]
pub struct MatchResult {
    /// Contiguous device-resident blocks from the root.
    pub dev: u32,
    /// Further blocks resident on host (or device after a host gap).
    pub host: u32,
    /// Matched path: (node, matched blocks in that node).
    pub path: Vec<(NodeId, u32)>,
}

impl MatchResult {
    pub fn total(&self) -> u32 {
        self.dev + self.host
    }
}

#[derive(Debug, Clone)]
pub struct TwoTierRadix {
    nodes: Vec<Node>,
    free: Vec<NodeId>,
    pub dev_cap: u64,
    pub host_cap: u64,
    pub dev_used: u64,
    pub host_used: u64,
    dh: BinaryHeap<HeapKey>,
    hh: BinaryHeap<HeapKey>,
    seq: u64,
    live: u64,
    pub demote_scope: DemoteScope,
    pub stats: CacheStats,
    pub evictions: Vec<HostEviction>,
    pub record_evictions: bool,
    /// Split refreshes both fragments' access time (engine behavior); false = v2 behavior.
    pub split_refresh: bool,
    /// A local match that ends inside a node splits it there (engine: unified_tree_core.py
    /// _match_prefix_helper :871 @f8a08865b1): only the matched upper fragment is refreshed and
    /// un-demoted by this and later matches (:911-918); the rest keeps its own access time and
    /// demote flag (_split_node :1358-1362). false = v1-v3 behavior (whole node touched).
    pub split_on_match: bool,
}

/// Longest common prefix of node runs `eruns` and prompt runs `pruns[ri..]` (offset `off` into
/// run `ri`). Returns (matched blocks, new ri, new off).
fn lcp(eruns: &[(u64, u32)], pruns: &[(u64, u32)], mut ri: usize, mut off: u32) -> (u32, usize, u32) {
    let mut m: u32 = 0;
    let mut ei = 0usize;
    let mut eo: u32 = 0;
    while ei < eruns.len() && ri < pruns.len() {
        let (es, ec) = eruns[ei];
        let (ps, pc) = pruns[ri];
        if es + eo as u64 != ps + off as u64 {
            break;
        }
        let k = (ec - eo).min(pc - off);
        m += k;
        eo += k;
        off += k;
        if eo == ec {
            ei += 1;
            eo = 0;
        }
        if off == pc {
            ri += 1;
            off = 0;
        }
    }
    (m, ri, off)
}

fn split_runs(runs: &[(u64, u32)], m: u32) -> (Vec<(u64, u32)>, Vec<(u64, u32)>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut acc: u32 = 0;
    for &(s, c) in runs {
        if acc >= m {
            b.push((s, c));
        } else if acc + c <= m {
            a.push((s, c));
        } else {
            let k = m - acc;
            a.push((s, k));
            b.push((s + k as u64, c - k));
        }
        acc += c;
    }
    (a, b)
}

/// Blocks in a run list.
pub fn run_blocks(runs: &[(u64, u32)]) -> u64 {
    runs.iter().map(|&(_, c)| c as u64).sum()
}

/// Prefix of `runs` with exactly `m` blocks.
pub fn runs_prefix(runs: &[(u64, u32)], m: u32) -> Vec<(u64, u32)> {
    split_runs(runs, m).0
}

impl TwoTierRadix {
    pub fn new(dev_cap_blocks: u64, host_cap_blocks: u64) -> Self {
        let mut root = Node::new(Vec::new(), 0, NONE);
        root.dev = true;
        root.host = true;
        root.last = 1e18;
        Self {
            nodes: vec![root],
            free: Vec::new(),
            dev_cap: dev_cap_blocks,
            host_cap: host_cap_blocks,
            dev_used: 0,
            host_used: 0,
            dh: BinaryHeap::new(),
            hh: BinaryHeap::new(),
            seq: 0,
            live: 1,
            demote_scope: DemoteScope::Host,
            stats: CacheStats::default(),
            evictions: Vec::new(),
            record_evictions: false,
            split_refresh: true,
            split_on_match: false,
        }
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    pub fn live_nodes(&self) -> u64 {
        self.live
    }

    fn alloc(&mut self, node: Node) -> NodeId {
        self.live += 1;
        if let Some(id) = self.free.pop() {
            self.nodes[id as usize] = node;
            id
        } else {
            self.nodes.push(node);
            (self.nodes.len() - 1) as NodeId
        }
    }

    fn release(&mut self, id: NodeId) {
        let n = &mut self.nodes[id as usize];
        n.alive = false;
        n.ch = FxHashMap::default();
        n.runs = Vec::new();
        n.par = NONE;
        self.live -= 1;
        self.free.push(id);
    }

    fn class(&self, id: NodeId, tier_host: bool) -> u8 {
        let n = &self.nodes[id as usize];
        if n.demoted && (tier_host || self.demote_scope == DemoteScope::DeviceAndHost) {
            0
        } else {
            1
        }
    }

    /// Push a node onto the heap of the tier where it is an eviction candidate.
    fn push(&mut self, id: NodeId) {
        if id == ROOT {
            return;
        }
        let (dev, host, leaf, last) = {
            let n = &self.nodes[id as usize];
            (n.dev, n.host, n.ch.is_empty(), n.last)
        };
        self.seq += 1;
        if dev {
            let class = self.class(id, false);
            self.dh.push(HeapKey { class, last, seq: self.seq, node: id });
        } else if host && leaf {
            let class = self.class(id, true);
            self.hh.push(HeapKey { class, last, seq: self.seq, node: id });
        }
    }

    fn set_dev(&mut self, id: NodeId, v: bool) {
        let (cur, nb, par) = {
            let n = &self.nodes[id as usize];
            (n.dev, n.nb, n.par)
        };
        if cur == v {
            return;
        }
        self.nodes[id as usize].dev = v;
        if v {
            self.dev_used += nb as u64;
            self.nodes[par as usize].ndev += 1;
        } else {
            self.dev_used -= nb as u64;
            self.nodes[par as usize].ndev -= 1;
        }
    }

    fn walk(&mut self, runs: &[(u64, u32)], touch: Option<f64>, clear_demote: bool) -> MatchResult {
        let mut node = ROOT;
        let mut ri = 0usize;
        let mut off: u32 = 0;
        let mut res = MatchResult::default();
        let mut indev = true;
        while ri < runs.len() {
            let key = runs[ri].0 + off as u64;
            let Some(&ch) = self.nodes[node as usize].ch.get(&key) else {
                break;
            };
            let (m, ri2, off2) = lcp(&self.nodes[ch as usize].runs, runs, ri, off);
            ri = ri2;
            off = off2;
            let mut ch = ch;
            if let Some(t) = touch {
                let nb0 = self.nodes[ch as usize].nb;
                if m < nb0 {
                    self.stats.partial_local_matches += 1;
                    self.stats.partial_rest_blocks += (nb0 - m) as u64;
                    if self.split_on_match {
                        // Engine: split at the match end; the upper fragment is the matched node.
                        ch = self.split(node, key, ch, m, t);
                        self.stats.match_splits += 1;
                    }
                }
            }
            let (cdev, chost, cnb) = {
                let c = &self.nodes[ch as usize];
                (c.dev, c.host, c.nb)
            };
            if indev && cdev {
                res.dev += m;
            } else {
                indev = false;
                if chost || cdev {
                    res.host += m;
                } else {
                    break;
                }
            }
            res.path.push((ch, m));
            if let Some(t) = touch {
                let c = &mut self.nodes[ch as usize];
                c.last = t;
                if clear_demote && c.demoted {
                    c.demoted = false;
                    self.stats.demote_cleared_nodes += 1;
                    self.stats.demote_cleared_blocks += cnb as u64;
                }
            }
            if m < cnb {
                break;
            }
            node = ch;
        }
        res
    }

    /// Local match (a request on this worker): touches the path and clears demote flags.
    pub fn match_local(&mut self, runs: &[(u64, u32)], t: f64) -> MatchResult {
        let r = self.walk(runs, Some(t), true);
        // Touched nodes get fresh heap entries when they are next unpinned; push now too so
        // unpinned touched nodes keep a valid entry.
        for &(n, _) in &r.path {
            if !self.nodes[n as usize].pinned() {
                self.push(n);
            }
        }
        r
    }

    /// Read-only match (router index view, peer-pull source view). Never refreshes recency.
    pub fn match_peek(&mut self, runs: &[(u64, u32)]) -> MatchResult {
        self.walk(runs, None, false)
    }

    /// Load host-only nodes of a matched path to the device tier (load-back) and touch them.
    /// Returns loaded blocks.
    pub fn load_back(&mut self, path: &[(NodeId, u32)], t: f64) -> u64 {
        let mut loaded = 0u64;
        for &(n, _) in path {
            self.nodes[n as usize].last = t;
            if !self.nodes[n as usize].dev {
                loaded += self.nodes[n as usize].nb as u64;
                self.set_dev(n, true);
            }
        }
        self.stats.loaded_back_blocks += loaded;
        for &(n, _) in path {
            if !self.nodes[n as usize].pinned() {
                self.push(n);
            }
        }
        self.evict_dev(t);
        loaded
    }

    fn pin_kind(&mut self, mut node: NodeId, d: i64, kind: u8) {
        while node != NONE && node != ROOT {
            let n = &mut self.nodes[node as usize];
            let cnt = match kind {
                1 => &mut n.lease,
                2 => &mut n.hlock,
                _ => &mut n.lock,
            };
            let v = *cnt as i64 + d;
            debug_assert!(v >= 0, "pin count underflow (kind {kind})");
            *cnt = v.max(0) as u32;
            let unpinned = !n.pinned();
            let par = n.par;
            if unpinned {
                self.push(node);
            }
            node = par;
        }
    }

    /// Lock (d=+1) or unlock (d=-1) the path from `node` to the root (in-flight request).
    pub fn lock(&mut self, node: Option<NodeId>, d: i64) {
        if let Some(n) = node {
            self.pin_kind(n, d, 0);
        }
    }

    /// Host lock (d=+1/-1) on the path from `node` to the root: host eviction skips it.
    pub fn host_lock(&mut self, node: Option<NodeId>, d: i64) {
        if let Some(n) = node {
            self.pin_kind(n, d, 2);
        }
    }

    /// Lease (d=+1) or release (d=-1) the path from `node` to the root (peer-pull source).
    pub fn lease(&mut self, node: Option<NodeId>, d: i64) {
        if let Some(n) = node {
            self.pin_kind(n, d, 1);
        }
    }

    /// The deepest node of a path covering at least `blocks` blocks from the root, if any.
    pub fn path_node_covering(path: &[(NodeId, u32)], blocks: u32) -> Option<NodeId> {
        let mut acc = 0u32;
        let mut last = None;
        for &(n, m) in path {
            if acc >= blocks {
                break;
            }
            last = Some(n);
            acc += m;
        }
        last
    }

    /// Demote-source: flag the nodes of `path` strictly below the deepest branch point on it
    /// (a node with two or more children). Returns (flagged nodes, flagged blocks).
    pub fn demote_below_branch(&mut self, path: &[(NodeId, u32)]) -> (u64, u64) {
        // Deepest branch point: index of the last node on the path with >= 2 children; the
        // root counts as a branch point only if it has >= 2 children.
        let mut start = 0usize;
        for (i, &(n, _)) in path.iter().enumerate() {
            if self.nodes[n as usize].ch.len() >= 2 {
                start = i + 1;
            }
        }
        let mut fn_ = 0u64;
        let mut fb = 0u64;
        for &(n, _) in &path[start.min(path.len())..] {
            let node = &mut self.nodes[n as usize];
            if !node.alive || node.demoted {
                continue;
            }
            node.demoted = true;
            fn_ += 1;
            fb += node.nb as u64;
        }
        self.stats.demote_flagged_nodes += fn_;
        self.stats.demote_flagged_blocks += fb;
        for &(n, _) in &path[start.min(path.len())..] {
            if !self.nodes[n as usize].pinned() {
                self.push(n);
            }
        }
        (fn_, fb)
    }

    /// Demote-source walk (peer_kv_demote.py:117-148): from the deepest leased node toward the
    /// root. Stop: no host copy; the deepest node has children; an ancestor with another child
    /// (branch); accessed at or after the lookup time; root. Already flagged nodes pass through.
    /// Returns (flagged nodes, flagged blocks, stop reason 0 no_node 1 no_host 2 children 3 branch 4 access 5 root).
    pub fn demote_walk(&mut self, path: &[(NodeId, u32)], lookup_t: f64) -> (u64, u64, u8) {
        if path.is_empty() {
            return (0, 0, 0);
        }
        // A lease ending inside a node, or a gone node -> "no_node", nothing flagged.
        let (last_id, last_m) = path[path.len() - 1];
        let ln = &self.nodes[last_id as usize];
        if !ln.alive || last_m < ln.nb {
            return (0, 0, 0);
        }
        let mut fn_ = 0u64;
        let mut fb = 0u64;
        let deepest = path.len() - 1;
        let mut reason = 5u8;
        let mut flagged: Vec<NodeId> = Vec::new();
        for i in (0..=deepest).rev() {
            let id = path[i].0;
            let n = &self.nodes[id as usize];
            if !n.alive || !n.host {
                reason = 1;
                break;
            }
            if i == deepest && !n.ch.is_empty() {
                reason = 2;
                break;
            }
            if i < deepest && n.ch.len() >= 2 {
                reason = 3;
                break;
            }
            if n.last >= lookup_t {
                reason = 4;
                break;
            }
            if n.demoted {
                continue;
            }
            flagged.push(id);
        }
        for id in flagged {
            let n = &mut self.nodes[id as usize];
            n.demoted = true;
            fn_ += 1;
            fb += n.nb as u64;
            if !n.host_pinned() {
                self.push(id);
            }
        }
        self.stats.demote_flagged_nodes += fn_;
        self.stats.demote_flagged_blocks += fb;
        (fn_, fb, reason)
    }

    /// Demote walk as the engine runs it at RELEASE time (peer_kv_demote.py select_demote_path,
    /// :117-148 + docstring step 3): find the live node whose last page is the last leased page by
    /// a read-only match of the leased prefix `runs` NOW (a lease ending inside a node, or a gone or
    /// shorter path -> no_node), then walk PARENT links toward the root. Flag a node only if it has
    /// no child outside the path and no local access at or after `lookup_t`; stop at the first node
    /// that fails (children / branch / access), at a node without a host copy (no_host) or at the
    /// root. Returns (flagged nodes, flagged blocks, reason as demote_walk).
    pub fn demote_from(&mut self, runs: &[(u64, u32)], lookup_t: f64) -> (u64, u64, u8) {
        let want = run_blocks(runs) as u32;
        let m = self.walk(runs, None, false);
        let Some(&(last_id, last_m)) = m.path.last() else { return (0, 0, 0) };
        if m.total() < want || !self.nodes[last_id as usize].alive || last_m < self.nodes[last_id as usize].nb {
            return (0, 0, 0);
        }
        let mut node = last_id;
        let mut path_child = NONE;
        let mut reason = 5u8;
        let mut flagged: Vec<NodeId> = Vec::new();
        while node != NONE && node != ROOT {
            let n = &self.nodes[node as usize];
            if !n.alive || !n.host {
                reason = 1;
                break;
            }
            let allowed = if path_child == NONE { 0 } else { 1 };
            if n.ch.len() > allowed {
                reason = if path_child == NONE { 2 } else { 3 };
                break;
            }
            if n.last >= lookup_t {
                reason = 4;
                break;
            }
            if !n.demoted {
                flagged.push(node);
            }
            path_child = node;
            node = n.par;
        }
        let (mut fn_, mut fb) = (0u64, 0u64);
        for id in flagged {
            let n = &mut self.nodes[id as usize];
            n.demoted = true;
            fn_ += 1;
            fb += n.nb as u64;
            if !n.host_pinned() {
                self.push(id);
            }
        }
        self.stats.demote_flagged_nodes += fn_;
        self.stats.demote_flagged_blocks += fb;
        (fn_, fb, reason)
    }

    /// ORACLE O2 "perfect move" (NOT engine behaviour; bypasses the flag-then-evict-first path of
    /// peer_kv_demote.py :117-148 and the eviction order of unified_tree_core.py :1833-1835 @f8a08865b1):
    /// free the source copy of a transferred prefix at once. Read-only match of `runs` (the prefix the
    /// source still holds); the deepest matched node is taken whole if the prefix ends inside it and it
    /// has no children (the engine walk would stop with no_node). Walk parent links and stop like the
    /// engine walk where a node is shared or in use: another child (branch / children), a local access
    /// at or after `lookup_t`, no host copy, the root. Free every collected node deepest first (device
    /// and host copy); a pinned node (lock / lease / host lock) is flagged as demoted instead and ends
    /// the freeing. Returns (freed blocks, flagged-instead blocks).
    pub fn move_free(&mut self, runs: &[(u64, u32)], lookup_t: f64) -> (u64, u64) {
        let m = self.walk(runs, None, false);
        let Some(&(last_id, _)) = m.path.last() else { return (0, 0) };
        let mut node = last_id;
        let mut path_child = NONE;
        let mut col: Vec<NodeId> = Vec::new();
        while node != NONE && node != ROOT {
            let n = &self.nodes[node as usize];
            if !n.alive || !n.host {
                break;
            }
            let allowed = if path_child == NONE { 0 } else { 1 };
            if n.ch.len() > allowed || n.last >= lookup_t {
                break;
            }
            col.push(node);
            path_child = node;
            node = n.par;
        }
        let (mut freed, mut flagged) = (0u64, 0u64);
        for id in col {
            let n = &self.nodes[id as usize];
            if !n.alive || !n.ch.is_empty() {
                break;
            }
            if n.pinned() || n.host_pinned() {
                if !n.demoted {
                    let nb = n.nb as u64;
                    self.nodes[id as usize].demoted = true;
                    self.stats.demote_flagged_nodes += 1;
                    self.stats.demote_flagged_blocks += nb;
                    flagged += nb;
                }
                break;
            }
            freed += self.drop_node(id);
        }
        self.stats.moved_freed_blocks += freed;
        self.stats.moved_flagged_blocks += flagged;
        (freed, flagged)
    }

    /// Remove an unpinned leaf node from both tiers (oracle O2 only).
    fn drop_node(&mut self, id: NodeId) -> u64 {
        let (nb, par, key, dev, host) = {
            let n = &self.nodes[id as usize];
            (n.nb, n.par, n.runs[0].0, n.dev, n.host)
        };
        if dev {
            self.set_dev(id, false);
        }
        if host {
            self.host_used -= nb as u64;
        }
        if self.nodes[par as usize].ch.get(&key) == Some(&id) {
            self.nodes[par as usize].ch.remove(&key);
        }
        self.release(id);
        let p = &self.nodes[par as usize];
        if par != ROOT && p.ch.is_empty() && !p.pinned() {
            self.push(par);
        }
        nb as u64
    }

    /// Insert a run list (write-through: device + host) and touch its path. With
    /// `host_only` the new and existing-host-only nodes stay host-only (pull landing on host).
    /// Returns the path nodes.
    pub fn insert(&mut self, runs: &[(u64, u32)], t: f64, host_only: bool) -> Vec<NodeId> {
        let mut node = ROOT;
        let mut ri = 0usize;
        let mut off: u32 = 0;
        let mut path = Vec::new();
        while ri < runs.len() {
            let key = runs[ri].0 + off as u64;
            let ch = self.nodes[node as usize].ch.get(&key).copied();
            let Some(mut ch) = ch else {
                // New leaf with the remaining runs.
                let mut rest = Vec::with_capacity(runs.len() - ri);
                rest.push((runs[ri].0 + off as u64, runs[ri].1 - off));
                rest.extend_from_slice(&runs[ri + 1..]);
                let nb: u32 = rest.iter().map(|&(_, c)| c).sum();
                let mut leaf = Node::new(rest, nb, node);
                leaf.last = t;
                leaf.host = true;
                leaf.host_t = t;
                let id = self.alloc(leaf);
                self.nodes[node as usize].ch.insert(key, id);
                self.host_used += nb as u64;
                self.stats.host_inserted_blocks += nb as u64;
                if !host_only {
                    self.set_dev(id, true);
                }
                path.push(id);
                break;
            };
            let (m, ri2, off2) = lcp(&self.nodes[ch as usize].runs, runs, ri, off);
            let cnb = self.nodes[ch as usize].nb;
            if m < cnb {
                ch = self.split(node, key, ch, m, t);
            }
            {
                let c = &self.nodes[ch as usize];
                let (cdev, chost, cnb) = (c.dev, c.host, c.nb);
                if !chost {
                    self.nodes[ch as usize].host = true;
                    self.nodes[ch as usize].host_t = t;
                    self.host_used += cnb as u64;
                    self.stats.host_inserted_blocks += cnb as u64;
                }
                if !host_only && !cdev {
                    self.set_dev(ch, true);
                }
            }
            {
                let c = &mut self.nodes[ch as usize];
                c.last = t;
                if c.demoted {
                    c.demoted = false;
                    self.stats.demote_cleared_nodes += 1;
                    self.stats.demote_cleared_blocks += c.nb as u64;
                }
            }
            path.push(ch);
            node = ch;
            ri = ri2;
            off = off2;
        }
        for &n in &path {
            self.push(n);
        }
        self.evict_dev(t);
        path
    }

    /// Split `ch` (child of `parent` under `key`) after `m` blocks; returns the upper node.
    fn split(&mut self, parent: NodeId, key: u64, ch: NodeId, m: u32, t: f64) -> NodeId {
        let (a, b) = split_runs(&self.nodes[ch as usize].runs, m);
        let (cdev, chost, clast, clock, clease, chlock, cdem, chost_t) = {
            let c = &self.nodes[ch as usize];
            (c.dev, c.host, c.last, c.lock, c.lease, c.hlock, c.demoted, c.host_t)
        };
        let mut mid = Node::new(a, m, parent);
        mid.dev = cdev;
        mid.host = chost;
        mid.last = clast;
        mid.lock = clock;
        mid.lease = clease;
        // Path-wide pin counts: the new upper fragment lies on every pinned path through `ch`, so it
        // carries all three counts; else a later release would underflow it (permanent pin).
        mid.hlock = chlock;
        mid.demoted = cdem;
        mid.host_t = chost_t;
        mid.ndev = if cdev { 1 } else { 0 };
        let bkey = b[0].0;
        let mid_id = self.alloc(mid);
        {
            let c = &mut self.nodes[ch as usize];
            c.runs = b;
            c.nb -= m;
            c.par = mid_id;
        }
        self.nodes[mid_id as usize].ch.insert(bkey, ch);
        self.nodes[parent as usize].ch.insert(key, mid_id);
        // Engine: a new node is created with a fresh access time and the split child gets a fresh
        // time too (unified_tree_core.py:123 and 1285-1335 @831bac2694): both fragments are recent.
        if self.split_refresh {
            self.nodes[mid_id as usize].last = t;
            self.nodes[ch as usize].last = t;
            if !self.nodes[ch as usize].pinned() {
                self.push(ch);
            }
        }
        // Parent's device-child count is unchanged: `mid` replaces `ch` with the same residency.
        mid_id
    }

    /// Device eviction until within capacity.
    pub fn evict_dev(&mut self, t: f64) {
        while self.dev_used > self.dev_cap {
            let Some(k) = self.dh.pop() else { break };
            let id = k.node;
            let n = &self.nodes[id as usize];
            if !n.alive || !n.dev || n.ndev > 0 || n.pinned() || n.last != k.last || n.par == NONE
            {
                continue;
            }
            if self.class(id, false) != k.class {
                continue;
            }
            self.stats.dev_evicted_blocks += n.nb as u64;
            let par = n.par;
            let leaf = n.ch.is_empty();
            self.set_dev(id, false);
            if leaf {
                self.push(id);
            }
            let p = &self.nodes[par as usize];
            if par != ROOT && p.ndev == 0 && p.dev && !p.pinned() {
                self.push(par);
            }
        }
        if self.dh.len() as u64 > 8 * self.live + 10000 {
            self.rebuild();
        }
        self.evict_host(t);
    }

    /// Host eviction until within capacity: host-only, unpinned leaves; demoted first, then LRU.
    /// Host eviction; returns the remaining shortfall in blocks (0 when within capacity).
    pub fn evict_host(&mut self, t: f64) -> u64 {
        while self.host_used > self.host_cap {
            let Some(k) = self.hh.pop() else { break };
            let id = k.node;
            let n = &self.nodes[id as usize];
            if !n.alive
                || n.dev
                || !n.host
                || !n.ch.is_empty()
                || n.host_pinned()
                || n.last != k.last
                || n.par == NONE
            {
                continue;
            }
            if self.class(id, true) != k.class {
                continue;
            }
            let par = n.par;
            let nb = n.nb;
            let key = n.runs[0].0;
            let demoted = n.demoted;
            let age_last = (t - n.last).max(0.0);
            let age_write = (t - n.host_t).max(0.0);
            self.stats.host_evicted_blocks += nb as u64;
            self.stats.host_evicted_nodes += 1;
            self.stats.host_age_last_sum += age_last * nb as f64;
            self.stats.host_age_write_sum += age_write * nb as f64;
            if demoted {
                self.stats.demote_evicted_nodes += 1;
                self.stats.demote_evicted_blocks += nb as u64;
            }
            if self.record_evictions {
                self.evictions.push(HostEviction { t, blocks: nb, age_last, age_write, demoted, ns: key >> 32 });
            }
            if self.nodes[par as usize].ch.get(&key) == Some(&id) {
                self.nodes[par as usize].ch.remove(&key);
            }
            self.host_used -= nb as u64;
            self.release(id);
            let p = &self.nodes[par as usize];
            if par != ROOT && p.ch.is_empty() && !p.dev && !p.host_pinned() {
                self.push(par);
            }
        }
        if self.hh.len() as u64 > 8 * self.live + 10000 {
            self.rebuild();
        }
        let short = self.host_used.saturating_sub(self.host_cap);
        if short > 0 {
            self.stats.overcommit_events += 1;
            self.stats.overcommit_blocks += short;
        }
        short
    }

    fn rebuild(&mut self) {
        self.dh.clear();
        self.hh.clear();
        let mut st = vec![ROOT];
        while let Some(n) = st.pop() {
            let kids: Vec<NodeId> = self.nodes[n as usize].ch.values().copied().collect();
            st.extend(kids);
            if n != ROOT {
                self.push(n);
            }
        }
    }

    /// Host-pool census: used, pinned (locked or leased), device-resident (not host-evictable)
    /// and demoted blocks.
    pub fn census(&self) -> PoolCensus {
        let mut c = PoolCensus { host_used: self.host_used, ..Default::default() };
        let mut st = vec![ROOT];
        while let Some(n) = st.pop() {
            let node = &self.nodes[n as usize];
            st.extend(node.ch.values().copied());
            if n == ROOT || !node.host {
                continue;
            }
            let nb = node.nb as u64;
            if node.host_pinned() {
                c.pinned_host += nb;
            } else if node.dev {
                c.device_resident += nb;
            }
            if node.demoted {
                c.demoted_host += nb;
            }
        }
        c
    }

    /// Pin audit: expected path-wide lock / host-lock counts from the holders' anchor nodes vs the
    /// counts stored on every live node. Returns (lock mismatches, host-lock mismatches, nodes).
    pub fn pin_audit(&self, locks: &[NodeId], hlocks: &[NodeId]) -> (u64, u64, u64) {
        let mut el: FxHashMap<NodeId, u32> = FxHashMap::default();
        let mut eh: FxHashMap<NodeId, u32> = FxHashMap::default();
        for (anchors, map) in [(locks, &mut el), (hlocks, &mut eh)] {
            for &a in anchors {
                let mut n = a;
                while n != NONE && n != ROOT {
                    *map.entry(n).or_insert(0) += 1;
                    n = self.nodes[n as usize].par;
                }
            }
        }
        let (mut ml, mut mh, mut nodes) = (0u64, 0u64, 0u64);
        let mut st = vec![ROOT];
        while let Some(n) = st.pop() {
            let node = &self.nodes[n as usize];
            st.extend(node.ch.values().copied());
            if n == ROOT {
                continue;
            }
            nodes += 1;
            if node.lock != el.get(&n).copied().unwrap_or(0) {
                ml += 1;
            }
            if node.hlock != eh.get(&n).copied().unwrap_or(0) {
                mh += 1;
            }
        }
        (ml, mh, nodes)
    }

    /// Debug invariant check (tests).
    pub fn check(&self) -> Result<(), String> {
        let mut dev = 0u64;
        let mut host = 0u64;
        let mut st = vec![ROOT];
        while let Some(n) = st.pop() {
            let node = &self.nodes[n as usize];
            let mut nd = 0u32;
            for &c in node.ch.values() {
                let cn = &self.nodes[c as usize];
                if cn.par != n {
                    return Err(format!("parent link broken at {c}"));
                }
                if cn.dev {
                    nd += 1;
                    if !node.dev {
                        return Err(format!("device child {c} under non-device parent {n}"));
                    }
                }
                st.push(c);
            }
            if nd != node.ndev {
                return Err(format!("ndev mismatch at {n}: {} vs {nd}", node.ndev));
            }
            if n != ROOT {
                if node.dev {
                    dev += node.nb as u64;
                }
                if node.host {
                    host += node.nb as u64;
                }
                if !node.dev && !node.host {
                    return Err(format!("node {n} on no tier"));
                }
            }
        }
        if dev != self.dev_used || host != self.host_used {
            return Err(format!(
                "usage mismatch dev {dev}/{} host {host}/{}",
                self.dev_used, self.host_used
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(a: u64, n: u32) -> Vec<(u64, u32)> {
        vec![(a, n)]
    }

    #[test]
    fn write_through_and_device_lru_keeps_host_copy() {
        let mut c = TwoTierRadix::new(10, 100);
        c.insert(&runs(0, 6), 1.0, false);
        c.insert(&runs(100, 6), 2.0, false); // device over capacity -> evict LRU (block 0..6)
        c.check().unwrap();
        assert_eq!(c.dev_used, 6);
        assert_eq!(c.host_used, 12);
        let m = c.match_peek(&runs(0, 6));
        assert_eq!((m.dev, m.host), (0, 6));
        let m = c.match_peek(&runs(100, 6));
        assert_eq!((m.dev, m.host), (6, 0));
    }

    #[test]
    fn host_eviction_is_leaf_up_lru_by_last_local_use() {
        let mut c = TwoTierRadix::new(0, 12);
        // Chain A: 0..4 then 4..8 (two nodes after the split by a sibling).
        c.insert(&[(0, 8)], 1.0, false);
        c.insert(&[(0, 4), (50, 4)], 2.0, false); // splits A at 4: [0..4] -> {[4..8], [50..54]}
        c.check().unwrap();
        assert_eq!(c.host_used, 12);
        // Local use of [0..8] at t=3 refreshes the shared node and the A leaf.
        c.match_local(&[(0, 8)], 3.0);
        // A new chain forces eviction: the [50..54] leaf (last use 2.0) goes first.
        c.insert(&runs(200, 4), 4.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&[(0, 4), (50, 4)]).total(), 4);
        assert_eq!(c.match_peek(&[(0, 8)]).total(), 8);
        // Leaf-up: the shared parent [0..4] (last use 3.0) is not evicted while it has a
        // child; its child [4..8] goes first.
        c.insert(&runs(300, 4), 5.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&[(0, 8)]).total(), 4, "child evicted, parent kept");
        // Now the parent is a leaf and the LRU host-only node: evicted next.
        c.insert(&runs(400, 4), 6.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&[(0, 4)]).total(), 0, "parent evicted after its last child");
    }

    #[test]
    fn peek_does_not_refresh_the_source() {
        let mut c = TwoTierRadix::new(0, 8);
        c.insert(&runs(0, 4), 1.0, false);
        c.insert(&runs(10, 4), 2.0, false);
        // A peer pull reads chain 0 at t=3 (peek): no refresh.
        let m = c.match_peek(&runs(0, 4));
        assert_eq!(m.total(), 4);
        c.insert(&runs(20, 4), 4.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 4)).total(), 0, "peeked chain is still LRU");
        assert_eq!(c.match_peek(&runs(10, 4)).total(), 4);
    }

    #[test]
    fn lease_pins_source_until_release() {
        let mut c = TwoTierRadix::new(0, 8);
        c.insert(&runs(0, 4), 1.0, false);
        c.insert(&runs(10, 4), 2.0, false);
        let m = c.match_peek(&runs(0, 4));
        let deep = TwoTierRadix::path_node_covering(&m.path, 4);
        c.lease(deep, 1);
        c.insert(&runs(20, 4), 3.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 4)).total(), 4, "leased chain survives");
        assert_eq!(c.match_peek(&runs(10, 4)).total(), 0);
        c.lease(deep, -1);
        c.insert(&runs(30, 4), 4.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 4)).total(), 0, "released chain evictable again");
    }

    #[test]
    fn demote_flag_evicts_first_and_local_match_clears_it() {
        let mut c = TwoTierRadix::new(0, 12);
        c.insert(&runs(0, 4), 1.0, false);
        c.insert(&runs(10, 4), 2.0, false);
        c.insert(&runs(20, 4), 3.0, false);
        // Demote the most recent chain (20): it is evicted before the LRU chain (0).
        let m = c.match_peek(&runs(20, 4));
        let (n, b) = c.demote_below_branch(&m.path);
        assert_eq!((n, b), (1, 4));
        c.insert(&runs(30, 4), 4.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(20, 4)).total(), 0, "demoted chain evicted first");
        assert_eq!(c.match_peek(&runs(0, 4)).total(), 4);
        assert_eq!(c.stats.demote_evicted_blocks, 4);
        // Clear on local match: demote chain 10, then use it locally -> plain LRU again.
        let m = c.match_peek(&runs(10, 4));
        c.demote_below_branch(&m.path);
        c.match_local(&runs(10, 4), 5.0);
        assert_eq!(c.stats.demote_cleared_blocks, 4);
        c.insert(&runs(40, 4), 6.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 4)).total(), 0, "LRU chain evicted");
        assert_eq!(c.match_peek(&runs(10, 4)).total(), 4, "cleared chain kept");
    }

    #[test]
    fn match_split_leaves_the_unmatched_rest_aging_and_demoted() {
        for split in [false, true] {
            let mut c = TwoTierRadix::new(0, 100);
            c.split_on_match = split;
            c.insert(&runs(0, 10), 1.0, false);
            let m = c.match_peek(&runs(0, 10));
            c.demote_below_branch(&m.path);
            // A local match of the first 4 blocks only (diverges at block 4).
            let m = c.match_local(&[(0, 4), (500, 2)], 2.0);
            assert_eq!(m.total(), 4);
            assert_eq!(c.stats.partial_local_matches, 1);
            assert_eq!(c.stats.partial_rest_blocks, 6);
            c.match_local(&[(0, 4), (600, 2)], 3.0);
            c.check().unwrap();
            let p = c.match_peek(&runs(0, 10)).path;
            if split {
                assert_eq!(p.len(), 2, "split at the match end");
                assert_eq!((c.node(p[0].0).nb, c.node(p[1].0).nb), (4, 6));
                assert_eq!(c.node(p[0].0).last, 3.0);
                assert!(!c.node(p[0].0).demoted, "matched fragment cleared");
                assert_eq!(c.node(p[1].0).last, 2.0, "rest refreshed only at the split");
                assert!(c.node(p[1].0).demoted, "rest keeps the demote flag");
                assert_eq!(c.stats.match_splits, 1);
            } else {
                assert_eq!(p.len(), 1);
                assert_eq!(c.node(p[0].0).last, 3.0);
                assert!(!c.node(p[0].0).demoted);
            }
        }
    }

    #[test]
    fn oracle_move_frees_the_private_suffix_and_keeps_shared_or_pinned_nodes() {
        let mut c = TwoTierRadix::new(100, 100);
        c.insert(&[(0, 4), (100, 4)], 1.0, false);
        c.insert(&[(0, 4), (200, 6)], 2.0, false); // [0..4] is a branch point
        let used = c.host_used;
        // Move the [0..4]+[200..206] prefix: only the private suffix goes, the shared head stays.
        let (freed, flagged) = c.move_free(&[(0, 4), (200, 6)], 3.0);
        assert_eq!((freed, flagged), (6, 0));
        assert_eq!(c.host_used, used - 6);
        c.check().unwrap();
        assert_eq!(c.match_peek(&[(0, 4), (200, 6)]).total(), 4);
        assert_eq!(c.match_peek(&[(0, 4), (100, 4)]).total(), 8);
        // A prefix ending inside a childless node takes the whole node (no_node in the engine walk).
        let (freed, _) = c.move_free(&[(0, 4), (100, 2)], 3.0);
        assert_eq!(freed, 4 + 4, "suffix node whole, then the head (now a leaf)");
        c.check().unwrap();
        // Pinned node: flagged, not freed; accessed after the lookup: kept.
        c.insert(&runs(300, 4), 4.0, false);
        let m = c.match_peek(&runs(300, 4));
        let deep = TwoTierRadix::path_node_covering(&m.path, 4);
        c.lock(deep, 1);
        assert_eq!(c.move_free(&runs(300, 4), 5.0), (0, 4));
        c.lock(deep, -1);
        c.insert(&runs(400, 4), 6.0, false);
        assert_eq!(c.move_free(&runs(400, 4), 5.0), (0, 0), "used at or after the lookup");
        c.check().unwrap();
    }

    #[test]
    fn demote_spares_shared_prefix_above_the_branch_point() {
        let mut c = TwoTierRadix::new(0, 100);
        c.insert(&[(0, 4), (100, 4)], 1.0, false);
        c.insert(&[(0, 4), (200, 4)], 2.0, false); // [0..4] is a branch point
        let m = c.match_peek(&[(0, 4), (200, 4)]);
        let (n, b) = c.demote_below_branch(&m.path);
        assert_eq!((n, b), (1, 4), "only the private suffix is flagged");
        assert!(!c.node(m.path[0].0).demoted);
        assert!(c.node(m.path[1].0).demoted);
    }

    #[test]
    fn demote_walk_follows_the_engine_stop_rules() {
        let mut c = TwoTierRadix::new(0, 1000);
        // Session A: [0..4] -> [4..8]; session B shares nothing.
        c.insert(&[(0, 8)], 1.0, false);
        c.insert(&[(0, 4), (50, 4)], 2.0, false); // branch at [0..4]
        // Lease of A's full path, lookup at t=3: the leaf [4..8] is flagged, the walk stops at the branch.
        let m = c.match_peek(&[(0, 8)]);
        let (n, b, why) = c.demote_walk(&m.path, 3.0);
        assert_eq!((n, b, why), (1, 4, 3));
        // Accessed after the lookup -> stop "access", nothing flagged.
        let mut c2 = TwoTierRadix::new(0, 1000);
        c2.insert(&[(0, 8)], 1.0, false);
        let m = c2.match_peek(&[(0, 8)]);
        c2.match_local(&[(0, 8)], 5.0);
        assert_eq!(c2.demote_walk(&m.path, 4.0).2, 4);
        // Deepest leased node with children -> stop "children".
        let mut c3 = TwoTierRadix::new(0, 1000);
        c3.insert(&[(0, 8)], 1.0, false);
        c3.insert(&[(0, 8), (100, 4)], 1.5, false);
        let m = c3.match_peek(&[(0, 8)]);
        assert_eq!(c3.demote_walk(&m.path, 2.0).2, 2);
        // A lease ending inside a node -> "no_node".
        let m = c3.match_peek(&[(0, 8), (100, 2)]);
        assert_eq!(c3.demote_walk(&m.path, 2.0).2, 0);
        // No branch: the whole chain up to the root is flagged.
        let mut c4 = TwoTierRadix::new(0, 1000);
        c4.insert(&[(0, 8)], 1.0, false);
        let m = c4.match_peek(&[(0, 8)]);
        let (n, b, why) = c4.demote_walk(&m.path, 2.0);
        assert_eq!((n, b, why), (1, 8, 5));
        // A later insert walk (touch) clears the flag.
        c4.insert(&[(0, 8)], 3.0, false);
        assert_eq!(c4.stats.demote_cleared_blocks, 8);
    }

    #[test]
    fn host_lock_blocks_host_eviction_but_not_device_eviction() {
        let mut c = TwoTierRadix::new(4, 8);
        let p = c.insert(&runs(0, 4), 1.0, false);
        c.host_lock(p.last().copied(), 1);
        c.insert(&runs(10, 4), 2.0, false); // device over cap -> chain 0 demoted to host-only
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 4)).dev, 0, "host lock does not pin the device copy");
        c.insert(&runs(20, 4), 3.0, false); // host over cap -> chain 10 (unlocked) goes, chain 0 stays
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 4)).total(), 4);
        c.host_lock(p.last().copied(), -1);
        c.insert(&runs(30, 4), 4.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 4)).total(), 0);
    }

    #[test]
    fn demote_from_walks_parents_at_release_time() {
        // Leased prefix = blocks 0..8 of one chain (lookup at t = 2).
        let mut c = TwoTierRadix::new(100, 100);
        c.insert(&runs(0, 8), 1.0, false);
        // A split between lookup and release creates a branch point at block 4 (other session
        // part inserted at t = 1.5, before the lookup, so no access after the lookup on 0..4
        // except the touch of the shared upper part at 1.5).
        c.insert(&[(0, 4), (500, 2)], 1.5, false);
        let (n, b, why) = c.demote_from(&runs(0, 8), 2.0);
        assert_eq!((n, b, why), (1, 4, 3), "only the lower fragment; stop at the new branch point");
        // Without a branch the walk reaches the root and flags the whole path.
        let mut d = TwoTierRadix::new(100, 100);
        d.insert(&runs(0, 8), 1.0, false);
        assert_eq!(d.demote_from(&runs(0, 8), 2.0), (1, 8, 5));
        // A lease that ends inside a node flags nothing.
        let mut e = TwoTierRadix::new(100, 100);
        e.insert(&runs(0, 8), 1.0, false);
        assert_eq!(e.demote_from(&runs(0, 6), 2.0), (0, 0, 0));
        // An access after the lookup stops the walk.
        let mut f = TwoTierRadix::new(100, 100);
        f.insert(&runs(0, 8), 3.0, false);
        assert_eq!(f.demote_from(&runs(0, 8), 2.0).2, 4);
    }

    #[test]
    fn split_keeps_pin_counts_and_release_leaves_no_pin() {
        let mut c = TwoTierRadix::new(100, 100);
        let p = c.insert(&runs(0, 8), 1.0, false);
        let deep = p.last().copied();
        c.host_lock(deep, 1);
        c.lock(deep, 1);
        // A diverging insert splits the locked node at block 4.
        c.insert(&[(0, 4), (500, 4)], 2.0, false);
        c.check().unwrap();
        c.host_lock(deep, -1);
        c.lock(deep, -1);
        let cen = c.census();
        assert_eq!(cen.pinned_host, 0, "no pin may survive the release after a split");
    }

    #[test]
    fn load_back_moves_host_nodes_to_device() {
        let mut c = TwoTierRadix::new(8, 100);
        c.insert(&runs(0, 8), 1.0, false);
        c.insert(&runs(100, 8), 2.0, false);
        let m = c.match_local(&runs(0, 8), 3.0);
        assert_eq!((m.dev, m.host), (0, 8));
        let loaded = c.load_back(&m.path, 3.0);
        assert_eq!(loaded, 8);
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 8)).dev, 8);
        assert_eq!(c.match_peek(&runs(100, 8)).dev, 0);
    }

    #[test]
    fn host_only_landing_does_not_use_device() {
        let mut c = TwoTierRadix::new(8, 100);
        c.insert(&runs(0, 8), 1.0, true);
        c.check().unwrap();
        assert_eq!(c.dev_used, 0);
        assert_eq!(c.host_used, 8);
        let m = c.match_peek(&runs(0, 8));
        assert_eq!((m.dev, m.host), (0, 8));
    }

    #[test]
    fn locked_path_is_not_evicted_on_either_tier() {
        let mut c = TwoTierRadix::new(4, 4);
        let p = c.insert(&runs(0, 4), 1.0, false);
        c.lock(p.last().copied(), 1);
        c.insert(&runs(10, 4), 2.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 4)).dev, 4);
        c.lock(p.last().copied(), -1);
        c.insert(&runs(20, 4), 3.0, false);
        c.check().unwrap();
        assert_eq!(c.match_peek(&runs(0, 4)).total(), 0);
    }
}
