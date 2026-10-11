// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event loop of the HiCache + P2P P/D topology runtime.
//!
//! Request path: client send -> router (P policy, D policy at arrival) -> P recv / D recv
//! (ingress) -> D FIFO pre-allocation -> P bootstrap ready at max(P recv, D alloc) + handshake
//! [-> peer pull] -> P FCFS waiting queue -> chunked prefill (continuing chunk first, then FIFO
//! adds; one formed batch of look-ahead) -> prefill done (+ fixed latency) -> KV tail -> D first
//! step -> decode at TPOT(B, S) -> client completion.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use super::cache::{DemoteScope, MatchResult, NodeId, TwoTierRadix, run_blocks, runs_prefix};
use super::trie::{Event, PyRandom, Workload};
use crate::replay::report::percentile_in_place;

/// Linear fit `a + b * tokens / 1e5` (seconds).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinFit {
    pub a: f64,
    pub b_per_100k: f64,
}

impl LinFit {
    fn at(&self, tokens: u64) -> f64 {
        (self.a + self.b_per_100k * tokens as f64 / 1e5).max(0.0)
    }
}

/// Simulation configuration (JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SimConfig {
    pub label: String,
    pub seed: u64,
    /// Seed of the trie client's warm-start draws (`random.Random(client_seed)`).
    pub client_seed: u64,
    pub n_p: usize,
    pub n_d: usize,
    pub concurrency: usize,
    pub duration_s: f64,
    pub warm_s: f64,
    pub score_s: f64,
    pub wait_cap_s: f64,
    pub block_tokens: u64,
    pub chunk_tokens: u64,
    /// Client think/route lag before the router sees a request (s).
    pub client_lag_s: f64,
    // ---- P cache
    pub p_dev_tokens: u64,
    pub p_host_tokens: u64,
    pub demote_scope_device: bool,
    // ---- P timing: per-P [c0 s, c1 s per 1k new tokens, c2 s per 1e9 pairs, c3 s per 100k host tokens, c4 s per 100k pulled tokens]
    pub p_beta: Vec<[f64; 5]>,
    /// Multiplicative per-iteration noise: lognormal sigma (0 = none).
    pub p_noise_sigma: f64,
    /// Fixed non-GPU latency per request before prefill done (s).
    pub p_lat_s: f64,
    /// Fraction of host load-back cost hidden behind the bootstrap wait (bootstrap prefetch).
    pub lb_hidden: f64,
    /// Max tokens the bootstrap prefetch may cover (262144) and fraction of the prompt (0.66).
    pub prefetch_max_tokens: u64,
    pub prefetch_max_fraction: f64,
    // ---- network / bootstrap / tails
    pub ing_p: LinFit,
    pub ing_d: LinFit,
    pub handshake_s: f64,
    pub kv_tail: LinFit,
    pub d_first_s: f64,
    // ---- D
    pub d_cap_tokens: u64,
    /// D reservation = prompt + min(max_tokens, d_reserve_out).
    pub d_reserve_out: u64,
    pub tpot_s: f64,
    pub step_ref_ms: f64,
    pub tpot_fb: bool,
    // ---- router P
    /// "smgca" | "cosched" | "sticky" | "rr"
    pub p_policy: String,
    pub smg_threshold: f64,
    pub smg_abs: i64,
    pub smg_rel: f64,
    pub smg_tree_tokens: u64,
    pub smg_evict_interval_s: f64,
    /// SMG low-match target: "smallest_tenant" (SMG code) or "min_load".
    pub smg_low_match: String,
    /// SMG load decrement point: "prefill_done" or "request_done".
    pub smg_load_release: String,
    pub cs_margin_tokens: u64,
    pub cs_load_poll_s: f64,
    pub pull_min_tokens: u64,
    pub pull_min_incremental_tokens: u64,
    pub pull_enabled: bool,
    // ---- router D
    /// "po2" | "rr" | "d1"
    pub d_policy: String,
    /// power_of_two load: "requests" (router in-flight) or "snapshot" (refreshed every d_load_refresh_s)
    pub d_load_mode: String,
    pub d_load_refresh_s: f64,
    // ---- pulls
    pub pull_fit: LinFit,
    pub pull_check_s: f64,
    /// "p_recv" (pull starts when P receives the request) or "queue" (JIT: when the request enters the P queue)
    pub pull_start: String,
    /// "host" (pulled KV lands on the destination host tier) or "device"
    pub pull_landing: String,
    /// Re-pull at admission when the pulled range was evicted before the first chunk (fix 2).
    pub repull_on_admission: bool,
    /// Lock the matched prefix of queued requests (fix 3).
    pub queued_prefix_protect: bool,
    pub demote_source: bool,
    pub record_rows: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            label: String::new(),
            seed: 0,
            client_seed: 0,
            n_p: 3,
            n_d: 6,
            concurrency: 90,
            duration_s: 5400.0,
            warm_s: 1800.0,
            score_s: 3600.0,
            wait_cap_s: 120.0,
            block_tokens: 64,
            chunk_tokens: 8192,
            client_lag_s: 0.0,
            p_dev_tokens: 2_067_584,
            p_host_tokens: 12_405_504,
            demote_scope_device: false,
            p_beta: vec![[0.08, 0.06, 0.0, 0.05, 0.05]],
            p_noise_sigma: 0.0,
            p_lat_s: 0.3,
            lb_hidden: 0.0,
            prefetch_max_tokens: 262_144,
            prefetch_max_fraction: 0.66,
            ing_p: LinFit { a: 0.01, b_per_100k: 0.0 },
            ing_d: LinFit { a: 0.01, b_per_100k: 0.0 },
            handshake_s: 0.05,
            kv_tail: LinFit { a: 0.1, b_per_100k: 0.1 },
            d_first_s: 0.05,
            d_cap_tokens: 2_278_656,
            d_reserve_out: 8192,
            tpot_s: 0.0054,
            step_ref_ms: 23.3,
            tpot_fb: true,
            p_policy: "smgca".into(),
            smg_threshold: 0.3,
            smg_abs: 64,
            smg_rel: 1.5,
            smg_tree_tokens: 99_244_544,
            smg_evict_interval_s: 120.0,
            smg_low_match: "smallest_tenant".into(),
            smg_load_release: "prefill_done".into(),
            cs_margin_tokens: 32768,
            cs_load_poll_s: 0.1,
            pull_min_tokens: 256,
            pull_min_incremental_tokens: 256,
            pull_enabled: true,
            d_policy: "po2".into(),
            d_load_mode: "requests".into(),
            d_load_refresh_s: 5.0,
            pull_fit: LinFit { a: 0.257, b_per_100k: 0.244 },
            pull_check_s: 0.1,
            pull_start: "p_recv".into(),
            pull_landing: "host".into(),
            repull_on_admission: false,
            queued_prefix_protect: false,
            demote_source: false,
            record_rows: true,
        }
    }
}

// ------------------------------------------------------------------------------------ events
#[derive(Debug, Clone, Copy)]
enum Ev {
    AgentStep(u32),
    WaitDone(u32, u32),
    SendAt(u32),
    Route(u32),
    PRecv(u32),
    DRecv(u32),
    PReady(u32),
    PullDone(u32),
    EndIter(u16),
    PrefillDone(u16, u32),
    KvArrived(u32),
    DecJoin(u32),
    DecTick(u16, u64),
    ReqDone(u32),
    PollTick,
    DLoadTick,
    SmgEvict,
}

struct Item {
    t: f64,
    seq: u64,
    ev: Ev,
}

impl PartialEq for Item {
    fn eq(&self, o: &Self) -> bool {
        self.t == o.t && self.seq == o.seq
    }
}
impl Eq for Item {}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.t.total_cmp(&self.t).then_with(|| o.seq.cmp(&self.seq))
    }
}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

// ------------------------------------------------------------------------------------ entities
#[derive(Debug, Clone, Default)]
struct Req {
    task: u32,
    agent_state: u32,
    idx: u32,
    warm: bool,
    dec_lb: f64,
    runs: Vec<(u64, u32)>,
    pt: u64,
    #[allow(dead_code)]
    maxt: u32,
    ct: u32,
    need: u64,
    admit: f64,
    start: f64,
    p: u16,
    d: u16,
    p_recv: f64,
    d_recv: f64,
    t_alloc: Option<f64>,
    p_in: bool,
    p_ready: Option<f64>,
    // pulls
    pull_src: Option<u16>,
    pull_hint: u64,
    pulled: u64,
    pull_start_t: Option<f64>,
    pull_end: Option<f64>,
    pull_lease: Option<NodeId>,
    pull_dst_lock: Option<NodeId>,
    pull_src_path: Vec<(NodeId, u32)>,
    pull_dst_have: u64,
    boot_ready: Option<f64>,
    pull_pending: bool,
    // prefill
    t_fwd: Option<f64>,
    t_pd: Option<f64>,
    t_kv: Option<f64>,
    first: Option<f64>,
    done: Option<f64>,
    cancel: bool,
    dev: u64,
    host: u64,
    stor: u64,
    unc: u64,
    sched: u64,
    computed: u64,
    lbcost: f64,
    locked: Option<NodeId>,
    qlock: Option<NodeId>,
    unc_est: u64,
    ideal: u64,
    // routing / gap cause
    owner: u16,
    detour: bool,
    route_best: u64,
    route_tgt: u64,
    fc_best: u64,
    nin_held: bool,
    d_counted: bool,
    pa_on: bool,
}

#[derive(Debug, Clone)]
struct TaskInst {
    row: u32,
    inst: u64,
    cut: Option<f64>,
    comp: FxHashSet<(u32, u32)>,
    waiters: FxHashMap<(u32, u32), Vec<u32>>,
    inc: FxHashMap<(u32, u32), Vec<(u32, u32)>>,
    left: u32,
}

#[derive(Debug, Clone)]
struct AgentState {
    task: u32,
    agent: u32,
    idx: u32,
    cutting: bool,
    pend: f64,
    warm: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct BItem {
    r: u32,
    n: u64,
    pre: u64,
}

struct PW {
    beta: [f64; 5],
    radix: TwoTierRadix,
    wq: VecDeque<u32>,
    chunked: Option<u32>,
    run_items: Option<Vec<BItem>>,
    next_items: Option<Vec<BItem>>,
    recent: VecDeque<(f64, u64)>,
    snap: u64,
    snap_t: f64,
    nin: i64,
    busy_s: f64,
    win_busy: f64,
    host_ins_win: u64,
}

struct DW {
    cap: u64,
    used: u64,
    fifo: VecDeque<u32>,
    nreq: i64,
    load_snap: i64,
    b: i64,
    s: u64,
    v: f64,
    vt: f64,
    rate: f64,
    dh: BinaryHeap<std::cmp::Reverse<(u64, u64, u32)>>,
    ver: u64,
    hseq: u64,
}

/// Aggregate output.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SimReport {
    pub label: String,
    pub seed: u64,
    pub n_scored: usize,
    pub n_admitted_window: usize,
    pub rps: f64,
    pub hit_sum_a_over_i: f64,
    pub hit_actual_over_prompt: f64,
    pub ideal_over_prompt: f64,
    pub dev_share: f64,
    pub host_share: f64,
    pub pulled_share: f64,
    pub zero_hit_row_share: f64,
    pub zero_hit_row_share_of_ideal_rows: f64,
    pub zero_hit_deficit_share: f64,
    pub deficit_tokens: f64,
    pub e_share_route: f64,
    pub r_share_route: f64,
    pub pull_short_share_route: f64,
    pub other_share_route: f64,
    pub e_share_fc: f64,
    pub ttft: [f64; 5],
    pub e2e: [f64; 2],
    pub e2e_mean: f64,
    pub ttft_mean: f64,
    pub p_queue_mean: f64,
    pub d_wait_mean: f64,
    pub prefill_mean: f64,
    pub decode_mean: f64,
    pub pulls_scored: u64,
    pub pulled_tokens_scored: u64,
    pub pulls_all: u64,
    pub pulled_tokens_all: u64,
    pub pull_short_all: u64,
    pub detour_share: f64,
    pub per_p_host_retention_cap_over_inflow_s: Vec<f64>,
    pub per_p_host_age_last_s: Vec<f64>,
    pub per_p_host_age_write_s: Vec<f64>,
    pub per_p_host_age_last_p50_s: Vec<f64>,
    pub per_p_busy: Vec<f64>,
    pub per_p_share: Vec<f64>,
    pub per_p_host_inserted_tokens_win: Vec<u64>,
    pub per_p_host_evicted_tokens: Vec<u64>,
    pub per_p_loaded_back_tokens: Vec<u64>,
    pub demote_flagged_nodes: u64,
    pub demote_flagged_tokens: u64,
    pub demote_evicted_nodes: u64,
    pub demote_evicted_tokens: u64,
    pub demote_cleared_nodes: u64,
    pub demote_cleared_tokens: u64,
    pub tasks_started: u64,
    pub requests_sent: u64,
    pub host_fill: Vec<f64>,
    pub dev_fill: Vec<f64>,
    pub wall_s: f64,
}

/// One scored request row (optional CSV output).
#[derive(Debug, Clone, Serialize)]
pub struct Row {
    pub task_row: u32,
    pub agent: u32,
    pub idx: u32,
    pub admit: f64,
    pub p: u16,
    pub d: u16,
    pub pt: u64,
    pub actual: u64,
    pub ideal: u64,
    pub dev: u64,
    pub host: u64,
    pub pulled: u64,
    pub route_best: u64,
    pub route_tgt: u64,
    pub fc_best: u64,
    pub pull_hint: u64,
    pub detour: bool,
    pub ttft: f64,
    pub e2e: f64,
    pub ct: u32,
    pub completed: bool,
}

pub struct Sim<'a> {
    c: SimConfig,
    wl: &'a Workload,
    ct_map: &'a FxHashMap<(u32, u32, u32), u32>,
    heap: BinaryHeap<Item>,
    seq: u64,
    now: f64,
    p: Vec<PW>,
    d: Vec<DW>,
    reqs: Vec<Req>,
    tasks: Vec<TaskInst>,
    agents: Vec<AgentState>,
    next_row: usize,
    tasks_started: u64,
    inst: u64,
    crng: PyRandom,
    rng: PyRandom,
    ideal: TwoTierRadix,
    smg: Vec<TwoTierRadix>,
    sticky: FxHashMap<(u64, String), u16>,
    rr: usize,
    drr: usize,
    pulls_all: u64,
    pulled_all: u64,
    pull_short_all: u64,
    lo: f64,
    hi: f64,
}

fn step_ms(b: i64, s_tok: u64) -> f64 {
    let lk = (1.0f64 - 8.0 / 256.0).ln();
    let b = b as f64;
    9.71 + 0.467 * b + 10.59 * (1.0 - (6.0 * b * lk).exp()) + 0.996 * s_tok as f64 / 1e6
}

impl<'a> Sim<'a> {
    pub fn new(c: SimConfig, wl: &'a Workload, ct_map: &'a FxHashMap<(u32, u32, u32), u32>) -> Self {
        let bt = c.block_tokens;
        let mut p = Vec::new();
        for i in 0..c.n_p {
            let beta = c.p_beta[i % c.p_beta.len()];
            let mut radix = TwoTierRadix::new(c.p_dev_tokens / bt, c.p_host_tokens / bt);
            radix.demote_scope =
                if c.demote_scope_device { DemoteScope::DeviceAndHost } else { DemoteScope::Host };
            radix.record_evictions = true;
            p.push(PW {
                beta,
                radix,
                wq: VecDeque::new(),
                chunked: None,
                run_items: None,
                next_items: None,
                recent: VecDeque::new(),
                snap: 0,
                snap_t: -1.0,
                nin: 0,
                busy_s: 0.0,
                win_busy: 0.0,
                host_ins_win: 0,
            });
        }
        let d = (0..c.n_d)
            .map(|_| DW {
                cap: c.d_cap_tokens,
                used: 0,
                fifo: VecDeque::new(),
                nreq: 0,
                load_snap: 0,
                b: 0,
                s: 0,
                v: 0.0,
                vt: 0.0,
                rate: 0.0,
                dh: BinaryHeap::new(),
                ver: 0,
                hseq: 0,
            })
            .collect();
        let smg = (0..c.n_p).map(|_| TwoTierRadix::new(u64::MAX / 4, u64::MAX / 4)).collect();
        let lo = c.warm_s;
        let hi = c.warm_s + c.score_s;
        Self {
            crng: PyRandom::new(c.client_seed),
            rng: PyRandom::new(0x5eed_0000 + c.seed),
            c,
            wl,
            ct_map,
            heap: BinaryHeap::new(),
            seq: 0,
            now: 0.0,
            p,
            d,
            reqs: Vec::new(),
            tasks: Vec::new(),
            agents: Vec::new(),
            next_row: 0,
            tasks_started: 0,
            inst: 0,
            ideal: TwoTierRadix::new(u64::MAX / 4, u64::MAX / 4),
            smg,
            sticky: FxHashMap::default(),
            rr: 0,
            drr: 0,
            pulls_all: 0,
            pulled_all: 0,
            pull_short_all: 0,
            lo,
            hi,
        }
    }

    fn at(&mut self, t: f64, ev: Ev) {
        self.seq += 1;
        self.heap.push(Item { t, seq: self.seq, ev });
    }

    fn urand(&mut self) -> f64 {
        self.rng.random()
    }

    fn urange(&mut self, n: usize) -> usize {
        self.rng.randbelow(n as u32) as usize
    }

    pub fn run(&mut self) {
        for _ in 0..self.c.concurrency {
            self.start_task(0.0);
        }
        if self.c.p_policy == "cosched" {
            self.at(0.0, Ev::PollTick);
        }
        if self.c.d_load_mode == "snapshot" {
            self.at(0.0, Ev::DLoadTick);
        }
        if self.c.p_policy == "smgca" && self.c.smg_evict_interval_s > 0.0 {
            self.at(self.c.smg_evict_interval_s, Ev::SmgEvict);
        }
        while let Some(it) = self.heap.pop() {
            if it.t > self.c.duration_s + 1.0 {
                break;
            }
            self.now = it.t;
            let t = it.t;
            match it.ev {
                Ev::AgentStep(a) => self.agent_step(t, a),
                Ev::WaitDone(a, i) => {
                    let task = self.agents[a as usize].task;
                    let agent = self.agents[a as usize].agent;
                    self.complete(task, agent, i, t);
                    self.agents[a as usize].idx += 1;
                    self.agent_step(t, a);
                }
                Ev::SendAt(a) => {
                    let i = self.agents[a as usize].idx;
                    self.send(t, a, i, false, 0.0);
                }
                Ev::Route(r) => self.route(t, r),
                Ev::PRecv(r) => self.on_p_recv(t, r),
                Ev::DRecv(r) => {
                    let d = self.reqs[r as usize].d as usize;
                    self.d[d].fifo.push_back(r);
                    self.d_try_admit(t, d);
                }
                Ev::PReady(r) => self.p_ready(t, r),
                Ev::PullDone(r) => self.pull_done(t, r),
                Ev::EndIter(p) => self.end_iter(t, p as usize),
                Ev::PrefillDone(p, r) => self.prefill_done(t, p as usize, r),
                Ev::KvArrived(r) => self.kv_arrived(t, r),
                Ev::DecJoin(r) => self.dec_join(t, r),
                Ev::DecTick(d, ver) => self.dec_tick(t, d as usize, ver),
                Ev::ReqDone(r) => self.req_done(t, r),
                Ev::PollTick => {
                    for p in &mut self.p {
                        p.snap = Self::backlog_of(p, &self.reqs);
                        p.snap_t = t;
                    }
                    if t < self.c.duration_s {
                        let dt = self.c.cs_load_poll_s;
                        self.at(t + dt, Ev::PollTick);
                    }
                }
                Ev::DLoadTick => {
                    for d in &mut self.d {
                        d.load_snap = d.nreq;
                    }
                    if t < self.c.duration_s {
                        let dt = self.c.d_load_refresh_s;
                        self.at(t + dt, Ev::DLoadTick);
                    }
                }
                Ev::SmgEvict => {
                    let lim = self.c.smg_tree_tokens / self.c.block_tokens;
                    for tree in &mut self.smg {
                        tree.dev_cap = lim;
                        tree.host_cap = lim;
                        tree.evict_dev(t);
                        tree.dev_cap = u64::MAX / 4;
                        tree.host_cap = u64::MAX / 4;
                    }
                    if t < self.c.duration_s {
                        let dt = self.c.smg_evict_interval_s;
                        self.at(t + dt, Ev::SmgEvict);
                    }
                }
            }
        }
        let dur = self.c.duration_s;
        for r in &mut self.reqs {
            if r.done.is_none() || r.done.unwrap() > dur {
                r.cancel = true;
            }
        }
    }

    // ------------------------------------------------------------------------------ client
    fn start_task(&mut self, t: f64) {
        if t >= self.c.duration_s || self.wl.tasks.is_empty() {
            return;
        }
        let row = self.next_row % self.wl.tasks.len();
        self.next_row += 1;
        self.inst += 1;
        let task = &self.wl.tasks[row];
        let mut cut = None;
        if (self.tasks_started as usize) < self.c.concurrency {
            let draw = self.crng.random();
            let mut st = f64::INFINITY;
            let mut en = f64::NEG_INFINITY;
            for a in &task.agents {
                for e in &a.events {
                    if let Event::Generate { src_t: Some(s), src_api, .. } = e {
                        st = st.min(*s);
                        en = en.max(*s + src_api.unwrap_or(0.0));
                    }
                }
            }
            if st.is_finite() {
                cut = Some(st + draw * (en - st));
            }
        }
        self.tasks_started += 1;
        let mut inc: FxHashMap<(u32, u32), Vec<(u32, u32)>> = FxHashMap::default();
        for &(ba, bi, aa, ai) in &task.deps {
            inc.entry((aa as u32, ai as u32)).or_default().push((ba as u32, bi as u32));
        }
        let ti = self.tasks.len() as u32;
        let nag = task.agents.len();
        self.tasks.push(TaskInst {
            row: row as u32,
            inst: self.inst,
            cut,
            comp: FxHashSet::default(),
            waiters: FxHashMap::default(),
            inc,
            left: nag as u32,
        });
        for ag in 0..nag {
            let a = self.agents.len() as u32;
            self.agents.push(AgentState {
                task: ti,
                agent: ag as u32,
                idx: 0,
                cutting: cut.is_some(),
                pend: 0.0,
                warm: None,
            });
            self.at(t, Ev::AgentStep(a));
        }
    }

    fn agent_step(&mut self, t: f64, a: u32) {
        loop {
            let (task, agent, i) = {
                let s = &self.agents[a as usize];
                (s.task, s.agent, s.idx)
            };
            let row = self.tasks[task as usize].row as usize;
            let evs = &self.wl.tasks[row].agents[agent as usize].events;
            if i as usize >= evs.len() {
                let ts = &mut self.tasks[task as usize];
                ts.left -= 1;
                if ts.left == 0 {
                    self.start_task(t);
                }
                return;
            }
            if let Some(deps) = self.tasks[task as usize].inc.get(&(agent, i)) {
                let deps = deps.clone();
                for dep in deps {
                    if !self.tasks[task as usize].comp.contains(&dep) {
                        self.tasks[task as usize].waiters.entry(dep).or_default().push(a);
                        return;
                    }
                }
            }
            let e = evs[i as usize].clone();
            match e {
                Event::Wait { seconds } => {
                    let d = seconds.min(self.c.wait_cap_s);
                    if self.agents[a as usize].cutting {
                        self.agents[a as usize].pend = d;
                        self.complete(task, agent, i, t);
                        self.agents[a as usize].idx += 1;
                        continue;
                    }
                    self.at(t + d, Ev::WaitDone(a, i));
                    return;
                }
                Event::Other => {
                    self.complete(task, agent, i, t);
                    self.agents[a as usize].idx += 1;
                    continue;
                }
                Event::Generate { src_t, .. } => {
                    let cut = self.tasks[task as usize].cut;
                    let cutting = self.agents[a as usize].cutting;
                    if cutting {
                        if let (Some(s), Some(cv)) = (src_t, cut) {
                            if s < cv {
                                let st = &mut self.agents[a as usize];
                                st.warm = Some(i);
                                st.pend = 0.0;
                                self.complete(task, agent, i, t);
                                self.agents[a as usize].idx += 1;
                                continue;
                            }
                        }
                        let st = &mut self.agents[a as usize];
                        st.cutting = false;
                        let residual = match (src_t, cut) {
                            (Some(s), Some(cv)) => st.pend.min(s - cv).max(0.0),
                            _ => 0.0,
                        };
                        if let Some(w) = st.warm {
                            self.send(t, a, w, true, residual);
                            return;
                        }
                        if residual > 0.0 {
                            self.at(t + residual, Ev::SendAt(a));
                            return;
                        }
                    }
                    self.send(t, a, i, false, 0.0);
                    return;
                }
            }
        }
    }

    fn complete(&mut self, task: u32, agent: u32, i: u32, t: f64) {
        let ts = &mut self.tasks[task as usize];
        ts.comp.insert((agent, i));
        if let Some(ws) = ts.waiters.remove(&(agent, i)) {
            for a in ws {
                self.at(t, Ev::AgentStep(a));
            }
        }
    }

    fn send(&mut self, t: f64, a: u32, i: u32, warm: bool, then: f64) {
        if t >= self.c.duration_s {
            return;
        }
        let (task, agent) = {
            let s = &self.agents[a as usize];
            (s.task, s.agent)
        };
        let (row, inst) = {
            let ts = &self.tasks[task as usize];
            (ts.row, ts.inst)
        };
        let Event::Generate { runs, max_tokens, .. } =
            &self.wl.tasks[row as usize].agents[agent as usize].events[i as usize]
        else {
            return;
        };
        let off = inst << 32;
        let runs: Vec<(u64, u32)> = runs.iter().map(|&(s, c)| (s + off, c)).collect();
        let pt = run_blocks(&runs) * self.c.block_tokens;
        let maxt = if warm { 1 } else { (*max_tokens).max(1) };
        let ct = if warm {
            1
        } else {
            self.ct_map.get(&(row, agent, i)).copied().unwrap_or(maxt).max(1)
        };
        let need = pt + (maxt as u64).min(self.c.d_reserve_out);
        let r = Req {
            task,
            agent_state: a,
            idx: i,
            warm,
            dec_lb: then,
            runs,
            pt,
            maxt,
            ct,
            need,
            admit: t,
            start: t + self.c.client_lag_s,
            ..Default::default()
        };
        let id = self.reqs.len() as u32;
        let start = r.start;
        self.reqs.push(r);
        self.at(start, Ev::Route(id));
    }

    // ------------------------------------------------------------------------------ router
    fn backlog_of(p: &PW, reqs: &[Req]) -> u64 {
        let mut s: u64 = p.wq.iter().map(|&r| reqs[r as usize].unc_est).sum();
        if let Some(r) = p.chunked {
            let r = &reqs[r as usize];
            s += r.unc.saturating_sub(r.sched);
        }
        if let Some(items) = &p.next_items {
            s += items.iter().map(|x| x.n).sum::<u64>();
        }
        if let Some(items) = &p.run_items {
            s += items.iter().map(|x| x.n).sum::<u64>();
        }
        s
    }

    fn live(&mut self, i: usize, t: f64) -> u64 {
        let poll = self.c.cs_load_poll_s;
        let p = &mut self.p[i];
        while let Some(&(tt, _)) = p.recent.front() {
            if tt < t - (2.0 * poll).max(0.1) {
                p.recent.pop_front();
            } else {
                break;
            }
        }
        let st = p.snap_t;
        p.snap + p.recent.iter().filter(|&&(tt, _)| tt >= st).map(|&(_, x)| x).sum::<u64>()
    }

    fn route(&mut self, t: f64, r: u32) {
        self.pick_p(t, r);
        self.pick_d(r);
        let pt = self.reqs[r as usize].pt;
        let pr = t + self.c.ing_p.at(pt);
        let dr = t + self.c.ing_d.at(pt);
        self.reqs[r as usize].p_recv = pr;
        self.reqs[r as usize].d_recv = dr;
        self.at(pr, Ev::PRecv(r));
        self.at(dr, Ev::DRecv(r));
    }

    fn actual_hits(&mut self, r: u32) -> Vec<u64> {
        let bt = self.c.block_tokens;
        let runs = self.reqs[r as usize].runs.clone();
        self.p.iter_mut().map(|p| p.radix.match_peek(&runs).total() as u64 * bt).collect()
    }

    fn pick_p(&mut self, t: f64, r: u32) {
        let np = self.p.len();
        let bt = self.c.block_tokens;
        let hits = self.actual_hits(r);
        let best = *hits.iter().max().unwrap_or(&0);
        let pol = self.c.p_policy.clone();
        let pt = self.reqs[r as usize].pt;
        let tgt: usize;
        let mut own: usize;
        match pol.as_str() {
            "smgca" => {
                let runs = self.reqs[r as usize].runs.clone();
                let ld: Vec<i64> = self.p.iter().map(|p| p.nin).collect();
                let mx = *ld.iter().max().unwrap();
                let mn = *ld.iter().min().unwrap();
                let imbalanced = mx - mn > self.c.smg_abs && (mx as f64) > (mn as f64) * self.c.smg_rel;
                let min_load = |s: &mut Self| -> usize {
                    let c: Vec<usize> = (0..np).filter(|&k| ld[k] == mn).collect();
                    c[s.urange(c.len())]
                };
                if imbalanced {
                    tgt = min_load(self);
                } else {
                    let m: Vec<u64> =
                        self.smg.iter_mut().map(|tr| tr.match_peek(&runs).total() as u64 * bt).collect();
                    let mb = *m.iter().max().unwrap();
                    if mb > 0 && (mb as f64) / (pt.max(1) as f64) > self.c.smg_threshold {
                        let c: Vec<usize> = (0..np).filter(|&k| m[k] == mb).collect();
                        tgt = c[self.urange(c.len())];
                    } else if self.c.smg_low_match == "smallest_tenant" {
                        let sz: Vec<u64> = self.smg.iter().map(|tr| tr.host_used).collect();
                        let smin = *sz.iter().min().unwrap();
                        let c: Vec<usize> = (0..np).filter(|&k| sz[k] == smin).collect();
                        tgt = c[self.urange(c.len())];
                    } else {
                        tgt = min_load(self);
                    }
                }
                self.smg[tgt].insert(&runs, t, false);
                own = tgt;
                self.p[tgt].nin += 1;
                self.reqs[r as usize].nin_held = true;
            }
            "cosched" => {
                let live: Vec<u64> = (0..np).map(|i| self.live(i, t)).collect();
                own = (0..np)
                    .min_by(|&a, &b| hits[b].cmp(&hits[a]).then(live[a].cmp(&live[b])).then(a.cmp(&b)))
                    .unwrap();
                let least = (0..np)
                    .min_by(|&a, &b| live[a].cmp(&live[b]).then(hits[b].cmp(&hits[a])).then(a.cmp(&b)))
                    .unwrap();
                let mut tt = own;
                if hits[own] > 0 && live[own].saturating_sub(live[least]) > self.c.cs_margin_tokens {
                    tt = least;
                }
                if hits[own] == 0 {
                    tt = least;
                }
                tgt = tt;
            }
            "sticky" => {
                let (task, agent) = {
                    let rq = &self.reqs[r as usize];
                    (rq.task, self.agents[rq.agent_state as usize].agent)
                };
                let row = self.tasks[task as usize].row as usize;
                let inst = self.tasks[task as usize].inst;
                let rk = self.wl.tasks[row].agents[agent as usize]
                    .routing_key
                    .clone()
                    .unwrap_or_else(|| format!("a{agent}"));
                let key = (inst, rk);
                let v = if let Some(&v) = self.sticky.get(&key) {
                    v
                } else {
                    let v = (self.rr % np) as u16;
                    self.rr += 1;
                    self.sticky.insert(key, v);
                    v
                };
                tgt = v as usize;
                own = tgt;
            }
            _ => {
                tgt = self.rr % np;
                self.rr += 1;
                own = tgt;
            }
        }
        if pol != "cosched" {
            own = (0..np).min_by(|&a, &b| hits[b].cmp(&hits[a]).then(a.cmp(&b))).unwrap();
            if hits[own] == hits[tgt] {
                own = tgt;
            }
        }
        let rq = &mut self.reqs[r as usize];
        rq.p = tgt as u16;
        rq.owner = own as u16;
        rq.detour = tgt != own && hits[own] > 0;
        rq.route_best = best;
        rq.route_tgt = hits[tgt];
        // Pull hint (co-schedule router: scope any, best holder, min tokens / min incremental).
        if pol == "cosched" && self.c.pull_enabled {
            let gain = best.saturating_sub(hits[tgt]);
            if best >= self.c.pull_min_tokens && gain >= self.c.pull_min_incremental_tokens {
                let src = (0..np)
                    .filter(|&k| k != tgt)
                    .min_by(|&a, &b| hits[b].cmp(&hits[a]).then(a.cmp(&b)))
                    .unwrap();
                if hits[src] == best {
                    rq.pull_src = Some(src as u16);
                    rq.pull_hint = gain;
                }
            }
        }
        // Router-side estimate of the uncached work (live backlog ledger).
        let est = hits[tgt] + rq.pull_hint;
        rq.unc_est = pt.saturating_sub(est).max(1);
        let ue = rq.unc_est;
        self.p[tgt].recent.push_back((t, ue));
    }

    fn pick_d(&mut self, r: u32) {
        let nd = self.d.len();
        let d = match self.c.d_policy.as_str() {
            "rr" => {
                let d = self.drr % nd;
                self.drr += 1;
                d
            }
            "d1" => (0..nd).min_by_key(|&k| (self.d[k].used, k)).unwrap(),
            _ => {
                let i = self.urange(nd);
                let mut j = self.urange(nd - 1);
                if j >= i {
                    j += 1;
                }
                let snap = self.c.d_load_mode == "snapshot";
                let load = |k: usize| if snap { self.d[k].load_snap } else { self.d[k].nreq };
                if load(i) <= load(j) { i } else { j }
            }
        };
        self.reqs[r as usize].d = d as u16;
        self.reqs[r as usize].d_counted = true;
        self.d[d].nreq += 1;
    }

    // ------------------------------------------------------------------------------ D side
    fn d_try_admit(&mut self, t: f64, d: usize) {
        while let Some(&r) = self.d[d].fifo.front() {
            if self.reqs[r as usize].cancel {
                self.d[d].fifo.pop_front();
                continue;
            }
            let need = self.reqs[r as usize].need;
            if self.d[d].used + need > self.d[d].cap && self.d[d].used > 0 {
                break;
            }
            self.d[d].fifo.pop_front();
            self.d[d].used += need;
            self.reqs[r as usize].t_alloc = Some(t);
            self.reqs[r as usize].pa_on = true;
            if self.reqs[r as usize].p_in {
                self.boot_done(t, r);
            }
        }
    }

    fn d_release(&mut self, t: f64, r: u32) {
        let d = self.reqs[r as usize].d as usize;
        if self.reqs[r as usize].pa_on {
            self.reqs[r as usize].pa_on = false;
            self.d[d].used -= self.reqs[r as usize].need;
        }
        self.d_try_admit(t, d);
    }

    // ------------------------------------------------------------------------------ P bootstrap / pulls
    fn on_p_recv(&mut self, t: f64, r: u32) {
        self.reqs[r as usize].p_in = true;
        if self.reqs[r as usize].pull_src.is_some() && self.c.pull_start == "p_recv" {
            self.start_pull(t, r);
        }
        if self.reqs[r as usize].t_alloc.is_some() {
            self.boot_done(t, r);
        }
    }

    fn start_pull(&mut self, t: f64, r: u32) {
        let bt = self.c.block_tokens;
        let (src, dst) = {
            let rq = &self.reqs[r as usize];
            (rq.pull_src.unwrap() as usize, rq.p as usize)
        };
        let runs = self.reqs[r as usize].runs.clone();
        let have = self.p[dst].radix.match_peek(&runs).total() as u64;
        let sm: MatchResult = self.p[src].radix.match_peek(&runs);
        let avail = sm.total() as u64;
        let want_blocks = (self.reqs[r as usize].pull_hint / bt) + have;
        let upto = avail.min(want_blocks);
        let pulled_blocks = upto.saturating_sub(have);
        self.pulls_all += 1;
        if pulled_blocks * bt < self.reqs[r as usize].pull_hint {
            self.pull_short_all += 1;
        }
        let rq = &mut self.reqs[r as usize];
        rq.pull_start_t = Some(t);
        rq.pull_dst_have = have * bt;
        if pulled_blocks == 0 {
            rq.pulled = 0;
            rq.pull_end = Some(t);
            rq.pull_pending = false;
            return;
        }
        let lease = TwoTierRadix::path_node_covering(&sm.path, upto as u32);
        self.p[src].radix.lease(lease, 1);
        let rq = &mut self.reqs[r as usize];
        rq.pull_lease = lease;
        rq.pull_src_path = sm.path.clone();
        rq.pulled = pulled_blocks * bt;
        self.pulled_all += pulled_blocks * bt;
        let dur = self.c.pull_fit.at(pulled_blocks * bt);
        rq.pull_end = Some(t + dur);
        rq.pull_pending = true;
        self.at(t + dur, Ev::PullDone(r));
    }

    fn pull_done(&mut self, t: f64, r: u32) {
        let bt = self.c.block_tokens;
        let (src, dst, lease, pulled, have) = {
            let rq = &self.reqs[r as usize];
            (rq.pull_src.unwrap() as usize, rq.p as usize, rq.pull_lease, rq.pulled, rq.pull_dst_have)
        };
        // Land the pulled range on the destination, then lock it until the first chunk.
        let upto_blocks = ((have + pulled) / bt) as u32;
        let runs = runs_prefix(&self.reqs[r as usize].runs, upto_blocks);
        let host_only = self.c.pull_landing == "host";
        let path = self.p[dst].radix.insert(&runs, t, host_only);
        let deep = path.last().copied();
        self.p[dst].radix.lock(deep, 1);
        self.reqs[r as usize].pull_dst_lock = deep;
        // Release the source lease; demote the source copy below the deepest branch point.
        if self.c.demote_source {
            let sp = self.reqs[r as usize].pull_src_path.clone();
            // Only the leased part of the path (up to the pulled end).
            let mut acc = 0u32;
            let mut leased = Vec::new();
            for &(n, m) in &sp {
                if acc >= upto_blocks {
                    break;
                }
                leased.push((n, m));
                acc += m;
            }
            self.p[src].radix.lease(lease, -1);
            self.p[src].radix.demote_below_branch(&leased);
        } else {
            self.p[src].radix.lease(lease, -1);
        }
        self.reqs[r as usize].pull_lease = None;
        self.reqs[r as usize].pull_pending = false;
        self.p[src].radix.evict_dev(t);
        // If the request was already bootstrap-ready, it was waiting for the pull.
        if let Some(rdy) = self.reqs[r as usize].boot_ready {
            let at = rdy.max(t + self.c.pull_check_s);
            if self.reqs[r as usize].p_ready.is_none() {
                self.at(at, Ev::PReady(r));
            }
        }
    }

    fn boot_done(&mut self, t: f64, r: u32) {
        let rq = &self.reqs[r as usize];
        let rdy = rq.p_recv.max(rq.t_alloc.unwrap_or(t)) + self.c.handshake_s;
        let rdy = rdy.max(t);
        self.reqs[r as usize].boot_ready = Some(rdy);
        if self.reqs[r as usize].pull_src.is_some() {
            if self.c.pull_start == "queue" && self.reqs[r as usize].pull_start_t.is_none() {
                // JIT pull: issued when the request is ready to enter the queue.
                self.start_pull(rdy.max(t), r);
            }
            if self.reqs[r as usize].pull_pending {
                return; // PReady is scheduled at pull completion.
            }
            if let Some(pe) = self.reqs[r as usize].pull_end {
                let at = rdy.max(pe + if self.reqs[r as usize].pulled > 0 { self.c.pull_check_s } else { 0.0 });
                self.at(at, Ev::PReady(r));
                return;
            }
        }
        self.at(rdy, Ev::PReady(r));
    }

    fn p_ready(&mut self, t: f64, r: u32) {
        if self.reqs[r as usize].p_ready.is_some() {
            return;
        }
        self.reqs[r as usize].p_ready = Some(t);
        let p = self.reqs[r as usize].p as usize;
        // Queued-prefix protect (fix 3): lock the currently matched prefix while queued.
        if self.c.queued_prefix_protect {
            let runs = self.reqs[r as usize].runs.clone();
            let m = self.p[p].radix.match_peek(&runs);
            let deep = m.path.last().map(|&(n, _)| n);
            self.p[p].radix.lock(deep, 1);
            self.reqs[r as usize].qlock = deep;
        }
        let bt = self.c.block_tokens;
        let runs = self.reqs[r as usize].runs.clone();
        let est = self.p[p].radix.match_peek(&runs).total() as u64 * bt;
        self.reqs[r as usize].unc_est = self.reqs[r as usize].pt.saturating_sub(est).max(1);
        self.p[p].wq.push_back(r);
        if self.p[p].run_items.is_none() {
            if let Some(items) = self.form(t, p) {
                self.start_iter(t, p, items);
            }
        }
    }

    // ------------------------------------------------------------------------------ P engine
    fn first_chunk(&mut self, t: f64, p: usize, r: u32) {
        let bt = self.c.block_tokens;
        let runs = self.reqs[r as usize].runs.clone();
        let id = self.ideal.match_peek(&runs).total() as u64 * bt;
        self.reqs[r as usize].t_fwd = Some(t);
        self.reqs[r as usize].ideal = id;
        // Best actual prefix anywhere at first chunk (gap-cause input).
        let fcb = self.p.iter_mut().map(|w| w.radix.match_peek(&runs).total() as u64 * bt).max().unwrap_or(0);
        self.reqs[r as usize].fc_best = fcb;
        let m = self.p[p].radix.match_local(&runs, t);
        let mut dev = m.dev as u64 * bt;
        let mut host = m.host as u64 * bt;
        let pt = self.reqs[r as usize].pt;
        // Cap like SGLang: at least one token is computed.
        if dev + host >= pt {
            let over = dev + host + 1 - pt;
            if host >= over {
                host -= over;
            } else {
                dev -= over - host;
                host = 0;
            }
        }
        // Pulled part = host-tier blocks that came from the pull (landing on host), bounded by
        // what was pulled.
        let pulled = self.reqs[r as usize].pulled;
        let mut stor = 0;
        if pulled > 0 {
            let beyond_have = (dev + host).saturating_sub(self.reqs[r as usize].pull_dst_have);
            stor = beyond_have.min(pulled).min(host);
            host -= stor;
        }
        self.p[p].radix.load_back(&m.path, t);
        let deep = m.path.last().map(|&(n, _)| n);
        self.p[p].radix.lock(deep, 1);
        // Release queue-protect and pull-landing locks now that the request holds its own.
        if let Some(q) = self.reqs[r as usize].qlock.take() {
            self.p[p].radix.lock(Some(q), -1);
        }
        if let Some(q) = self.reqs[r as usize].pull_dst_lock.take() {
            self.p[p].radix.lock(Some(q), -1);
        }
        let beta = self.p[p].beta;
        let mut lb = beta[3] * host as f64 / 1e5 + beta[4] * stor as f64 / 1e5;
        if self.c.lb_hidden > 0.0 {
            let cap = (self.c.prefetch_max_tokens as f64).min(self.c.prefetch_max_fraction * pt as f64);
            let frac = if host + stor > 0 { (cap / (host + stor) as f64).min(1.0) } else { 0.0 };
            lb -= self.c.lb_hidden * frac * lb;
        }
        let rq = &mut self.reqs[r as usize];
        rq.locked = deep;
        rq.dev = dev;
        rq.host = host;
        rq.stor = stor;
        rq.unc = pt.saturating_sub(dev + host + stor).max(1);
        rq.sched = 0;
        rq.computed = 0;
        rq.lbcost = lb.max(0.0);
    }

    fn form(&mut self, t: f64, p: usize) -> Option<Vec<BItem>> {
        let mut budget = self.c.chunk_tokens;
        let mut items = Vec::new();
        if let Some(r) = self.p[p].chunked {
            let (left, pre) = {
                let rq = &self.reqs[r as usize];
                (rq.unc - rq.sched, rq.pt - rq.unc + rq.sched)
            };
            let n = left.min(budget);
            items.push(BItem { r, n, pre });
            self.reqs[r as usize].sched += n;
            budget -= n;
            if self.reqs[r as usize].sched < self.reqs[r as usize].unc {
                return Some(items);
            }
            self.p[p].chunked = None;
        }
        while budget > 0 {
            let Some(r) = self.p[p].wq.pop_front() else { break };
            if self.reqs[r as usize].cancel {
                continue;
            }
            self.first_chunk(t, p, r);
            let (unc, pt) = {
                let rq = &self.reqs[r as usize];
                (rq.unc, rq.pt)
            };
            let n = unc.min(budget);
            items.push(BItem { r, n, pre: pt - unc });
            self.reqs[r as usize].sched = n;
            budget -= n;
            if n < unc {
                self.p[p].chunked = Some(r);
                break;
            }
        }
        if items.is_empty() { None } else { Some(items) }
    }

    fn iter_secs(&mut self, p: usize, items: &[BItem]) -> f64 {
        let b = self.p[p].beta;
        let mut nk = 0u64;
        let mut pairs = 0f64;
        let mut lb = 0f64;
        for it in items {
            nk += it.n;
            pairs += it.n as f64 * it.pre as f64 + it.n as f64 * (it.n as f64 + 1.0) / 2.0;
            let rq = &self.reqs[it.r as usize];
            if it.pre == rq.pt - rq.unc {
                lb += rq.lbcost;
            }
        }
        let mut svc = (b[0] + b[1] * nk as f64 / 1e3 + b[2] * pairs / 1e9 + lb).max(0.02);
        if self.c.p_noise_sigma > 0.0 {
            // Box-Muller lognormal with mean 1.
            let u1 = self.urand().max(1e-12);
            let u2 = self.urand();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let s = self.c.p_noise_sigma;
            svc *= (s * z - 0.5 * s * s).exp();
        }
        svc
    }

    fn start_iter(&mut self, t: f64, p: usize, items: Vec<BItem>) {
        let svc = self.iter_secs(p, &items);
        let pw = &mut self.p[p];
        pw.busy_s += svc;
        if t >= self.lo && t < self.hi {
            pw.win_busy += svc;
        }
        pw.run_items = Some(items);
        self.at(t + svc, Ev::EndIter(p as u16));
        let nxt = self.form(t, p);
        self.p[p].next_items = nxt;
    }

    fn end_iter(&mut self, t: f64, p: usize) {
        let items = self.p[p].run_items.take().unwrap_or_default();
        for it in &items {
            let rq = &mut self.reqs[it.r as usize];
            rq.computed += it.n;
            if rq.computed >= rq.unc && rq.t_pd.is_none() {
                rq.t_pd = Some(-1.0);
                let at = t + self.c.p_lat_s;
                self.at(at, Ev::PrefillDone(p as u16, it.r));
            }
        }
        let nxt = match self.p[p].next_items.take() {
            Some(n) => Some(n),
            None => self.form(t, p),
        };
        if let Some(n) = nxt {
            self.start_iter(t, p, n);
        }
    }

    fn prefill_done(&mut self, t: f64, p: usize, r: u32) {
        self.reqs[r as usize].t_pd = Some(t);
        if self.reqs[r as usize].nin_held && self.c.smg_load_release == "prefill_done" {
            self.reqs[r as usize].nin_held = false;
            self.p[p].nin -= 1;
        }
        let runs = self.reqs[r as usize].runs.clone();
        self.ideal.insert(&runs, t, false);
        let before = self.p[p].radix.stats.host_inserted_blocks;
        let path = self.p[p].radix.insert(&runs, t, false);
        let ins = (self.p[p].radix.stats.host_inserted_blocks - before) * self.c.block_tokens;
        if t >= self.lo && t < self.hi {
            self.p[p].host_ins_win += ins;
        }
        let nd = path.last().copied();
        self.p[p].radix.lock(nd, 1);
        let old = self.reqs[r as usize].locked;
        self.p[p].radix.lock(old, -1);
        self.reqs[r as usize].locked = nd;
        let tail = self.c.kv_tail.at(self.reqs[r as usize].pt);
        self.at(t + tail, Ev::KvArrived(r));
    }

    fn kv_arrived(&mut self, t: f64, r: u32) {
        let p = self.reqs[r as usize].p as usize;
        self.reqs[r as usize].t_kv = Some(t);
        let lk = self.reqs[r as usize].locked.take();
        self.p[p].radix.lock(lk, -1);
        self.p[p].radix.evict_dev(t);
        let first = t + self.c.d_first_s;
        self.reqs[r as usize].first = Some(first);
        let ct = self.reqs[r as usize].ct;
        if self.c.tpot_fb {
            if ct <= 1 {
                self.reqs[r as usize].done = Some(first);
                self.at(first, Ev::ReqDone(r));
            } else {
                self.at(first, Ev::DecJoin(r));
            }
        } else {
            let done = first + (ct.saturating_sub(1)) as f64 * self.c.tpot_s;
            self.reqs[r as usize].done = Some(done);
            self.at(done, Ev::ReqDone(r));
        }
    }

    fn d_adv(&mut self, t: f64, d: usize) {
        let dw = &mut self.d[d];
        if dw.b > 0 {
            dw.v += (t - dw.vt) * dw.rate;
        }
        dw.vt = t;
    }

    fn d_resched(&mut self, t: f64, d: usize) {
        self.d[d].ver += 1;
        let ver = self.d[d].ver;
        let Some(std::cmp::Reverse((tgt_bits, _, _))) = self.d[d].dh.peek().copied() else {
            return;
        };
        let tgt = f64::from_bits(tgt_bits);
        let step = step_ms(self.d[d].b, self.d[d].s);
        self.d[d].rate = 1.0 / (self.c.tpot_s * step / self.c.step_ref_ms);
        let dt = ((tgt - self.d[d].v) / self.d[d].rate).max(0.0);
        self.at(t + dt, Ev::DecTick(d as u16, ver));
    }

    fn dec_join(&mut self, t: f64, r: u32) {
        let d = self.reqs[r as usize].d as usize;
        let n = (self.reqs[r as usize].ct - 1) as f64;
        self.d_adv(t, d);
        let dw = &mut self.d[d];
        dw.b += 1;
        dw.s += self.reqs[r as usize].pt;
        dw.hseq += 1;
        let key = (dw.v + n).to_bits();
        dw.dh.push(std::cmp::Reverse((key, dw.hseq, r)));
        self.d_resched(t, d);
    }

    fn dec_tick(&mut self, t: f64, d: usize, ver: u64) {
        if ver != self.d[d].ver {
            return;
        }
        self.d_adv(t, d);
        let mut out = Vec::new();
        while let Some(std::cmp::Reverse((k, _, r))) = self.d[d].dh.peek().copied() {
            if f64::from_bits(k) <= self.d[d].v + 1e-6 {
                self.d[d].dh.pop();
                self.d[d].b -= 1;
                self.d[d].s -= self.reqs[r as usize].pt;
                out.push(r);
            } else {
                break;
            }
        }
        self.d_resched(t, d);
        for r in out {
            self.reqs[r as usize].done = Some(t);
            self.at(t, Ev::ReqDone(r));
        }
    }

    fn req_done(&mut self, t: f64, r: u32) {
        if t > self.c.duration_s {
            self.reqs[r as usize].cancel = true;
            return;
        }
        let d = self.reqs[r as usize].d as usize;
        self.d_release(t, r);
        if self.reqs[r as usize].d_counted {
            self.reqs[r as usize].d_counted = false;
            self.d[d].nreq -= 1;
        }
        let p = self.reqs[r as usize].p as usize;
        if self.reqs[r as usize].nin_held {
            self.reqs[r as usize].nin_held = false;
            self.p[p].nin -= 1;
        }
        let a = self.reqs[r as usize].agent_state;
        if self.reqs[r as usize].warm {
            let lb = self.reqs[r as usize].dec_lb;
            if lb > 0.0 {
                self.at(t + lb, Ev::SendAt(a));
            } else {
                let i = self.agents[a as usize].idx;
                self.send(t, a, i, false, 0.0);
            }
            return;
        }
        let (task, agent) = {
            let s = &self.agents[a as usize];
            (s.task, s.agent)
        };
        let idx = self.reqs[r as usize].idx;
        self.complete(task, agent, idx, t);
        self.agents[a as usize].idx += 1;
        self.agent_step(t, a);
    }

    // ------------------------------------------------------------------------------ report
    pub fn report(&mut self, wall_s: f64) -> (SimReport, Vec<Row>) {
        let lo = self.lo;
        let hi = self.hi;
        let dur = self.c.duration_s;
        let w: Vec<&Req> =
            self.reqs.iter().filter(|r| !r.warm && r.admit >= lo && r.admit < hi).collect();
        let cmp: Vec<&Req> = w
            .iter()
            .copied()
            .filter(|r| !r.cancel && r.done.is_some_and(|d| d <= dur) && r.t_fwd.is_some())
            .collect();
        let mut rep = SimReport { label: self.c.label.clone(), seed: self.c.seed, ..Default::default() };
        rep.n_scored = cmp.len();
        rep.n_admitted_window = w.len();
        rep.rps = cmp.len() as f64 / self.c.score_s;
        let sum = |f: &dyn Fn(&Req) -> f64| -> f64 { cmp.iter().map(|r| f(r)).sum() };
        let pt = sum(&|r| r.pt as f64).max(1.0);
        let actual = |r: &Req| (r.dev + r.host + r.stor) as f64;
        let sa = sum(&|r| actual(r));
        let si = sum(&|r| r.ideal.min(r.pt) as f64);
        rep.hit_sum_a_over_i = if si > 0.0 { sa / si } else { 0.0 };
        rep.hit_actual_over_prompt = sa / pt;
        rep.ideal_over_prompt = si / pt;
        rep.dev_share = sum(&|r| r.dev as f64) / pt;
        rep.host_share = sum(&|r| r.host as f64) / pt;
        rep.pulled_share = sum(&|r| r.stor as f64) / pt;
        let n = cmp.len().max(1) as f64;
        let zero: Vec<&&Req> = cmp.iter().filter(|r| actual(r) == 0.0 && r.ideal > 0).collect();
        rep.zero_hit_row_share = zero.len() as f64 / n;
        let ideal_rows = cmp.iter().filter(|r| r.ideal > 0).count().max(1) as f64;
        rep.zero_hit_row_share_of_ideal_rows = zero.len() as f64 / ideal_rows;
        let deficit = |r: &Req| (r.ideal.min(r.pt) as f64 - actual(r)).max(0.0);
        let def_total = sum(&|r| deficit(r));
        rep.deficit_tokens = def_total;
        let zdef: f64 = zero.iter().map(|r| deficit(r)).sum();
        rep.zero_hit_deficit_share = if def_total > 0.0 { zdef / def_total } else { 0.0 };
        // Gap causes at routing time (token-weighted shares of the deficit):
        // E = prefix not on any P (ideal - best), R = best holder not chosen and no pull,
        // pull-short = pull hinted but realized < best, other = remainder.
        let (mut e, mut rr, mut ps, mut ef) = (0.0, 0.0, 0.0, 0.0);
        for r in &cmp {
            let dfc = deficit(r);
            if dfc <= 0.0 {
                continue;
            }
            let id = r.ideal.min(r.pt) as f64;
            let ed = (id - r.route_best as f64).max(0.0).min(dfc);
            e += ed;
            let rest = dfc - ed;
            let miss_choice = (r.route_best as f64 - r.route_tgt as f64).max(0.0).min(rest);
            if r.pull_src.is_none() {
                rr += miss_choice;
            } else {
                let got = r.pulled as f64;
                ps += (miss_choice - got).max(0.0);
            }
            ef += (id - r.fc_best as f64).max(0.0).min(dfc);
        }
        if def_total > 0.0 {
            rep.e_share_route = e / def_total;
            rep.r_share_route = rr / def_total;
            rep.pull_short_share_route = ps / def_total;
            rep.other_share_route = 1.0 - rep.e_share_route - rep.r_share_route - rep.pull_short_share_route;
            rep.e_share_fc = ef / def_total;
        }
        let mut ttft: Vec<f64> = cmp.iter().map(|r| r.first.unwrap() - r.start).collect();
        let mut e2e: Vec<f64> = cmp.iter().map(|r| r.done.unwrap() - r.start).collect();
        rep.ttft_mean = ttft.iter().sum::<f64>() / n;
        rep.e2e_mean = e2e.iter().sum::<f64>() / n;
        if !ttft.is_empty() {
            for (k, q) in [25.0, 50.0, 90.0, 95.0, 99.0].iter().enumerate() {
                rep.ttft[k] = percentile_in_place(&mut ttft, *q);
            }
            rep.e2e[0] = percentile_in_place(&mut e2e, 50.0);
            rep.e2e[1] = percentile_in_place(&mut e2e, 99.0);
        }
        rep.p_queue_mean = sum(&|r| r.t_fwd.unwrap() - r.p_ready.unwrap_or(r.t_fwd.unwrap())) / n;
        rep.d_wait_mean = sum(&|r| r.t_alloc.unwrap_or(r.d_recv) - r.d_recv) / n;
        rep.prefill_mean = sum(&|r| r.t_pd.unwrap() - r.t_fwd.unwrap()) / n;
        rep.decode_mean = sum(&|r| r.done.unwrap() - r.first.unwrap()) / n;
        rep.pulls_scored = w.iter().filter(|r| r.pulled > 0).count() as u64;
        rep.pulled_tokens_scored = w.iter().map(|r| r.pulled).sum();
        rep.pulls_all = self.pulls_all;
        rep.pulled_tokens_all = self.pulled_all;
        rep.pull_short_all = self.pull_short_all;
        rep.detour_share = w.iter().filter(|r| r.detour).count() as f64 / w.len().max(1) as f64;
        let bt = self.c.block_tokens;
        for (i, pw) in self.p.iter().enumerate() {
            let cap = pw.radix.host_cap as f64 * bt as f64;
            let inflow = pw.host_ins_win as f64 / self.c.score_s;
            rep.per_p_host_retention_cap_over_inflow_s.push(if inflow > 0.0 { cap / inflow } else { 0.0 });
            let ev: Vec<_> = pw.radix.evictions.iter().filter(|e| e.t >= lo && e.t < hi).collect();
            let blk: f64 = ev.iter().map(|e| e.blocks as f64).sum();
            let al: f64 = ev.iter().map(|e| e.age_last * e.blocks as f64).sum();
            let aw: f64 = ev.iter().map(|e| e.age_write * e.blocks as f64).sum();
            rep.per_p_host_age_last_s.push(if blk > 0.0 { al / blk } else { 0.0 });
            rep.per_p_host_age_write_s.push(if blk > 0.0 { aw / blk } else { 0.0 });
            let mut ages: Vec<f64> = ev.iter().map(|e| e.age_last).collect();
            rep.per_p_host_age_last_p50_s.push(if ages.is_empty() { 0.0 } else { percentile_in_place(&mut ages, 50.0) });
            rep.per_p_busy.push(pw.win_busy / self.c.score_s);
            rep.per_p_share.push(cmp.iter().filter(|r| r.p as usize == i).count() as f64 / n);
            rep.per_p_host_inserted_tokens_win.push(pw.host_ins_win);
            rep.per_p_host_evicted_tokens.push(blk as u64 * bt);
            rep.per_p_loaded_back_tokens.push(pw.radix.stats.loaded_back_blocks * bt);
            rep.host_fill.push(pw.radix.host_used as f64 / pw.radix.host_cap.max(1) as f64);
            rep.dev_fill.push(pw.radix.dev_used as f64 / pw.radix.dev_cap.max(1) as f64);
            let s = &pw.radix.stats;
            rep.demote_flagged_nodes += s.demote_flagged_nodes;
            rep.demote_flagged_tokens += s.demote_flagged_blocks * bt;
            rep.demote_evicted_nodes += s.demote_evicted_nodes;
            rep.demote_evicted_tokens += s.demote_evicted_blocks * bt;
            rep.demote_cleared_nodes += s.demote_cleared_nodes;
            rep.demote_cleared_tokens += s.demote_cleared_blocks * bt;
        }
        rep.tasks_started = self.tasks_started;
        rep.requests_sent = self.reqs.len() as u64;
        rep.wall_s = wall_s;
        let mut rows = Vec::new();
        if self.c.record_rows {
            for r in &w {
                let ok = !r.cancel && r.done.is_some_and(|d| d <= dur) && r.t_fwd.is_some();
                rows.push(Row {
                    task_row: self.tasks[r.task as usize].row,
                    agent: self.agents[r.agent_state as usize].agent,
                    idx: r.idx,
                    admit: r.admit,
                    p: r.p,
                    d: r.d,
                    pt: r.pt,
                    actual: r.dev + r.host + r.stor,
                    ideal: r.ideal,
                    dev: r.dev,
                    host: r.host,
                    pulled: r.stor,
                    route_best: r.route_best,
                    route_tgt: r.route_tgt,
                    fc_best: r.fc_best,
                    pull_hint: r.pull_hint,
                    detour: r.detour,
                    ttft: r.first.map(|f| f - r.start).unwrap_or(-1.0),
                    e2e: r.done.map(|f| f - r.start).unwrap_or(-1.0),
                    ct: r.ct,
                    completed: ok,
                });
            }
        }
        (rep, rows)
    }
}

/// Run one simulation.
pub fn run(c: SimConfig, wl: &Workload, ct_map: &FxHashMap<(u32, u32, u32), u32>) -> (SimReport, Vec<Row>) {
    let t0 = std::time::Instant::now();
    let mut sim = Sim::new(c, wl, ct_map);
    sim.run();
    sim.report(t0.elapsed().as_secs_f64())
}
