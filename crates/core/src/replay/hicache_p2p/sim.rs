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

/// Fix-3 guard pass constants (queued_prefix_guard.py@831bac2694): candidates examined per pass
/// (:76), passes before an unlocked request is examined again (:78), passes without an arrival
/// lock after a pressure release (:80), release limit as a multiple of the shortfall (:82).
const QPG_MAX_EXAMINE_PER_PASS: usize = 16;
const QPG_RETRY_AFTER_PASSES: u64 = 64;
const QPG_PRESSURE_NO_ARRIVE_PASSES: u64 = 256;
const QPG_PRESSURE_RELEASE_FACTOR: u64 = 2;

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
    /// Per-P empirical service-time ratio quantiles (201 points, psim res_q); used when non-empty.
    pub p_res_q: Vec<Vec<f64>>,
    /// Empirical distributions (201 quantiles): handshake, D first step, ingress residuals.
    pub hs_q: Vec<f64>,
    pub dfirst_q: Vec<f64>,
    pub ing_p_res_q: Vec<f64>,
    pub ing_d_res_q: Vec<f64>,
    pub kv_tail_res_q: Vec<f64>,
    /// Ideal-hit registry granularity (client prefix_block_size) in tokens.
    pub ideal_gran_tokens: u64,
    /// Multiplier from simulated host write-through tokens to the measured D->H counter.
    pub dh_counter_scale: f64,
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
    /// "smgca" | "cosched" | "sticky" | "rr" | "eft" | "ideal_eft" (the last two: SWEEP-PDLOAD variants)
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
    /// Refresh the local LRU at P intake and waiting-queue entry (arrival match / BWP re-match).
    pub touch_at_intake: bool,
    /// F3 guard: minimum depth and token budget (min(3.0 M, 0.3 x usable)).
    pub guard_min_tokens: u64,
    pub guard_budget_tokens: u64,
    /// F3 guard per-pass step (QPG step(), :254-280): trim the budget newest-first (:275-276) and
    /// re-examine unlocked queued requests (:277-278, :404-421). Flags only for A/B diagnosis.
    pub guard_trim: bool,
    pub guard_examine: bool,
    /// Demote walk located at RELEASE time from the live node holding the last leased page,
    /// walking parent links (peer_kv_demote.py select_demote_path); false = walk the path stored at
    /// lookup time (v2). Split refreshes both fragments (unified_tree_core.py:123, 1285-1335);
    /// false = v2. Flags only for A/B diagnosis; defaults = engine behavior.
    pub demote_relocate: bool,
    pub split_refresh: bool,
    /// Post-readout v4 (engine alignment): a local match that ends inside a node splits it
    /// (unified_tree_core.py :871, :911-918, :1312-1370 @f8a08865b1); request-local matches use
    /// the engine key (prompt minus its last token, page aligned). false = v1-v3.
    pub split_on_match: bool,
    /// E1 deferred demote (an engine VARIANT, not current engine behaviour; definitions of the RETURNS
    /// workflow model srcN<k> / srcT<s>, wf-complete/returns/returns-20261011.txt section 3): at the
    /// complete release the source copy is NOT flagged; a pending demote per (agent chain, source P)
    /// runs the release-time demote walk (same lookup time and stop rules) when the chain has served
    /// demote_defer_n later requests on other P, or demote_defer_t seconds after the move; a later
    /// request of the chain on the source P cancels it. 0 = off (immediate demote).
    pub demote_defer_n: u32,
    pub demote_defer_t: f64,
    /// ORACLE bounds (post-readout counterfactuals, NOT engine behaviour; all false = engine path):
    /// O1 oracle_pull: every hinted pull delivers max(owner's current host run, hint) up to len - 1
    /// at the normal transfer time: no lease-budget refusal (SV:996-1017), no lookup miss / short
    /// grant after source eviction (SV:931-937, SV:17-27, SLOT:4-35), no < 256 revocation
    /// (URC:2852-2858, 3639-3645). O2 oracle_move: at transfer the source copy is freed at once
    /// (TwoTierRadix::move_free) instead of the demote flag (DM:117-148). O3 pull_at_route: the
    /// pull starts at the routing decision, no JIT deferral (JIT:269-335, SCH:3971-3978, 4129-4134)
    /// and no P-intake start (SCH:3353-3383).
    pub oracle_pull: bool,
    pub oracle_move: bool,
    pub pull_at_route: bool,
    /// O3b: a landed pull stays host-locked until admission: the F3 landing lock is exempt from the
    /// budget trim (QPG :275-276) and from the pressure release (URC:2069-2097; QPG :282-316).
    pub oracle_protect_landed: bool,
    /// co_schedule workload signal: "ledger" (cs4 as run on GPU: charge = prompt - expected hit,
    /// kept in full from dispatch to the P response) or "true" (idealized live backlog: charges of
    /// requests in the P waiting queue + not-yet-computed uncached tokens of admitted prefills;
    /// bootstrap queue, computed chunk tokens and the KV-transfer tail excluded; no poll lag).
    pub cs_workload: String,
    /// R3 pull penalty: detour iff saved > margin + pulled_est x (rho + lambda), pulled_est = owner
    /// hit - alternative's own hit. lambda < 0 = off.
    pub cs_r3_lambda: f64,
    pub cs_r3_rho: f64,
    /// R1 session pair (REROUTE reroute_cf.py replay(): pair): once a session (scope "session" =
    /// task instance, "key" = agent chain) has used 2 distinct P, the detour target is chosen among
    /// those 2 only. "off" = no restriction.
    pub cs_pair: String,
    /// R2 hysteresis per agent chain: within N requests (cs_hyst_n > 0) or T seconds (cs_hyst_t > 0)
    /// of the last move, the detour threshold is k x margin (reroute_cf.py replay(): N / T / k).
    pub cs_hyst_n: u32,
    pub cs_hyst_t: f64,
    pub cs_hyst_k: f64,
    /// SWEEP-PDLOAD (2026-10-12) router VARIANTS, not engine behaviour; defaults keep every older arm.
    /// p_policy "eft" = co-schedule + P2P pull with an earliest-finish detour: per candidate P the
    /// router predicts the P-side finish time = max(backlog seconds, pull seconds) + own prefill
    /// seconds (pass model c0 amortized per chunk, c1, c2 attention, c3 host / c4 pulled load-back)
    /// and adds a duplication charge = cs_eft_dup_rho x pulled tokens x per-token seconds; it detours
    /// to the best other P iff that total + cs_eft_margin_s < the owner's total. Pull seconds =
    /// cs_eft_pull_fixed_s + cs_eft_pull_s_per_100k x pulled / 1e5. One-way moves (cs_eft_one_way):
    /// the agent chain's home P (its last target) is the owner, and the P the chain moved away from
    /// last is not a detour target while it still holds >= pull_min_tokens of the prompt.
    pub cs_eft_pull_s_per_100k: f64,
    pub cs_eft_pull_fixed_s: f64,
    pub cs_eft_dup_rho: f64,
    pub cs_eft_margin_s: f64,
    pub cs_eft_one_way: bool,
    /// Burst-only detour: no detour unless the owner's predicted wait (backlog seconds) is >= this
    /// (s); 0 = off. Small-prefix-first: a move may pull at most this many tokens; 0 = off.
    pub cs_eft_burst_wait_s: f64,
    pub cs_eft_max_pull_tokens: u64,
    /// P waiting-queue order (engine VARIANT): "fcfs" (engine) or "sjf_aging" = least
    /// (uncached estimate - sjf_aging_tok_per_s x seconds in the waiting queue) first; held requests
    /// are still skipped, the continuing chunk still goes first.
    pub p_queue_order: String,
    pub sjf_aging_tok_per_s: f64,
    /// IDEAL bound (oracle, not an engine path): one two-tier pool shared by all P (device and host
    /// capacity = n_p x the per-P sizes; every P reads it at local cost: no duplicate copies, no
    /// pulls, no transfer time). Requires p_policy "ideal_eft" (no pulls) and fix 3 off.
    pub shared_pool: bool,
    /// p_policy "ideal_eft": earliest predicted finish over all P with the same backlog-seconds model
    /// as "eft"; ideal_backlog "true" = waiting queue + admitted prefills (as cs_workload "true"),
    /// "all" = also the bootstrap queue.
    pub ideal_backlog: String,
    /// Owner lease budget (64 GiB at 2,912,256 B per 64-token page = 1.51 M tokens).
    pub lease_budget_tokens: u64,
    /// Leases pin the source tree nodes (false = engine behavior: eviction is lease-blind).
    pub lease_pins: bool,
    /// F1 lead tokens (131,072) and F2 minimum deficit (8,192).
    pub jit_lead_tokens: u64,
    pub repull_min_tokens: u64,
    /// Pool size used in the turnover metric (12,405,568; eviction capacity is p_host_tokens).
    pub turnover_pool_tokens: u64,
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
            client_lag_s: 0.0042,
            p_dev_tokens: 2_067_584,
            p_host_tokens: 11_164_992,
            demote_scope_device: false,
            p_beta: vec![[0.08, 0.06, 0.0, 0.05, 0.05]],
            p_noise_sigma: 0.0,
            p_res_q: Vec::new(),
            hs_q: Vec::new(),
            dfirst_q: Vec::new(),
            ing_p_res_q: Vec::new(),
            ing_d_res_q: Vec::new(),
            kv_tail_res_q: Vec::new(),
            ideal_gran_tokens: 256,
            dh_counter_scale: 1.0,
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
            smg_tree_tokens: 12_405_568,
            smg_evict_interval_s: 120.0,
            smg_low_match: "min_load".into(),
            smg_load_release: "kv_arrived".into(),
            cs_margin_tokens: 32768,
            cs_load_poll_s: 0.1,
            pull_min_tokens: 256,
            pull_min_incremental_tokens: 256,
            pull_enabled: true,
            d_policy: "po2".into(),
            d_load_mode: "tokens_snapshot".into(),
            d_load_refresh_s: 5.0,
            pull_fit: LinFit { a: 0.257, b_per_100k: 0.244 },
            pull_check_s: 0.1,
            pull_start: "p_recv".into(),
            pull_landing: "host".into(),
            repull_on_admission: false,
            queued_prefix_protect: false,
            demote_source: false,
            record_rows: true,
            touch_at_intake: true,
            guard_min_tokens: 8192,
            guard_budget_tokens: 3_000_000,
            guard_trim: true,
            guard_examine: true,
            demote_relocate: true,
            split_refresh: true,
            split_on_match: false,
            demote_defer_n: 0,
            demote_defer_t: 0.0,
            oracle_pull: false,
            oracle_move: false,
            pull_at_route: false,
            oracle_protect_landed: false,
            cs_workload: "ledger".into(),
            cs_r3_lambda: -1.0,
            cs_r3_rho: 0.10,
            cs_pair: "off".into(),
            cs_hyst_n: 0,
            cs_hyst_t: 0.0,
            cs_hyst_k: 1.0,
            cs_eft_pull_s_per_100k: 0.74,
            cs_eft_pull_fixed_s: 0.0,
            cs_eft_dup_rho: 0.10,
            cs_eft_margin_s: 0.0,
            cs_eft_one_way: true,
            cs_eft_burst_wait_s: 0.0,
            cs_eft_max_pull_tokens: 0,
            p_queue_order: "fcfs".into(),
            sjf_aging_tok_per_s: 8192.0,
            shared_pool: false,
            ideal_backlog: "true".into(),
            lease_budget_tokens: 1_510_000,
            lease_pins: false,
            jit_lead_tokens: 131_072,
            repull_min_tokens: 8192,
            turnover_pool_tokens: 12_405_568,
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
    Census,
    /// Score-window start / end: snapshot the demote counters (output only).
    WinMark(bool),
    /// E1 deferred demote due (pending-demote id).
    DemoteDue(u64),
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
    pull_upto: u32,
    pull_issue_blocks: u64,
    pull_first_issue_t: Option<f64>,
    /// Age of the owner's copy at the first pull issue: time since the owner last used this
    /// session's prefix locally (first chunk or prefill done), if it ever did.
    pull_src_age: Option<f64>,
    pull_outcome: u8,
    jit_pending: bool,
    pull_attempts: u32,
    repulled: bool,
    hint_expected: u64,
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
    qlock: Option<(NodeId, u64)>,
    qlock_landed: bool,
    /// F3: arrival number at the P (qp_seq), pass from which an unlocked request may be examined
    /// again (_retry; 0 = no entry), pass until which no arrival lock is taken (_no_arrive).
    qp_seq: u64,
    qp_retry: u64,
    qp_noarrive: u64,
    /// P holding >= 50 % / >= 90 % of the prompt at the routing decision (device + host).
    copies50: u8,
    copies90: u8,
    unc_est: u64,
    ideal: u64,
    // routing / gap cause
    owner: u16,
    detour: bool,
    route_best: u64,
    route_tgt: u64,
    fc_best: u64,
    nin_held: bool,
    prev_p: Option<u16>,
    prev_gap: Option<f64>,
    local_age: Option<f64>,
    cs_cost: Option<u64>,
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
    pull_ins_win: u64,
    cs_pending: u64,
    cs_active: u64,
    bq_n: u64,
    bq_tok: u64,
    guard_used: u64,
    leased_tok: u64,
    guard_order: Vec<u32>,
    guard_released_pressure: u64,
    guard_released_budget: u64,
    guard_examined: u64,
    guard_skip_budget: u64,
    guard_grants: [u64; 3],
    /// Scheduler passes (one per batch formation), arrival numbering, bootstrap-queue members.
    pass_n: u64,
    qp_next: u64,
    bq: Vec<u32>,
    /// Admitted requests whose prefill is not done (first chunk .. prefill done).
    pf_set: Vec<u32>,
    /// Open idle segment (start, class) and idle seconds in the window by class
    /// [no queued work, only bootstrap-queue work (D allocation pending), waiting work all held].
    idle_since: Option<(f64, u8)>,
    idle_win: [f64; 3],
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
    ib: f64,
    ib_inv: f64,
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
    pub per_p_turnover_counter_s: Vec<f64>,
    pub dec_step_tokw_ms: f64,
    pub dec_b_mean: f64,
    pub p_boot_mean: f64,
    pub tail_mean: f64,
    pub d_alloc_to_kv_mean: f64,
    pub same_p_as_prev_share: f64,
    /// Zero-hit rows (I>0, A=0) by mechanism: [never on chosen P, chosen P != previous P (spread),
    /// same P as previous (LRU horizon on the owner), no previous prefill (concurrent / first)].
    pub zero_hit_mech: [u64; 4],
    /// Zero-hit rows by prev gap: [<0 or none, 0-60, 60-100, 100-140, >=140] s.
    pub zero_hit_gap_bins: [u64; 5],
    /// Age (s) of the chosen P's local copy at the zero-hit request (since last local use): p25/p50/p75.
    pub zero_hit_local_age: [f64; 3],
    /// Zero-hit rows: mechanism (rows as zero_hit_mech) x gap bin (columns as zero_hit_gap_bins),
    /// and hinted (pull hint given) rows per gap bin.
    pub zero_hit_mech_by_gap: [[u64; 5]; 4],
    pub zero_hit_hinted_by_gap: [u64; 5],
    /// D census over the score window: per D mean used fraction of the KV pool and mean FIFO
    /// (pre-allocation queue) length.
    pub per_d_used_frac_mean: Vec<f64>,
    pub per_d_fifo_mean: Vec<f64>,
    pub per_p_host_age_last_pct_s: Vec<[f64; 5]>,
    /// Census over the score window: mean fractions of the host pool (pinned, device-resident, demoted)
    /// and mean / p10 usable fraction (= 1 - pinned - device-resident).
    pub pool_pinned_frac: f64,
    pub pool_device_resident_frac: f64,
    pub pool_demoted_frac: f64,
    pub pool_usable_frac_mean: f64,
    pub pool_usable_frac_p10: f64,
    /// GCWS (gapcause v1.3): per 30-s sample and P, sum over sessions with a prefill on that P in
    /// the last w s of the session's largest prompt there, / 12,405,568. [p50, p90] pooled over
    /// samples x P, for w = 120 s and w = 300 s.
    pub gcws_w120: [f64; 2],
    pub gcws_w300: [f64; 2],
    pub per_p_turnover_dh_only_s: Vec<f64>,
    pub per_p_pull_landed_tokens_win: Vec<u64>,
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
    /// Same counters over the score window only: flagged nodes / tokens, evicted-demoted nodes /
    /// tokens, cleared nodes / tokens; and the demote-walk stop reasons in the window.
    pub demote_window: [u64; 6],
    pub demote_stop_window: [u64; 6],
    /// Hinted requests admitted in the window: [hinted, full, partial, lookup miss, never issued].
    pub hint_outcomes: [u64; 5],
    /// First pull issue delay after routing (s) for hinted window requests: p50 / p90; miss share by
    /// delay bucket [<5, 5-15, 15-30, >=30 s] as (n, misses).
    pub pull_issue_delay: [f64; 2],
    pub miss_by_delay: [[u64; 2]; 4],
    /// Hinted window requests by the owner-copy age at the first issue (s): [<45, 45-90, 90-150,
    /// 150-175, 175-200, 200-250, >=250, unknown] as (n, lookup misses).
    pub miss_by_src_age: [[u64; 2]; 8],
    /// p50 of [send -> first issue, P intake -> waiting queue (D allocation + handshake),
    /// waiting queue -> first issue (JIT deferral)] for full pulls and for lookup misses (s).
    pub pull_decomp_full_p50: [f64; 3],
    pub pull_decomp_miss_p50: [f64; 3],
    pub overcommit_events: u64,
    pub overcommit_tokens_mean: f64,
    pub guard_released_pressure: u64,
    /// F3 guard over the score window, per P: mean / p90 locked tokens (cost = path depth per
    /// lock, as the QPG summary's locked_tokens), mean locks; bootstrap / waiting queue means.
    pub per_p_guard_locked_mean: Vec<f64>,
    pub per_p_guard_locked_p90: Vec<f64>,
    pub per_p_guard_locks_mean: Vec<f64>,
    pub per_p_bq_mean: Vec<f64>,
    pub per_p_wq_mean: Vec<f64>,
    /// Whole-run guard counters (all P): grants [arrive, examine, landing], budget skips,
    /// budget releases, examined candidates.
    pub guard_grants: [u64; 3],
    pub guard_skip_budget: u64,
    pub guard_released_budget: u64,
    pub guard_examined: u64,
    /// Per P idle fraction of the score window: [nothing queued, only bootstrap work (D alloc
    /// pending), waiting work all held].
    pub per_p_idle_frac: Vec<[f64; 3]>,
    /// End-of-run invariant audit per P: [radix check errors, nodes whose lock count differs from
    /// the live holders, nodes whose host-lock count differs, live nodes].
    pub per_p_audit: Vec<[u64; 4]>,
    /// User-matrix metrics (scored rows). TPOT = (E2E - TTFT) / (output - 1), output > 1: p50 / p99.
    pub tpot: [f64; 2],
    pub uncached_per_req: f64,
    /// Hit distributions [mean, weighted mean, p5, p10, p15, p20, p25, p30, p50, p75] (type-7
    /// percentiles): actual = A / prompt (weighted = sumA / sum prompt), ideal = I / prompt,
    /// ratio = A / I over rows with I > 0 (weighted by prompt).
    pub hit_dist_actual: [f64; 10],
    pub hit_dist_ideal: [f64; 10],
    pub hit_dist_ratio: [f64; 10],
    /// Rerouted (detour) share of scored rows; per session reroutes per 10 requests [p50, mean].
    pub rerouted_share: f64,
    pub reroutes_per10_session: [f64; 2],
    /// Distinct P serving the session in the last 120 s / 300 s at each scored request [mean, p50].
    pub distinct_p_w120: [f64; 2],
    pub distinct_p_w300: [f64; 2],
    /// P with >= 50 % / >= 90 % of the prompt at the decision: mean, share of rows with >= 2.
    pub copies50: [f64; 2],
    pub copies90: [f64; 2],
    /// P queue wait (waiting queue -> first forward) p50 / p95; D wait p50 / p95.
    pub p_queue_pct: [f64; 2],
    pub d_wait_pct: [f64; 2],
    /// Spread metrics with REROUTE's offline definitions (reroute_cf.py): engine-clock TTFT
    /// (P entry -> first token) p25/p50/p90/p99; distinct P per (session, fixed 120 / 300-s bin)
    /// [mean, % bins with >= 3 P]; reroute = detour to a P with less hit than the owner; reroutes
    /// per 10 requests for sessions with >= 10 requests [p25, p50, p75, p90, mean]; per agent chain:
    /// after a reroute A->B, share returning to A within 1 / 2 / 3 requests, and of the next moves
    /// the share going to a third P (A->B->C) [ret1, ret2, ret3, abc share].
    pub ttft_engine: [f64; 4],
    pub dp_bins_w120: [f64; 2],
    pub dp_bins_w300: [f64; 2],
    pub rr_per10_ge10: [f64; 5],
    pub aba: [f64; 4],
    pub lookup_miss_all: u64,
    /// Whole run, all P: local matches ending inside a node, the unmatched rest of those nodes
    /// (tokens), and match-time splits (split_on_match).
    pub partial_local_matches: u64,
    pub partial_rest_tokens: u64,
    pub match_splits: u64,
    /// E1 whole run: deferred demotes run, cancelled by a return to the source, still pending at the end.
    pub demote_deferred: [u64; 3],
    /// Oracle O2: source tokens freed at transfer / only flagged (pinned at that moment), whole run.
    pub moved_freed_tokens: u64,
    pub moved_flagged_tokens: u64,
    pub lease_refused_all: u64,
    pub repulls_all: u64,
    /// Demote walk stop reasons: no_node, no_host, children, branch, access, root.
    pub demote_stop: [u64; 6],
    pub tasks_started: u64,
    pub requests_sent: u64,
    pub host_fill: Vec<f64>,
    pub dev_fill: Vec<f64>,
    /// SWEEP-PDLOAD output additions (scored rows): E2E p90 / p95 (same percentile rule as `ttft`),
    /// prompt / uncached / output token sums; EFT counters (whole run): [moves, kept at the owner by
    /// the one-way rule, by the pull cap, by the burst threshold].
    pub e2e_p90_p95: [f64; 2],
    pub prompt_tokens_scored: u64,
    pub uncached_tokens_scored: u64,
    pub output_tokens_scored: u64,
    pub eft_counts: [u64; 4],
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
    /// Output additions (s; -1 = n/a): P waiting queue (ready -> first forward), D allocation wait,
    /// prefill (first forward -> prefill done), decode (first token -> done).
    pub p_queue: f64,
    pub d_wait: f64,
    pub prefill_s: f64,
    pub decode_s: f64,
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
    /// R1: distinct P used per session scope; R2: (last P, move time, requests since the move) per
    /// agent chain.
    pair_set: FxHashMap<(u64, u32), Vec<u16>>,
    rr_last: FxHashMap<(u64, u32), (u16, f64, u32)>,
    /// E1: pending source demotes per agent chain: (id, source P, leased runs, lookup time, chain
    /// requests served elsewhere since the move, moving request); id -> chain for timed ones.
    dem_pend: FxHashMap<(u64, u32), Vec<(u64, u16, Vec<(u64, u32)>, f64, u32, u32)>>,
    dem_due: FxHashMap<u64, (u64, u32)>,
    dem_seq: u64,
    dem_deferred: [u64; 3],
    rr: usize,
    drr: usize,
    pulls_all: u64,
    pulled_all: u64,
    pull_short_all: u64,
    census: Vec<(f64, u16, u64, u64, u64, u64)>,
    /// Per 30-s sample and P: (t, P, guard locked tokens (cost), guard locks, bootstrap queue,
    /// waiting queue).
    pstate: Vec<(f64, u16, u64, u32, u32, u32)>,
    /// Per 30-s sample and D: (t, D, used tokens, FIFO length).
    dstate: Vec<(f64, u16, u64, u32)>,
    /// Demote counters (flagged/evicted/cleared nodes and blocks, then the six stop reasons) at
    /// the score-window start [0] and end [1].
    dm_mark: [[u64; 12]; 2],
    lookup_miss: u64,
    lease_refused: u64,
    repulls: u64,
    demote_stop: [u64; 6],
    /// Per (P, task admission): last local use time (first chunk or prefill done).
    last_local: FxHashMap<(u16, u64), f64>,
    /// Per task admission: (P, prefill-done time) of the latest prefill done.
    sess_last_pd: FxHashMap<u64, (u16, f64)>,
    /// EFT one-way moves: per agent chain (home P, P the chain moved away from last); counters as
    /// SimReport::eft_counts.
    eft_home: FxHashMap<(u64, u32), (Option<u16>, Option<u16>)>,
    eft_counts: [u64; 4],
    lo: f64,
    hi: f64,
}

fn step_ms(b: i64, s_tok: u64) -> f64 {
    let lk = (1.0f64 - 8.0 / 256.0).ln();
    let b = b as f64;
    9.71 + 0.467 * b + 10.59 * (1.0 - (6.0 * b * lk).exp()) + 0.996 * s_tok as f64 / 1e6
}

/// EFT / IDEAL router model: P seconds to prefill `unc` new tokens after a `pre`-token prefix on a P
/// with pass model `beta` (c0 amortized over full chunks, c1 per 1k new tokens, c2 per 1e9 attention
/// pairs, c3 per 100k host-loaded tokens, c4 per 100k pulled tokens loaded).
pub(crate) fn eft_secs(beta: &[f64; 5], chunk: u64, unc: u64, pre: u64, host: u64, pulled: u64) -> f64 {
    let u = unc as f64;
    beta[0] * u / chunk.max(1) as f64
        + beta[1] * u / 1e3
        + beta[2] * (u * pre as f64 + u * (u + 1.0) / 2.0) / 1e9
        + beta[3] * host as f64 / 1e5
        + beta[4] * pulled as f64 / 1e5
}

/// EFT detour decision (pure). Per candidate P: `wait[w]` backlog seconds, `own_s[w]` own prefill
/// seconds, `pull_s[w]` pull seconds (0 = no pull), `dup[w]` duplication charge (s), `pulled[w]`
/// tokens a move to w pulls. Total[w] = max(wait, pull) + own + dup. Returns (target, reason):
/// reason 0 = stay (no better P), 1 = move, 2 = stay because the only better P is `excluded`
/// (one-way), 3 = stay because a better P needs a pull above `max_pull` (cap; 0 = no cap), 4 = stay
/// because the owner's wait is below `burst_s` (0 = off). A move needs total + margin < owner total.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eft_choose(
    wait: &[f64],
    own_s: &[f64],
    pull_s: &[f64],
    dup: &[f64],
    pulled: &[u64],
    own: usize,
    excluded: Option<usize>,
    burst_s: f64,
    max_pull: u64,
    margin_s: f64,
) -> (usize, u8) {
    let tot = |w: usize| wait[w].max(pull_s[w]) + own_s[w] + dup[w];
    let base = tot(own);
    let better = |w: usize| w != own && tot(w) + margin_s < base;
    let pick = |ok: &dyn Fn(usize) -> bool| -> Option<usize> {
        (0..wait.len()).filter(|&w| better(w) && ok(w)).min_by(|&a, &b| tot(a).total_cmp(&tot(b)).then(a.cmp(&b)))
    };
    if pick(&|_| true).is_none() {
        return (own, 0);
    }
    if burst_s > 0.0 && wait[own] < burst_s {
        return (own, 4);
    }
    let not_excl = |w: usize| Some(w) != excluded;
    let capped = |w: usize| max_pull == 0 || pulled[w] <= max_pull;
    if let Some(w) = pick(&|w| not_excl(w) && capped(w)) {
        return (w, 1);
    }
    if pick(&|w| capped(w)).is_some() {
        return (own, 2);
    }
    (own, 3)
}

/// IDEAL choice (pure): least predicted finish `f`, ties -> fewer active requests, then lower index.
pub(crate) fn ideal_choose(f: &[f64], act: &[u64]) -> usize {
    (0..f.len()).min_by(|&a, &b| f[a].total_cmp(&f[b]).then(act[a].cmp(&act[b])).then(a.cmp(&b))).unwrap_or(0)
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
            radix.split_refresh = c.split_refresh;
            radix.split_on_match = c.split_on_match;
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
                pull_ins_win: 0,
                cs_pending: 0,
                cs_active: 0,
                bq_n: 0,
                bq_tok: 0,
                guard_used: 0,
                leased_tok: 0,
                guard_order: Vec::new(),
                guard_released_pressure: 0,
                guard_released_budget: 0,
                guard_examined: 0,
                guard_skip_budget: 0,
                guard_grants: [0; 3],
                pass_n: 0,
                qp_next: 0,
                bq: Vec::new(),
                pf_set: Vec::new(),
                idle_since: None,
                idle_win: [0.0; 3],
            });
        }
        if c.shared_pool {
            // IDEAL bound: P 0's tree is the one pool of all P (sizes x n_p); the others stay empty.
            assert!(
                c.p_policy != "cosched" && c.p_policy != "eft" && !c.queued_prefix_protect && !c.demote_source,
                "shared_pool is an oracle without pulls: use p_policy ideal_eft (or smgca / sticky / rr), fix 3 and demote off"
            );
            p[0].radix.dev_cap = c.n_p as u64 * (c.p_dev_tokens / bt);
            p[0].radix.host_cap = c.n_p as u64 * (c.p_host_tokens / bt);
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
                ib: 0.0,
                ib_inv: 0.0,
            })
            .collect();
        // Router-side trees (SMG cache_aware) keep their calibrated split behavior (no refresh).
        let smg = (0..c.n_p)
            .map(|_| {
                let mut t = TwoTierRadix::new(u64::MAX / 4, u64::MAX / 4);
                t.split_refresh = false;
                t
            })
            .collect();
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
            pair_set: FxHashMap::default(),
            rr_last: FxHashMap::default(),
            dem_pend: FxHashMap::default(),
            dem_due: FxHashMap::default(),
            dem_seq: 0,
            dem_deferred: [0; 3],
            rr: 0,
            drr: 0,
            pulls_all: 0,
            pulled_all: 0,
            pull_short_all: 0,
            census: Vec::new(),
            pstate: Vec::new(),
            dstate: Vec::new(),
            dm_mark: [[0; 12]; 2],
            lookup_miss: 0,
            lease_refused: 0,
            repulls: 0,
            demote_stop: [0; 6],
            last_local: FxHashMap::default(),
            sess_last_pd: FxHashMap::default(),
            eft_home: FxHashMap::default(),
            eft_counts: [0; 4],
            lo,
            hi,
        }
    }

    /// Radix index used for P `p`: P 0's tree for every P with the IDEAL shared pool, else P's own.
    fn rx(&self, p: usize) -> usize {
        if self.c.shared_pool { 0 } else { p }
    }

    /// Router backlog of P `w` in seconds of P time (EFT / IDEAL model): requests in the waiting
    /// queue (router charge or uncached estimate, at their prefix depth), the not-yet-computed part
    /// of admitted prefills, and with `with_bq` the bootstrap queue; pass model of P `w`.
    fn backlog_secs(&self, w: usize, with_bq: bool) -> f64 {
        let p = &self.p[w];
        let ch = self.c.chunk_tokens;
        let mut s = 0.0;
        for &r in p.wq.iter() {
            let rq = &self.reqs[r as usize];
            let u = rq.cs_cost.unwrap_or(rq.unc_est);
            s += eft_secs(&p.beta, ch, u, rq.pt.saturating_sub(u), 0, 0);
        }
        for &r in p.pf_set.iter() {
            let rq = &self.reqs[r as usize];
            let left = rq.unc.saturating_sub(rq.computed);
            s += eft_secs(&p.beta, ch, left, rq.pt.saturating_sub(left), 0, 0);
        }
        if with_bq {
            for &r in p.bq.iter() {
                let rq = &self.reqs[r as usize];
                let u = rq.cs_cost.unwrap_or(rq.unc_est);
                s += eft_secs(&p.beta, ch, u, rq.pt.saturating_sub(u), 0, 0);
            }
        }
        s
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

    /// Draw from a 201-point quantile table (linear interpolation), psim `qdraw`.
    fn qdraw(&mut self, which: u8) -> f64 {
        let u = self.urand() * 200.0;
        let q: &Vec<f64> = match which {
            0 => &self.c.hs_q,
            1 => &self.c.dfirst_q,
            2 => &self.c.ing_p_res_q,
            3 => &self.c.ing_d_res_q,
            _ => &self.c.kv_tail_res_q,
        };
        if q.len() < 201 {
            return f64::NAN;
        }
        let k = (u as usize).min(199);
        q[k] + (q[k + 1] - q[k]) * (u - k as f64)
    }

    pub fn run(&mut self) {
        for _ in 0..self.c.concurrency {
            self.start_task(0.0);
        }
        if self.c.p_policy == "cosched_live" {
            self.at(0.0, Ev::PollTick);
        }
        if self.c.d_load_mode == "tokens_snapshot" {
            self.at(0.0, Ev::DLoadTick);
        }
        self.at(self.lo, Ev::Census);
        self.at(self.lo, Ev::WinMark(false));
        self.at(self.hi, Ev::WinMark(true));
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
                Ev::DemoteDue(id) => self.demote_due(id),
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
                        let q: u64 = d.fifo.iter().map(|&r| self.reqs[r as usize].need).sum();
                        d.load_snap = (d.used + q) as i64;
                    }
                    if t < self.c.duration_s {
                        let dt = self.c.d_load_refresh_s;
                        self.at(t + dt, Ev::DLoadTick);
                    }
                }
                Ev::WinMark(end) => {
                    let mut v = [0u64; 12];
                    for p in &self.p {
                        let s = &p.radix.stats;
                        v[0] += s.demote_flagged_nodes;
                        v[1] += s.demote_flagged_blocks;
                        v[2] += s.demote_evicted_nodes;
                        v[3] += s.demote_evicted_blocks;
                        v[4] += s.demote_cleared_nodes;
                        v[5] += s.demote_cleared_blocks;
                    }
                    for k in 0..6 {
                        v[6 + k] = self.demote_stop[k];
                    }
                    self.dm_mark[end as usize] = v;
                }
                Ev::Census => {
                    let bt = self.c.block_tokens;
                    for (i, p) in self.p.iter().enumerate() {
                        let c = p.radix.census();
                        self.census.push((t, i as u16, c.host_used * bt, c.pinned_host * bt, c.device_resident * bt, c.demoted_host * bt));
                        self.pstate.push((t, i as u16, p.guard_used, p.guard_order.len() as u32, p.bq_n as u32, p.wq.len() as u32));
                    }
                    for (i, d) in self.d.iter().enumerate() {
                        self.dstate.push((t, i as u16, d.used, d.fifo.len() as u32));
                    }
                    if t + 30.0 < self.hi {
                        self.at(t + 30.0, Ev::Census);
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
        let mut r = r;
        let gran = self.c.ideal_gran_tokens.max(1);
        let m = self.ideal.match_peek(&r.runs).total() as u64 * self.c.block_tokens;
        r.ideal = (m.min(pt.saturating_sub(1)) / gran) * gran;
        let id = self.reqs.len() as u32;
        let start = r.start;
        self.reqs.push(r);
        self.at(start, Ev::Route(id));
    }

    // ------------------------------------------------------------------------------ router
    /// Idealized live backlog of P `w` (cs_workload = "true"): router charges of the requests in the
    /// waiting queue (past bootstrap, prefill not started) + not-yet-computed uncached tokens of the
    /// admitted prefills.
    fn true_backlog(&self, w: usize) -> u64 {
        let p = &self.p[w];
        let wait: u64 = p
            .wq
            .iter()
            .map(|&r| {
                let rq = &self.reqs[r as usize];
                rq.cs_cost.unwrap_or(rq.unc_est)
            })
            .sum();
        let run: u64 = p
            .pf_set
            .iter()
            .map(|&r| {
                let rq = &self.reqs[r as usize];
                rq.unc.saturating_sub(rq.computed)
            })
            .sum();
        wait + run
    }

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

    #[allow(dead_code)]
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
        if self.c.pull_at_route && self.c.pull_enabled && self.reqs[r as usize].pull_src.is_some() {
            // O3: the pull starts at the routing decision (oracle, no JIT deferral).
            self.start_pull(t, r);
        }
        let pt = self.reqs[r as usize].pt;
        let ip = self.qdraw(2);
        let id = self.qdraw(3);
        let base_p = self.c.ing_p.a + self.c.ing_p.b_per_100k * pt as f64 / 1e5;
        let base_d = self.c.ing_d.a + self.c.ing_d.b_per_100k * pt as f64 / 1e5;
        let pr = t + (base_p + if ip.is_nan() { 0.0 } else { ip }).max(0.0);
        let dr = t + (base_d + if id.is_nan() { 0.0 } else { id }).max(0.0);
        self.reqs[r as usize].p_recv = pr;
        self.reqs[r as usize].d_recv = dr;
        self.at(pr, Ev::PRecv(r));
        self.at(dr, Ev::DRecv(r));
    }

    fn gpu_hits(&mut self, r: u32) -> Vec<u64> {
        let bt = self.c.block_tokens;
        let runs = self.reqs[r as usize].runs.clone();
        if self.c.shared_pool {
            let h = self.p[0].radix.match_peek(&runs).dev as u64 * bt;
            return vec![h; self.p.len()];
        }
        self.p.iter_mut().map(|p| p.radix.match_peek(&runs).dev as u64 * bt).collect()
    }

    fn actual_hits(&mut self, r: u32) -> Vec<u64> {
        let bt = self.c.block_tokens;
        let runs = self.reqs[r as usize].runs.clone();
        if self.c.shared_pool {
            let h = self.p[0].radix.match_peek(&runs).total() as u64 * bt;
            return vec![h; self.p.len()];
        }
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
                // Experimental sgl-router co_schedule, reserved locality-margin mode on the router
                // ledger (dataplane.txt R3: co_schedule.rs:675-707, cost_balance.rs:294-321, 803-843).
                let l = pt;
                let gpu = self.gpu_hits(r);
                let local: Vec<u64> = hits.iter().map(|&h| h.min(l)).collect();
                let mut hint: Vec<Option<(usize, u64)>> = vec![None; np];
                let mut exp: Vec<u64> = vec![0; np];
                for w in 0..np {
                    let s_ = (0..np)
                        .filter(|&k| k != w && hits[k] > 0)
                        .min_by(|&a, &b| hits[b].cmp(&hits[a]).then(a.cmp(&b)));
                    if let Some(s_) = s_ {
                        if hits[s_] >= self.c.pull_min_tokens
                            && hits[s_].saturating_sub(hits[w]) >= self.c.pull_min_incremental_tokens
                        {
                            hint[w] = Some((s_, hits[s_]));
                        }
                    }
                    let hs = hint[w].map(|(_, h)| h).unwrap_or(0);
                    exp[w] = local[w].max(hs).min(l);
                }
                let pend: Vec<u64> = if self.c.cs_workload == "true" {
                    (0..np).map(|w| self.true_backlog(w)).collect()
                } else {
                    self.p.iter().map(|p| p.cs_pending).collect()
                };
                let act: Vec<u64> = self.p.iter().map(|p| p.cs_active).collect();
                let after: Vec<u64> = (0..np).map(|w| pend[w] + (l - exp[w])).collect();
                own = (0..np)
                    .min_by(|&a, &b| {
                        local[b].cmp(&local[a]).then(gpu[b].cmp(&gpu[a])).then(pend[a].cmp(&pend[b]))
                            .then(act[a].cmp(&act[b])).then(a.cmp(&b))
                    })
                    .unwrap();
                let (inst, agent) = {
                    let rq = &self.reqs[r as usize];
                    (self.tasks[rq.task as usize].inst, self.agents[rq.agent_state as usize].agent)
                };
                let ckey = (inst, agent);
                let pkey = match self.c.cs_pair.as_str() {
                    "key" => Some(ckey),
                    "session" => Some((inst, u32::MAX)),
                    _ => None,
                };
                let allowed: Vec<usize> = match pkey.and_then(|k| self.pair_set.get(&k)) {
                    Some(v) if v.len() >= 2 => v.iter().map(|&x| x as usize).collect(),
                    _ => (0..np).collect(),
                };
                let least = allowed
                    .iter()
                    .copied()
                    .min_by(|&a, &b| {
                        after[a].cmp(&after[b]).then(gpu[b].cmp(&gpu[a])).then(local[b].cmp(&local[a]))
                            .then(act[a].cmp(&act[b])).then(a.cmp(&b))
                    })
                    .unwrap();
                let saved = after[own].saturating_sub(after[least]);
                let mut thr = self.c.cs_margin_tokens as f64;
                if self.c.cs_hyst_n > 0 || self.c.cs_hyst_t > 0.0 {
                    if let Some(&(_, tm, ns)) = self.rr_last.get(&ckey) {
                        let hold = (self.c.cs_hyst_n > 0 && ns < self.c.cs_hyst_n)
                            || (self.c.cs_hyst_t > 0.0 && t - tm < self.c.cs_hyst_t);
                        if hold {
                            thr *= self.c.cs_hyst_k;
                        }
                    }
                }
                if self.c.cs_r3_lambda >= 0.0 {
                    let pulled_est = local[own].saturating_sub(local[least]) as f64;
                    thr += pulled_est * (self.c.cs_r3_rho + self.c.cs_r3_lambda);
                }
                tgt = if least != own && saved as f64 > thr { least } else { own };
                if let Some(k) = pkey {
                    let v = self.pair_set.entry(k).or_default();
                    if !v.contains(&(tgt as u16)) {
                        v.push(tgt as u16);
                    }
                }
                let nl = match self.rr_last.get(&ckey).copied() {
                    None => (tgt as u16, t, u32::MAX),
                    Some((lp, _, _)) if lp != tgt as u16 => (tgt as u16, t, 0),
                    Some((lp, tm, ns)) => (lp, tm, ns.saturating_add(1)),
                };
                self.rr_last.insert(ckey, nl);
                let cost = l - exp[tgt];
                self.p[tgt].cs_pending += cost;
                self.p[tgt].cs_active += 1;
                self.reqs[r as usize].cs_cost = Some(cost);
                if self.c.pull_enabled {
                    if let Some((src, hs)) = hint[tgt] {
                        self.reqs[r as usize].pull_src = Some(src as u16);
                        self.reqs[r as usize].pull_hint = hs.saturating_sub(hits[tgt]);
                        self.reqs[r as usize].hint_expected = hs;
                    }
                }
            }
            "ideal_eft" => {
                // IDEAL bound: with the shared pool every P sees the same prefix; earliest predicted
                // finish. (Without the shared pool: earliest finish on each P's own cache, no pulls.)
                let l = pt;
                let with_bq = self.c.ideal_backlog == "all";
                let ch = self.c.chunk_tokens;
                let f: Vec<f64> = (0..np)
                    .map(|w| {
                        let h = hits[w].min(l);
                        self.backlog_secs(w, with_bq) + eft_secs(&self.p[w].beta, ch, l - h, h, 0, 0)
                    })
                    .collect();
                let act: Vec<u64> = self.p.iter().map(|p| p.cs_active).collect();
                tgt = ideal_choose(&f, &act);
                own = tgt;
                let unc = l - hits[tgt].min(l);
                self.p[tgt].cs_pending += unc;
                self.p[tgt].cs_active += 1;
                self.reqs[r as usize].cs_cost = Some(unc);
            }
            "eft" => {
                // Co-schedule + P2P pull with an earliest-finish detour (SWEEP-PDLOAD router variant).
                // Hints and expected hits as "cosched" (co_schedule.rs:675-707).
                let l = pt;
                let bt_ = self.c.block_tokens;
                let runs = self.reqs[r as usize].runs.clone();
                let ms: Vec<MatchResult> = (0..np).map(|w| self.p[w].radix.match_peek(&runs)).collect();
                let gpu: Vec<u64> = ms.iter().map(|m| m.dev as u64 * bt_).collect();
                let local: Vec<u64> = hits.iter().map(|&h| h.min(l)).collect();
                let mut hint: Vec<Option<(usize, u64)>> = vec![None; np];
                let mut exp: Vec<u64> = vec![0; np];
                for w in 0..np {
                    let s_ = (0..np)
                        .filter(|&k| k != w && hits[k] > 0)
                        .min_by(|&a, &b| hits[b].cmp(&hits[a]).then(a.cmp(&b)));
                    if let Some(s_) = s_ {
                        if hits[s_] >= self.c.pull_min_tokens
                            && hits[s_].saturating_sub(hits[w]) >= self.c.pull_min_incremental_tokens
                        {
                            hint[w] = Some((s_, hits[s_]));
                        }
                    }
                    let hs = hint[w].map(|(_, h)| h).unwrap_or(0);
                    exp[w] = local[w].max(hs).min(l);
                }
                let pend: Vec<u64> = (0..np).map(|w| self.true_backlog(w)).collect();
                let act: Vec<u64> = self.p.iter().map(|p| p.cs_active).collect();
                let own_hit = (0..np)
                    .min_by(|&a, &b| {
                        local[b].cmp(&local[a]).then(gpu[b].cmp(&gpu[a])).then(pend[a].cmp(&pend[b]))
                            .then(act[a].cmp(&act[b])).then(a.cmp(&b))
                    })
                    .unwrap();
                let ckey = self.chain_key(r);
                let (home, prev) = self.eft_home.get(&ckey).copied().unwrap_or((None, None));
                own = match home {
                    Some(h) if self.c.cs_eft_one_way && (h as usize) < np => h as usize,
                    _ => own_hit,
                };
                let ch = self.c.chunk_tokens;
                let mut wait = vec![0.0; np];
                let mut own_s = vec![0.0; np];
                let mut pull_s = vec![0.0; np];
                let mut dup = vec![0.0; np];
                let mut pulled = vec![0u64; np];
                for w in 0..np {
                    let b = self.p[w].beta;
                    let pl = exp[w].saturating_sub(local[w]);
                    let host_local = local[w].saturating_sub(gpu[w].min(local[w]));
                    wait[w] = self.backlog_secs(w, false);
                    own_s[w] = eft_secs(&b, ch, l - exp[w], exp[w], host_local, pl);
                    pull_s[w] = if pl > 0 {
                        self.c.cs_eft_pull_fixed_s + self.c.cs_eft_pull_s_per_100k * pl as f64 / 1e5
                    } else {
                        0.0
                    };
                    dup[w] = self.c.cs_eft_dup_rho * eft_secs(&b, ch, pl, 0, 0, 0);
                    pulled[w] = pl;
                }
                let excluded = match prev {
                    Some(pv) if self.c.cs_eft_one_way
                        && (pv as usize) < np
                        && pv as usize != own
                        && local[pv as usize] >= self.c.pull_min_tokens =>
                    {
                        Some(pv as usize)
                    }
                    _ => None,
                };
                let (t_, why) = eft_choose(
                    &wait,
                    &own_s,
                    &pull_s,
                    &dup,
                    &pulled,
                    own,
                    excluded,
                    self.c.cs_eft_burst_wait_s,
                    self.c.cs_eft_max_pull_tokens,
                    self.c.cs_eft_margin_s,
                );
                tgt = t_;
                match why {
                    1 => self.eft_counts[0] += 1,
                    2 => self.eft_counts[1] += 1,
                    3 => self.eft_counts[2] += 1,
                    4 => self.eft_counts[3] += 1,
                    _ => {}
                }
                let nh = if tgt != own { (Some(tgt as u16), Some(own as u16)) } else { (Some(tgt as u16), prev) };
                self.eft_home.insert(ckey, nh);
                let cost = l - exp[tgt];
                self.p[tgt].cs_pending += cost;
                self.p[tgt].cs_active += 1;
                self.reqs[r as usize].cs_cost = Some(cost);
                if self.c.pull_enabled {
                    if let Some((src, hs)) = hint[tgt] {
                        self.reqs[r as usize].pull_src = Some(src as u16);
                        self.reqs[r as usize].pull_hint = hs.saturating_sub(hits[tgt]);
                        self.reqs[r as usize].hint_expected = hs;
                    }
                }
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
        if pol != "cosched" && pol != "eft" {
            own = (0..np).min_by(|&a, &b| hits[b].cmp(&hits[a]).then(a.cmp(&b))).unwrap();
            if hits[own] == hits[tgt] {
                own = tgt;
            }
        }
        let c50 = hits.iter().filter(|&&h| 2 * h >= pt && h > 0).count() as u8;
        let c90 = hits.iter().filter(|&&h| 10 * h >= 9 * pt && h > 0).count() as u8;
        let rq = &mut self.reqs[r as usize];
        rq.copies50 = c50;
        rq.copies90 = c90;
        rq.p = tgt as u16;
        rq.owner = own as u16;
        rq.detour = tgt != own && hits[own] > 0;
        rq.route_best = best;
        rq.route_tgt = hits[tgt];
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
                let snap = self.c.d_load_mode == "tokens_snapshot";
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
        let p = self.reqs[r as usize].p as usize;
        self.p[p].bq_n += 1;
        self.p[p].bq_tok += self.reqs[r as usize].pt;
        self.p[p].bq.push(r);
        self.p[p].qp_next += 1;
        self.reqs[r as usize].qp_seq = self.p[p].qp_next;
        // Arrival match / bootstrap-prefetch re-match refresh the local LRU (H5) [approx: at intake].
        let runs = self.lkey(r);
        if self.c.touch_at_intake {
            let x = self.rx(p);
            self.p[x].radix.match_local(&runs, t);
        }
        if self.reqs[r as usize].pull_src.is_some() && self.c.pull_enabled && !self.c.pull_at_route {
            let jit = self.c.pull_start == "queue";
            let now_ok = if jit {
                // F1: pull now only if bootstrap + waiting <= 8 requests and queued input <= 1.1 M tokens.
                let n = self.p[p].bq_n + self.p[p].wq.len() as u64;
                let qt = self.p[p].bq_tok + self.queued_tokens(p);
                n <= 8 && qt <= 1_100_000
            } else {
                true
            };
            if now_ok {
                self.start_pull(t, r);
            } else {
                self.reqs[r as usize].jit_pending = true;
            }
        }
        // F3: S-ARRIVE lock after the arrival pull or its deferral (QPG on_arrival :219-229 ->
        // _try_arrive :378-402; skipped if a pull is in flight, depth < 8192, budget full). The
        // arrival match was the intake touch above.
        if self.c.queued_prefix_protect {
            self.guard_try_arrive(t, p, r, false);
        }
        if self.reqs[r as usize].t_alloc.is_some() {
            self.boot_done(t, r);
        }
        self.p_idle_update(t, p);
    }

    fn queued_tokens(&self, p: usize) -> u64 {
        let mut s: u64 = self.p[p].wq.iter().map(|&r| self.reqs[r as usize].pt).sum();
        if let Some(c) = self.p[p].chunked {
            let rq = &self.reqs[c as usize];
            s += rq.unc.saturating_sub(rq.sched);
        }
        s
    }

    /// F3 _try_arrive (QPG :378-402): set the retry pass; skip while pressure-suppressed or while a
    /// pull is in flight; match (the guard scan refreshes the LRU when `do_match`, H5); lock the
    /// deepest node with a host copy if depth >= 8192 and the budget allows.
    fn guard_try_arrive(&mut self, t: f64, p: usize, r: u32, do_match: bool) {
        if self.reqs[r as usize].qlock.is_some() {
            return;
        }
        let pass = self.p[p].pass_n;
        self.reqs[r as usize].qp_retry = pass + QPG_RETRY_AFTER_PASSES;
        if self.reqs[r as usize].qp_noarrive > pass || self.reqs[r as usize].pull_pending {
            return;
        }
        let bt = self.c.block_tokens;
        let runs = self.lkey(r);
        let m = if do_match { self.p[p].radix.match_local(&runs, t) } else { self.p[p].radix.match_peek(&runs) };
        let depth = m.total() as u64 * bt;
        let Some(deep) = m.path.last().map(|&(n, _)| n) else { return };
        if depth < self.c.guard_min_tokens {
            return;
        }
        if self.p[p].guard_used + depth > self.c.guard_budget_tokens {
            self.p[p].guard_skip_budget += 1;
            return;
        }
        self.guard_grant(p, r, deep, depth, if do_match { 1 } else { 0 });
        self.reqs[r as usize].qp_retry = 0;
    }

    /// F3 on_landed (QPG :231-252): lock the landed node (the deepest one with a host copy),
    /// replacing the request's older lock; may exceed the budget (the next pass trims).
    fn guard_landed(&mut self, p: usize, r: u32) {
        let bt = self.c.block_tokens;
        let runs = self.reqs[r as usize].runs.clone();
        let m = self.p[p].radix.match_peek(&runs);
        let depth = m.total() as u64 * bt;
        let Some(deep) = m.path.last().map(|&(n, _)| n) else { return };
        if depth < self.c.guard_min_tokens {
            return;
        }
        if let Some((old, _)) = self.reqs[r as usize].qlock {
            if old == deep {
                return;
            }
            self.guard_release(p, r);
        }
        self.guard_grant(p, r, deep, depth, 2);
    }

    fn guard_grant(&mut self, p: usize, r: u32, node: NodeId, depth: u64, source: usize) {
        self.p[p].radix.host_lock(Some(node), 1);
        self.p[p].guard_used += depth;
        self.p[p].guard_order.push(r);
        self.p[p].guard_grants[source] += 1;
        self.reqs[r as usize].qlock = Some((node, depth));
        self.reqs[r as usize].qlock_landed = source == 2;
    }

    /// F3 per-pass step after batch formation (QPG step :254-280; prefill.py:750-753): release the
    /// newest locks (highest arrival number) while over budget, with a retry after 64 passes
    /// (:275-276, :454-455); then examine up to 16 unlocked queued requests (waiting + bootstrap,
    /// oldest first) whose retry pass has come, while >= 8192 tokens of budget remain (:277-278,
    /// :404-421).
    fn guard_step(&mut self, t: f64, p: usize) {
        if !self.c.queued_prefix_protect {
            return;
        }
        self.p[p].pass_n += 1;
        let pass = self.p[p].pass_n;
        if self.c.guard_trim {
            while self.p[p].guard_used > self.c.guard_budget_tokens && !self.p[p].guard_order.is_empty() {
                let prot = self.c.oracle_protect_landed;
                let Some(r) = self.p[p]
                    .guard_order
                    .iter()
                    .copied()
                    .filter(|&x| !(prot && self.reqs[x as usize].qlock_landed))
                    .max_by_key(|&x| self.reqs[x as usize].qp_seq)
                else {
                    break;
                };
                self.guard_release(p, r);
                self.reqs[r as usize].qp_retry = pass + QPG_RETRY_AFTER_PASSES;
                self.p[p].guard_released_budget += 1;
            }
        }
        let room = |s: &Self| s.c.guard_budget_tokens.saturating_sub(s.p[p].guard_used);
        if self.c.guard_examine && room(self) >= self.c.guard_min_tokens {
            let mut cands: Vec<u32> = self.p[p]
                .wq
                .iter()
                .chain(self.p[p].bq.iter())
                .copied()
                .filter(|&x| {
                    let rq = &self.reqs[x as usize];
                    rq.qlock.is_none() && rq.qp_retry <= pass && !rq.cancel
                })
                .collect();
            cands.sort_by_key(|&x| self.reqs[x as usize].qp_seq);
            let mut examined = 0usize;
            for r in cands {
                if examined >= QPG_MAX_EXAMINE_PER_PASS || room(self) < self.c.guard_min_tokens {
                    break;
                }
                examined += 1;
                self.p[p].guard_examined += 1;
                self.guard_try_arrive(t, p, r, true);
            }
        }
    }

    /// F3 pressure: if host eviction left a shortfall, release the guard's NEWEST locks (highest
    /// arrival number; never the waiting-queue head's or the requester's) one at a time and evict
    /// again, up to 2x the shortfall (URC:2069-2097; QPG relieve_host_pressure :282-304,
    /// release_newest :306-317); a released request takes no arrival lock for 256 passes (:456-457).
    fn guard_pressure(&mut self, t: f64, p: usize, requester: Option<u32>) {
        let x = self.rx(p);
        let short = self.p[x].radix.host_used.saturating_sub(self.p[x].radix.host_cap);
        if short == 0 || !self.c.queued_prefix_protect {
            return;
        }
        let limit = QPG_PRESSURE_RELEASE_FACTOR * short * self.c.block_tokens;
        let head = self.p[p].wq.front().copied();
        let pass = self.p[p].pass_n;
        let mut order = self.p[p].guard_order.clone();
        order.sort_by_key(|&x| std::cmp::Reverse(self.reqs[x as usize].qp_seq));
        let mut released = 0u64;
        for r in order {
            if released >= limit || self.p[p].radix.host_used <= self.p[p].radix.host_cap {
                break;
            }
            if Some(r) == head || Some(r) == requester {
                continue;
            }
            if self.c.oracle_protect_landed && self.reqs[r as usize].qlock_landed {
                continue;
            }
            let d = self.reqs[r as usize].qlock.map(|x| x.1).unwrap_or(0);
            self.guard_release(p, r);
            self.reqs[r as usize].qp_noarrive = pass + QPG_PRESSURE_NO_ARRIVE_PASSES;
            released += d;
            self.p[p].guard_released_pressure += 1;
            self.p[p].radix.evict_host(t);
        }
    }

    fn guard_release(&mut self, p: usize, r: u32) {
        if let Some((n, d)) = self.reqs[r as usize].qlock.take() {
            self.reqs[r as usize].qlock_landed = false;
            self.p[p].radix.host_lock(Some(n), -1);
            self.p[p].guard_used -= d;
            self.p[p].guard_order.retain(|&x| x != r);
        }
    }

    /// Close the open idle segment of P `p` and, if P is idle now, open one classified by its
    /// queues: 0 nothing queued, 1 only bootstrap-queue work, 2 waiting work but none admissible.
    fn p_idle_update(&mut self, t: f64, p: usize) {
        if let Some((t0, cls)) = self.p[p].idle_since.take() {
            let a = t0.max(self.lo);
            let b = t.min(self.hi);
            if b > a {
                self.p[p].idle_win[cls as usize] += b - a;
            }
        }
        if self.p[p].run_items.is_none() {
            let cls = if !self.p[p].wq.is_empty() || self.p[p].chunked.is_some() {
                2
            } else if self.p[p].bq_n > 0 {
                1
            } else {
                0
            };
            self.p[p].idle_since = Some((t, cls));
        }
    }

    /// Issue a peer pull for request `r` (P1/P2): span from the destination's current match end
    /// to len-1 (page aligned); the owner grants the longest run it holds on host.
    fn start_pull(&mut self, t: f64, r: u32) {
        let bt = self.c.block_tokens;
        self.reqs[r as usize].jit_pending = false;
        let (src, dst) = {
            let rq = &self.reqs[r as usize];
            (rq.pull_src.unwrap() as usize, rq.p as usize)
        };
        let runs = self.reqs[r as usize].runs.clone();
        let pt = self.reqs[r as usize].pt;
        // Issue-time local match (refreshes the destination path).
        let lk = self.lkey(r);
        let have = self.p[dst].radix.match_local(&lk, t).total() as u64;
        let cap_blocks = pt.saturating_sub(1) / bt;
        if (cap_blocks.saturating_sub(have)) * bt < self.c.pull_min_tokens {
            return; // span < 256 tokens: no pull
        }
        self.reqs[r as usize].pull_attempts += 1;
        if self.reqs[r as usize].pull_first_issue_t.is_none() {
            self.reqs[r as usize].pull_first_issue_t = Some(t);
            let ns = self.tasks[self.reqs[r as usize].task as usize].inst;
            self.reqs[r as usize].pull_src_age = self.last_local.get(&(src as u16, ns)).map(|&x| t - x);
        }
        self.pulls_all += 1;
        let sm: MatchResult = self.p[src].radix.match_peek(&runs);
        let mut avail = (sm.total() as u64).min(cap_blocks);
        if self.c.oracle_pull {
            // O1: the hinted range is always there (oracle).
            let hb = (self.reqs[r as usize].hint_expected / bt).min(cap_blocks);
            avail = avail.max(hb);
        }
        let mut pulled_blocks = avail.saturating_sub(have);
        if self.c.oracle_pull && pulled_blocks * bt < self.c.pull_min_tokens {
            // Destination already holds (nearly) all of the hinted range: nothing to pull, no failure.
            let rq = &mut self.reqs[r as usize];
            rq.pull_start_t = Some(t);
            if rq.pull_outcome == 0 {
                rq.pull_outcome = 1;
            }
            return;
        }
        // Owner lease budget (64 GiB = 1.51 M tokens): the whole lookup is refused over budget.
        if !self.c.oracle_pull && pulled_blocks > 0 && self.p[src].leased_tok + pulled_blocks * bt > self.c.lease_budget_tokens {
            pulled_blocks = 0;
            self.lease_refused += 1;
        }
        if pulled_blocks * bt < self.c.pull_min_tokens {
            pulled_blocks = 0; // hit < 256 tokens is revoked
        }
        let hint_target = self.reqs[r as usize].hint_expected.min(pt.saturating_sub(1));
        if (have + pulled_blocks) * bt + bt <= hint_target {
            self.pull_short_all += 1;
        }
        let rq = &mut self.reqs[r as usize];
        rq.pull_start_t = Some(t);
        if rq.pull_attempts == 1 {
            rq.pull_dst_have = have * bt;
        }
        if pulled_blocks == 0 {
            self.lookup_miss += 1;
            if self.reqs[r as usize].pull_outcome == 0 {
                self.reqs[r as usize].pull_outcome = 3;
            }
            return;
        }
        {
            let full = (have + pulled_blocks) * bt + bt > hint_target;
            let rq = &mut self.reqs[r as usize];
            if rq.pull_outcome == 0 || rq.pull_outcome == 3 {
                rq.pull_outcome = if full { 1 } else { 2 };
            }
        }
        let upto = (have + pulled_blocks) as u32;
        let lease = TwoTierRadix::path_node_covering(&sm.path, upto);
        if self.c.lease_pins {
            self.p[src].radix.lease(lease, 1);
        }
        self.p[src].leased_tok += pulled_blocks * bt;
        // Leased path (root .. deepest leased node) for the demote walk.
        let mut acc = 0u32;
        let mut leased = Vec::new();
        for &(n, m) in &sm.path {
            if acc >= upto {
                break;
            }
            leased.push((n, m));
            acc += m;
        }
        let rq = &mut self.reqs[r as usize];
        rq.pull_lease = lease;
        rq.pull_src_path = leased;
        rq.pull_upto = upto;
        rq.pull_issue_blocks = pulled_blocks;
        rq.pulled += pulled_blocks * bt;
        self.pulled_all += pulled_blocks * bt;
        let dur = self.c.pull_fit.at(pulled_blocks * bt);
        rq.pull_end = Some(t + dur);
        rq.pull_pending = true;
        self.at(t + dur, Ev::PullDone(r));
    }

    fn pull_done(&mut self, t: f64, r: u32) {
        let bt = self.c.block_tokens;
        let (src, dst, lease, upto, lookup_t) = {
            let rq = &self.reqs[r as usize];
            (rq.pull_src.unwrap() as usize, rq.p as usize, rq.pull_lease, rq.pull_upto, rq.pull_start_t.unwrap_or(t))
        };
        let pulled_now = self.reqs[r as usize].pull_issue_blocks * bt;
        // P3: one host-only node with a fresh access time on the destination (anchor touched).
        let runs = runs_prefix(&self.reqs[r as usize].runs, upto);
        let host_only = self.c.pull_landing == "host";
        let before = self.p[dst].radix.stats.host_inserted_blocks;
        self.p[dst].radix.insert(&runs, t, host_only);
        self.guard_pressure(t, dst, Some(r));
        let ins = (self.p[dst].radix.stats.host_inserted_blocks - before) * bt;
        if t >= self.lo && t < self.hi {
            self.p[dst].pull_ins_win += ins;
        }
        // F3: lock at pull landing replaces the older lock (may exceed the budget).
        if self.c.queued_prefix_protect && self.reqs[r as usize].t_fwd.is_none() {
            self.guard_landed(dst, r);
        }
        // Complete release at the owner; demote walk (X) with the lookup time.
        if self.c.lease_pins {
            self.p[src].radix.lease(lease, -1);
        }
        self.p[src].leased_tok = self.p[src].leased_tok.saturating_sub(pulled_now);
        let defer = self.c.demote_defer_n > 0 || self.c.demote_defer_t > 0.0;
        if self.c.oracle_move {
            // O2: free the source copy of the transferred prefix now (oracle, not the engine path).
            let leased = runs_prefix(&self.reqs[r as usize].runs, upto);
            self.p[src].radix.move_free(&leased, lookup_t);
        } else if self.c.demote_source && defer {
            // E1: keep a pending demote for (chain, source); a newer move from the same source replaces it.
            let key = self.chain_key(r);
            let leased = runs_prefix(&self.reqs[r as usize].runs, upto);
            self.dem_seq += 1;
            let id = self.dem_seq;
            let v = self.dem_pend.entry(key).or_default();
            v.retain(|e| e.1 != src as u16);
            v.push((id, src as u16, leased, lookup_t, 0, r));
            if self.c.demote_defer_t > 0.0 {
                self.dem_due.insert(id, key);
                self.at(t + self.c.demote_defer_t, Ev::DemoteDue(id));
            }
        } else if self.c.demote_source {
            let sp = std::mem::take(&mut self.reqs[r as usize].pull_src_path);
            let (_, _, why) = if self.c.demote_relocate {
                let leased = runs_prefix(&self.reqs[r as usize].runs, upto);
                self.p[src].radix.demote_from(&leased, lookup_t)
            } else {
                self.p[src].radix.demote_walk(&sp, lookup_t)
            };
            self.demote_stop[why as usize] += 1;
        }
        self.reqs[r as usize].pull_lease = None;
        self.reqs[r as usize].pull_pending = false;
        self.p[src].radix.evict_dev(t);
        // A held request becomes admissible again: wake an idle P.
        if self.p[dst].run_items.is_none() && !self.p[dst].wq.is_empty() {
            if let Some(items) = self.form(t, dst) {
                self.start_iter(t, dst, items);
            }
        }
        self.p_idle_update(t, dst);
    }

    fn chain_key(&self, r: u32) -> (u64, u32) {
        let rq = &self.reqs[r as usize];
        (self.tasks[rq.task as usize].inst, self.agents[rq.agent_state as usize].agent)
    }

    /// E1: run a pending demote (release-time walk with the stored lookup time).
    fn dem_run(&mut self, src: usize, runs: &[(u64, u32)], lookup_t: f64) {
        let (_, _, why) = self.p[src].radix.demote_from(runs, lookup_t);
        self.demote_stop[why as usize] += 1;
        self.dem_deferred[0] += 1;
    }

    /// E1 timed demote (srcT): flag unless the chain came back to the source first (entry gone).
    fn demote_due(&mut self, id: u64) {
        let Some(key) = self.dem_due.remove(&id) else { return };
        let Some(v) = self.dem_pend.get_mut(&key) else { return };
        let Some(i) = v.iter().position(|e| e.0 == id) else { return };
        let e = v.swap_remove(i);
        self.dem_run(e.1 as usize, &e.2, e.3);
    }

    /// E1 at a chain request's first chunk on P: a request on the source cancels its pending demote
    /// (the chain came back); a request elsewhere counts toward srcN (the moving request excluded).
    fn dem_on_serve(&mut self, p: usize, r: u32) {
        if !(self.c.demote_defer_n > 0 || self.c.demote_defer_t > 0.0) {
            return;
        }
        let key = self.chain_key(r);
        let Some(v) = self.dem_pend.get_mut(&key) else { return };
        let mut run: Vec<(usize, Vec<(u64, u32)>, f64)> = Vec::new();
        let mut cancelled = 0u64;
        let n_need = self.c.demote_defer_n;
        v.retain_mut(|e| {
            if e.1 as usize == p {
                cancelled += 1;
                return false;
            }
            if e.5 != r {
                e.4 += 1;
                if n_need > 0 && e.4 >= n_need {
                    run.push((e.1 as usize, std::mem::take(&mut e.2), e.3));
                    return false;
                }
            }
            true
        });
        if v.is_empty() {
            self.dem_pend.remove(&key);
        }
        self.dem_deferred[1] += cancelled;
        for (src, runs, lt) in run {
            self.dem_run(src, &runs, lt);
        }
    }

    fn boot_done(&mut self, t: f64, r: u32) {
        let rq = &self.reqs[r as usize];
        let base = rq.p_recv.max(rq.t_alloc.unwrap_or(t));
        let hs = self.qdraw(0);
        let rdy = base + if hs.is_nan() { self.c.handshake_s } else { hs.max(0.0) };
        let rdy = rdy.max(t);
        self.reqs[r as usize].boot_ready = Some(rdy);
        self.at(rdy, Ev::PReady(r));
    }

    fn p_ready(&mut self, t: f64, r: u32) {
        if self.reqs[r as usize].p_ready.is_some() {
            return;
        }
        self.reqs[r as usize].p_ready = Some(t);
        let p = self.reqs[r as usize].p as usize;
        self.p[p].bq_n -= 1;
        self.p[p].bq_tok -= self.reqs[r as usize].pt;
        if let Some(i) = self.p[p].bq.iter().position(|&x| x == r) {
            self.p[p].bq.swap_remove(i);
        }
        let bt = self.c.block_tokens;
        let runs = self.reqs[r as usize].runs.clone();
        // Waiting-queue entry is not a match point in the engine (H5): no LRU refresh here.
        let x = self.rx(p);
        let est = self.p[x].radix.match_peek(&runs).total() as u64 * bt;
        self.reqs[r as usize].unc_est = self.reqs[r as usize].pt.saturating_sub(est).max(1);
        self.p[p].wq.push_back(r);
        if self.p[p].run_items.is_none() {
            if let Some(items) = self.form(t, p) {
                self.start_iter(t, p, items);
            }
        }
        self.p_idle_update(t, p);
    }

    /// F1 JIT scan (each scheduler pass): issue pending pulls of the first 8 waiting entries at
    /// position p <= 4 or with <= 131,072 full-input tokens ahead.
    fn jit_scan(&mut self, t: f64, p: usize) {
        let mut ahead: u64 = self.p[p].chunked.map(|c| {
            let rq = &self.reqs[c as usize];
            rq.unc.saturating_sub(rq.sched)
        }).unwrap_or(0);
        let n = self.p[p].wq.len().min(8);
        let mut issue = Vec::new();
        for pos in 0..n {
            let r = self.p[p].wq[pos];
            let rq = &self.reqs[r as usize];
            if rq.jit_pending && (pos <= 4 || ahead <= self.c.jit_lead_tokens) {
                issue.push(r);
            }
            ahead += rq.pt;
        }
        for r in issue {
            self.start_pull(t, r);
        }
    }

    /// Key of a request-local radix match: the engine matches the prompt minus its last token,
    /// page aligned (max_prefix_len = input_len - 1); only used with split_on_match (v4), where
    /// the split point depends on it. v1-v3: the full prompt runs.
    fn lkey(&self, r: u32) -> Vec<(u64, u32)> {
        let rq = &self.reqs[r as usize];
        if self.c.split_on_match {
            let cap = (rq.pt.saturating_sub(1) / self.c.block_tokens) as u32;
            runs_prefix(&rq.runs, cap)
        } else {
            rq.runs.clone()
        }
    }

    // ------------------------------------------------------------------------------ P engine
    fn first_chunk(&mut self, t: f64, p: usize, r: u32) {
        let bt = self.c.block_tokens;
        let runs = self.reqs[r as usize].runs.clone();
        self.reqs[r as usize].t_fwd = Some(t);
        // Best actual prefix anywhere at first chunk (gap-cause input).
        let fcb = self.p.iter_mut().map(|w| w.radix.match_peek(&runs).total() as u64 * bt).max().unwrap_or(0);
        self.reqs[r as usize].fc_best = fcb;
        let ns = self.tasks[self.reqs[r as usize].task as usize].inst;
        {
            let prev = self.sess_last_pd.get(&ns).copied();
            let ll = self.last_local.get(&(p as u16, ns)).copied();
            let rq = &mut self.reqs[r as usize];
            rq.prev_p = prev.map(|x| x.0);
            rq.prev_gap = prev.map(|x| rq.admit - x.1);
            rq.local_age = ll.map(|x| t - x);
        }
        self.last_local.insert((p as u16, ns), t);
        self.dem_on_serve(p, r);
        let lk = self.lkey(r);
        let x = self.rx(p);
        let m = self.p[x].radix.match_local(&lk, t);
        let mut dev = m.dev as u64 * bt;
        let mut host = m.host as u64 * bt;
        let pt = self.reqs[r as usize].pt;
        // Cap like SGLang: match at most len-1 tokens, page aligned -> the last page is recomputed.
        let capm = (pt.saturating_sub(1) / bt) * bt;
        if dev + host > capm {
            let over = dev + host - capm;
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
        self.p[x].radix.load_back(&m.path, t);
        let deep = m.path.last().map(|&(n, _)| n);
        self.p[x].radix.lock(deep, 1);
        // F3: the queued-prefix host lock is released at admission.
        self.guard_release(p, r);
        let beta = self.p[p].beta;
        let mut lb = beta[3] * host as f64 / 1e5 + beta[4] * stor as f64 / 1e5;
        if self.c.lb_hidden > 0.0 {
            let cap = (self.c.prefetch_max_tokens as f64).min(self.c.prefetch_max_fraction * pt as f64);
            let frac = if host + stor > 0 { (cap / (host + stor) as f64).min(1.0) } else { 0.0 };
            lb -= self.c.lb_hidden * frac * lb;
        }
        self.p[p].pf_set.push(r);
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

    /// One scheduler pass: form a batch, then the F3 guard step (after batch formation).
    fn form(&mut self, t: f64, p: usize) -> Option<Vec<BItem>> {
        let res = self.form_batch(t, p);
        self.guard_step(t, p);
        res
    }

    fn form_batch(&mut self, t: f64, p: usize) -> Option<Vec<BItem>> {
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
        if self.c.p_queue_order == "sjf_aging" && self.p[p].wq.len() > 1 {
            // Engine VARIANT (not SGLang behaviour): least (uncached estimate - aging x wait) first;
            // stable sort keeps FCFS order among equal keys.
            let a = self.c.sjf_aging_tok_per_s;
            let key = |s: &Self, r: u32| {
                let rq = &s.reqs[r as usize];
                rq.cs_cost.unwrap_or(rq.unc_est) as f64 - a * (t - rq.p_ready.unwrap_or(t)).max(0.0)
            };
            let mut v: Vec<(f64, u32)> = self.p[p].wq.iter().map(|&r| (key(self, r), r)).collect();
            v.sort_by(|x, y| x.0.total_cmp(&y.0));
            self.p[p].wq = v.into_iter().map(|x| x.1).collect();
        }
        if self.c.pull_start == "queue" {
            self.jit_scan(t, p);
        }
        let mut idx = 0usize;
        while budget > 0 && idx < self.p[p].wq.len() {
            let r = self.p[p].wq[idx];
            if self.reqs[r as usize].cancel {
                self.p[p].wq.remove(idx);
                continue;
            }
            // Hold: a request whose pull is in flight is skipped; later requests pass it.
            if self.reqs[r as usize].pull_pending {
                idx += 1;
                continue;
            }
            if self.reqs[r as usize].jit_pending {
                self.start_pull(t, r);
                if self.reqs[r as usize].pull_pending {
                    idx += 1;
                    continue;
                }
            }
            // F2: re-pull at admission from the same owner when the local match is short of the
            // hint by >= 8192 tokens (one attempt).
            if self.c.repull_on_admission && self.reqs[r as usize].pull_src.is_some()
                && self.reqs[r as usize].pull_attempts < 2 && !self.reqs[r as usize].repulled
            {
                let bt = self.c.block_tokens;
                let runs = self.reqs[r as usize].runs.clone();
                let local = self.p[p].radix.match_peek(&runs).total() as u64 * bt;
                let pt = self.reqs[r as usize].pt;
                let expected = self.reqs[r as usize].hint_expected.min(pt.saturating_sub(1));
                if expected > 0 && expected.saturating_sub(local) >= self.c.repull_min_tokens {
                    self.reqs[r as usize].repulled = true;
                    self.repulls += 1;
                    self.start_pull(t, r);
                    if self.reqs[r as usize].pull_pending {
                        idx += 1;
                        continue;
                    }
                }
            }
            self.p[p].wq.remove(idx);
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
        if !self.c.p_res_q.is_empty() {
            let u = self.urand() * 200.0;
            let q = &self.c.p_res_q[p % self.c.p_res_q.len()];
            if q.len() >= 201 {
                let norm = q.iter().sum::<f64>() / q.len() as f64;
                let k = (u as usize).min(199);
                svc *= (q[k] + (q[k + 1] - q[k]) * (u - k as f64)) / norm;
            }
        } else if self.c.p_noise_sigma > 0.0 {
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
        self.p_idle_update(t, p);
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
        self.p_idle_update(t, p);
    }

    fn prefill_done(&mut self, t: f64, p: usize, r: u32) {
        self.reqs[r as usize].t_pd = Some(t);
        if let Some(i) = self.p[p].pf_set.iter().position(|&x| x == r) {
            self.p[p].pf_set.swap_remove(i);
        }
        if self.reqs[r as usize].nin_held && self.c.smg_load_release == "prefill_done" {
            self.reqs[r as usize].nin_held = false;
            self.p[p].nin -= 1;
        }
        let runs = self.reqs[r as usize].runs.clone();
        let ns = self.tasks[self.reqs[r as usize].task as usize].inst;
        self.last_local.insert((p as u16, ns), t);
        if !self.reqs[r as usize].warm {
            self.sess_last_pd.insert(ns, (p as u16, t));
        }
        let x = self.rx(p);
        let before = self.p[x].radix.stats.host_inserted_blocks;
        let path = self.p[x].radix.insert(&runs, t, false);
        self.guard_pressure(t, p, Some(r));
        let ins = (self.p[x].radix.stats.host_inserted_blocks - before) * self.c.block_tokens;
        if t >= self.lo && t < self.hi {
            self.p[p].host_ins_win += ins;
        }
        let nd = path.last().copied();
        self.p[x].radix.lock(nd, 1);
        let old = self.reqs[r as usize].locked;
        self.p[x].radix.lock(old, -1);
        self.reqs[r as usize].locked = nd;
        let pt = self.reqs[r as usize].pt;
        let tr = self.qdraw(4);
        let tail = if tr.is_nan() {
            self.c.kv_tail.at(pt)
        } else {
            (self.c.kv_tail.a + self.c.kv_tail.b_per_100k * pt as f64 / 1e5 + tr).max(0.0)
        };
        self.at(t + tail, Ev::KvArrived(r));
    }

    fn kv_arrived(&mut self, t: f64, r: u32) {
        let p = self.reqs[r as usize].p as usize;
        if self.reqs[r as usize].nin_held && self.c.smg_load_release == "kv_arrived" {
            self.reqs[r as usize].nin_held = false;
            self.p[p].nin -= 1;
        }
        if let Some(cost) = self.reqs[r as usize].cs_cost.take() {
            self.p[p].cs_pending -= cost;
            self.p[p].cs_active -= 1;
        }
        self.reqs[r as usize].t_kv = Some(t);
        let lk = self.reqs[r as usize].locked.take();
        let x = self.rx(p);
        self.p[x].radix.lock(lk, -1);
        self.p[x].radix.evict_dev(t);
        let df = self.qdraw(1);
        let first = t + if df.is_nan() { self.c.d_first_s } else { df.max(0.0) };
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
        let (lo, hi) = (self.lo, self.hi);
        let dw = &mut self.d[d];
        if dw.b > 0 {
            dw.v += (t - dw.vt) * dw.rate;
            let a = dw.vt.max(lo);
            let b = t.min(hi);
            if b > a {
                let st = step_ms(dw.b, dw.s);
                dw.ib += dw.b as f64 * (b - a);
                dw.ib_inv += dw.b as f64 * (b - a) / st;
            }
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
        let runs = std::mem::take(&mut self.reqs[r as usize].runs);
        self.ideal.insert(&runs, t, false);
        self.reqs[r as usize].runs = runs;
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
            rep.e2e_p90_p95 = [percentile_in_place(&mut e2e, 90.0), percentile_in_place(&mut e2e, 95.0)];
        }
        rep.p_queue_mean = sum(&|r| r.t_fwd.unwrap() - r.p_ready.unwrap_or(r.t_fwd.unwrap())) / n;
        {
            // ---- user-matrix metrics (output only)
            fn pct7(v: &mut Vec<f64>, q: f64) -> f64 {
                if v.is_empty() {
                    return 0.0;
                }
                v.sort_by(|a, b| a.total_cmp(b));
                let h = (v.len() - 1) as f64 * q;
                let lo_i = h.floor() as usize;
                let hi_i = (lo_i + 1).min(v.len() - 1);
                v[lo_i] + (v[hi_i] - v[lo_i]) * (h - lo_i as f64)
            }
            let qs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.75];
            let mut tp: Vec<f64> = cmp
                .iter()
                .filter(|r| r.ct > 1)
                .map(|r| (r.done.unwrap() - r.first.unwrap()) / (r.ct - 1) as f64)
                .collect();
            rep.tpot = [pct7(&mut tp, 0.50), pct7(&mut tp, 0.99)];
            rep.uncached_per_req = sum(&|r| (r.pt as f64 - actual(r)).max(0.0)) / n;
            rep.prompt_tokens_scored = cmp.iter().map(|r| r.pt).sum();
            rep.uncached_tokens_scored = cmp.iter().map(|r| r.pt.saturating_sub(r.dev + r.host + r.stor)).sum();
            rep.output_tokens_scored = cmp.iter().map(|r| r.ct as u64).sum();
            let fill = |vals: Vec<(f64, f64)>, out: &mut [f64; 10]| {
                // (value, weight)
                if vals.is_empty() {
                    return;
                }
                let m = vals.iter().map(|x| x.0).sum::<f64>() / vals.len() as f64;
                let wsum: f64 = vals.iter().map(|x| x.1).sum();
                let wm = if wsum > 0.0 { vals.iter().map(|x| x.0 * x.1).sum::<f64>() / wsum } else { 0.0 };
                let mut v: Vec<f64> = vals.iter().map(|x| x.0).collect();
                out[0] = m;
                out[1] = wm;
                for (k, q) in qs.iter().enumerate() {
                    out[2 + k] = pct7(&mut v, *q);
                }
            };
            let mut o = [0.0; 10];
            fill(cmp.iter().map(|r| (actual(r) / r.pt.max(1) as f64, r.pt as f64)).collect(), &mut o);
            rep.hit_dist_actual = o;
            let mut o = [0.0; 10];
            fill(cmp.iter().map(|r| (r.ideal.min(r.pt) as f64 / r.pt.max(1) as f64, r.pt as f64)).collect(), &mut o);
            rep.hit_dist_ideal = o;
            let mut o = [0.0; 10];
            fill(
                cmp.iter().filter(|r| r.ideal > 0).map(|r| (actual(r) / r.ideal.min(r.pt) as f64, r.pt as f64)).collect(),
                &mut o,
            );
            rep.hit_dist_ratio = o;
            rep.rerouted_share = cmp.iter().filter(|r| r.detour).count() as f64 / n;
            let mut per: FxHashMap<u32, (u32, u32)> = FxHashMap::default();
            for r in &cmp {
                let e = per.entry(r.task).or_insert((0, 0));
                e.0 += 1;
                if r.detour {
                    e.1 += 1;
                }
            }
            let mut v: Vec<f64> = per.values().map(|&(a, b)| 10.0 * b as f64 / a as f64).collect();
            let mean = if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
            rep.reroutes_per10_session = [pct7(&mut v, 0.50), mean];
            // Distinct P per session over the last w seconds (all non-warm requests of the session).
            let mut by_task: FxHashMap<u32, Vec<(f64, u16)>> = FxHashMap::default();
            for r in self.reqs.iter().filter(|r| !r.warm) {
                by_task.entry(r.task).or_default().push((r.admit, r.p));
            }
            for v in by_task.values_mut() {
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
            for (wi, wsec) in [120.0f64, 300.0].iter().enumerate() {
                let mut d: Vec<f64> = Vec::new();
                for r in &cmp {
                    let v = &by_task[&r.task];
                    let mut seen = [false; 64];
                    let mut k = 0u32;
                    for &(a, pp) in v.iter() {
                        if a > r.admit {
                            break;
                        }
                        if a > r.admit - wsec && !seen[pp as usize % 64] {
                            seen[pp as usize % 64] = true;
                            k += 1;
                        }
                    }
                    d.push(k as f64);
                }
                let mean = if d.is_empty() { 0.0 } else { d.iter().sum::<f64>() / d.len() as f64 };
                let p50 = pct7(&mut d, 0.50);
                if wi == 0 {
                    rep.distinct_p_w120 = [mean, p50];
                } else {
                    rep.distinct_p_w300 = [mean, p50];
                }
            }
            rep.copies50 = [
                sum(&|r| r.copies50 as f64) / n,
                cmp.iter().filter(|r| r.copies50 >= 2).count() as f64 / n,
            ];
            rep.copies90 = [
                sum(&|r| r.copies90 as f64) / n,
                cmp.iter().filter(|r| r.copies90 >= 2).count() as f64 / n,
            ];
            let mut pq: Vec<f64> =
                cmp.iter().map(|r| r.t_fwd.unwrap() - r.p_ready.unwrap_or(r.t_fwd.unwrap())).collect();
            rep.p_queue_pct = [pct7(&mut pq, 0.50), pct7(&mut pq, 0.95)];
            let mut dw: Vec<f64> = cmp.iter().map(|r| r.t_alloc.unwrap_or(r.d_recv) - r.d_recv).collect();
            rep.d_wait_pct = [pct7(&mut dw, 0.50), pct7(&mut dw, 0.95)];
            let mut te: Vec<f64> = cmp.iter().map(|r| r.first.unwrap() - r.p_recv).collect();
            rep.ttft_engine = [pct7(&mut te, 0.25), pct7(&mut te, 0.50), pct7(&mut te, 0.90), pct7(&mut te, 0.99)];
            for (wi, wsec) in [120.0f64, 300.0].iter().enumerate() {
                let mut bins: FxHashMap<(u32, i64), u64> = FxHashMap::default();
                for r in &cmp {
                    let b = ((r.admit - lo) / wsec).floor() as i64;
                    *bins.entry((r.task, b)).or_insert(0) |= 1u64 << (r.p as u64 % 64);
                }
                let v: Vec<u32> = bins.values().map(|m| m.count_ones()).collect();
                let nb = v.len().max(1) as f64;
                let out = [v.iter().map(|&x| x as f64).sum::<f64>() / nb, 100.0 * v.iter().filter(|&&x| x >= 3).count() as f64 / nb];
                if wi == 0 {
                    rep.dp_bins_w120 = out;
                } else {
                    rep.dp_bins_w300 = out;
                }
            }
            let is_rr = |r: &Req| r.detour && r.route_tgt < r.route_best;
            let mut per: FxHashMap<u32, (u32, u32)> = FxHashMap::default();
            for r in &cmp {
                let e = per.entry(r.task).or_insert((0, 0));
                e.0 += 1;
                if is_rr(r) {
                    e.1 += 1;
                }
            }
            let mut v: Vec<f64> = per.values().filter(|x| x.0 >= 10).map(|&(a, b)| 10.0 * b as f64 / a as f64).collect();
            let mean = if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };
            rep.rr_per10_ge10 = [pct7(&mut v, 0.25), pct7(&mut v, 0.50), pct7(&mut v, 0.75), pct7(&mut v, 0.90), mean];
            let mut chains: FxHashMap<(u32, u32), Vec<(f64, u16, u16, bool)>> = FxHashMap::default();
            for r in &cmp {
                let ag = self.agents[r.agent_state as usize].agent;
                chains.entry((r.task, ag)).or_default().push((r.admit, r.p, r.owner, is_rr(r)));
            }
            let (mut nrr, mut ret, mut aba, mut abc) = (0u64, [0u64; 4], 0u64, 0u64);
            for xs in chains.values_mut() {
                xs.sort_by(|a, b| a.0.total_cmp(&b.0));
                for i in 0..xs.len() {
                    if !xs[i].3 {
                        continue;
                    }
                    nrr += 1;
                    let (a_, b_) = (xs[i].2, xs[i].1);
                    for m in 1..=3usize {
                        if xs[i + 1..(i + 1 + m).min(xs.len())].iter().any(|y| y.1 == a_) {
                            ret[m] += 1;
                        }
                    }
                    for y in &xs[i + 1..] {
                        if y.1 != b_ {
                            if y.1 == a_ {
                                aba += 1;
                            } else {
                                abc += 1;
                            }
                            break;
                        }
                    }
                }
            }
            let d = nrr.max(1) as f64;
            rep.aba = [100.0 * ret[1] as f64 / d, 100.0 * ret[2] as f64 / d, 100.0 * ret[3] as f64 / d, 100.0 * abc as f64 / (aba + abc).max(1) as f64];
        }
        rep.d_wait_mean = sum(&|r| r.t_alloc.unwrap_or(r.d_recv) - r.d_recv) / n;
        rep.prefill_mean = sum(&|r| r.t_pd.unwrap() - r.t_fwd.unwrap()) / n;
        rep.decode_mean = sum(&|r| r.done.unwrap() - r.first.unwrap()) / n;
        rep.p_boot_mean = sum(&|r| r.p_ready.unwrap_or(r.p_recv) - r.p_recv) / n;
        rep.tail_mean = sum(&|r| r.t_kv.unwrap() - r.t_pd.unwrap()) / n;
        rep.d_alloc_to_kv_mean = sum(&|r| r.t_kv.unwrap() - r.t_alloc.unwrap_or(r.t_kv.unwrap())) / n;
        let ib: f64 = self.d.iter().map(|d| d.ib).sum();
        let ibi: f64 = self.d.iter().map(|d| d.ib_inv).sum();
        rep.dec_step_tokw_ms = if ibi > 0.0 { ib / ibi } else { 0.0 };
        rep.dec_b_mean = ib / (self.c.score_s * self.d.len().max(1) as f64);
        // Same P as the previous request of the same task admission (by admit order, all non-warm requests).
        let mut last: FxHashMap<u32, u16> = FxHashMap::default();
        let mut same = 0usize;
        let mut tot = 0usize;
        let mut order: Vec<&Req> = self.reqs.iter().filter(|r| !r.warm).collect();
        order.sort_by(|a, b| a.admit.total_cmp(&b.admit));
        for r in order {
            if let Some(&pp) = last.get(&r.task) {
                if r.admit >= lo && r.admit < hi && !r.cancel && r.done.is_some_and(|d| d <= dur) {
                    tot += 1;
                    if pp == r.p {
                        same += 1;
                    }
                }
            }
            last.insert(r.task, r.p);
        }
        rep.same_p_as_prev_share = same as f64 / tot.max(1) as f64;
        let mut ages: Vec<f64> = Vec::new();
        for r in &cmp {
            if !(r.ideal > 0 && actual(r) == 0.0) {
                continue;
            }
            let mech = match (r.local_age, r.prev_p) {
                (_, None) => 3,
                (None, Some(_)) => 0,
                (Some(_), Some(pp)) if pp != r.p => 1,
                _ => 2,
            };
            rep.zero_hit_mech[mech] += 1;
            let gb = match r.prev_gap {
                None => 0,
                Some(g) if g < 0.0 => 0,
                Some(g) if g < 60.0 => 1,
                Some(g) if g < 100.0 => 2,
                Some(g) if g < 140.0 => 3,
                _ => 4,
            };
            rep.zero_hit_gap_bins[gb] += 1;
            rep.zero_hit_mech_by_gap[mech][gb] += 1;
            if r.pull_hint > 0 {
                rep.zero_hit_hinted_by_gap[gb] += 1;
            }
            if let Some(a) = r.local_age {
                ages.push(a);
            }
        }
        if !ages.is_empty() {
            rep.zero_hit_local_age = [
                percentile_in_place(&mut ages, 25.0),
                percentile_in_place(&mut ages, 50.0),
                percentile_in_place(&mut ages, 75.0),
            ];
        }
        // GCWS footprint.
        {
            let pool = self.c.turnover_pool_tokens as f64;
            let np = self.p.len();
            let mut ev: Vec<Vec<(f64, u64, u64)>> = vec![Vec::new(); np];
            for r in &self.reqs {
                if let Some(tp) = r.t_pd {
                    if tp >= 0.0 {
                        ev[r.p as usize].push((tp, self.tasks[r.task as usize].inst, r.pt));
                    }
                }
            }
            for e in &mut ev {
                e.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
            for (wi, w) in [120.0f64, 300.0].iter().enumerate() {
                let mut vals: Vec<f64> = Vec::new();
                let mut tt = lo;
                while tt < hi {
                    for e in &ev {
                        let a = e.partition_point(|x| x.0 <= tt - w);
                        let b = e.partition_point(|x| x.0 <= tt);
                        let mut mx: FxHashMap<u64, u64> = FxHashMap::default();
                        for x in &e[a..b] {
                            let v = mx.entry(x.1).or_insert(0);
                            if x.2 > *v {
                                *v = x.2;
                            }
                        }
                        vals.push(mx.values().sum::<u64>() as f64 / pool);
                    }
                    tt += 30.0;
                }
                if !vals.is_empty() {
                    let out = [percentile_in_place(&mut vals, 50.0), percentile_in_place(&mut vals, 90.0)];
                    if wi == 0 {
                        rep.gcws_w120 = out;
                    } else {
                        rep.gcws_w300 = out;
                    }
                }
            }
        }
        let cs: Vec<_> = self.census.iter().filter(|c| c.0 >= lo && c.0 < hi).collect();
        if !cs.is_empty() {
            let cap = self.c.p_host_tokens as f64;
            let k = cs.len() as f64;
            rep.pool_pinned_frac = cs.iter().map(|c| c.3 as f64 / cap).sum::<f64>() / k;
            rep.pool_device_resident_frac = cs.iter().map(|c| c.4 as f64 / cap).sum::<f64>() / k;
            rep.pool_demoted_frac = cs.iter().map(|c| c.5 as f64 / cap).sum::<f64>() / k;
            let mut us: Vec<f64> = cs.iter().map(|c| 1.0 - (c.3 + c.4) as f64 / cap).collect();
            rep.pool_usable_frac_mean = us.iter().sum::<f64>() / k;
            rep.pool_usable_frac_p10 = percentile_in_place(&mut us, 10.0);
        }
        rep.pulls_scored = w.iter().filter(|r| r.pulled > 0).count() as u64;
        rep.pulled_tokens_scored = w.iter().map(|r| r.pulled).sum();
        rep.pulls_all = self.pulls_all;
        rep.pulled_tokens_all = self.pulled_all;
        rep.pull_short_all = self.pull_short_all;
        rep.detour_share = w.iter().filter(|r| r.detour).count() as f64 / w.len().max(1) as f64;
        let bt = self.c.block_tokens;
        for (i, pw) in self.p.iter().enumerate() {
            let cap = self.c.turnover_pool_tokens as f64;
            let inflow = (pw.host_ins_win + pw.pull_ins_win) as f64 / self.c.score_s;
            rep.per_p_host_retention_cap_over_inflow_s.push(if inflow > 0.0 { cap / inflow } else { 0.0 });
            let dh = self.c.dh_counter_scale * pw.host_ins_win as f64 / self.c.score_s;
            let pl = pw.pull_ins_win as f64 / self.c.score_s;
            rep.per_p_turnover_counter_s.push(if dh + pl > 0.0 { cap / (dh + pl) } else { 0.0 });
            rep.per_p_turnover_dh_only_s.push(if dh > 0.0 { cap / dh } else { 0.0 });
            rep.per_p_pull_landed_tokens_win.push(pw.pull_ins_win);
            let ev: Vec<_> = pw.radix.evictions.iter().filter(|e| e.t >= lo && e.t < hi).collect();
            let blk: f64 = ev.iter().map(|e| e.blocks as f64).sum();
            let al: f64 = ev.iter().map(|e| e.age_last * e.blocks as f64).sum();
            let aw: f64 = ev.iter().map(|e| e.age_write * e.blocks as f64).sum();
            rep.per_p_host_age_last_s.push(if blk > 0.0 { al / blk } else { 0.0 });
            rep.per_p_host_age_write_s.push(if blk > 0.0 { aw / blk } else { 0.0 });
            let mut ages: Vec<f64> = ev.iter().map(|e| e.age_last).collect();
            rep.per_p_host_age_last_p50_s.push(if ages.is_empty() { 0.0 } else { percentile_in_place(&mut ages, 50.0) });
            let mut pc = [0.0; 5];
            if !ages.is_empty() {
                for (k, q) in [10.0, 25.0, 50.0, 75.0, 90.0].iter().enumerate() {
                    pc[k] = percentile_in_place(&mut ages, *q);
                }
            }
            rep.per_p_host_age_last_pct_s.push(pc);
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
        for k in 0..6 {
            let d = self.dm_mark[1][k].saturating_sub(self.dm_mark[0][k]);
            rep.demote_window[k] = if k % 2 == 1 { d * bt } else { d };
            rep.demote_stop_window[k] = self.dm_mark[1][6 + k].saturating_sub(self.dm_mark[0][6 + k]);
        }
        {
            let mut delays: Vec<f64> = Vec::new();
            let mut dec: [[Vec<f64>; 3]; 2] = Default::default();
            for r in &w {
                if r.pull_src.is_none() || r.hint_expected == 0 {
                    continue;
                }
                rep.hint_outcomes[0] += 1;
                match r.pull_outcome {
                    1 => rep.hint_outcomes[1] += 1,
                    2 => rep.hint_outcomes[2] += 1,
                    3 => rep.hint_outcomes[3] += 1,
                    _ => rep.hint_outcomes[4] += 1,
                }
                if let Some(ti) = r.pull_first_issue_t {
                    let ab = match r.pull_src_age {
                        None => 7,
                        Some(a) if a < 45.0 => 0,
                        Some(a) if a < 90.0 => 1,
                        Some(a) if a < 150.0 => 2,
                        Some(a) if a < 175.0 => 3,
                        Some(a) if a < 200.0 => 4,
                        Some(a) if a < 250.0 => 5,
                        Some(_) => 6,
                    };
                    rep.miss_by_src_age[ab][0] += 1;
                    if r.pull_outcome == 3 {
                        rep.miss_by_src_age[ab][1] += 1;
                    }
                    let k = if r.pull_outcome == 1 { Some(0) } else if r.pull_outcome == 3 { Some(1) } else { None };
                    if let (Some(k), Some(pr)) = (k, r.p_ready) {
                        dec[k][0].push(ti - r.start);
                        dec[k][1].push(pr - r.p_recv);
                        dec[k][2].push((ti - pr).max(0.0));
                    }
                    let dl = ti - r.start;
                    delays.push(dl);
                    let b = if dl < 5.0 { 0 } else if dl < 15.0 { 1 } else if dl < 30.0 { 2 } else { 3 };
                    rep.miss_by_delay[b][0] += 1;
                    if r.pull_outcome == 3 {
                        rep.miss_by_delay[b][1] += 1;
                    }
                }
            }
            if !delays.is_empty() {
                rep.pull_issue_delay = [percentile_in_place(&mut delays, 50.0), percentile_in_place(&mut delays, 90.0)];
            }
            for k in 0..2 {
                for j in 0..3 {
                    let v = &mut dec[k][j];
                    let x = if v.is_empty() { 0.0 } else { percentile_in_place(v, 50.0) };
                    if k == 0 {
                        rep.pull_decomp_full_p50[j] = x;
                    } else {
                        rep.pull_decomp_miss_p50[j] = x;
                    }
                }
            }
        }
        rep.tasks_started = self.tasks_started;
        rep.lookup_miss_all = self.lookup_miss;
        for w in &self.p {
            rep.moved_freed_tokens += w.radix.stats.moved_freed_blocks * self.c.block_tokens;
            rep.moved_flagged_tokens += w.radix.stats.moved_flagged_blocks * self.c.block_tokens;
            rep.partial_local_matches += w.radix.stats.partial_local_matches;
            rep.partial_rest_tokens += w.radix.stats.partial_rest_blocks * self.c.block_tokens;
            rep.match_splits += w.radix.stats.match_splits;
        }
        let oe: u64 = self.p.iter().map(|w| w.radix.stats.overcommit_events).sum();
        let ob: u64 = self.p.iter().map(|w| w.radix.stats.overcommit_blocks).sum();
        rep.overcommit_events = oe;
        rep.overcommit_tokens_mean = if oe > 0 { ob as f64 * self.c.block_tokens as f64 / oe as f64 } else { 0.0 };
        rep.guard_released_pressure = self.p.iter().map(|w| w.guard_released_pressure).sum();
        for (i, w) in self.p.iter().enumerate() {
            let shared = self.c.shared_pool;
            if shared && i > 0 {
                rep.per_p_audit.push([0, 0, 0, 0]);
                continue;
            }
            let mine = |r: &&Req| shared || r.p as usize == i;
            let locks: Vec<NodeId> = self.reqs.iter().filter(mine).filter_map(|r| r.locked).collect();
            let hlocks: Vec<NodeId> = self.reqs.iter().filter(mine).filter_map(|r| r.qlock.map(|x| x.0)).collect();
            let (ml, mh, nodes) = w.radix.pin_audit(&locks, &hlocks);
            let chk = if w.radix.check().is_err() { 1 } else { 0 };
            rep.per_p_audit.push([chk, ml, mh, nodes]);
        }
        {
            let win = (self.hi - self.lo).max(1e-9);
            for (i, w) in self.p.iter().enumerate() {
                let mut idle = w.idle_win;
                if let Some((t0, cls)) = w.idle_since {
                    let a = t0.max(self.lo);
                    if self.hi > a {
                        idle[cls as usize] += self.hi - a;
                    }
                }
                rep.per_p_idle_frac.push([idle[0] / win, idle[1] / win, idle[2] / win]);
                let xs: Vec<_> =
                    self.pstate.iter().filter(|x| x.1 as usize == i && x.0 >= self.lo && x.0 < self.hi).collect();
                let n = xs.len().max(1) as f64;
                let mut lk: Vec<f64> = xs.iter().map(|x| x.2 as f64).collect();
                rep.per_p_guard_locked_mean.push(lk.iter().sum::<f64>() / n);
                rep.per_p_guard_locked_p90.push(if lk.is_empty() { 0.0 } else { percentile_in_place(&mut lk, 90.0) });
                rep.per_p_guard_locks_mean.push(xs.iter().map(|x| x.3 as f64).sum::<f64>() / n);
                rep.per_p_bq_mean.push(xs.iter().map(|x| x.4 as f64).sum::<f64>() / n);
                rep.per_p_wq_mean.push(xs.iter().map(|x| x.5 as f64).sum::<f64>() / n);
                for k in 0..3 {
                    rep.guard_grants[k] += w.guard_grants[k];
                }
                rep.guard_skip_budget += w.guard_skip_budget;
                rep.guard_released_budget += w.guard_released_budget;
                rep.guard_examined += w.guard_examined;
            }
            for (i, d) in self.d.iter().enumerate() {
                let xs: Vec<_> =
                    self.dstate.iter().filter(|x| x.1 as usize == i && x.0 >= self.lo && x.0 < self.hi).collect();
                let n = xs.len().max(1) as f64;
                rep.per_d_used_frac_mean.push(xs.iter().map(|x| x.2 as f64).sum::<f64>() / n / d.cap.max(1) as f64);
                rep.per_d_fifo_mean.push(xs.iter().map(|x| x.3 as f64).sum::<f64>() / n);
            }
        }
        rep.lease_refused_all = self.lease_refused;
        rep.repulls_all = self.repulls;
        rep.demote_stop = self.demote_stop;
        rep.demote_deferred = [self.dem_deferred[0], self.dem_deferred[1], self.dem_pend.values().map(|v| v.len() as u64).sum()];
        rep.requests_sent = self.reqs.len() as u64;
        rep.eft_counts = self.eft_counts;
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
                    p_queue: match (r.t_fwd, r.p_ready) {
                        (Some(f), Some(p)) => f - p,
                        _ => -1.0,
                    },
                    d_wait: r.t_alloc.map(|a| a - r.d_recv).unwrap_or(-1.0),
                    prefill_s: match (r.t_pd, r.t_fwd) {
                        (Some(d), Some(f)) if d >= 0.0 => d - f,
                        _ => -1.0,
                    },
                    decode_s: match (r.done, r.first) {
                        (Some(d), Some(f)) => d - f,
                        _ => -1.0,
                    },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::hicache_p2p::trie::{Agent, Task};

    const B: [f64; 5] = [0.12, 0.023, 0.117, 0.0855, 0.0855];

    #[test]
    fn eft_secs_is_the_pass_model_with_amortized_pass_cost() {
        let u = 8192u64;
        let want = 0.12 + 0.023 * 8.192 + 0.117 * (8192.0 * 8193.0 / 2.0) / 1e9;
        assert!((eft_secs(&B, 8192, u, 0, 0, 0) - want).abs() < 1e-12);
        // Prefix depth adds attention pairs; host / pulled load-back add linearly.
        let deep = eft_secs(&B, 8192, u, 100_000, 0, 0);
        assert!((deep - want - 0.117 * 8192.0 * 100_000.0 / 1e9).abs() < 1e-12);
        let lb = eft_secs(&B, 8192, 0, 0, 200_000, 100_000);
        assert!((lb - (0.0855 * 2.0 + 0.0855)).abs() < 1e-12);
        assert_eq!(eft_secs(&B, 8192, 0, 50_000, 0, 0), 0.0);
    }

    #[test]
    fn eft_choose_moves_only_to_a_strictly_earlier_finish() {
        let pulled = [0u64, 100_000, 100_000];
        // Owner 0 waits 10 s; P1 finishes at max(1, 0.5) + 2 + 0.1 = 3.1 s; P2 at 7.1 s.
        let (t, why) = eft_choose(&[10.0, 1.0, 5.0], &[2.0; 3], &[0.0, 0.5, 0.5], &[0.0, 0.1, 0.1], &pulled, 0, None, 0.0, 0, 0.0);
        assert_eq!((t, why), (1, 1));
        // A long pull is not hidden by a short queue: P1 = max(0.2, 4.0) + 2 + 0.1 = 6.1 > owner 3 + 2.
        let (t, why) = eft_choose(&[3.0, 0.2], &[2.0; 2], &[0.0, 4.0], &[0.0, 0.1], &[0, 400_000], 0, None, 0.0, 0, 0.0);
        assert_eq!((t, why), (0, 0));
        // Ties and gains below the margin stay at the owner.
        let (t, why) = eft_choose(&[1.0, 1.0], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[0, 0], 0, None, 0.0, 0, 0.0);
        assert_eq!((t, why), (0, 0));
        let (t, _) = eft_choose(&[2.0, 1.0], &[1.0; 2], &[0.0; 2], &[0.0; 2], &[0, 0], 0, None, 0.0, 0, 1.5);
        assert_eq!(t, 0);
        // The duplication charge can cancel a small gain.
        let (t, _) = eft_choose(&[2.0, 1.5], &[1.0; 2], &[0.0; 2], &[0.0, 0.6], &[0, 50_000], 0, None, 0.0, 0, 0.0);
        assert_eq!(t, 0);
    }

    #[test]
    fn eft_choose_one_way_cap_and_burst_rules() {
        let wait = [10.0, 1.0, 4.0];
        let own_s = [1.0; 3];
        let z = [0.0; 3];
        // The best P (1) is the one the chain left: the next better P (2) is taken.
        let (t, why) = eft_choose(&wait, &own_s, &z, &z, &[0, 10, 10], 0, Some(1), 0.0, 0, 0.0);
        assert_eq!((t, why), (2, 1));
        // Only the excluded P is better: stay, reason one-way.
        let (t, why) = eft_choose(&[10.0, 1.0, 12.0], &own_s, &z, &z, &[0, 10, 10], 0, Some(1), 0.0, 0, 0.0);
        assert_eq!((t, why), (0, 2));
        // Pull cap: P1 would pull 120k > 100k, P2 pulls 20k and is still better than the owner.
        let (t, why) = eft_choose(&wait, &own_s, &z, &z, &[0, 120_000, 20_000], 0, None, 0.0, 100_000, 0.0);
        assert_eq!((t, why), (2, 1));
        let (t, why) = eft_choose(&wait, &own_s, &z, &z, &[0, 120_000, 120_000], 0, None, 0.0, 100_000, 0.0);
        assert_eq!((t, why), (0, 3));
        // Burst-only: no move while the owner's wait is below the threshold.
        let (t, why) = eft_choose(&wait, &own_s, &z, &z, &[0, 10, 10], 0, None, 12.0, 0, 0.0);
        assert_eq!((t, why), (0, 4));
        let (t, why) = eft_choose(&wait, &own_s, &z, &z, &[0, 10, 10], 0, None, 8.0, 0, 0.0);
        assert_eq!((t, why), (1, 1));
    }

    #[test]
    fn ideal_choose_least_finish_then_fewer_active_then_index() {
        assert_eq!(ideal_choose(&[3.0, 1.0, 2.0], &[0, 9, 0]), 1);
        assert_eq!(ideal_choose(&[1.0, 1.0, 2.0], &[5, 2, 0]), 1);
        assert_eq!(ideal_choose(&[1.0, 1.0, 1.0], &[2, 2, 2]), 0);
    }

    fn synth_workload(tasks: usize, turns: usize) -> Workload {
        let mut wl = Workload::default();
        for i in 0..tasks {
            let mut ev = Vec::new();
            for k in 0..turns {
                ev.push(Event::Generate {
                    runs: vec![(i as u64 * 1_000_000, 40 + 40 * k as u32)],
                    max_tokens: 64,
                    out_blocks: 1,
                    src_t: None,
                    src_api: None,
                });
                ev.push(Event::Wait { seconds: 3.0 });
            }
            wl.tasks.push(Task {
                id: format!("t{i}"),
                block_size: 64,
                agents: vec![Agent { id: "a".into(), routing_key: None, events: ev }],
                deps: Vec::new(),
            });
        }
        wl
    }

    fn small_cfg() -> SimConfig {
        SimConfig {
            n_p: 3,
            n_d: 3,
            concurrency: 9,
            duration_s: 400.0,
            warm_s: 50.0,
            score_s: 300.0,
            p_lat_s: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn shared_pool_ideal_spreads_requests_and_keeps_every_prefix_hit() {
        let wl = synth_workload(12, 40);
        let ct = FxHashMap::default();
        let c = SimConfig { p_policy: "ideal_eft".into(), shared_pool: true, ..small_cfg() };
        let (rep, rows) = run(c, &wl, &ct);
        assert!(rep.n_scored > 100, "scored {}", rep.n_scored);
        // One pool: a returning session hits its whole earlier prompt on whichever P it lands.
        assert!(rep.hit_sum_a_over_i > 0.99, "hit {}", rep.hit_sum_a_over_i);
        let used = rep.per_p_share.iter().filter(|&&s| s > 0.05).count();
        assert_eq!(used, 3, "share {:?}", rep.per_p_share);
        // Requests of one session move between P without losing the prefix.
        let mut moved = 0;
        let mut last: FxHashMap<u32, u16> = FxHashMap::default();
        for r in rows.iter().filter(|r| r.completed) {
            if let Some(&p) = last.get(&r.task_row) {
                if p != r.p && r.ideal > 0 {
                    moved += 1;
                    assert!(r.actual + 64 >= r.ideal.min(r.pt.saturating_sub(64)), "row {:?}", r);
                }
            }
            last.insert(r.task_row, r.p);
        }
        assert!(moved > 0);
        // Pin audit: no check error and no lock mismatch on the shared tree.
        for a in &rep.per_p_audit {
            assert_eq!(a[..3], [0, 0, 0], "audit {:?}", rep.per_p_audit);
        }
        assert!(rows.iter().filter(|r| r.completed).all(|r| r.p_queue >= 0.0 && r.decode_s >= 0.0));
    }

    #[test]
    fn eft_policy_runs_with_pulls_and_records_moves() {
        let wl = synth_workload(12, 40);
        let ct = FxHashMap::default();
        let c = SimConfig {
            p_policy: "eft".into(),
            cs_workload: "true".into(),
            pull_start: "queue".into(),
            repull_on_admission: true,
            queued_prefix_protect: true,
            demote_source: true,
            split_on_match: true,
            ..small_cfg()
        };
        let (rep, _) = run(c, &wl, &ct);
        assert!(rep.n_scored > 100);
        assert!(rep.hit_sum_a_over_i > 0.9, "hit {}", rep.hit_sum_a_over_i);
        for a in &rep.per_p_audit {
            assert_eq!(a[..3], [0, 0, 0], "audit {:?}", rep.per_p_audit);
        }
        // Moves happen only with a strictly earlier predicted finish; every move is counted.
        assert!(rep.eft_counts[0] as f64 >= rep.detour_share * rep.n_admitted_window as f64 - 1.0);
    }
}
