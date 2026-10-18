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

use super::cache::{
    DemoteScope, KV_FINISHED, KV_PREFETCH, KV_READY, MatchResult, NodeId, TwoTierRadix, run_blocks, runs_prefix,
};
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
    /// Attribution option (NOT the SMG gateway): cache-aware matches each P's actual radix (device +
    /// host, exact and current, like the co-schedule / EFT routers here) instead of the SMG
    /// approximate tree. Threshold, balance rule and load counter unchanged. false = SMG tree.
    pub smg_exact_hits: bool,
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
    /// One-way rule part 2 (with cs_eft_one_way): the P the chain moved away from last is not a
    /// detour target. false = keep the home P as the owner but allow any target (session-sticky
    /// hysteresis arm: stay home unless another P finishes > cs_eft_margin_s earlier).
    pub cs_eft_exclude_prev: bool,
    /// co_schedule pull cap (router VARIANT): a detour target whose pull (expected hit - own hit)
    /// exceeds this many tokens is not a candidate; the request stays on the cache-holding P unless
    /// another candidate within the cap wins. 0 = off.
    pub cs_max_pull_tokens: u64,
    /// P waiting-queue order (engine VARIANT): "fcfs" (engine) or "sjf_aging" = least
    /// (uncached estimate - sjf_aging_tok_per_s x seconds in the waiting queue) first; held requests
    /// are still skipped, the continuing chunk still goes first.
    pub p_queue_order: String,
    pub sjf_aging_tok_per_s: f64,
    /// Late D (PR #39 data plane; SWEEP-LATED 2026-10-12). "d0" = today (D picked at arrival, D KV
    /// reserved at bootstrap before the P waiting queue). "alloc" = D picked at arrival, its KV is
    /// allocated at the bind (lead window) instead. "choose" = D picked at the bind by d_bind_rule
    /// ("po2": power of two on the polled token snapshot; "d1": fewest in-flight tokens, router
    /// counter) with the KV allocated at the bind. Defaults keep every older arm.
    pub d_mode: String,
    pub d_bind_rule: String,
    /// P binder, lead mode "work" (late_d.py @e126b456d4 :1510-1593, 1700-1827, 1941-1974): late rooms
    /// wait in P's bootstrap queue; once per P pass (and when a room turns bindable or an ADMIT turns
    /// HELD) unbound rooms with an INTENT are scanned in arrival order and bound while the window is
    /// open: chunk_left <= target, A < ld_lead_max_reqs, and (A < ld_lead_min_reqs or chunk_left +
    /// T x (1 - hit) < target). A = bound requests in the waiting queue + decisions younger than
    /// hold_s; T = their prompt tokens; target = need_iters x chunk; need_iters = clamp(ceil(T_bind /
    /// T_iter) + margin + boost, 1, ld_need_iters_cap); T_bind = EWMA(decision -> bootstrap done,
    /// samples clipped at 2 s) + 2 x EWMA |dev|; T_iter = EWMA of back-to-back busy passes; hit = EWMA
    /// cached / prompt at launch; hold_s = max(ld_hold_detect_s, min(T_bind, 2 s)). Boost +1 per
    /// starved pass (no runnable request, bindable rooms waiting), -1 after ld_starve_decay calm
    /// passes, cap 6. Rule 4: a D with >= ld_max_outstanding_per_d outstanding binds from this P
    /// (decided, not bound) is not eligible; a room without an eligible D is skipped.
    pub ld_lead_margin_iters: u32,
    pub ld_lead_min_reqs: u32,
    pub ld_lead_max_reqs: u32,
    pub ld_need_iters_cap: u32,
    pub ld_max_outstanding_per_d: u32,
    pub ld_ewma_alpha: f64,
    pub ld_hold_detect_s: f64,
    pub ld_starve_decay: u32,
    /// Latencies: decision -> ADMIT processed at D (choose: + router round trip), D allocation ->
    /// bootstrap done on P (metadata + P poll).
    pub ld_admit_latency_s: f64,
    pub ld_choose_latency_s: f64,
    pub ld_meta_latency_s: f64,
    /// D admission of late entries: first fit in ADMIT order (a failed budget check skips); the
    /// oldest entry held >= ld_claim_passes D passes (at step_ref_ms each) becomes the claim, and
    /// others allocate only if free - claim need >= need.
    pub ld_claim_passes: u32,
    /// D step overhead per held entry and per DEFERRED leg (ms per decode step; 0 = off).
    pub ld_held_scan_ms: f64,
    /// Fixed D step overhead with late D on (ms per decode step; the per-pass prealloc walk, handshake
    /// poll and inbox work of late D; fitted on the GB300 TPOT shift; 0 = off).
    pub ld_step_overhead_ms: f64,
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
    /// S2 serving-D model (option; false = no D cache, as before). Each D keeps a two-tier radix:
    /// device tier = the D KV pool not held by active reservations, host tier = d_host_ratio x the
    /// pool, write-through, LRU. The prompt is inserted at decode finish (the trace's output blocks
    /// never recur in a later prompt, so they are not inserted); the prefix is matched and locked at
    /// D prealloc and the P sends only [D prefix, len) (KV tail). Reservations are unchanged (no D
    /// memory saving for a cached prefix).
    pub d_cache: bool,
    pub d_host_ratio: f64,
    /// S2 read before P admission of [local, avail) from a D copy (needs d_cache): "off"; "a" = from
    /// the serving D after a hinted pull left the P short; "b" = "a" + the same read for any request;
    /// "c" = "b" with the D that served the chain's previous turn as the source (oracle bound). A read
    /// happens when avail - local >= s2_min_tokens; it takes pull_fit time and lands like a pull.
    pub s2_read: String,
    pub s2_min_tokens: u64,
    /// Session retention (option; sglang-private PR #49 engine + Morphosis PR #25 router, adoption
    /// plan wf_b21a8cdb-0dc; default off = every older arm bit-identical). Session = the replay's
    /// agent trajectory (agent state), version = its request counter (+1 per request). The router
    /// registers (session, version) on the chosen P; the P counts one reference per session on the
    /// anchor path of the full input (cache.rs). On a P move the old P keeps its references until the
    /// new P reports the current version READY (the full input on the host tier, after write-through
    /// or pull landing): the router polls the current P every kvret_poll_s and then sends 'release'
    /// to each retired P (delivered after kvret_ctl_s). The agent's last request completing sends
    /// 'end' to every P the session used (kvret_end). Host eviction on P: class 0 (known zero) first,
    /// class 1 keeps the base order; #49 host recency; GPU-backed duplicates reclaimable
    /// (kvret_dup_reclaim). D side unchanged (no tags reach D). Requires demote-source off.
    pub kv_retention: bool,
    /// With kv_retention: "ready_end" (default: #25 release after READY on a move + END),
    /// "transfer_end" (the user's design: on a move WITH a pull, the source P's registration of the
    /// session is released when the pulled pages are committed on the destination, at the same moment
    /// as the destination's +1; moves without a pull keep ready_end; END as ready_end),
    /// "end_only" (no release on a move; END only), "recency_only" (the #49 engine without session
    /// tags, as with SMG cache-aware: every range untracked = class 1; only the host-recency changes
    /// and the class-1 order apply; no router control traffic).
    pub kvret_mode: String,
    pub kvret_poll_s: f64,
    pub kvret_ctl_s: f64,
    pub kvret_dup_reclaim: bool,
    pub kvret_end: bool,
    /// S4 idle END (option; 0 = off): with session tags, an agent whose last request completed
    /// kvret_idle_s seconds ago and that sent no newer request is treated as ended (END to every P
    /// the session used, as the trajectory END). A later request of the same agent starts a new
    /// session identity (version 1), so its references come back on the P it is placed on.
    pub kvret_idle_s: f64,
    /// S3 (d) attribution option (default off): pull landings do not displace anything; the host
    /// blocks a pull adds on the destination are outside its host capacity while they stay resident.
    pub pull_free_capacity: bool,
    /// S5 router options (defaults off = every older arm bit-identical).
    /// co_schedule move mode: "pull" (default: a move to a P with less cache pulls from the best
    /// holder) or "recompute" (pull hints ignored: expected hit = the P's own cache, a move recomputes).
    pub cs_move_mode: String,
    /// co_schedule router-state error: each P's backlog estimate x (1 + U(-e, e)), drawn per P per
    /// decision (0 = off).
    pub cs_noise_queue: f64,
    /// co_schedule predictor error: each candidate's total (backlog + own uncached tokens) x
    /// exp(sigma Z), Z ~ N(0, 1) per P per decision (0 = off). MAPE = exp(s^2/2) (2 Phi(s) - 1).
    pub cs_noise_sigma: f64,
    /// co_schedule state age (s, 0 = off): the router sees each P's backlog as sampled every 0.1 s,
    /// cs_state_age_s old, plus (cs_state_own_fix) its own charges routed to that P since the sample.
    pub cs_state_age_s: f64,
    pub cs_state_own_fix: bool,
    /// EFT-path arms from OR-MODELS-P-ROUTING (wf_077a930f): "off" | "fpi" (first policy iteration:
    /// a = start + x + lambda x (2u + x) / (2 (1 - rho)) over (owner, local), (other, pull),
    /// (other, recompute); argmin, ties to the owner) | "wfa" (work-function algorithm per agent chain
    /// on FPI costs) | "ski" (ski-rental: move when the excess cost built up on the holder >= the
    /// move cost) | "skir" (randomized threshold M ln(1 + (e - 1) U)). lambda_p, rho_p are measured
    /// under owner-always routing during the first cs_fpi_pilot_s seconds (the router routes
    /// owner-always then); rho capped at cs_fpi_rho_cap. Ski excess decays with cs_ski_tau_s.
    pub cs_fpi: String,
    pub cs_fpi_pilot_s: f64,
    pub cs_fpi_rho_cap: f64,
    pub cs_ski_tau_s: f64,
    /// wfa / ski / skir move cost M(k): "seconds" (default; OR-MODELS S2 as written: min(pull wait
    /// beyond the target backlog + restore, recompute extra) in plain P seconds) or "fpi" (the FPI
    /// cost of executing the move minus the as-if-local FPI cost on k, i.e. in the same units as the
    /// per-turn costs, including the externality of the extra work).
    pub cs_fpi_move_cost: String,
    /// S6 options (defaults off = every older arm bit-identical).
    /// EFT path: duplicate-token charge, seconds per pulled token (the pulled tokens then exist on
    /// >= 2 P), added to each candidate's predicted finish (TRADE lambda).
    pub cs_eft_dup_per_token_s: f64,
    /// SMG cache-aware hybrids for the decomposition of the P1 no-pull gain: comma list of request
    /// classes routed by the P1 rule (least queued uncached tokens, no pull, P1 no-pull candidates)
    /// instead of the SMG rule: "cold" (best hit < pull_min_tokens), "low" (best hit <= smg_threshold
    /// x prompt), "multi" (>= 2 P within pull_min_incremental_tokens of the best hit), "single" (the
    /// rest). Empty = off.
    pub smg_p1_classes: String,
    /// SMG load signal: "requests" (default: in-flight requests, smg_abs) or "uncached_tokens"
    /// (CA-U: live queued uncached tokens of each P, threshold smg_abs_tokens; smg_rel unchanged).
    pub smg_load: String,
    pub smg_abs_tokens: u64,
    /// S6 diagnostic (EFT path): the backlog seconds of each P also count the requests the router
    /// already sent to it that are still in transit or in its bootstrap queue (an in-flight ledger;
    /// without it, near-simultaneous decisions do not see each other).
    pub cs_eft_wait_transit: bool,
    /// S7 PM-EFT options on the EFT path (all default off = older arms bit-identical).
    /// H: hand-off seconds added to the predicted finish of a P when the request pulls there.
    pub cs_eft_handoff_s: f64,
    /// Noise-aware move margin: move only if C_o - C_w > margin_s + k * sqrt(B_o^2 + B_w^2).
    pub cs_eft_margin_k: f64,
    /// Owner ties (equal cached prefix) go to the P with the smaller predicted finish.
    pub cs_eft_owner_tie_c: bool,
    /// Owner = the P with the least backlog among P within this many tokens of the best cached
    /// prefix (C4 no-pull tie-break; 0 = off).
    pub cs_eft_owner_near_tokens: u64,
    /// Predictor error on the EFT path: B and S each x exp(sigma Z) per P per decision; B also
    /// x (1 + U(-e, e)) (backlog error).
    pub cs_eft_noise_sigma: f64,
    pub cs_eft_noise_queue: f64,
    /// State age on the EFT path (s, 0 = off): B = max(0, B sampled every 0.1 s, age old, minus the
    /// time since the sample) + seconds the router itself routed to that P since the sample.
    pub cs_eft_state_age_s: f64,
    /// R2 lambda-pressure (0 = off): lambda_w = lambda x (1 + gain x max(0, rate / ref - 1)), rate =
    /// EWMA (tau) of class-1 host-evicted tok/s on the destination P ("worker") or on all P ("global").
    pub cs_eft_lp_ref: f64,
    pub cs_eft_lp_scope: String,
    pub cs_eft_lp_tau_s: f64,
    pub cs_eft_lp_gain: f64,
    /// B2 C4-gated (0 = off): a request leaves its owner only when an item with >= big uncached
    /// tokens and >= theta predicted seconds is queued or running on the owner.
    pub cs_eft_gate_big_tokens: u64,
    pub cs_eft_gate_theta_s: f64,
    /// Multiplier on the trace's think waits (after the cap; 1.0 = as recorded). Used to match req/s.
    pub wait_scale: f64,
    /// B1 C7-hold (0 = off; EFT path): dispatch to the owner (most cached prefix, ties -> least backlog)
    /// when its backlog <= cs_hold_passes full-chunk passes; else hold the request in the router. Every
    /// cs_hold_tick_s each P releases its own held requests (FIFO) while under that limit; with
    /// cs_hold_steal a P with backlog <= cs_hold_idle_s takes the held request with the largest gain
    /// g = B_o + S_o + w S_o min(n_behind, 3) - (T_pull + H + S_j) - lambda x, if g > cs_hold_margin_s.
    /// Cool-downs: a P that received a steal is no steal source for cs_hold_worker_cd_s; a session
    /// that moved is not stolen again for cs_hold_sess_cd_s.
    pub cs_hold_passes: f64,
    pub cs_hold_steal: bool,
    pub cs_hold_margin_s: f64,
    pub cs_hold_w: f64,
    pub cs_hold_idle_s: f64,
    pub cs_hold_tick_s: f64,
    pub cs_hold_worker_cd_s: f64,
    pub cs_hold_sess_cd_s: f64,
    /// Output only: share of P queue time (ready -> forward) that overlaps a fully idle other P.
    pub idle_overlap_metric: bool,
    /// S7: the copy price lambda applies to every token a candidate would duplicate (best cached
    /// prefix on any P - its own), pulled or recomputed; false = pulled tokens only.
    pub cs_eft_dup_on_recompute: bool,
    /// S8 PRED-GATE (EFT path; all default off). Forecast error: each P's forecast (B and S together) is
    /// multiplied per decision by exp(scale x e), e drawn from a table of 201 log(pred / actual) quantiles
    /// (uniform probabilities 0..1) of a real predictor; the idle table applies when the P has no
    /// backlog (empty = the busy table). Empty busy table = no noise (oracle forecast).
    pub cs_pred_noise: Vec<f64>,
    pub cs_pred_noise_idle: Vec<f64>,
    pub cs_pred_noise_scale: f64,
    /// Gate: owner o (most cached prefix, ties by C); w* = argmin over w != o of C_w; move only if
    /// (C_o - C_w*) - m sqrt(b_o^2 + b_w*^2) > gmin, b_x = k(idle/busy) x forecast_x + floor, where
    /// forecast_x = max(B_x, T_pull_x) + S_x (the copy price is not part of the forecast).
    pub cs_pred_gate: bool,
    pub cs_pred_m: f64,
    pub cs_pred_gmin_s: f64,
    pub cs_pred_k_idle: f64,
    pub cs_pred_k_busy: f64,
    pub cs_pred_floor_s: f64,
    /// Output only: record the routing-time forecast of the selected P (ingress base + max(B, T_pull) + S;
    /// noisy and noise-free) and score it against the actual decision -> prefill-done time.
    pub pred_metric: bool,
    /// LATE-P (late P binding, 2026-10-17; all default off = every older arm bit-identical; needs p_policy "eft").
    /// Requests wait in one router queue. The router dispatches request r to P w only when w is predicted to be
    /// nearly free: D_lo(w) = D(w) - lpb_bnd_scale x b_busy(D(w)) <= lead(w, r) = recv(r) + max(boot, T_pull(r, w))
    /// + T_iter(w). D(w) = backlog seconds incl. bootstrap queue + in-flight ledger - elapsed part of the running
    /// pass, x the forecast-error factor of w (redrawn from cs_pred_noise at every pass end of w = one push
    /// message). recv = ingress fit + online residual (EWMA mean + lpb_rx_k x EWMA |dev|); boot = receive ->
    /// waiting-queue entry (same calibration; before samples: handshake + late-D latencies); T_iter = EWMA of
    /// back-to-back passes. Queue order lpb_order: "fifo" (arrival) or "hrrn" (highest (wait + S_own) / S_own).
    /// A request goes to its owner (most cached prefix) when the owner is open; a cold request (best hit < the
    /// pull floor) to the open P with the earliest predicted finish; with lpb_moves to an open non-owner w only if
    /// (C_o - C_w - dup_w) - sqrt(b_o^2 + b_w^2) > cs_pred_gmin_s, C_o = completion when waiting for the owner
    /// behind the owner's older queued requests. dup_w = cs_eft_dup_per_token_s x duplicated tokens.
    pub lpb: bool,
    pub lpb_order: String,
    pub lpb_moves: bool,
    pub lpb_rx_k: f64,
    pub lpb_tick_s: f64,
    /// Sensitivity: receipt-delay forecast x scale; dispatch bound x scale (0 = trust the forecast).
    pub lpb_rx_scale: f64,
    pub lpb_bnd_scale: f64,
    /// PRED-GWQ (2026-10-18, default off): predicted-drain fill. Only a request that is free to go to any P enters
    /// the LATE-P router queue; a request whose largest local hit exceeds the second largest by at least
    /// chunk_tokens (bound to its owner) takes the normal P choice (p_policy eft = TRADE) at arrival and waits in the
    /// engine queue (queued prefix protected, as without the router queue).
    pub lpb_hold_free: bool,
    /// PRED-GWQ: extra lead (s) added to every P's lead (a deeper engine queue: + R x extra tokens).
    pub lpb_lead_extra_s: f64,
    /// r5 bound curves (deploy mode, level 0.8): knots (s) and bounds (s) per bucket (idle, busy). Used by LATE-P
    /// and, with cs_pred_curve, by the PRED-GATE instead of k x forecast. Empty = 0.
    pub bnd_idle_k: Vec<f64>,
    pub bnd_idle_b: Vec<f64>,
    pub bnd_busy_k: Vec<f64>,
    pub bnd_busy_b: Vec<f64>,
    pub cs_pred_curve: bool,
    /// PRED-EFT+HOL: each candidate's cost gets mu x a_w x S_w x (B_w + S_w / 2) (head-of-line externality on the
    /// requests that arrive at w while the request waits and runs); a_w = requests routed to w per second over the
    /// last cs_hol_window_s. 0 = off.
    pub cs_hol_mu: f64,
    pub cs_hol_window_s: f64,
    /// HOL form: "light" = a_w S (B + S / 2) (one delay per later arrival, light load); "bp" = a_w S (B + S) /
    /// (1 - rho_w) (the delay propagates through the busy period), rho_w = (sum of the predicted S of requests routed
    /// to w in the window) / window, capped at cs_hol_rho_cap. Both are computed every decision (shadow counters).
    pub cs_hol_form: String,
    pub cs_hol_rho_cap: f64,
    /// HOL arrival-rate input: "routed" (requests the router sent to w; feeds back on routing) or "uniform" (all
    /// routed requests / n_p for every P, same for rho: no feedback from the router's own placements).
    pub cs_hol_rate: String,
    /// Output only: every 0.1 s of the score window, P idle while work waits elsewhere (router queue or another
    /// P's waiting queue) and the per-P outstanding uncached tokens max / mean.
    pub iww_metric: bool,
    /// LAMBDA study (2026-10-17; all default off = every older arm bit-identical).
    /// Ghost list: per P the blocks most recently deleted from the P (gone from both tiers), up to ghost_frac x the
    /// host capacity, FIFO by deletion. A request's ghost hit = blocks beyond its actual match (local + pulled) on the
    /// serving P, contiguous from the match end, that are in that P's ghost: they would have hit with a host pool
    /// larger by the ghost size. h = ghost-hit tokens per second / ghost tokens = marginal hit rate of host capacity.
    /// 0 = off.
    pub ghost_frac: f64,
    /// Online estimators (EWMA time constant lam_tau_s, sampled every lam_tick_s): per P ghost-hit rate, ghost size,
    /// class-0 (known-zero) host tokens, class-0 and all host-evicted tok/s, busy fraction, requests on the P
    /// (bootstrap + waiting + prefilling), prefill tokens per second; busy periods (P fully idle at both ends).
    pub lam_tau_s: f64,
    pub lam_tick_s: f64,
    /// lambda_auto (EFT path): "off" | "global" | "pair". The copy price per pulled token (s) becomes
    /// lam_scale x q x h x T_dup / R. h as above; T_dup = class-0 host tokens / class-0 host-evicted tok/s (residence
    /// of the released source copy; capped at the host turnover); q = E2E seconds per second of extra P work (lam_q);
    /// R = prefill tokens per busy second. "global" = pool-wide estimates; "pair" = h, q, R of the destination and
    /// T_dup of the pull source. Until lam_min_s seconds of samples exist, cs_eft_dup_per_token_s applies.
    pub lam_auto: String,
    /// q proxy: "sat" = concurrency / n_p x busy fraction; "bp" = requests from a random request to the end of its
    /// busy period (sum n(n+1)/2 / sum n over busy periods); "L" = 1 + requests on the P; "one" = 1.
    pub lam_q: String,
    /// Induced recompute per pulled token: "ghost" = h x T_dup (local marginal value); "one" = 1 (each pulled
    /// token displaces one live token that is recomputed later: the full-displacement bound). Price = q x mu / R.
    /// lam_q "bpsat" = min(busy-period count incl. the open period, concurrency / n_p x busy fraction).
    pub lam_mu: String,
    pub lam_scale: f64,
    pub lam_min_s: f64,
    /// Forced moves (counterfactual probe; separate RNG stream): an EFT decision that keeps the owner although the
    /// best non-owner P with a pull (least price-free total max(wait, pull) + own) is cheaper, with owner hit >=
    /// fm_min_tokens, is sent to that P with probability fm_rate. 0 = off.
    pub fm_rate: f64,
    pub fm_min_tokens: u64,
    /// Work injection (counterfactual probe; same separate stream): with probability inj_rate a non-warm request's
    /// first prefill pass takes inj_s more seconds (no cache effect). 0 = off.
    pub inj_rate: f64,
    pub inj_s: f64,
    /// P2PMEM audit (2026-10-18; options, NOT engine behaviour; default off = every older arm bit-identical; need
    /// kv_retention with kvret_mode "transfer_end"). At the transfer_end source release of a move with a pull:
    /// kvret_move_free = TRUE MOVE: the released class-0 source copy (the transferred prefix and the subtree below
    /// it) is freed on both tiers at once (TwoTierRadix::kv_move_free); kvret_src_first = the released class-0
    /// source copy becomes the first host-eviction candidate (TwoTierRadix::kv_src_first; device tier unchanged).
    pub kvret_move_free: bool,
    pub kvret_src_first: bool,
    /// P2PMEM audit, forced-move probe variants (options, NOT engine behaviour; defaults keep every older arm):
    /// fm_dst = target of a forced move among the non-owner P with a pull: "best" (least price-free total, the LAMBDA
    /// probe), "random", "worst" (largest total); fm_cap_follow = the host capacity of the pulled blocks moves with
    /// a forced move (destination +K before the landing, source -K after its release): a memory-neutral move.
    pub fm_dst: String,
    pub fm_cap_follow: bool,
    /// P2PMEM audit ORACLE bound: fm_cap_follow for every move with a pull (not only forced ones).
    pub cap_follow_all: bool,
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
            smg_exact_hits: false,
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
            cs_eft_exclude_prev: true,
            cs_max_pull_tokens: 0,
            p_queue_order: "fcfs".into(),
            sjf_aging_tok_per_s: 8192.0,
            d_mode: "d0".into(),
            d_bind_rule: "po2".into(),
            ld_lead_margin_iters: 1,
            ld_lead_min_reqs: 2,
            ld_lead_max_reqs: 8,
            ld_need_iters_cap: 8,
            ld_max_outstanding_per_d: 4,
            ld_ewma_alpha: 0.2,
            ld_hold_detect_s: 0.2,
            ld_starve_decay: 50,
            ld_admit_latency_s: 0.02,
            ld_choose_latency_s: 0.015,
            ld_meta_latency_s: 0.03,
            ld_claim_passes: 20,
            ld_held_scan_ms: 0.0,
            ld_step_overhead_ms: 0.0,
            shared_pool: false,
            ideal_backlog: "true".into(),
            lease_budget_tokens: 1_510_000,
            lease_pins: false,
            jit_lead_tokens: 131_072,
            repull_min_tokens: 8192,
            turnover_pool_tokens: 12_405_568,
            d_cache: false,
            d_host_ratio: 6.0,
            s2_read: "off".into(),
            s2_min_tokens: 8192,
            kv_retention: false,
            kvret_mode: "ready_end".into(),
            kvret_poll_s: 0.1,
            kvret_ctl_s: 0.01,
            kvret_dup_reclaim: true,
            kvret_end: true,
            kvret_idle_s: 0.0,
            pull_free_capacity: false,
            cs_move_mode: "pull".into(),
            cs_noise_queue: 0.0,
            cs_noise_sigma: 0.0,
            cs_state_age_s: 0.0,
            cs_state_own_fix: true,
            cs_fpi: "off".into(),
            cs_fpi_pilot_s: 900.0,
            cs_fpi_rho_cap: 0.95,
            cs_ski_tau_s: 60.0,
            cs_fpi_move_cost: "seconds".into(),
            cs_eft_dup_per_token_s: 0.0,
            smg_p1_classes: String::new(),
            smg_load: "requests".into(),
            smg_abs_tokens: 0,
            cs_eft_wait_transit: false,
            cs_eft_handoff_s: 0.0,
            cs_eft_margin_k: 0.0,
            cs_eft_owner_tie_c: false,
            cs_eft_owner_near_tokens: 0,
            cs_eft_noise_sigma: 0.0,
            cs_eft_noise_queue: 0.0,
            cs_eft_state_age_s: 0.0,
            cs_eft_lp_ref: 0.0,
            cs_eft_lp_scope: "worker".into(),
            cs_eft_lp_tau_s: 60.0,
            cs_eft_lp_gain: 1.0,
            cs_eft_gate_big_tokens: 0,
            cs_eft_gate_theta_s: 3.0,
            wait_scale: 1.0,
            cs_hold_passes: 0.0,
            cs_hold_steal: true,
            cs_hold_margin_s: 1.0,
            cs_hold_w: 0.0,
            cs_hold_idle_s: 0.3,
            cs_hold_tick_s: 0.05,
            cs_hold_worker_cd_s: 10.0,
            cs_hold_sess_cd_s: 60.0,
            idle_overlap_metric: false,
            cs_eft_dup_on_recompute: false,
            cs_pred_noise: Vec::new(),
            cs_pred_noise_idle: Vec::new(),
            cs_pred_noise_scale: 1.0,
            cs_pred_gate: false,
            cs_pred_m: 0.0,
            cs_pred_gmin_s: 0.0,
            cs_pred_k_idle: 0.0,
            cs_pred_k_busy: 0.0,
            cs_pred_floor_s: 0.0,
            pred_metric: false,
            lpb: false,
            lpb_order: "fifo".into(),
            lpb_moves: true,
            lpb_rx_k: 2.0,
            lpb_tick_s: 0.02,
            lpb_rx_scale: 1.0,
            lpb_bnd_scale: 1.0,
            lpb_hold_free: false,
            lpb_lead_extra_s: 0.0,
            bnd_idle_k: Vec::new(),
            bnd_idle_b: Vec::new(),
            bnd_busy_k: Vec::new(),
            bnd_busy_b: Vec::new(),
            cs_pred_curve: false,
            cs_hol_mu: 0.0,
            cs_hol_window_s: 60.0,
            cs_hol_form: "light".into(),
            cs_hol_rho_cap: 0.95,
            cs_hol_rate: "routed".into(),
            iww_metric: false,
            ghost_frac: 0.0,
            lam_tau_s: 120.0,
            lam_tick_s: 5.0,
            lam_auto: "off".into(),
            lam_q: "sat".into(),
            lam_mu: "ghost".into(),
            lam_scale: 1.0,
            lam_min_s: 300.0,
            fm_rate: 0.0,
            fm_min_tokens: 32_768,
            inj_rate: 0.0,
            inj_s: 1.0,
            kvret_move_free: false,
            kvret_src_first: false,
            fm_dst: "best".into(),
            fm_cap_follow: false,
            cap_follow_all: false,
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
    /// S2: a read from a D copy landed on the P.
    DReadDone(u32),
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
    /// Late D: the ADMIT reached the D (allocation attempt); the D metadata reached P (bound).
    LdAlloc(u32),
    LdBound(u32),
    /// Late D: re-run P's binder (an ADMIT stops counting in the window after hold_s).
    LdTick(u16),
    /// Session retention: router READY poll; control op delivered to a P (session, P, version,
    /// 0 release / 1 end).
    KvTick,
    KvCtl(u32, u16, u32, u8),
    /// S4 idle END check for (session, version of the completed request).
    KvIdle(u32, u32),
    /// S5: router backlog sample (state age); end of the FPI pilot.
    StateTick,
    FpiPilotEnd,
    /// S7 B1: router hold queue check.
    HoldTick,
    /// LATE-P: router queue timer; idle-while-waiting sample (output only).
    LpbTick,
    IwwTick,
    /// LAMBDA: online estimator tick.
    LamTick,
}

/// S5 per agent chain router state for the work-function / ski-rental arms: work function per P,
/// holder, built-up excess, current threshold draw U, last decision time.
#[derive(Debug, Clone)]
struct WfState {
    w: Vec<f64>,
    h: usize,
    r: f64,
    u: f64,
    last: f64,
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
    /// S6: routing class at the decision (0 cold, 1 low match, 2 several near-best holders, 3 single
    /// holder) and cache given up by the choice (best hit - hit on the chosen P, tokens).
    rclass: u8,
    lost: u64,
    /// S6: router seconds charged to the target's in-transit ledger (removed at P receive).
    transit_s: f64,
    /// S7 B1: seconds held in the router; taken by a P other than its owner.
    hold_s: f64,
    stolen: bool,
    /// LATE-P: router-queue entry and dispatch times; dispatch class (1 owner, 2 cold, 3 move).
    lpb_arr: f64,
    lpb_disp: f64,
    lpb_cls: u8,
    /// S8: routing time and forecast of the selected P (noisy, noise-free), P idle at the decision.
    pred_t: f64,
    pred_c: f64,
    pred_c0: f64,
    pred_idle: bool,
    pred_b0: f64,
    pred_s0: f64,
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
    /// Late D: bindable (INTENT and sender ready) since, bind decision time, ADMIT at D, bootstrap
    /// done (entered the waiting queue), router in-flight tokens charged (d1 rule).
    ld_bindable_t: Option<f64>,
    ld_bind_t: Option<f64>,
    ld_held_since: Option<f64>,
    ld_bound_t: Option<f64>,
    ld_inflight: u64,
    /// S2: D prefix matched and locked at D prealloc (tokens, at most len-1 page aligned) and its lock;
    /// D read: tokens read, P local match at issue, in flight, decision taken, tokens counted as hit at
    /// the first chunk, source D, source lease (oracle "c"), blocks covered.
    d_prefix: u64,
    d_lock: Option<NodeId>,
    dread: u64,
    dread_have: u64,
    dread_pending: bool,
    dread_tried: bool,
    drd: u64,
    dread_src: Option<u16>,
    dread_lease: Option<NodeId>,
    dread_upto: u32,
    /// Session retention: the version the router registered for this request, and the session
    /// identity (the agent state; a new identity after an S4 idle END).
    kv_ver: u32,
    kv_sid: u32,
    /// LAMBDA: forced move (probe) with its predicted self gain (s) and pulled tokens; work injected; ghost-hit
    /// tokens at the first chunk.
    forced: bool,
    fm_gain: f64,
    fm_pulled: u64,
    inj: bool,
    ghost: u64,
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

/// Router-side session state (Morphosis #25 KvSessions): last version, current P, version held per
/// P, pending retirements (P, version, time of the move), ended.
#[derive(Debug, Clone, Default)]
struct KvSess {
    ver: u32,
    cur: Option<u16>,
    workers: Vec<(u16, u32)>,
    retire: Vec<(u16, u32, f64)>,
    ended: bool,
}

/// LATE-P cached candidate view of a queued request (see Sim::lpb_cand).
#[derive(Debug, Clone)]
struct LpbC {
    t: f64,
    own: usize,
    cold: bool,
    s0: Vec<f64>,
    ps: Vec<f64>,
    dup: Vec<f64>,
    pt: u64,
}

/// LATE-P: a queued request's candidate view (cache matches, pull hints) is recomputed after this many seconds.
const LPB_CACHE_S: f64 = 0.2;

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
    /// Late-D binder state: EWMA prefix-hit ratio, T_bind mean / |dev|, T_iter, starvation boost and
    /// calm-pass count, rooms with a decision but not bound yet, outstanding binds per D.
    ld_hit: f64,
    ld_tb: f64,
    ld_tb_dev: f64,
    ld_ti: f64,
    ld_boost: u32,
    ld_calm: u32,
    ld_sent: Vec<u32>,
    ld_out: Vec<u32>,
    ld_last_end: f64,
    /// S6: in-transit router seconds (requests sent here, not yet received).
    transit_s: f64,
    /// S7 lambda-pressure: EWMA class-1 host-evicted tok/s, its time and the eviction-log cursor.
    lp_rate: f64,
    lp_t: f64,
    lp_cur: usize,
    /// LATE-P: forecast-error factor of this P (redrawn per pass); start time of the running pass.
    lpb_nf: f64,
    pass_t0: f64,
    /// LAMBDA: ghost FIFO (block, seq), live ghost entries (block -> seq), next seq; ghost-hit tokens (whole run,
    /// score window); prefill tokens computed (whole run); online EWMAs [ghost-hit tok/s, ghost tokens, class-0 host
    /// tokens, class-0 evicted tok/s, all host-evicted tok/s, busy fraction, requests on the P, prefill tok/s], the
    /// last raw counters [ghost hits, class-0 evicted, all evicted, busy s, prefill tokens], first-sample flag; busy
    /// period: requests prefilled in the open period, open flag, decayed sums of n(n+1)/2 and n, last close time,
    /// window sums [n(n+1)/2, n, periods].
    ghost_q: VecDeque<(u64, u64)>,
    ghost_map: FxHashMap<u64, u64>,
    ghost_seq: u64,
    ghost_hit_all: u64,
    ghost_hit_win: u64,
    tok_done: u64,
    lam_e: [f64; 8],
    lam_last: [f64; 5],
    lam_ok: bool,
    bp_n: u64,
    bp_open: bool,
    bp_a: f64,
    bp_b: f64,
    bp_t: f64,
    bp_win: [f64; 3],
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
    /// Late D: bound requests waiting for KV memory (bind order) with their arrival time at D (f64
    /// bits); router in-flight tokens (d1 rule); legs registered without KV (choose: all candidates).
    held: VecDeque<(u32, u64)>,
    inflight_tok: u64,
    legs: i64,
    /// S2 D cache (used only with d_cache).
    radix: TwoTierRadix,
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
    /// Late D (scored rows): held at D = ADMIT at D -> allocation [mean, p50, p95, p99] s (engine
    /// late_d_held_duration); share held > 0.1 s; means of [D receipt -> ADMIT at D (engine
    /// late_d_deferred_duration), P bindable -> bind decision, bind decision -> bootstrap done].
    pub ld_held: [f64; 4],
    pub ld_held_share: f64,
    pub ld_waits: [f64; 3],
    /// S2 serving-D model (scored rows; 0 without d_cache): D reads (count, tokens read, tokens counted
    /// as hit at the first chunk); D prefix at prealloc as a share of prompt tokens; the static serving-D
    /// bound Dsrv = sum max(0, min(I, D prefix) - A) in tokens, over sumI and over P-computed tokens;
    /// share of rows with D prefix > A; D-local reuse = sum max(0, D prefix - A) / prompt tokens; D reads
    /// issued in the whole run (count, tokens).
    pub dreads_scored: u64,
    pub dread_tokens_scored: u64,
    pub dread_hit_tokens_scored: u64,
    pub d_prefix_share: f64,
    pub dsrv_tokens: f64,
    pub dsrv_over_ideal: f64,
    pub dsrv_over_computed: f64,
    pub d_prefix_gt_a_rows: f64,
    pub d_local_reuse: f64,
    pub dreads_all: u64,
    pub dread_tokens_all: u64,
    /// Session retention (0 without kv_retention). Router: releases sent [whole run, score window];
    /// READY waits of the releases sent in the window [count, mean, p50, p90] (s, move -> release
    /// sent); END ops [whole run, window]; retirements cancelled by a return to the P; retirements
    /// dropped by END; status polls. Engine (all P, whole run): release outcomes [applied, fence,
    /// stale, ended]; tracker [admits, fenced, retries, end ops, wakes, anchors moved on delete, late
    /// marks]; host eviction by class [class 0 host-only leaf, class 0 duplicate, class 1 host-only
    /// leaf, class 1 duplicate] in tokens (window, whole run); device-only nodes deleted (tokens);
    /// class-0 host tokens / host pool (census mean); audit per P [count mismatches, dead anchors,
    /// bad registrations, live registrations].
    pub kv_releases_sent: [u64; 2],
    pub kv_ready_wait: [f64; 4],
    /// transfer_end: releases at pull commit [whole run, window] and their wait (move -> release) in
    /// the window [count, mean, p50, p90] s; kv_ready_wait then covers the READY releases only.
    /// co_schedule decisions kept on the owner by the pull cap (whole run).
    pub kv_transfer_releases: [u64; 2],
    pub kv_transfer_wait: [f64; 4],
    pub cs_cap_kept: u64,
    pub kv_end_ops: [u64; 2],
    pub kv_retire_cancelled: u64,
    pub kv_retire_dropped_end: u64,
    pub kv_polls: u64,
    pub kv_release_outcome: [u64; 4],
    pub kv_tracker: [u64; 7],
    pub kv_evicted_tokens_win: [u64; 4],
    pub kv_evicted_tokens_all: [u64; 4],
    pub kv_devonly_deleted_tokens: u64,
    pub kv_class0_host_frac: f64,
    pub kv_audit: Vec<[u64; 4]>,
    /// Cross-P duplicates (all arms; census every 30 s in the score window): tokens resident (any
    /// tier) on >= 2 P (mean), extra copies sum(copies - 1) (mean), token-seconds of the first over
    /// the window, union of resident tokens over all P (mean).
    pub dup_tokens_mean: f64,
    pub dup_extra_tokens_mean: f64,
    pub dup_token_s: f64,
    pub resident_union_tokens_mean: f64,
    /// P moves per session (agent trajectory; consecutive requests on different P, the later one
    /// admitted in the score window): [sessions with a window request, mean moves, share with >= 1
    /// move, p90 moves, moves per 100 window requests].
    pub sess_moves: [f64; 5],
    /// S4 idle END: [sessions ended by the idle rule (whole run), the same in the window, new
    /// session identities after an idle END (whole run)].
    pub kv_idle: [u64; 3],
    /// Agent idle gaps (all arms): route time of an agent's request in the score window - completion
    /// of the same agent's previous request: [count, mean, p10, p25, p50, p75, p90, p95, p99, max] (s)
    /// and the share of gaps > 30, 60, 120, 300, 600 s.
    pub agent_gap_s: [f64; 10],
    pub agent_gap_gt: [f64; 5],
    /// S3 (d) pull_free_capacity (tokens, all P): added by pull landings outside the capacity, removed
    /// later, still resident at the end.
    pub pull_free_tokens: [u64; 3],
    /// S5 FPI family: measured (lambda req/s, rho) per P after the pilot; counters (whole run)
    /// [pull moves, recompute moves, holder resets, decisions after the pilot].
    pub fpi_lambda_rho: Vec<[f64; 2]>,
    pub fpi_counts: [u64; 4],
    /// S5 regime diagnostics (all arms): host-evicted tokens per second over all P in the score
    /// window; 300-s series over the whole run: [bin start s, host-evicted tok/s, mean E2E of the
    /// completed non-warm requests sent in the bin, their hit (actual / prompt, tok-w), completions/s].
    pub host_evicted_tok_s_win: f64,
    pub series_300s: Vec<[f64; 5]>,
    /// S6 diagnostics. Routing classes of the requests sent in the score window (cold, low match,
    /// several near-best holders, single holder): [requests, sent to a P with less cache than the
    /// best P, cache given up (tokens)]; share of scored requests sent to a P with less cache.
    pub route_class: Vec<[f64; 3]>,
    pub lost_share: f64,
    /// Hit deficit of the scored requests (tokens per request, means): [ideal, best P at the
    /// decision, chosen P at the decision, pulled and used, actual, E_global = ideal - best (cached
    /// nowhere), E_route = best - chosen - pulled (given up by routing, not recovered by a pull),
    /// E_queue = chosen + pulled - actual (lost between the decision and admission)]; capped like
    /// the engine (prompt - 1, page aligned). E_global + E_route + E_queue = ideal - actual.
    pub deficit: [f64; 8],
    /// The same by the agent chain's move at this request: rows [first request of the chain in the
    /// run, stay (same P as the chain's previous request), move to a P not used before by the chain,
    /// move back to a P the chain used before]: [share, E_global, E_route, E_queue, ideal, actual].
    pub deficit_by_move: Vec<[f64; 6]>,
    /// Host-evicted tokens/s in the score window by trigger (rows: prefill write-through insert, pull
    /// landing insert, other) and class (columns: retention class 0, class 1, LRU order).
    pub host_ev_cause_tok_s: Vec<[f64; 3]>,
    /// Device-evicted and device-only-deleted tokens/s over the whole run by trigger (same rows).
    pub dev_ev_cause_tok_s: [f64; 3],
    pub devonly_ev_cause_tok_s: [f64; 3],
    /// S7 E2E split by the chain's move at the request: rows [first request of the chain, not moved,
    /// moved without a pull, moved with a pull]: [share, mean E2E, mean TTFT, mean pulled tokens].
    pub e2e_by_move: Vec<[f64; 4]>,
    /// S7: mean lambda-pressure multiplier over priced pull candidates; share of decisions where the
    /// B2 gate fired.
    pub lp_mult_mean: f64,
    pub gate_fire_share: f64,
    /// S7 B1: share of scored requests held in the router, share taken by a non-owner P (steal),
    /// mean hold seconds of the held ones; share of P queue time overlapping a fully idle other P.
    pub held_share: f64,
    pub stolen_share: f64,
    pub hold_s_mean: f64,
    pub idle_overlap_share: f64,
    /// S8 forecast scoring (pred_metric; scored requests routed on the EFT path): rows [noisy forecast,
    /// noise-free forecast] x [n, MAPE all, MAPE idle, MAPE busy, APE p50, APE p90, bias s, n idle];
    /// |actual - forecast| / forecast quantiles q 0.5 / 0.6827 / 0.8 / 0.9, rows [noisy idle, noisy busy,
    /// noise-free idle, noise-free busy]; gate counters [decisions, cheaper non-owner P exists, moves].
    pub pred_err: Vec<[f64; 8]>,
    pub pred_kq: Vec<[f64; 4]>,
    pub pred_gate: [u64; 3],
    /// S8: share of scored requests sent to a P other than their owner (most cached prefix).
    pub moved_share: f64,
    /// S7 B1 debug at the end of the run: held count, then per P [backlog s without ledger, ledger s,
    /// waiting, bootstrap, prefilling counts].
    pub hold_end: Vec<f64>,
    /// LATE-P (scored requests): [router wait mean, p50, p95 (s), share dispatched to the owner, cold, moved,
    /// calibrated recv residual mean (s), calibrated boot mean (s)]; counters (whole run) [dispatches, owner, cold,
    /// move with pull, move without pull, full scans].
    pub lpb_stats: [f64; 8],
    pub lpb_counts: [u64; 6],
    /// PRED-GWQ (lpb_hold_free): arrivals bound to their owner that skipped the router queue (whole run).
    pub lpb_bypassed: u64,
    /// PRED-EFT+HOL shadow (whole run): [gate decisions, the other HOL form picks another P, big cold requests
    /// (uncached on the owner >= 32,768 and best hit < the pull floor), big cold and differ, differ with the selected
    /// form's P less loaded, differ with the selected form keeping the owner].
    pub hol_shadow: [u64; 6],
    /// iww_metric: [share of P-samples idle while work waits elsewhere, samples, per-P outstanding uncached max /
    /// mean p50, p90].
    pub iww: [f64; 4],
    /// LAMBDA study (score window). ghost: [ghost-hit tok/s (all P), mean ghost tokens (all P), h = their ratio
    /// (1/s)]. lam_est (means of the online estimators over the window ticks, pool-wide): [h (1/s), T_dup (s), q_sat,
    /// q_bp, q_L, R (prefill tok per busy s), lambda_auto global with lam_q (s per token, mean of ticks), class-0 host
    /// tokens, class-0 evicted tok/s, all host-evicted tok/s]; lam_auto_pct: p10 / p90 of the per-tick global value;
    /// q_bp_win: [sum n(n+1)/2 / sum n over the window's busy periods, busy periods, mean n]; lam_used: [mean price
    /// used on priced pull candidates in the window (s per token), count]; fm: [eligible decisions (whole run), forced
    /// (whole run), forced in the window, mean pulled tokens, mean predicted self gain (s)]; inj: [whole run, window];
    /// per-request ghost-hit tokens of the scored requests (mean); prefill tokens computed in the window per second.
    pub ghost: [f64; 3],
    pub lam_est: [f64; 10],
    pub lam_auto_pct: [f64; 2],
    pub q_bp_win: [f64; 3],
    pub lam_used: [f64; 2],
    pub fm: [f64; 5],
    /// P2PMEM audit: per-P host capacity at the end (tokens) and host capacity moved by fm_cap_follow (tokens).
    pub p_host_cap_end: Vec<u64>,
    pub cap_moved_tokens: u64,
    pub inj: [u64; 2],
    pub ghost_per_req: f64,
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
    /// Late D: held at D = ADMIT at D -> allocation (s; -1 = no bind).
    pub d_held: f64,
    /// S2: D prefix at prealloc and the D-read part of `actual` (tokens; 0 without d_cache).
    pub d_prefix: u64,
    pub dread: u64,
    /// S8 (pred_metric; -1 = n/a): actual decision -> prefill done; forecasts (noisy, noise-free), the
    /// noise-free backlog and own-service parts, P idle at the decision; decision -> P receive, P receive
    /// -> ready, owner P.
    pub pred_lab: f64,
    pub pred_c: f64,
    pub pred_c0: f64,
    pub pred_b0: f64,
    pub pred_s0: f64,
    pub pred_idle: bool,
    pub recv_s: f64,
    pub ready_s: f64,
    pub owner: u16,
    /// LAMBDA: forced move (probe), its predicted self gain (s) and pulled tokens; work injected; ghost-hit tokens.
    pub forced: bool,
    pub fm_gain: f64,
    pub fm_pulled: u64,
    pub inj: bool,
    pub ghost: u64,
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
    /// co_schedule decisions kept on the owner by cs_max_pull_tokens.
    cs_cap_kept: u64,
    /// S2: per agent chain, the D that served its latest finished turn; D reads issued / tokens.
    last_d: FxHashMap<(u64, u32), u16>,
    dreads_all: u64,
    dread_tok_all: u64,
    /// Session retention: router sessions, sessions with pending retirements, releases sent (time,
    /// READY wait), END op times, retirements cancelled / dropped by END, polls, control-op log
    /// (delivery time, session, P, version, op), last generate index per (task row, agent).
    kv_sess: FxHashMap<u32, KvSess>,
    kv_pend: Vec<u32>,
    kv_rel: Vec<(f64, f64, u8)>,
    kv_end_t: Vec<f64>,
    kv_cancel: u64,
    kv_drop_end: u64,
    kv_polls: u64,
    kv_log: Vec<(f64, u32, u16, u32, u8)>,
    kv_lastgen: FxHashMap<(u32, u32), u32>,
    /// Duplicate census (t, dup blocks, extra-copy blocks, union blocks, class-0 host blocks).
    dupc: Vec<(f64, u64, u64, u64, u64)>,
    /// S4: current session identity per agent state (absent = the agent state), next new identity,
    /// sessions ended by the idle rule, new identities after an idle END.
    kv_sid_of: FxHashMap<u32, u32>,
    kv_sid_next: u32,
    kv_idle_ends: Vec<f64>,
    kv_idle_new: u64,
    /// Per agent state: completion time of its latest (non-warm) request; idle gaps (route time of
    /// the next request - that completion) of requests routed in the score window.
    agent_last_done: FxHashMap<u32, f64>,
    agent_gaps: Vec<f64>,
    /// S5: noise RNG (separate stream), backlog samples (t, per-P backlog), charges routed per P since
    /// (t, tokens), FPI pilot counts / measured (lambda, rho) per P, per-chain WFA / ski state,
    /// counters [pull moves, recompute moves, holder resets, decisions after the pilot].
    nrng: PyRandom,
    st_hist: VecDeque<(f64, Vec<u64>)>,
    st_routed: Vec<VecDeque<(f64, u64)>>,
    /// S7 EFT state age: sampled backlog seconds and router seconds routed per P since.
    eft_hist: VecDeque<(f64, Vec<f64>)>,
    eft_routed: Vec<VecDeque<(f64, f64)>>,
    /// S7: lambda-pressure multipliers used (sum, count) and gate firings (decisions, fired).
    lp_stats: (f64, u64),
    gate_stats: (u64, u64),
    /// S7 B1: held requests (request, owner, hold time), tick scheduled, forced dispatch target,
    /// last steal received per P, last move per session; idle intervals (start, end, P).
    held: Vec<(u32, u16, f64)>,
    hold_tick_on: bool,
    /// S8 gate counters: decisions, decisions with a cheaper non-owner P, moves.
    pred_stats: [u64; 3],
    force_tgt: Option<usize>,
    steal_rx_t: Vec<f64>,
    sess_move_t: FxHashMap<(u64, u32), f64>,
    idle_log: Vec<(f64, f64, u16)>,
    fpi_n: Vec<u64>,
    fpi_lr: Option<Vec<(f64, f64)>>,
    wf: FxHashMap<(u64, u32), WfState>,
    fpi_counts: [u64; 4],
    /// LATE-P: router queue, timer flag, receipt calibration [recv residual mean, |dev|, boot mean, |dev|] and sample
    /// counts, counters [dispatches, owner, cold, move with pull, move without pull, full scans]; HOL routing times per
    /// P; idle-while-waiting samples [ticks, idle-while-waiting P-samples, P-samples] and per-sample load max / mean.
    lpb_q: Vec<u32>,
    lpb_cache: FxHashMap<u32, LpbC>,
    lpb_tick_on: bool,
    lpb_rx: [f64; 4],
    lpb_rx_n: [u64; 2],
    lpb_counts: [u64; 6],
    lpb_bypassed: u64,
    hol_hist: Vec<VecDeque<(f64, f64)>>,
    hol_sh: [u64; 6],
    iww: [f64; 3],
    lmm: Vec<f64>,
    lo: f64,
    hi: f64,
    /// LAMBDA: estimator tick time, first tick, score-window samples [ghost-hit tok/s, ghost tokens, class-0 host
    /// tokens, class-0 evicted tok/s, all evicted tok/s (sums over P), busy fraction, requests on a P (means over P),
    /// prefill tok/s (sum), q_bp (pool), lambda_auto global (s per token), h, T_dup, q (lam_q), R]; lambda_auto used on
    /// priced pull candidates (sum, count, window sum, window count); forced moves [eligible, forced whole run,
    /// forced in window], their pulled tokens and predicted self gain in the window; injections [whole run, window].
    lam_t: f64,
    lam_t0: f64,
    lam_win: Vec<[f64; 14]>,
    lam_used: (f64, u64, f64, u64),
    fm_rng: PyRandom,
    /// P2PMEM audit: host capacity blocks moved by fm_cap_follow (whole run).
    cap_moved: u64,
    fm_elig: u64,
    fm_n: [u64; 2],
    fm_tok_win: u64,
    fm_gain_win: f64,
    inj_n: [u64; 2],
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
#[allow(clippy::too_many_arguments, dead_code)]
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
    eft_choose_k(wait, own_s, pull_s, dup, pulled, own, excluded, burst_s, max_pull, margin_s, 0.0)
}

/// eft_choose with the S7 noise-aware margin: a move needs C_o - C_w > margin_s + k sqrt(B_o^2 + B_w^2).
#[allow(clippy::too_many_arguments)]
pub(crate) fn eft_choose_k(
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
    margin_k: f64,
) -> (usize, u8) {
    let tot = |w: usize| wait[w].max(pull_s[w]) + own_s[w] + dup[w];
    let base = tot(own);
    let mg = |w: usize| {
        if margin_k > 0.0 {
            margin_s + margin_k * (wait[own] * wait[own] + wait[w] * wait[w]).sqrt()
        } else {
            margin_s
        }
    };
    let better = |w: usize| w != own && tot(w) + mg(w) < base;
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

/// r5 bound curve (router_e3.bound_curve): values[0] at and below knots[0]; linear between knots; values[-1] x
/// x / knots[-1] above the last knot. Empty = 0.
pub(crate) fn bound_curve(knots: &[f64], values: &[f64], x: f64) -> f64 {
    let n = knots.len().min(values.len());
    if n == 0 {
        return 0.0;
    }
    if x <= knots[0] {
        return values[0];
    }
    if x >= knots[n - 1] {
        return values[n - 1] * x / knots[n - 1];
    }
    let i = knots[..n].partition_point(|&k| k <= x);
    let (x0, x1, y0, y1) = (knots[i - 1], knots[i], values[i - 1], values[i]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

/// S8: inverse-CDF draw from a table of n >= 2 quantiles at uniform probabilities 0..1 (linear).
pub(crate) fn table_draw(tab: &[f64], u: f64) -> f64 {
    let n = tab.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return tab[0];
    }
    let x = u.clamp(0.0, 1.0) * (n - 1) as f64;
    let i = (x as usize).min(n - 2);
    tab[i] + (tab[i + 1] - tab[i]) * (x - i as f64)
}

/// S5 FPI admission cost (pure): start + x + lambda x (2 u + x) / (2 (1 - rho)).
pub(crate) fn fpi_score(start: f64, x: f64, u: f64, lambda: f64, rho: f64) -> f64 {
    start + x + lambda * x * (2.0 * u + x) / (2.0 * (1.0 - rho))
}

/// IDEAL choice (pure): least predicted finish `f`, ties -> fewer active requests, then lower index.
pub(crate) fn ideal_choose(f: &[f64], act: &[u64]) -> usize {
    (0..f.len()).min_by(|&a, &b| f[a].total_cmp(&f[b]).then(act[a].cmp(&act[b])).then(a.cmp(&b))).unwrap_or(0)
}

impl<'a> Sim<'a> {
    pub fn new(c: SimConfig, wl: &'a Workload, ct_map: &'a FxHashMap<(u32, u32, u32), u32>) -> Self {
        let bt = c.block_tokens;
        let (s5_np, s5_seed) = (c.n_p, c.seed);
        let mut p = Vec::new();
        for i in 0..c.n_p {
            let beta = c.p_beta[i % c.p_beta.len()];
            let mut radix = TwoTierRadix::new(c.p_dev_tokens / bt, c.p_host_tokens / bt);
            radix.demote_scope =
                if c.demote_scope_device { DemoteScope::DeviceAndHost } else { DemoteScope::Host };
            radix.record_evictions = true;
            radix.ghost_on = c.ghost_frac > 0.0;
            radix.split_refresh = c.split_refresh;
            radix.split_on_match = c.split_on_match;
            if c.kv_retention {
                radix.enable_kvret(c.kvret_dup_reclaim);
            }
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
                ld_hit: 0.9,
                ld_tb: -1.0,
                ld_tb_dev: 0.0,
                ld_ti: -1.0,
                ld_boost: 0,
                ld_calm: 0,
                ld_sent: Vec::new(),
                ld_out: vec![0; c.n_d],
                ld_last_end: -1.0,
                transit_s: 0.0,
                lp_rate: 0.0,
                lp_t: 0.0,
                lp_cur: 0,
                lpb_nf: 1.0,
                pass_t0: 0.0,
                ghost_q: VecDeque::new(),
                ghost_map: FxHashMap::default(),
                ghost_seq: 0,
                ghost_hit_all: 0,
                ghost_hit_win: 0,
                tok_done: 0,
                lam_e: [0.0; 8],
                lam_last: [0.0; 5],
                lam_ok: false,
                bp_n: 0,
                bp_open: false,
                bp_a: 0.0,
                bp_b: 0.0,
                bp_t: 0.0,
                bp_win: [0.0; 3],
            });
        }
        if c.kv_retention {
            assert!(
                !c.demote_source
                    && !c.oracle_move
                    && c.demote_defer_n == 0
                    && c.demote_defer_t == 0.0
                    && !c.shared_pool
                    && c.s2_read == "off",
                "kv_retention needs demote-source off (the engine refuses --peer-kv-demote-source with retention), \
                 no oracle move, no deferred demote, no shared pool, no S2 read"
            );
            assert!(
                matches!(c.kvret_mode.as_str(), "ready_end" | "transfer_end" | "end_only" | "recency_only"),
                "kvret_mode: ready_end | transfer_end | end_only | recency_only"
            );
        }
        assert!(
            c.kvret_idle_s <= 0.0 || (c.kv_retention && c.kvret_mode != "recency_only"),
            "kvret_idle_s needs kv_retention with session tags"
        );
        assert!(
            !(c.kvret_move_free || c.kvret_src_first) || (c.kv_retention && c.kvret_mode == "transfer_end"),
            "kvret_move_free / kvret_src_first need kv_retention with kvret_mode transfer_end"
        );
        assert!(!(c.kvret_move_free && c.kvret_src_first), "kvret_move_free and kvret_src_first are exclusive");
        assert!(matches!(c.fm_dst.as_str(), "best" | "random" | "worst"), "fm_dst: best | random | worst");
        assert!(!(c.pull_free_capacity && c.shared_pool), "pull_free_capacity needs per-P pools");
        assert!(matches!(c.cs_move_mode.as_str(), "pull" | "recompute"), "cs_move_mode: pull | recompute");
        assert!(
            matches!(c.cs_fpi.as_str(), "off" | "fpi" | "wfa" | "ski" | "skir"),
            "cs_fpi: off | fpi | wfa | ski | skir"
        );
        assert!(c.cs_fpi == "off" || c.p_policy == "eft", "cs_fpi needs p_policy eft");
        assert!(matches!(c.cs_fpi_move_cost.as_str(), "seconds" | "fpi"), "cs_fpi_move_cost: seconds | fpi");
        assert!(matches!(c.smg_load.as_str(), "requests" | "uncached_tokens"), "smg_load: requests | uncached_tokens");
        for k in c.smg_p1_classes.split(',').filter(|s| !s.is_empty()) {
            assert!(matches!(k, "cold" | "low" | "multi" | "single"), "smg_p1_classes: cold, low, multi, single");
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
                held: VecDeque::new(),
                inflight_tok: 0,
                legs: 0,
                radix: {
                    let mut r = TwoTierRadix::new(
                        c.d_cap_tokens / bt,
                        (c.d_host_ratio.max(0.0) * c.d_cap_tokens as f64) as u64 / bt,
                    );
                    r.split_refresh = c.split_refresh;
                    r.split_on_match = c.split_on_match;
                    r
                },
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
            cs_cap_kept: 0,
            last_d: FxHashMap::default(),
            dreads_all: 0,
            dread_tok_all: 0,
            kv_sess: FxHashMap::default(),
            kv_pend: Vec::new(),
            kv_rel: Vec::new(),
            kv_end_t: Vec::new(),
            kv_cancel: 0,
            kv_drop_end: 0,
            kv_polls: 0,
            kv_log: Vec::new(),
            kv_lastgen: FxHashMap::default(),
            dupc: Vec::new(),
            kv_sid_of: FxHashMap::default(),
            kv_sid_next: 0x8000_0000,
            kv_idle_ends: Vec::new(),
            kv_idle_new: 0,
            agent_last_done: FxHashMap::default(),
            agent_gaps: Vec::new(),
            nrng: PyRandom::new(0x6e01_0000 + s5_seed),
            st_hist: VecDeque::new(),
            st_routed: vec![VecDeque::new(); s5_np],
            eft_hist: VecDeque::new(),
            eft_routed: vec![VecDeque::new(); s5_np],
            lp_stats: (0.0, 0),
            gate_stats: (0, 0),
            held: Vec::new(),
            hold_tick_on: false,
            pred_stats: [0; 3],
            force_tgt: None,
            steal_rx_t: vec![f64::NEG_INFINITY; s5_np],
            sess_move_t: FxHashMap::default(),
            idle_log: Vec::new(),
            fpi_n: vec![0; s5_np],
            fpi_lr: None,
            wf: FxHashMap::default(),
            fpi_counts: [0; 4],
            lpb_q: Vec::new(),
            lpb_cache: FxHashMap::default(),
            lpb_tick_on: false,
            lpb_rx: [0.0; 4],
            lpb_rx_n: [0; 2],
            lpb_counts: [0; 6],
            lpb_bypassed: 0,
            hol_hist: vec![VecDeque::new(); s5_np],
            hol_sh: [0; 6],
            iww: [0.0; 3],
            lmm: Vec::new(),
            lo,
            hi,
            lam_t: 0.0,
            lam_t0: -1.0,
            lam_win: Vec::new(),
            lam_used: (0.0, 0, 0.0, 0),
            fm_rng: PyRandom::new(0x4c41_0000 + s5_seed),
            cap_moved: 0,
            fm_elig: 0,
            fm_n: [0; 2],
            fm_tok_win: 0,
            fm_gain_win: 0.0,
            inj_n: [0; 2],
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
        if self.kv_tagged() && (self.c.kvret_mode == "ready_end" || self.c.kvret_mode == "transfer_end") {
            self.at(self.c.kvret_poll_s, Ev::KvTick);
        }
        if self.c.cs_state_age_s > 0.0 || self.c.cs_eft_state_age_s > 0.0 {
            self.at(0.0, Ev::StateTick);
        }
        if self.c.cs_fpi != "off" {
            self.at(self.c.cs_fpi_pilot_s, Ev::FpiPilotEnd);
        }
        if self.c.lpb {
            assert!(self.c.p_policy == "eft" && self.c.cs_hold_passes <= 0.0, "lpb needs p_policy eft and no B1 hold");
            assert!(matches!(self.c.lpb_order.as_str(), "fifo" | "hrrn"), "lpb_order: fifo | hrrn");
        }
        if self.c.iww_metric {
            self.at(self.lo, Ev::IwwTick);
        }
        if self.c.ghost_frac > 0.0 || self.c.lam_auto != "off" {
            assert!(matches!(self.c.lam_auto.as_str(), "off" | "global" | "pair"), "lam_auto: off | global | pair");
            assert!(matches!(self.c.lam_q.as_str(), "sat" | "bp" | "bpsat" | "L" | "one"), "lam_q: sat | bp | bpsat | L | one");
            assert!(matches!(self.c.lam_mu.as_str(), "ghost" | "one"), "lam_mu: ghost | one");
            assert!(
                self.c.lam_auto == "off" || (self.c.ghost_frac > 0.0 && self.c.p_policy == "eft"),
                "lam_auto needs ghost_frac > 0 and p_policy eft"
            );
            self.at(self.c.lam_tick_s.max(0.1), Ev::LamTick);
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
                    if self.c.d_mode == "d0" {
                        let d = self.reqs[r as usize].d as usize;
                        self.d[d].fifo.push_back(r);
                        self.d_try_admit(t, d);
                    } else {
                        self.ld_register(r);
                    }
                }
                Ev::LdAlloc(r) => self.ld_alloc(t, r),
                Ev::LdBound(r) => self.ld_bound(t, r),
                Ev::LdTick(p) => {
                    self.ld_binder(t, p as usize);
                    self.p_idle_update(t, p as usize);
                }
                Ev::PReady(r) => self.p_ready(t, r),
                Ev::KvTick => {
                    self.kv_tick(t);
                    if t < self.c.duration_s {
                        let dt = self.c.kvret_poll_s;
                        self.at(t + dt, Ev::KvTick);
                    }
                }
                Ev::KvCtl(sid, p, ver, op) => self.kv_ctl(t, sid, p as usize, ver, op),
                Ev::KvIdle(sid, ver) => self.kv_idle(t, sid, ver),
                Ev::StateTick => {
                    if self.c.cs_state_age_s > 0.0 {
                        let v = self.cs_pend_now();
                        self.st_hist.push_back((t, v));
                        let keep = self.c.cs_state_age_s + 1.0;
                        while self.st_hist.len() > 2 && self.st_hist[1].0 < t - keep {
                            self.st_hist.pop_front();
                        }
                    }
                    if self.c.cs_eft_state_age_s > 0.0 {
                        let v: Vec<f64> = (0..self.p.len()).map(|w| self.backlog_secs(w, true)).collect();
                        self.eft_hist.push_back((t, v));
                        let keep = self.c.cs_eft_state_age_s + 1.0;
                        while self.eft_hist.len() > 2 && self.eft_hist[1].0 < t - keep {
                            self.eft_hist.pop_front();
                        }
                        for q in self.eft_routed.iter_mut() {
                            while q.front().is_some_and(|x| x.0 < t - keep) {
                                q.pop_front();
                            }
                        }
                    }
                    if t < self.c.duration_s {
                        self.at(t + 0.1, Ev::StateTick);
                    }
                }
                Ev::HoldTick => self.hold_tick(t),
                Ev::LpbTick => {
                    self.lpb_tick_on = false;
                    self.lpb_dispatch(t);
                }
                Ev::IwwTick => self.iww_tick(t),
                Ev::LamTick => self.lam_tick(t),
                Ev::FpiPilotEnd => {
                    let dt = self.c.cs_fpi_pilot_s.max(1e-9);
                    let cap = self.c.cs_fpi_rho_cap;
                    let lr = (0..self.p.len())
                        .map(|w| (self.fpi_n[w] as f64 / dt, (self.p[w].busy_s / dt).min(cap)))
                        .collect();
                    self.fpi_lr = Some(lr);
                }
                Ev::PullDone(r) => self.pull_done(t, r),
                Ev::DReadDone(r) => self.dread_done(t, r),
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
                        let h: u64 = d.held.iter().map(|&(r, _)| self.reqs[r as usize].need).sum();
                        d.load_snap = (d.used + q + h) as i64;
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
                    let (dup, extra, uni) = self.dup_census();
                    let c0: u64 = self.p.iter().map(|p| p.radix.kv_class0_host_blocks()).sum();
                    self.dupc.push((t, dup, extra, uni, c0));
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
                    let d = seconds.min(self.c.wait_cap_s) * self.c.wait_scale;
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

    /// PRED-GWQ: request r is bound to its owner (largest local hit - second largest >= chunk_tokens).
    fn lpb_bound_to_owner(&mut self, r: u32) -> bool {
        let bt = self.c.block_tokens;
        let l = self.reqs[r as usize].pt;
        let runs = self.reqs[r as usize].runs.clone();
        let (mut h1, mut h2) = (0u64, 0u64);
        for w in 0..self.p.len() {
            let h = (self.p[w].radix.match_peek(&runs).total() as u64 * bt).min(l);
            if h > h1 {
                h2 = h1;
                h1 = h;
            } else if h > h2 {
                h2 = h;
            }
        }
        h1 - h2 >= self.c.chunk_tokens
    }

    fn route(&mut self, t: f64, r: u32) {
        let lpb_bypass = self.c.lpb && self.c.lpb_hold_free && self.force_tgt.is_none() && self.lpb_bound_to_owner(r);
        if lpb_bypass {
            self.lpb_bypassed += 1;
        }
        if self.c.lpb && self.force_tgt.is_none() && !lpb_bypass {
            // LATE-P: the request enters the router queue; the dispatcher decides when and where.
            if !self.reqs[r as usize].warm {
                let a = self.reqs[r as usize].agent_state;
                if let Some(&ld) = self.agent_last_done.get(&a) {
                    if t >= self.lo && t < self.hi {
                        self.agent_gaps.push(t - ld);
                    }
                }
            }
            self.reqs[r as usize].lpb_arr = t;
            self.lpb_q.push(r);
            self.lpb_dispatch(t);
            return;
        }
        if self.c.cs_hold_passes > 0.0 && self.force_tgt.is_none() {
            let hits = self.actual_hits(r);
            let bs: Vec<f64> = (0..self.p.len()).map(|w| self.hold_backlog(w)).collect();
            let o = (0..self.p.len())
                .min_by(|&a, &b| hits[b].cmp(&hits[a]).then(bs[a].total_cmp(&bs[b])).then(a.cmp(&b)))
                .unwrap();
            if bs[o] > self.hold_limit(o) {
                self.held.push((r, o as u16, t));
                if !self.hold_tick_on {
                    self.hold_tick_on = true;
                    self.at(t + self.c.cs_hold_tick_s, Ev::HoldTick);
                }
                return;
            }
            self.dispatch_to(t, r, o);
            return;
        }
        if !self.reqs[r as usize].warm && !(self.c.lpb && self.force_tgt.is_some()) {
            let a = self.reqs[r as usize].agent_state;
            if let Some(&ld) = self.agent_last_done.get(&a) {
                if t >= self.lo && t < self.hi {
                    self.agent_gaps.push(t - ld);
                }
            }
        }
        self.pick_p(t, r);
        if self.c.d_mode != "choose" {
            self.pick_d(r);
        }
        if self.kv_tagged() {
            self.kv_route(t, r);
        }
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
        let mut tgt: usize;
        let mut own: usize;
        match pol.as_str() {
            "smgca" => {
                let runs = self.reqs[r as usize].runs.clone();
                let tok = self.c.smg_load == "uncached_tokens";
                let ld: Vec<i64> = if tok {
                    (0..np).map(|w| self.true_backlog(w) as i64).collect()
                } else {
                    self.p.iter().map(|p| p.nin).collect()
                };
                let mx = *ld.iter().max().unwrap();
                let mn = *ld.iter().min().unwrap();
                let abs = if tok { self.c.smg_abs_tokens as i64 } else { self.c.smg_abs };
                let imbalanced = mx - mn > abs && (mx as f64) > (mn as f64) * self.c.smg_rel;
                let min_load = |s: &mut Self| -> usize {
                    let c: Vec<usize> = (0..np).filter(|&k| ld[k] == mn).collect();
                    c[s.urange(c.len())]
                };
                if imbalanced {
                    tgt = min_load(self);
                } else {
                    let m: Vec<u64> = if self.c.smg_exact_hits {
                        hits.clone()
                    } else {
                        self.smg.iter_mut().map(|tr| tr.match_peek(&runs).total() as u64 * bt).collect()
                    };
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
                if !self.c.smg_p1_classes.is_empty() {
                    // S6 decomposition: this request's class is routed by the P1 no-pull rule.
                    let cl = self.route_class(&hits, pt);
                    let name = ["cold", "low", "multi", "single"][cl as usize];
                    if self.c.smg_p1_classes.split(',').any(|k| k == name) {
                        tgt = self.p1_choice(r, &hits, 1);
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
                            && self.c.cs_move_mode != "recompute"
                        {
                            hint[w] = Some((s_, hits[s_]));
                        }
                    }
                    let hs = hint[w].map(|(_, h)| h).unwrap_or(0);
                    exp[w] = local[w].max(hs).min(l);
                }
                let mut pend: Vec<u64> = if self.c.cs_state_age_s > 0.0 {
                    self.cs_pend_stale(t)
                } else {
                    self.cs_pend_now()
                };
                if self.c.cs_noise_queue > 0.0 {
                    let e = self.c.cs_noise_queue;
                    for x in pend.iter_mut() {
                        let u = self.nrng.random();
                        *x = (*x as f64 * (1.0 + e * (2.0 * u - 1.0))).max(0.0).round() as u64;
                    }
                }
                let act: Vec<u64> = self.p.iter().map(|p| p.cs_active).collect();
                let mut after: Vec<u64> = (0..np).map(|w| pend[w] + (l - exp[w])).collect();
                if self.c.cs_noise_sigma > 0.0 {
                    let s = self.c.cs_noise_sigma;
                    for x in after.iter_mut() {
                        let z = self.ngauss();
                        *x = (*x as f64 * (s * z).exp()).round() as u64;
                    }
                }
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
                let pick_least = |cand: &[usize]| -> usize {
                    cand.iter()
                        .copied()
                        .min_by(|&a, &b| {
                            after[a].cmp(&after[b]).then(gpu[b].cmp(&gpu[a])).then(local[b].cmp(&local[a]))
                                .then(act[a].cmp(&act[b])).then(a.cmp(&b))
                        })
                        .unwrap_or(own)
                };
                let mut least = pick_least(&allowed);
                let cap = self.c.cs_max_pull_tokens;
                let least_uncapped = least;
                if cap > 0 {
                    // Pull cap: a target whose pull (expected hit - own hit) exceeds the cap is no candidate.
                    let ok: Vec<usize> =
                        allowed.iter().copied().filter(|&w| w == own || exp[w].saturating_sub(local[w]) <= cap).collect();
                    least = pick_least(&ok);
                }
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
                if cap > 0 && least_uncapped != least {
                    let saved_u = after[own].saturating_sub(after[least_uncapped]);
                    let mut thr_u = self.c.cs_margin_tokens as f64;
                    if self.c.cs_r3_lambda >= 0.0 {
                        let pe = local[own].saturating_sub(local[least_uncapped]) as f64;
                        thr_u += pe * (self.c.cs_r3_rho + self.c.cs_r3_lambda);
                    }
                    let tgt_u = if least_uncapped != own && saved_u as f64 > thr_u { least_uncapped } else { own };
                    if tgt_u != tgt {
                        self.cs_cap_kept += 1;
                    }
                }
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
                if self.c.cs_state_age_s > 0.0 {
                    let keep = self.c.cs_state_age_s + 1.0;
                    let q = &mut self.st_routed[tgt];
                    q.push_back((t, cost));
                    while q.front().is_some_and(|x| x.0 < t - keep) {
                        q.pop_front();
                    }
                }
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
                let mut pidle = vec![false; np];
                let mut wait0 = vec![0.0; np];
                let mut own0 = vec![0.0; np];
                for w in 0..np {
                    let b = self.p[w].beta;
                    let pl = exp[w].saturating_sub(local[w]);
                    let host_local = local[w].saturating_sub(gpu[w].min(local[w]));
                    wait[w] = if self.c.cs_eft_state_age_s > 0.0 {
                        self.eft_wait_stale(t, w)
                    } else if self.c.cs_eft_wait_transit {
                        self.backlog_secs(w, true) + self.p[w].transit_s.max(0.0)
                    } else {
                        self.backlog_secs(w, false)
                    };
                    own_s[w] = eft_secs(&b, ch, l - exp[w], exp[w], host_local, pl);
                    if self.c.cs_eft_noise_queue > 0.0 {
                        let e = self.c.cs_eft_noise_queue;
                        let u = self.nrng.random();
                        wait[w] *= 1.0 + e * (2.0 * u - 1.0);
                    }
                    if self.c.cs_eft_noise_sigma > 0.0 {
                        let s = self.c.cs_eft_noise_sigma;
                        let z1 = self.ngauss();
                        let z2 = self.ngauss();
                        wait[w] *= (s * z1).exp();
                        own_s[w] *= (s * z2).exp();
                    }
                    pidle[w] = wait[w] <= 1e-9;
                    wait0[w] = wait[w];
                    own0[w] = own_s[w];
                    if !self.c.cs_pred_noise.is_empty() {
                        let u = self.nrng.random();
                        let tab = if pidle[w] && !self.c.cs_pred_noise_idle.is_empty() {
                            &self.c.cs_pred_noise_idle
                        } else {
                            &self.c.cs_pred_noise
                        };
                        let f = (self.c.cs_pred_noise_scale * table_draw(tab, u)).exp();
                        wait[w] *= f;
                        own_s[w] *= f;
                    }
                    pull_s[w] = if pl > 0 {
                        self.c.cs_eft_pull_fixed_s + self.c.cs_eft_pull_s_per_100k * pl as f64 / 1e5
                    } else {
                        0.0
                    };
                    let lam = if self.c.cs_eft_lp_ref > 0.0 && pl > 0 {
                        let m = self.lp_mult(t, w);
                        self.lp_stats.0 += m;
                        self.lp_stats.1 += 1;
                        self.c.cs_eft_dup_per_token_s * m
                    } else if self.c.lam_auto != "off" && pl > 0 {
                        let la = self.lam_auto_s(t, w, hint[w].map(|x| x.0));
                        self.lam_used.0 += la;
                        self.lam_used.1 += 1;
                        if t >= self.lo && t < self.hi {
                            self.lam_used.2 += la;
                            self.lam_used.3 += 1;
                        }
                        la
                    } else {
                        self.c.cs_eft_dup_per_token_s
                    };
                    let dtok = if self.c.cs_eft_dup_on_recompute {
                        local.iter().copied().max().unwrap_or(0).saturating_sub(local[w])
                    } else {
                        pl
                    };
                    let lam = if self.c.cs_eft_dup_on_recompute && pl == 0 && dtok > 0 && self.c.cs_eft_lp_ref > 0.0 {
                        self.c.cs_eft_dup_per_token_s * self.lp_mult(t, w)
                    } else {
                        lam
                    };
                    dup[w] = self.c.cs_eft_dup_rho * eft_secs(&b, ch, pl, 0, 0, 0) + lam * dtok as f64;
                    if pl > 0 {
                        dup[w] += self.c.cs_eft_handoff_s;
                    }
                    pulled[w] = pl;
                }
                let mut hol_alt = vec![0.0; np];
                if self.c.cs_hol_mu > 0.0 {
                    // PRED-EFT+HOL: head-of-line externality of placing the request on w (selected form in the
                    // cost; the other form's difference kept for the shadow decision).
                    let cap = self.c.cs_hol_rho_cap.clamp(0.0, 0.999);
                    for w in 0..np {
                        let (a, rho) = self.hol_rate(t, w);
                        let (s_, b_) = (own_s[w], wait[w]);
                        let light = a * s_ * (b_ + s_ / 2.0);
                        let bp = a * s_ * (b_ + s_) / (1.0 - rho.min(cap));
                        let (sel, alt) = if self.c.cs_hol_form == "bp" { (bp, light) } else { (light, bp) };
                        dup[w] += self.c.cs_hol_mu * sel;
                        hol_alt[w] = self.c.cs_hol_mu * (alt - sel);
                    }
                }
                let big_cold = l.saturating_sub(local[own_hit]) >= 32_768
                    && local[own_hit] < self.c.pull_min_incremental_tokens.max(self.c.pull_min_tokens);
                let tot_w = |w: usize| wait[w].max(pull_s[w]) + own_s[w] + dup[w];
                if self.c.cs_eft_owner_near_tokens > 0 && !self.c.cs_eft_one_way {
                    let best = local[own_hit];
                    let near = self.c.cs_eft_owner_near_tokens;
                    own = (0..np)
                        .filter(|&w| local[w] + near >= best)
                        .min_by(|&a, &b| wait[a].total_cmp(&wait[b]).then((a != own_hit).cmp(&(b != own_hit))).then(a.cmp(&b)))
                        .unwrap_or(own_hit);
                } else if self.c.cs_eft_owner_tie_c && !self.c.cs_eft_one_way {
                    let best = local[own_hit];
                    own = (0..np)
                        .filter(|&w| local[w] == best)
                        .min_by(|&a, &b| tot_w(a).total_cmp(&tot_w(b)).then((a != own_hit).cmp(&(b != own_hit))).then(a.cmp(&b)))
                        .unwrap_or(own_hit);
                }
                let gate_hold = if self.c.cs_eft_gate_big_tokens > 0 {
                    let fired = self.gate_fires(own);
                    self.gate_stats.0 += 1;
                    if fired {
                        self.gate_stats.1 += 1;
                    }
                    !fired
                } else {
                    false
                };
                let excluded = match prev {
                    Some(pv) if self.c.cs_eft_one_way
                        && self.c.cs_eft_exclude_prev
                        && (pv as usize) < np
                        && pv as usize != own
                        && local[pv as usize] >= self.c.pull_min_tokens =>
                    {
                        Some(pv as usize)
                    }
                    _ => None,
                };
                // S5 FPI family: (target, pull allowed) replaces the greedy EFT choice.
                let mut fpi_pull = true;
                let (t_, why) = if gate_hold {
                    (own, 0)
                } else if self.c.cs_pred_gate {
                    let fc = |w: usize| wait[w].max(pull_s[w]) + own_s[w];
                    let tot = |w: usize| fc(w) + dup[w];
                    let bnd = |w: usize| {
                        if self.c.cs_pred_curve {
                            let (kn, bv) = if pidle[w] && !self.c.bnd_idle_k.is_empty() {
                                (&self.c.bnd_idle_k, &self.c.bnd_idle_b)
                            } else {
                                (&self.c.bnd_busy_k, &self.c.bnd_busy_b)
                            };
                            return bound_curve(kn, bv, fc(w));
                        }
                        let k = if pidle[w] { self.c.cs_pred_k_idle } else { self.c.cs_pred_k_busy };
                        k * fc(w) + self.c.cs_pred_floor_s
                    };
                    self.pred_stats[0] += 1;
                    let res = match (0..np).filter(|&w| w != own).min_by(|&a, &b| tot(a).total_cmp(&tot(b)).then(a.cmp(&b))) {
                        Some(w) => {
                            let gain = tot(own) - tot(w);
                            if gain > 0.0 {
                                self.pred_stats[1] += 1;
                            }
                            let need = self.c.cs_pred_m * (bnd(own).powi(2) + bnd(w).powi(2)).sqrt() + self.c.cs_pred_gmin_s;
                            if gain - need > 0.0 {
                                self.pred_stats[2] += 1;
                                (w, 1)
                            } else {
                                (own, 0)
                            }
                        }
                        None => (own, 0),
                    };
                    if self.c.cs_hol_mu > 0.0 {
                        // Shadow: the same gate with the other HOL form; counters [decisions, differ, big cold,
                        // big cold and differ, differ with the selected P less loaded (B), differ with the selected
                        // form keeping the owner].
                        let tot2 = |w: usize| tot(w) + hol_alt[w];
                        let alt = match (0..np).filter(|&w| w != own).min_by(|&a, &b| tot2(a).total_cmp(&tot2(b)).then(a.cmp(&b))) {
                            Some(w) => {
                                let need =
                                    self.c.cs_pred_m * (bnd(own).powi(2) + bnd(w).powi(2)).sqrt() + self.c.cs_pred_gmin_s;
                                if tot2(own) - tot2(w) - need > 0.0 { w } else { own }
                            }
                            None => own,
                        };
                        self.hol_sh[0] += 1;
                        if big_cold {
                            self.hol_sh[2] += 1;
                        }
                        if alt != res.0 {
                            self.hol_sh[1] += 1;
                            if big_cold {
                                self.hol_sh[3] += 1;
                            }
                            if wait[res.0] < wait[alt] {
                                self.hol_sh[4] += 1;
                            }
                            if res.0 == own {
                                self.hol_sh[5] += 1;
                            }
                        }
                    }
                    res
                } else if self.c.cs_fpi != "off" {
                    let rec_s: Vec<f64> = (0..np)
                        .map(|w| {
                            let host_local = local[w].saturating_sub(gpu[w].min(local[w]));
                            eft_secs(&self.p[w].beta, ch, l - local[w], local[w], host_local, 0)
                        })
                        .collect();
                    let (k, use_pull) = self.fpi_route(t, r, own, l, &local, &wait, &own_s, &pull_s, &pulled, &rec_s);
                    fpi_pull = use_pull;
                    (k, if k != own { 1 } else { 0 })
                } else {
                    eft_choose_k(
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
                        self.c.cs_eft_margin_k,
                    )
                };
                tgt = t_;
                if self.c.fm_rate > 0.0 && tgt == own && self.force_tgt.is_none() && local[own] >= self.c.fm_min_tokens {
                    // LAMBDA probe: force a priced-out pull move to the best non-owner P (price-free total).
                    let tot0 = |w: usize| wait[w].max(pull_s[w]) + own_s[w];
                    let best = (0..np)
                        .filter(|&w| w != own && pulled[w] > 0)
                        .min_by(|&a, &b| tot0(a).total_cmp(&tot0(b)).then(a.cmp(&b)));
                    if let Some(b) = best {
                        if tot0(b) < tot0(own) {
                            self.fm_elig += 1;
                            let u = self.fm_rng.random();
                            if u < self.c.fm_rate {
                                // P2PMEM audit: fm_dst picks the forced target among the non-owner P with a pull.
                                let cands: Vec<usize> = (0..np).filter(|&w| w != own && pulled[w] > 0).collect();
                                let b = match self.c.fm_dst.as_str() {
                                    "random" => {
                                        let v = self.fm_rng.random();
                                        cands[((v * cands.len() as f64) as usize).min(cands.len() - 1)]
                                    }
                                    "worst" => *cands
                                        .iter()
                                        .max_by(|&&x, &&y| tot0(x).total_cmp(&tot0(y)).then(y.cmp(&x)))
                                        .unwrap(),
                                    _ => b,
                                };
                                tgt = b;
                                let g = tot0(own) - tot0(b);
                                let rq = &mut self.reqs[r as usize];
                                rq.forced = true;
                                rq.fm_gain = g;
                                rq.fm_pulled = pulled[b];
                                self.fm_n[0] += 1;
                                if t >= self.lo && t < self.hi {
                                    self.fm_n[1] += 1;
                                    self.fm_tok_win += pulled[b];
                                    self.fm_gain_win += g;
                                }
                            }
                        }
                    }
                }
                if let Some(f) = self.force_tgt {
                    tgt = f;
                    own = own_hit;
                }
                if self.c.cs_hol_mu > 0.0 {
                    self.hol_hist[tgt].push_back((t, own_s[tgt]));
                }
                match why {
                    1 => self.eft_counts[0] += 1,
                    2 => self.eft_counts[1] += 1,
                    3 => self.eft_counts[2] += 1,
                    4 => self.eft_counts[3] += 1,
                    _ => {}
                }
                if self.c.pred_metric {
                    let ing = self.c.ing_p.a + self.c.ing_p.b_per_100k * l as f64 / 1e5;
                    let rq = &mut self.reqs[r as usize];
                    rq.pred_t = t;
                    rq.pred_c = ing + wait[tgt].max(pull_s[tgt]) + own_s[tgt];
                    rq.pred_c0 = ing + wait0[tgt].max(pull_s[tgt]) + own0[tgt];
                    rq.pred_idle = pidle[tgt];
                    rq.pred_b0 = wait0[tgt];
                    rq.pred_s0 = own0[tgt];
                }
                let nh = if tgt != own { (Some(tgt as u16), Some(own as u16)) } else { (Some(tgt as u16), prev) };
                self.eft_home.insert(ckey, nh);
                let cost = if fpi_pull { l - exp[tgt] } else { l - local[tgt] };
                self.p[tgt].cs_pending += cost;
                self.p[tgt].cs_active += 1;
                self.reqs[r as usize].cs_cost = Some(cost);
                if self.c.pull_enabled && fpi_pull {
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
        let rcl = self.route_class(&hits, pt);
        let lost = best.min(pt).saturating_sub(hits[tgt].min(pt));
        if self.c.cs_eft_wait_transit || self.c.cs_eft_state_age_s > 0.0 {
            let u = self.reqs[r as usize].cs_cost.unwrap_or(pt.saturating_sub(hits[tgt].min(pt)));
            let s = eft_secs(&self.p[tgt].beta, self.c.chunk_tokens, u, pt.saturating_sub(u), 0, 0);
            if self.c.cs_eft_wait_transit {
                self.reqs[r as usize].transit_s = s;
                self.p[tgt].transit_s += s;
            }
            if self.c.cs_eft_state_age_s > 0.0 {
                self.eft_routed[tgt].push_back((t, s));
            }
        }
        let rq = &mut self.reqs[r as usize];
        rq.rclass = rcl;
        rq.lost = lost;
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
            self.d_prealloc(t, r);
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
        if self.reqs[r as usize].ld_inflight > 0 {
            self.d[d].inflight_tok = self.d[d].inflight_tok.saturating_sub(self.reqs[r as usize].ld_inflight);
            self.reqs[r as usize].ld_inflight = 0;
        }
        if self.c.d_mode == "d0" {
            self.d_try_admit(t, d);
        } else {
            self.ld_try_held(t, d);
        }
        if self.c.d_cache {
            self.d_cache_cap(t, d);
        }
    }

    // ------------------------------------------------------------------------------ S2 D cache
    /// S2 device tier of D `d` = the KV pool not held by active reservations; evict to fit.
    fn d_cache_cap(&mut self, t: f64, d: usize) {
        let bt = self.c.block_tokens;
        let dw = &mut self.d[d];
        dw.radix.dev_cap = dw.cap.saturating_sub(dw.used) / bt;
        dw.radix.evict_dev(t);
    }

    /// S2 D prealloc (engine _match_prefix_and_lock): match the prompt prefix on the D radix
    /// (refreshes it), lock it until the request ends, record the D prefix (at most len-1, page
    /// aligned).
    fn d_prealloc(&mut self, t: f64, r: u32) {
        if !self.c.d_cache {
            return;
        }
        let bt = self.c.block_tokens;
        let d = self.reqs[r as usize].d as usize;
        let lk = self.lkey(r);
        let m = self.d[d].radix.match_local(&lk, t);
        let deep = m.path.last().map(|&(n, _)| n);
        self.d[d].radix.lock(deep, 1);
        let pt = self.reqs[r as usize].pt;
        let capm = (pt.saturating_sub(1) / bt) * bt;
        self.reqs[r as usize].d_prefix = (m.total() as u64 * bt).min(capm);
        self.reqs[r as usize].d_lock = deep;
        self.d_cache_cap(t, d);
    }

    /// S2 decode finish: unlock the prealloc match and insert the prompt into the D radix (write-
    /// through); remember the chain's D for the oracle source.
    fn d_finish(&mut self, t: f64, r: u32) {
        if !self.c.d_cache {
            return;
        }
        let d = self.reqs[r as usize].d as usize;
        let lk = self.reqs[r as usize].d_lock.take();
        self.d[d].radix.lock(lk, -1);
        let runs = std::mem::take(&mut self.reqs[r as usize].runs);
        self.d[d].radix.insert(&runs, t, false);
        self.reqs[r as usize].runs = runs;
        let key = self.chain_key(r);
        self.last_d.insert(key, d as u16);
    }

    /// S2 read decision before P admission (one per request): source = the serving D ("a", "b";
    /// its prefix is locked since prealloc) or the D of the chain's previous turn ("c", oracle; leased
    /// for the read); "a" only for requests that had a pull hint (the pull has ended, short or failed).
    /// Reads [local, avail) when avail - local >= s2_min_tokens. Returns true when a read is in flight.
    fn s2_issue(&mut self, t: f64, p: usize, r: u32) -> bool {
        let bt = self.c.block_tokens;
        let mode = self.c.s2_read.as_str();
        if !self.c.d_cache || mode == "off" || (mode == "a" && self.reqs[r as usize].pull_src.is_none()) {
            return false;
        }
        let oracle = mode == "c";
        let lk = self.lkey(r);
        let x = self.rx(p);
        let local = self.p[x].radix.match_peek(&lk).total() as u64;
        let pt = self.reqs[r as usize].pt;
        let cap_blocks = pt.saturating_sub(1) / bt;
        let (src, avail, lease) = if oracle {
            let key = self.chain_key(r);
            let Some(&sd) = self.last_d.get(&key) else { return false };
            let m = self.d[sd as usize].radix.match_peek(&lk);
            let upto = (m.total() as u64).min(cap_blocks);
            (sd as usize, upto, TwoTierRadix::path_node_covering(&m.path, upto as u32))
        } else {
            (self.reqs[r as usize].d as usize, self.reqs[r as usize].d_prefix / bt, None)
        };
        if avail <= local || (avail - local) * bt < self.c.s2_min_tokens {
            return false;
        }
        if oracle {
            self.d[src].radix.lease(lease, 1);
        }
        let n = (avail - local) * bt;
        let rq = &mut self.reqs[r as usize];
        rq.dread_src = Some(src as u16);
        rq.dread_lease = lease;
        rq.dread_upto = avail as u32;
        rq.dread_have = local * bt;
        rq.dread = n;
        rq.dread_pending = true;
        self.dreads_all += 1;
        self.dread_tok_all += n;
        let dur = self.c.pull_fit.at(n);
        self.at(t + dur, Ev::DReadDone(r));
        true
    }

    /// S2 read landed: insert the read prefix on the P (as a pull landing), release the oracle
    /// source lease, wake the P.
    fn dread_done(&mut self, t: f64, r: u32) {
        let bt = self.c.block_tokens;
        let (src, dst, lease, upto) = {
            let rq = &self.reqs[r as usize];
            (rq.dread_src.unwrap_or(0) as usize, rq.p as usize, rq.dread_lease, rq.dread_upto)
        };
        let runs = runs_prefix(&self.reqs[r as usize].runs, upto);
        let host_only = self.c.pull_landing == "host";
        let x = self.rx(dst);
        let before = self.p[x].radix.stats.host_inserted_blocks;
        self.p[x].radix.insert(&runs, t, host_only);
        self.guard_pressure(t, dst, Some(r));
        let ins = (self.p[x].radix.stats.host_inserted_blocks - before) * bt;
        if t >= self.lo && t < self.hi {
            self.p[dst].pull_ins_win += ins;
        }
        if self.c.queued_prefix_protect && self.reqs[r as usize].t_fwd.is_none() {
            self.guard_landed(dst, r);
        }
        if lease.is_some() {
            self.d[src].radix.lease(lease, -1);
            self.d[src].radix.evict_dev(t);
        }
        self.reqs[r as usize].dread_lease = None;
        self.reqs[r as usize].dread_pending = false;
        if self.p[dst].run_items.is_none() && !self.p[dst].wq.is_empty() {
            if let Some(items) = self.form(t, dst) {
                self.start_iter(t, dst, items);
            }
        }
        self.p_idle_update(t, dst);
    }

    // ------------------------------------------------------------------------------ late D (PR #39)
    /// Late D: the request's leg(s) reach D without KV (alloc: the arrival D; choose: every D).
    fn ld_register(&mut self, r: u32) {
        if self.reqs[r as usize].ld_bind_t.is_some() {
            return; // already bound (bind can precede the D receipt in time)
        }
        if self.c.d_mode == "choose" {
            for d in &mut self.d {
                d.legs += 1;
            }
        } else {
            let d = self.reqs[r as usize].d as usize;
            self.d[d].legs += 1;
        }
    }

    /// D choice at the bind (choose mode) among the eligible candidates (rule 4 headroom; SMG
    /// late_bind.rs pick_among): "po2" = two distinct uniform draws, lower polled load (bound view,
    /// 5-s snapshot), tie -> first draw; "d1" = fewest charged in-flight tokens, ties -> fewer charged
    /// requests, then index (min_inflight_tokens.rs :250-272).
    fn ld_choose(&mut self, elig: &[usize]) -> usize {
        match self.c.d_bind_rule.as_str() {
            "d1" => *elig.iter().min_by_key(|&&k| (self.d[k].inflight_tok, self.d[k].nreq, k)).unwrap(),
            _ => {
                if elig.len() == 1 {
                    return elig[0];
                }
                let n = elig.len();
                let ii = self.urange(n);
                let mut jj = self.urange(n - 1);
                if jj >= ii {
                    jj += 1;
                }
                let (i, j) = (elig[ii], elig[jj]);
                if self.d[i].load_snap <= self.d[j].load_snap { i } else { j }
            }
        }
    }

    /// Eligible D for a bind from P `p`: alloc mode = the arrival D if under the outstanding cap;
    /// choose mode = every D under the cap (all D hold an INTENT once the room is bindable).
    fn ld_eligible(&self, p: usize, r: u32) -> Vec<usize> {
        let cap = self.c.ld_max_outstanding_per_d;
        if self.c.d_mode == "choose" {
            (0..self.d.len()).filter(|&k| self.p[p].ld_out[k] < cap).collect()
        } else {
            let d = self.reqs[r as usize].d as usize;
            if self.p[p].ld_out[d] < cap { vec![d] } else { Vec::new() }
        }
    }

    /// Bind decision (binder): choose the D among `elig` (choose mode: d_bind_rule; the other legs
    /// are released), charge the router in-flight counter (input + min(max_tokens, 8192) = need, d1
    /// ledger), send the ADMIT (arrives after ld_admit_latency_s, + ld_choose_latency_s in choose
    /// mode, not before the D leg), and re-run the binder when the decision stops counting (hold_s).
    fn ld_bind(&mut self, t: f64, r: u32, elig: &[usize]) {
        if self.reqs[r as usize].ld_bind_t.is_some() || self.c.d_mode == "d0" || elig.is_empty() {
            return;
        }
        self.reqs[r as usize].ld_bind_t = Some(t);
        let registered = self.reqs[r as usize].d_recv <= t;
        let mut lat = self.c.ld_admit_latency_s;
        if self.c.d_mode == "choose" {
            lat += self.c.ld_choose_latency_s;
            let d = self.ld_choose(elig);
            self.reqs[r as usize].d = d as u16;
            self.reqs[r as usize].d_counted = true;
            self.d[d].nreq += 1;
            if registered {
                for dw in &mut self.d {
                    dw.legs -= 1;
                }
            }
        } else if registered {
            let d = self.reqs[r as usize].d as usize;
            self.d[d].legs -= 1;
        }
        let d = self.reqs[r as usize].d as usize;
        let p = self.reqs[r as usize].p as usize;
        let it = self.reqs[r as usize].need;
        self.reqs[r as usize].ld_inflight = it;
        self.d[d].inflight_tok += it;
        self.p[p].ld_sent.push(r);
        self.p[p].ld_out[d] += 1;
        let at = (t + lat).max(self.reqs[r as usize].d_recv);
        self.at(at, Ev::LdAlloc(r));
        let hold = self.ld_hold_s(p);
        self.at(t + hold + 1e-6, Ev::LdTick(p as u16));
    }

    /// Binder's T_bind estimate (EWMA + 2 |dev|; before any sample: none) and hold_s.
    fn ld_tbind(&self, p: usize) -> Option<f64> {
        let pw = &self.p[p];
        if pw.ld_tb < 0.0 { None } else { Some(pw.ld_tb + 2.0 * pw.ld_tb_dev) }
    }

    fn ld_hold_s(&self, p: usize) -> f64 {
        let tb = self.ld_tbind(p).unwrap_or(0.0);
        self.c.ld_hold_detect_s.max(tb.min(2.0))
    }

    /// The ADMIT reaches D: queue the entry (ADMIT order) and run the late admission walk.
    fn ld_alloc(&mut self, t: f64, r: u32) {
        if self.reqs[r as usize].cancel {
            return;
        }
        let d = self.reqs[r as usize].d as usize;
        self.reqs[r as usize].ld_held_since = Some(t);
        self.d[d].held.push_back((r, t.to_bits()));
        self.ld_try_held(t, d);
    }

    /// D late admission walk (DESIGN 5.6): entries in ADMIT order; a failed budget check skips to
    /// the next entry; the oldest entry held >= ld_claim_passes D passes is the claim, and while it
    /// waits another entry allocates only if free - claim need >= need.
    fn ld_try_held(&mut self, t: f64, d: usize) {
        let claim_s = self.c.ld_claim_passes as f64 * self.c.step_ref_ms / 1e3;
        let mut claim_need = 0u64;
        let mut i = 0usize;
        while i < self.d[d].held.len() {
            let (r, t0) = self.d[d].held[i];
            if self.reqs[r as usize].cancel {
                self.d[d].held.remove(i);
                continue;
            }
            let need = self.reqs[r as usize].need;
            let free = self.d[d].cap.saturating_sub(self.d[d].used);
            let fits = if claim_need > 0 {
                free >= claim_need.saturating_add(need)
            } else {
                need <= free || self.d[d].used == 0
            };
            if fits {
                self.d[d].held.remove(i);
                self.d[d].used += need;
                self.reqs[r as usize].t_alloc = Some(t);
                self.reqs[r as usize].pa_on = true;
                self.d_prealloc(t, r);
                self.at(t + self.c.ld_meta_latency_s, Ev::LdBound(r));
                continue;
            }
            if i == 0 && claim_s >= 0.0 && t - f64::from_bits(t0) >= claim_s {
                claim_need = need;
            }
            i += 1;
        }
    }

    /// D metadata reached P: bootstrap done, the room leaves the bootstrap queue for the waiting
    /// queue; the binder's T_bind estimate is updated.
    fn ld_bound(&mut self, t: f64, r: u32) {
        let p = self.reqs[r as usize].p as usize;
        let d = self.reqs[r as usize].d as usize;
        self.reqs[r as usize].ld_bound_t = Some(t);
        if let Some(i) = self.p[p].ld_sent.iter().position(|&x| x == r) {
            self.p[p].ld_sent.swap_remove(i);
        }
        self.p[p].ld_out[d] = self.p[p].ld_out[d].saturating_sub(1);
        if let Some(b) = self.reqs[r as usize].ld_bind_t {
            let a = self.c.ld_ewma_alpha;
            let x = (t - b).min(2.0);
            let pw = &mut self.p[p];
            if pw.ld_tb < 0.0 {
                pw.ld_tb = x;
            } else {
                pw.ld_tb_dev = (1.0 - a) * pw.ld_tb_dev + a * (x - pw.ld_tb).abs();
                pw.ld_tb = (1.0 - a) * pw.ld_tb + a * x;
            }
        }
        self.p_enter_wq(t, r);
    }

    /// P binder, lead mode "work" (see SimConfig::ld_lead_margin_iters): runs once per P pass, when a
    /// room becomes bindable and when a decision stops counting (LdTick); binds bindable unbound
    /// bootstrap-queue rooms in arrival order while the window is open.
    fn ld_binder(&mut self, t: f64, p: usize) {
        if self.c.d_mode == "d0" {
            return;
        }
        let ch = self.c.chunk_tokens;
        let need_iters = match (self.ld_tbind(p), self.p[p].ld_ti) {
            (Some(tb), ti) if ti > 0.0 => {
                ((tb / ti).ceil() as u32 + self.c.ld_lead_margin_iters + self.p[p].ld_boost).clamp(1, self.c.ld_need_iters_cap)
            }
            _ => (2 + self.p[p].ld_boost).min(self.c.ld_need_iters_cap),
        };
        let target = need_iters as u64 * ch;
        let hold = self.ld_hold_s(p);
        let hit = self.p[p].ld_hit;
        let chunk_left: u64 = self.p[p].chunked.map(|c| {
            let rq = &self.reqs[c as usize];
            rq.unc.saturating_sub(rq.sched)
        }).unwrap_or(0);
        if chunk_left > target {
            return;
        }
        let mut a: u32 = 0;
        let mut tt: u64 = 0;
        for &r in self.p[p].wq.iter() {
            a += 1;
            tt += self.reqs[r as usize].pt;
        }
        for &r in self.p[p].ld_sent.iter() {
            let rq = &self.reqs[r as usize];
            if rq.ld_bind_t.map(|b| t - b < hold).unwrap_or(false) {
                a += 1;
                tt += rq.pt;
            }
        }
        let mut cands: Vec<u32> = self.p[p]
            .bq
            .iter()
            .copied()
            .filter(|&r| {
                let rq = &self.reqs[r as usize];
                rq.ld_bindable_t.is_some() && rq.ld_bind_t.is_none() && !rq.cancel
            })
            .collect();
        cands.sort_by_key(|&r| self.reqs[r as usize].qp_seq);
        for r in cands {
            let open = a < self.c.ld_lead_max_reqs
                && (a < self.c.ld_lead_min_reqs || chunk_left as f64 + tt as f64 * (1.0 - hit) < target as f64);
            if !open {
                break;
            }
            let elig = self.ld_eligible(p, r);
            if elig.is_empty() {
                continue; // rule 4: skip the room, try the next one
            }
            self.ld_bind(t, r, &elig);
            a += 1;
            tt += self.reqs[r as usize].pt;
        }
    }

    /// Starvation boost (per P pass): +1 when the pass found no runnable request while bindable
    /// rooms wait unbound; -1 after ld_starve_decay passes without starvation.
    fn ld_pass_boost(&mut self, p: usize, starved: bool) {
        let pw = &mut self.p[p];
        if starved {
            pw.ld_boost = (pw.ld_boost + 1).min(6);
            pw.ld_calm = 0;
        } else {
            pw.ld_calm += 1;
            if pw.ld_calm >= self.c.ld_starve_decay {
                pw.ld_boost = pw.ld_boost.saturating_sub(1);
                pw.ld_calm = 0;
            }
        }
    }

    // ------------------------------------------------------------------------------ P bootstrap / pulls
    fn on_p_recv(&mut self, t: f64, r: u32) {
        self.reqs[r as usize].p_in = true;
        if self.c.lpb && self.reqs[r as usize].lpb_disp > 0.0 {
            // LATE-P receipt calibration: residual of dispatch -> receive over the ingress fit.
            let pt = self.reqs[r as usize].pt;
            let base = self.c.ing_p.a + self.c.ing_p.b_per_100k * pt as f64 / 1e5;
            let x = t - self.reqs[r as usize].lpb_disp - base;
            self.lpb_ewma(0, x);
        }
        let p = self.reqs[r as usize].p as usize;
        if self.c.cs_eft_wait_transit {
            let s = std::mem::take(&mut self.reqs[r as usize].transit_s);
            self.p[p].transit_s -= s;
        }
        self.p[p].bq_n += 1;
        self.p[p].bq_tok += self.reqs[r as usize].pt;
        self.p[p].bq.push(r);
        self.p[p].qp_next += 1;
        self.reqs[r as usize].qp_seq = self.p[p].qp_next;
        if self.kv_tagged() {
            // #49 note_kv_session_arrival: register (session, version) with the FULL input.
            let (sid, ver) = (self.reqs[r as usize].kv_sid, self.reqs[r as usize].kv_ver);
            let key = self.reqs[r as usize].runs.clone();
            self.p[p].radix.kv_admit(sid, ver, &key);
        }
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
        if self.reqs[r as usize].t_alloc.is_some() || self.c.d_mode != "d0" {
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
        let (deep, depth) = if self.p[p].radix.kvret {
            // Retention: a GPU-backed node may have lost its host copy; the guard locks the deepest node
            // WITH a host copy (QPG locks host-backed nodes; base: every matched node has one).
            let (d, b) = self.p[p].radix.host_prefix(&m.path);
            (d, b * bt)
        } else {
            (m.path.last().map(|&(n, _)| n), m.total() as u64 * bt)
        };
        let Some(deep) = deep else { return };
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
        let (deep, depth) = if self.p[p].radix.kvret {
            let (d, b) = self.p[p].radix.host_prefix(&m.path);
            (d, b * bt)
        } else {
            (m.path.last().map(|&(n, _)| n), m.total() as u64 * bt)
        };
        let Some(deep) = deep else { return };
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
            if self.c.idle_overlap_metric && cls == 0 && t > t0 {
                self.idle_log.push((t0, t, p as u16));
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
            if cls == 0 && (self.c.ghost_frac > 0.0 || self.c.lam_auto != "off") {
                self.bp_close(t, p);
            }
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
        let tag = if self.kv_tagged() {
            Some((self.reqs[r as usize].kv_sid, self.reqs[r as usize].kv_ver, KV_PREFETCH))
        } else {
            None
        };
        // P2PMEM audit: a memory-neutral forced move carries the host capacity of its pulled blocks.
        let follow = (self.c.fm_cap_follow && self.reqs[r as usize].forced) || self.c.cap_follow_all;
        let pib = self.reqs[r as usize].pull_issue_blocks;
        if follow {
            self.p[dst].radix.host_cap += pib;
        }
        // S3 (d): the pulled pages are outside the destination's host capacity (no displacement).
        self.p[dst].radix.pf_mark = self.c.pull_free_capacity;
        self.p[dst].radix.ev_cause = 1;
        self.p[dst].radix.insert_kv(&runs, t, host_only, tag);
        self.p[dst].radix.ev_cause = 2;
        self.p[dst].radix.pf_mark = false;
        if self.c.kv_retention && self.c.kvret_mode == "transfer_end" {
            // The destination took its +1 at this commit (insert_kv tag); release the source now.
            self.kv_transfer_release(t, r, src);
        }
        if follow {
            self.p[src].radix.host_cap = self.p[src].radix.host_cap.saturating_sub(pib);
            self.p[src].radix.evict_host(t);
            self.cap_moved += pib;
        }
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
        // D0: after the D allocation; late D: after the D leg (its INTENT) reached the request.
        let base = if self.c.d_mode == "d0" { rq.p_recv.max(rq.t_alloc.unwrap_or(t)) } else { rq.p_recv.max(rq.d_recv) };
        let hs = self.qdraw(0);
        let rdy = base + if hs.is_nan() { self.c.handshake_s } else { hs.max(0.0) };
        let rdy = rdy.max(t);
        self.reqs[r as usize].boot_ready = Some(rdy);
        self.at(rdy, Ev::PReady(r));
    }

    fn p_ready(&mut self, t: f64, r: u32) {
        if self.c.d_mode != "d0" {
            // Late D: the room is bindable (INTENT and sender ready); it stays in the bootstrap
            // queue until the binder binds it and D allocates (ld_bound -> p_enter_wq).
            if self.reqs[r as usize].ld_bindable_t.is_none() {
                self.reqs[r as usize].ld_bindable_t = Some(t);
                let p = self.reqs[r as usize].p as usize;
                self.ld_binder(t, p);
                self.p_idle_update(t, p);
            }
            return;
        }
        self.p_enter_wq(t, r);
    }

    /// Bootstrap done: the request leaves the bootstrap queue for the FCFS waiting queue.
    fn p_enter_wq(&mut self, t: f64, r: u32) {
        if self.reqs[r as usize].p_ready.is_some() {
            return;
        }
        self.reqs[r as usize].p_ready = Some(t);
        if self.c.lpb && self.reqs[r as usize].lpb_disp > 0.0 {
            // LATE-P bootstrap calibration: receive -> waiting-queue entry (handshake, late-D bind latency) without
            // the late-D binder's queueing wait (P bindable -> bind decision): that wait is P queue time, and the
            // drain forecast already counts those rooms (bootstrap queue).
            let rq = &self.reqs[r as usize];
            let gate = match (rq.ld_bindable_t, rq.ld_bind_t) {
                (Some(a), Some(b)) if b > a => b - a,
                _ => 0.0,
            };
            let x = (t - rq.p_recv - gate).max(0.0);
            self.lpb_ewma(1, x);
        }
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
        // S2: the part read from a D copy (host-tier blocks beyond the local match at the read).
        let dread = self.reqs[r as usize].dread;
        let mut drd = 0;
        if dread > 0 {
            let beyond = (dev + host + stor).saturating_sub(self.reqs[r as usize].dread_have);
            drd = beyond.min(dread).min(host);
            host -= drd;
        }
        self.p[x].radix.load_back(&m.path, t);
        let deep = m.path.last().map(|&(n, _)| n);
        self.p[x].radix.lock(deep, 1);
        // F3: the queued-prefix host lock is released at admission.
        self.guard_release(p, r);
        let beta = self.p[p].beta;
        let mut lb = beta[3] * host as f64 / 1e5 + beta[4] * (stor + drd) as f64 / 1e5;
        if self.c.lb_hidden > 0.0 {
            let cap = (self.c.prefetch_max_tokens as f64).min(self.c.prefetch_max_fraction * pt as f64);
            let frac = if host + stor + drd > 0 { (cap / (host + stor + drd) as f64).min(1.0) } else { 0.0 };
            lb -= self.c.lb_hidden * frac * lb;
        }
        self.p[p].pf_set.push(r);
        {
            let a = self.c.ld_ewma_alpha;
            let h = (dev + host + stor + drd) as f64 / pt.max(1) as f64;
            let pw = &mut self.p[p];
            pw.ld_hit = (1.0 - a) * pw.ld_hit + a * h;
        }
        let rq = &mut self.reqs[r as usize];
        rq.locked = deep;
        rq.dev = dev;
        rq.host = host;
        rq.stor = stor;
        rq.drd = drd;
        rq.unc = pt.saturating_sub(dev + host + stor + drd).max(1);
        rq.sched = 0;
        rq.computed = 0;
        rq.lbcost = lb.max(0.0);
        if self.c.inj_rate > 0.0 && !self.reqs[r as usize].warm {
            let u = self.fm_rng.random();
            if u < self.c.inj_rate {
                let rq = &mut self.reqs[r as usize];
                rq.lbcost += self.c.inj_s;
                rq.inj = true;
                self.inj_n[0] += 1;
                if t >= self.lo && t < self.hi {
                    self.inj_n[1] += 1;
                }
            }
        }
        if self.c.ghost_frac > 0.0 {
            self.ghost_count(t, x, r);
        }
    }

    /// One scheduler pass: form a batch, then the F3 guard step (after batch formation).
    fn form(&mut self, t: f64, p: usize) -> Option<Vec<BItem>> {
        let res = self.form_batch(t, p);
        self.guard_step(t, p);
        if self.c.d_mode != "d0" {
            let waiting = self.p[p].bq.iter().any(|&r| {
                let rq = &self.reqs[r as usize];
                rq.ld_bindable_t.is_some() && rq.ld_bind_t.is_none()
            });
            self.ld_pass_boost(p, res.is_none() && waiting);
        }
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
        self.ld_binder(t, p);
        let mut idx = 0usize;
        while budget > 0 && idx < self.p[p].wq.len() {
            let r = self.p[p].wq[idx];
            if self.reqs[r as usize].cancel {
                self.p[p].wq.remove(idx);
                continue;
            }
            // Hold: a request whose pull (or S2 D read) is in flight is skipped; later requests pass it.
            if self.reqs[r as usize].pull_pending || self.reqs[r as usize].dread_pending {
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
            // S2: one read decision per request, after any pull has ended (the read lands like a pull).
            if self.c.s2_read != "off" && !self.reqs[r as usize].dread_tried {
                self.reqs[r as usize].dread_tried = true;
                if self.s2_issue(t, p, r) {
                    idx += 1;
                    continue;
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
        let a = self.c.ld_ewma_alpha;
        let pw = &mut self.p[p];
        if pw.ld_last_end >= 0.0 && (t - pw.ld_last_end).abs() < 1e-9 {
            pw.ld_ti = if pw.ld_ti < 0.0 { svc } else { (1.0 - a) * pw.ld_ti + a * svc };
        }
        pw.ld_last_end = t + svc;
        pw.busy_s += svc;
        if t >= self.lo && t < self.hi {
            pw.win_busy += svc;
        }
        pw.run_items = Some(items);
        pw.pass_t0 = t;
        pw.bp_open = true;
        self.p_idle_update(t, p);
        self.at(t + svc, Ev::EndIter(p as u16));
        let nxt = self.form(t, p);
        self.p[p].next_items = nxt;
    }

    fn end_iter(&mut self, t: f64, p: usize) {
        let items = self.p[p].run_items.take().unwrap_or_default();
        self.p[p].tok_done += items.iter().map(|x| x.n).sum::<u64>();
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
        if self.c.lpb {
            // LATE-P: a pass end = a new P state message; new forecast error draw, then dispatch.
            if !self.c.cs_pred_noise.is_empty() {
                let idle = self.p[p].run_items.is_none() && self.p[p].wq.is_empty();
                let u = self.nrng.random();
                let tab = if idle && !self.c.cs_pred_noise_idle.is_empty() {
                    &self.c.cs_pred_noise_idle
                } else {
                    &self.c.cs_pred_noise
                };
                self.p[p].lpb_nf = (self.c.cs_pred_noise_scale * table_draw(tab, u)).exp();
            }
            self.lpb_dispatch(t);
        }
    }

    fn prefill_done(&mut self, t: f64, p: usize, r: u32) {
        self.reqs[r as usize].t_pd = Some(t);
        self.p[p].bp_n += 1;
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
        let tag = if self.kv_tagged() {
            Some((self.reqs[r as usize].kv_sid, self.reqs[r as usize].kv_ver, KV_FINISHED))
        } else {
            None
        };
        self.p[x].radix.ev_cause = 0;
        let path = self.p[x].radix.insert_kv(&runs, t, false, tag);
        self.p[x].radix.ev_cause = 2;
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
        self.kv_send(t, r);
    }

    /// KV transfer tail to D (LDX-OFF fit + residual table) -> KV arrived. With the S2 D cache the
    /// P sends only [D prefix, len).
    fn kv_send(&mut self, t: f64, r: u32) {
        let pt = self.reqs[r as usize].pt.saturating_sub(self.reqs[r as usize].d_prefix);
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
        let mut step = step_ms(self.d[d].b, self.d[d].s);
        if self.c.ld_held_scan_ms > 0.0 {
            step += self.c.ld_held_scan_ms * (self.d[d].held.len() as f64 + self.d[d].legs.max(0) as f64);
        }
        if self.c.d_mode != "d0" {
            step += self.c.ld_step_overhead_ms;
        }
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
        self.d_finish(t, r);
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
        if !self.reqs[r as usize].warm {
            self.agent_last_done.insert(a, t);
        }
        let sid = self.reqs[r as usize].kv_sid;
        if self.kv_tagged() && self.c.kvret_idle_s > 0.0 {
            let ver = self.reqs[r as usize].kv_ver;
            self.at(t + self.c.kvret_idle_s, Ev::KvIdle(sid, ver));
        }
        if self.kv_tagged() && self.c.kvret_end && self.kv_is_last(r) {
            self.kv_end_session(t, sid);
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

    // ------------------------------------------------------------------------------ session retention
    /// Router registration at placement (Morphosis #25 register): version + 1 per request; on a P
    /// change the old P's version is retired (pending until READY on the new P); a P that serves the
    /// session again cancels its pending retirement.
    /// Session tags are sent (retention on and not recency_only).
    fn kv_tagged(&self) -> bool {
        self.c.kv_retention && self.c.kvret_mode != "recency_only"
    }

    fn kv_route(&mut self, t: f64, r: u32) {
        let a = self.reqs[r as usize].agent_state;
        let mut sid = a;
        if self.c.kvret_idle_s > 0.0 {
            // S4: after an idle END the agent's next request starts a new session identity.
            sid = self.kv_sid_of.get(&a).copied().unwrap_or(a);
            if self.kv_sess.get(&sid).is_some_and(|s| s.ended) {
                sid = self.kv_sid_next;
                self.kv_sid_next += 1;
                self.kv_sid_of.insert(a, sid);
                self.kv_idle_new += 1;
            }
        }
        self.reqs[r as usize].kv_sid = sid;
        let p = self.reqs[r as usize].p;
        let end_only = self.c.kvret_mode == "end_only";
        let s = self.kv_sess.entry(sid).or_default();
        s.ver += 1;
        let ver = s.ver;
        if let Some(cur) = s.cur {
            if cur != p && !end_only {
                if let Some(&(_, wv)) = s.workers.iter().find(|w| w.0 == cur) {
                    if !s.retire.iter().any(|x| x.0 == cur) {
                        s.retire.push((cur, wv, t));
                    }
                }
            }
        }
        let n0 = s.retire.len();
        s.retire.retain(|x| x.0 != p);
        let cancelled = (n0 - s.retire.len()) as u64;
        s.cur = Some(p);
        match s.workers.iter_mut().find(|w| w.0 == p) {
            Some(w) => w.1 = ver,
            None => s.workers.push((p, ver)),
        }
        let pend = !s.retire.is_empty();
        self.kv_cancel += cancelled;
        self.reqs[r as usize].kv_ver = ver;
        if pend && !self.kv_pend.contains(&sid) {
            self.kv_pend.push(sid);
        }
    }

    /// Router background worker (#25): for sessions with pending retirements, 'status' of the
    /// current version on the current P; READY -> 'release' of each retired (P, version).
    fn kv_tick(&mut self, t: f64) {
        if self.kv_pend.is_empty() {
            return;
        }
        let mut pend = std::mem::take(&mut self.kv_pend);
        pend.sort_unstable();
        pend.dedup();
        let dt = self.c.kvret_ctl_s;
        let mut keep = Vec::new();
        for sid in pend {
            let (cur, ver, idle) = {
                let s = &self.kv_sess[&sid];
                (s.cur, s.ver, s.retire.is_empty() || s.ended)
            };
            let Some(cur) = cur else { continue };
            if idle {
                continue;
            }
            self.kv_polls += 1;
            let (st, _, _) = self.p[cur as usize].radix.kv_status(sid, ver);
            if st == KV_READY {
                let items = std::mem::take(&mut self.kv_sess.get_mut(&sid).unwrap().retire);
                for (wp, wv, t0) in items {
                    self.at(t + dt, Ev::KvCtl(sid, wp, wv, 0));
                    self.kv_rel.push((t, t - t0, 0));
                }
            } else {
                keep.push(sid);
            }
        }
        self.kv_pend.extend(keep);
    }

    /// transfer_end: a pull of the session's current version has landed on its current P; if the pull
    /// source holds a pending retirement of this session (a move with a pull), release it at once
    /// (log op 2) instead of waiting for READY.
    fn kv_transfer_release(&mut self, t: f64, r: u32, src: usize) {
        let (sid, ver, dst) = {
            let rq = &self.reqs[r as usize];
            (rq.kv_sid, rq.kv_ver, rq.p)
        };
        let Some(s) = self.kv_sess.get_mut(&sid) else { return };
        if s.ended || s.ver != ver || s.cur != Some(dst) {
            return;
        }
        let Some(i) = s.retire.iter().position(|x| x.0 as usize == src) else { return };
        let (wp, wv, t0) = s.retire.remove(i);
        self.p[src].radix.kv_release(sid, wv);
        if self.c.kvret_move_free || self.c.kvret_src_first {
            // P2PMEM audit: the released source copy of the transferred prefix is freed / put first in eviction order.
            let upto = self.reqs[r as usize].pull_upto;
            let leased = runs_prefix(&self.reqs[r as usize].runs, upto);
            if self.c.kvret_move_free {
                self.p[src].radix.kv_move_free(&leased);
            } else {
                self.p[src].radix.kv_src_first(&leased);
            }
        }
        self.kv_log.push((t, sid, wp, wv, 2));
        self.kv_rel.push((t, t - t0, 1));
    }

    /// A control op reaches P `p`.
    fn kv_ctl(&mut self, t: f64, sid: u32, p: usize, ver: u32, op: u8) {
        if op == 0 {
            self.p[p].radix.kv_release(sid, ver);
        } else {
            self.p[p].radix.kv_end(sid, ver);
        }
        self.kv_log.push((t, sid, p as u16, ver, op));
    }

    /// S6 routing class of a request from the exact hits: 0 cold (best < pull_min_tokens), 1 low match
    /// (best <= smg_threshold x prompt), 2 several P within pull_min_incremental_tokens of the best,
    /// 3 a single best holder.
    fn route_class(&self, hits: &[u64], pt: u64) -> u8 {
        let h: Vec<u64> = hits.iter().map(|&x| x.min(pt)).collect();
        let best = *h.iter().max().unwrap_or(&0);
        if best < self.c.pull_min_tokens {
            return 0;
        }
        if best as f64 <= self.c.smg_threshold * pt as f64 {
            return 1;
        }
        let near = h.iter().filter(|&&x| x + self.c.pull_min_incremental_tokens > best).count();
        if near >= 2 {
            2
        } else {
            3
        }
    }

    /// S6: the co_schedule decision of P1 (least queued uncached tokens on the live backlog, margin 0,
    /// no R3 penalty, no pair / hysteresis) with pull cap `cap`, without side effects.
    fn p1_choice(&mut self, r: u32, hits: &[u64], cap: u64) -> usize {
        let np = self.p.len();
        let l = self.reqs[r as usize].pt;
        let gpu = self.gpu_hits(r);
        let local: Vec<u64> = hits.iter().map(|&h| h.min(l)).collect();
        let mut exp = vec![0u64; np];
        for w in 0..np {
            let s_ = (0..np).filter(|&k| k != w && hits[k] > 0).min_by(|&a, &b| hits[b].cmp(&hits[a]).then(a.cmp(&b)));
            let mut hs = 0;
            if let Some(s_) = s_ {
                if hits[s_] >= self.c.pull_min_tokens
                    && hits[s_].saturating_sub(hits[w]) >= self.c.pull_min_incremental_tokens
                {
                    hs = hits[s_];
                }
            }
            exp[w] = local[w].max(hs).min(l);
        }
        let pend: Vec<u64> = (0..np).map(|w| self.true_backlog(w)).collect();
        // In-flight requests per P (route -> KV arrived): the co_schedule ledger count, or SMG's own
        // counter on the cache-aware path (the ledger is not kept there; same release point).
        let act: Vec<u64> = if self.c.p_policy == "smgca" {
            self.p.iter().map(|p| p.nin.max(0) as u64).collect()
        } else {
            self.p.iter().map(|p| p.cs_active).collect()
        };
        let after: Vec<u64> = (0..np).map(|w| pend[w] + (l - exp[w])).collect();
        let own = (0..np)
            .min_by(|&a, &b| {
                local[b].cmp(&local[a]).then(gpu[b].cmp(&gpu[a])).then(pend[a].cmp(&pend[b])).then(act[a].cmp(&act[b])).then(a.cmp(&b))
            })
            .unwrap();
        let least = (0..np)
            .filter(|&w| w == own || exp[w].saturating_sub(local[w]) <= cap)
            .min_by(|&a, &b| {
                after[a].cmp(&after[b]).then(gpu[b].cmp(&gpu[a])).then(local[b].cmp(&local[a])).then(act[a].cmp(&act[b])).then(a.cmp(&b))
            })
            .unwrap_or(own);
        if least != own && after[own].saturating_sub(after[least]) > 0 {
            least
        } else {
            own
        }
    }

    /// S5: standard normal draw from the noise stream (Box-Muller).
    fn ngauss(&mut self) -> f64 {
        let u1 = self.nrng.random().max(1e-300);
        let u2 = self.nrng.random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// S5: co_schedule backlog signal now (cs_workload "true" = live backlog, else the ledger).
    fn cs_pend_now(&self) -> Vec<u64> {
        if self.c.cs_workload == "true" {
            (0..self.p.len()).map(|w| self.true_backlog(w)).collect()
        } else {
            self.p.iter().map(|p| p.cs_pending).collect()
        }
    }

    /// S7: EFT backlog seconds as sampled cs_eft_state_age_s ago, minus the time since the sample,
    /// plus the router's own seconds routed to that P since the sample.
    fn eft_wait_stale(&self, t: f64, w: usize) -> f64 {
        let due = t - self.c.cs_eft_state_age_s;
        let Some(&(ts, ref v)) = self.eft_hist.iter().rev().find(|x| x.0 <= due).or(self.eft_hist.front()) else {
            return self.backlog_secs(w, true);
        };
        let own: f64 = self.eft_routed[w].iter().filter(|x| x.0 > ts).map(|x| x.1).sum();
        (v[w] - (t - ts)).max(0.0) + own
    }

    /// S7 R2: lambda multiplier from the EWMA class-1 host-eviction rate (worker or global scope).
    fn lp_mult(&mut self, t: f64, w: usize) -> f64 {
        let tau = self.c.cs_eft_lp_tau_s.max(1e-3);
        let bt = self.c.block_tokens as f64;
        for k in 0..self.p.len() {
            let pw = &mut self.p[k];
            if pw.lp_t == t && pw.lp_cur == pw.radix.evictions.len() {
                continue;
            }
            let dt = (t - pw.lp_t).max(0.0);
            let mut add = 0.0;
            for e in &pw.radix.evictions[pw.lp_cur..] {
                if e.class == 1 {
                    add += e.blocks as f64 * bt;
                }
            }
            pw.lp_cur = pw.radix.evictions.len();
            pw.lp_rate = pw.lp_rate * (-dt / tau).exp() + add / tau;
            pw.lp_t = t;
        }
        let rate = if self.c.cs_eft_lp_scope == "global" {
            self.p.iter().map(|p| p.lp_rate).sum::<f64>()
        } else {
            self.p[w].lp_rate
        };
        1.0 + self.c.cs_eft_lp_gain * (rate / self.c.cs_eft_lp_ref - 1.0).max(0.0)
    }

    /// S7 B2: the owner holds (queued, bootstrap or running) an item with >= big uncached tokens
    /// whose predicted prefill is >= theta seconds.
    fn gate_fires(&self, own: usize) -> bool {
        let p = &self.p[own];
        let ch = self.c.chunk_tokens;
        let big = self.c.cs_eft_gate_big_tokens;
        let th = self.c.cs_eft_gate_theta_s;
        let hit = |u: u64, pt: u64| u >= big && eft_secs(&p.beta, ch, u, pt.saturating_sub(u), 0, 0) >= th;
        p.wq.iter().chain(p.bq.iter()).any(|&r| {
            let rq = &self.reqs[r as usize];
            hit(rq.cs_cost.unwrap_or(rq.unc_est), rq.pt)
        }) || p.pf_set.iter().any(|&r| {
            let rq = &self.reqs[r as usize];
            hit(rq.unc.saturating_sub(rq.computed), rq.pt)
        })
    }

    /// S7 B1: router view of a P's backlog seconds (queues + running + in-flight ledger).
    fn hold_backlog(&self, w: usize) -> f64 {
        self.backlog_secs(w, true) + self.p[w].transit_s.max(0.0)
    }

    fn hold_limit(&self, w: usize) -> f64 {
        let ch = self.c.chunk_tokens;
        self.c.cs_hold_passes * eft_secs(&self.p[w].beta, ch, ch, 0, 0, 0)
    }

    /// S7 B1: route request r now, to P w (pull from the best other holder as the EFT path hints).
    fn dispatch_to(&mut self, t: f64, r: u32, w: usize) {
        self.force_tgt = Some(w);
        self.route(t, r);
        self.force_tgt = None;
    }

    /// S7 B1: release own held requests while the owner is under its limit, then let near-idle P steal.
    fn hold_tick(&mut self, t: f64) {
        let np = self.p.len();
        for w in 0..np {
            while let Some(i) = self.held.iter().position(|h| h.1 as usize == w) {
                if self.hold_backlog(w) > self.hold_limit(w) {
                    break;
                }
                let (r, _, t0) = self.held.remove(i);
                self.reqs[r as usize].hold_s = t - t0;
                self.dispatch_to(t, r, w);
            }
        }
        if self.c.cs_hold_steal {
            let ch = self.c.chunk_tokens;
            for j in 0..np {
                while !self.held.is_empty() && self.hold_backlog(j) <= self.c.cs_hold_idle_s {
                    let mut best: Option<(usize, f64)> = None;
                    for i in 0..self.held.len() {
                        let (r, o, t0) = self.held[i];
                        let o = o as usize;
                        if o == j || self.steal_rx_t[o] + self.c.cs_hold_worker_cd_s > t {
                            continue;
                        }
                        let ck = self.chain_key(r);
                        if self.sess_move_t.get(&ck).is_some_and(|&x| x + self.c.cs_hold_sess_cd_s > t) {
                            continue;
                        }
                        let hits = self.actual_hits(r);
                        let pt = self.reqs[r as usize].pt;
                        let (mo, mj) = (hits[o].min(pt), hits[j].min(pt));
                        let x = if mo > mj && mo - mj >= self.c.pull_min_incremental_tokens && mo >= self.c.pull_min_tokens {
                            mo - mj
                        } else {
                            0
                        };
                        let so = eft_secs(&self.p[o].beta, ch, pt - mo, mo, 0, 0);
                        let have = mj + x;
                        let sj = eft_secs(&self.p[j].beta, ch, pt - have, have, 0, x);
                        let behind = self.held.iter().filter(|h| h.1 as usize == o && h.2 > t0).count().min(3) as f64;
                        let tp = if x > 0 {
                            self.c.cs_eft_pull_fixed_s + self.c.cs_eft_pull_s_per_100k * x as f64 / 1e5 + self.c.cs_eft_handoff_s
                        } else {
                            0.0
                        };
                        let g = self.hold_backlog(o) + so + self.c.cs_hold_w * so * behind
                            - (tp + sj)
                            - self.c.cs_eft_dup_per_token_s * x as f64;
                        if g > self.c.cs_hold_margin_s && best.map_or(true, |b| g > b.1) {
                            best = Some((i, g));
                        }
                    }
                    let Some((i, _)) = best else { break };
                    let (r, _, t0) = self.held.remove(i);
                    self.reqs[r as usize].hold_s = t - t0;
                    self.reqs[r as usize].stolen = true;
                    self.steal_rx_t[j] = t;
                    let ck = self.chain_key(r);
                    self.sess_move_t.insert(ck, t);
                    self.dispatch_to(t, r, j);
                }
            }
        }
        if !self.held.is_empty() && t < self.c.duration_s {
            self.at(t + self.c.cs_hold_tick_s, Ev::HoldTick);
        } else {
            self.hold_tick_on = false;
        }
    }

    /// PRED-EFT+HOL: (requests routed to P w per second, utilization = predicted service seconds routed to w per
    /// second) over the last cs_hol_window_s.
    fn hol_rate(&mut self, t: f64, w: usize) -> (f64, f64) {
        let win = self.c.cs_hol_window_s.max(1e-3);
        let span = win.min(t.max(1e-3));
        let ks: Vec<usize> = if self.c.cs_hol_rate == "uniform" { (0..self.p.len()).collect() } else { vec![w] };
        let (mut n, mut s) = (0.0, 0.0);
        for &k in &ks {
            let q = &mut self.hol_hist[k];
            while q.front().is_some_and(|x| x.0 < t - win) {
                q.pop_front();
            }
            n += q.len() as f64;
            s += q.iter().map(|x| x.1).sum::<f64>();
        }
        let m = ks.len() as f64;
        (n / span / m, s / span / m)
    }

    /// LATE-P: EWMA calibration (alpha = ld_ewma_alpha) of the receipt residual (k = 0) or the bootstrap delay (1):
    /// mean and mean absolute deviation.
    fn lpb_ewma(&mut self, k: usize, x: f64) {
        let a = self.c.ld_ewma_alpha;
        let (m, d) = (2 * k, 2 * k + 1);
        if self.lpb_rx_n[k] == 0 {
            self.lpb_rx[m] = x;
            self.lpb_rx[d] = 0.0;
        } else {
            self.lpb_rx[d] = (1.0 - a) * self.lpb_rx[d] + a * (x - self.lpb_rx[m]).abs();
            self.lpb_rx[m] = (1.0 - a) * self.lpb_rx[m] + a * x;
        }
        self.lpb_rx_n[k] += 1;
    }

    /// LATE-P: forecast seconds until P w has no work (router view, with the forecast-error factor).
    fn lpb_drain(&self, t: f64, w: usize) -> f64 {
        let p = &self.p[w];
        let mut d = self.backlog_secs(w, true) + p.transit_s.max(0.0);
        if p.run_items.is_some() {
            d -= (t - p.pass_t0).max(0.0);
        }
        d.max(0.0) * p.lpb_nf
    }

    /// LATE-P: predicted dispatch -> receive and receive -> waiting-queue delays of a `pt`-token request (calibrated
    /// mean + lpb_rx_k x mean absolute deviation; before samples: fit / handshake + late-D latencies).
    fn lpb_lead_parts(&self, pt: u64) -> (f64, f64) {
        let k = self.c.lpb_rx_k;
        let base = self.c.ing_p.a + self.c.ing_p.b_per_100k * pt as f64 / 1e5;
        let recv = if self.lpb_rx_n[0] > 0 { base + self.lpb_rx[0] + k * self.lpb_rx[1] } else { base };
        let boot = if self.lpb_rx_n[1] > 0 {
            self.lpb_rx[2] + k * self.lpb_rx[3]
        } else {
            self.c.handshake_s + self.c.ld_admit_latency_s + self.c.ld_choose_latency_s + self.c.ld_meta_latency_s
        };
        (recv.max(0.0) * self.c.lpb_rx_scale, boot.max(0.0) * self.c.lpb_rx_scale)
    }

    fn lpb_titer(&self, w: usize) -> f64 {
        let ti = self.p[w].ld_ti;
        if ti > 0.0 { ti } else { eft_secs(&self.p[w].beta, self.c.chunk_tokens, self.c.chunk_tokens, 0, 0, 0) }
    }

    fn lpb_bound(&self, idle: bool, x: f64) -> f64 {
        let (kn, bv) = if idle && !self.c.bnd_idle_k.is_empty() {
            (&self.c.bnd_idle_k, &self.c.bnd_idle_b)
        } else {
            (&self.c.bnd_busy_k, &self.c.bnd_busy_b)
        };
        bound_curve(kn, bv, x)
    }

    /// LATE-P candidate view of request r: owner, cold flag, per-P own seconds (noise-free), pull seconds,
    /// duplicated-token charge (same hints as the EFT path). Cached for LPB_CACHE_S.
    fn lpb_cand(&mut self, t: f64, r: u32, d: &[f64]) -> LpbC {
        if let Some(c) = self.lpb_cache.get(&r) {
            if t - c.t <= LPB_CACHE_S {
                return c.clone();
            }
        }
        let c = self.lpb_cand_new(t, r, d);
        self.lpb_cache.insert(r, c.clone());
        c
    }

    fn lpb_cand_new(&mut self, t: f64, r: u32, d: &[f64]) -> LpbC {
        let np = self.p.len();
        let bt = self.c.block_tokens;
        let ch = self.c.chunk_tokens;
        let l = self.reqs[r as usize].pt;
        let runs = self.reqs[r as usize].runs.clone();
        let ms: Vec<MatchResult> = (0..np).map(|w| self.p[w].radix.match_peek(&runs)).collect();
        let hits: Vec<u64> = ms.iter().map(|m| m.total() as u64 * bt).collect();
        let gpu: Vec<u64> = ms.iter().map(|m| m.dev as u64 * bt).collect();
        let local: Vec<u64> = hits.iter().map(|&h| h.min(l)).collect();
        let best = local.iter().copied().max().unwrap_or(0);
        let own = (0..np)
            .min_by(|&a, &b| local[b].cmp(&local[a]).then(gpu[b].cmp(&gpu[a])).then(d[a].total_cmp(&d[b])).then(a.cmp(&b)))
            .unwrap();
        let cold = best < self.c.pull_min_incremental_tokens.max(self.c.pull_min_tokens);
        let mut s = vec![0.0; np];
        let mut ps = vec![0.0; np];
        let mut dup = vec![0.0; np];
        for w in 0..np {
            let s_ = (0..np).filter(|&k| k != w && hits[k] > 0).min_by(|&a, &b| hits[b].cmp(&hits[a]).then(a.cmp(&b)));
            let mut hs = 0;
            if let Some(s_) = s_ {
                if self.c.pull_enabled
                    && hits[s_] >= self.c.pull_min_tokens
                    && hits[s_].saturating_sub(hits[w]) >= self.c.pull_min_incremental_tokens
                {
                    hs = hits[s_];
                }
            }
            let exp = local[w].max(hs).min(l);
            let pl = exp.saturating_sub(local[w]);
            let host_local = local[w].saturating_sub(gpu[w].min(local[w]));
            s[w] = eft_secs(&self.p[w].beta, ch, l - exp, exp, host_local, pl);
            ps[w] = if pl > 0 { self.c.cs_eft_pull_fixed_s + self.c.cs_eft_pull_s_per_100k * pl as f64 / 1e5 } else { 0.0 };
            let dtok = if self.c.cs_eft_dup_on_recompute { best.saturating_sub(local[w]) } else { pl };
            dup[w] = self.c.cs_eft_dup_per_token_s * dtok as f64;
        }
        LpbC { t, own, cold, s0: s, ps, dup, pt: l }
    }

    /// LATE-P dispatcher (see SimConfig::lpb). Runs on arrivals, pass ends and the router timer.
    fn lpb_dispatch(&mut self, t: f64) {
        let np = self.p.len();
        loop {
            if self.lpb_q.is_empty() {
                break;
            }
            let d: Vec<f64> = (0..np).map(|w| self.lpb_drain(t, w)).collect();
            let dlo: Vec<f64> =
                (0..np).map(|w| (d[w] - self.c.lpb_bnd_scale * self.lpb_bound(false, d[w])).max(0.0)).collect();
            let ti: Vec<f64> = (0..np).map(|w| self.lpb_titer(w)).collect();
            // Cheap pre-check: can any P open for the largest lead in the queue (largest prompt, full pull)?
            let ptmax = self.lpb_q.iter().map(|&r| self.reqs[r as usize].pt).max().unwrap_or(0);
            let (rm, bm) = self.lpb_lead_parts(ptmax);
            let pm = self.c.cs_eft_pull_fixed_s + self.c.cs_eft_pull_s_per_100k * ptmax as f64 / 1e5;
            if (0..np).all(|w| dlo[w] > rm + bm.max(pm) + ti[w] + self.c.lpb_lead_extra_s) {
                break;
            }
            self.lpb_counts[5] += 1;
            let q = self.lpb_q.clone();
            let nf: Vec<f64> = (0..np).map(|w| self.p[w].lpb_nf).collect();
            let mut cands = Vec::with_capacity(q.len());
            for &r in &q {
                let mut c = self.lpb_cand(t, r, &d);
                for w in 0..np {
                    c.s0[w] *= nf[w];
                }
                cands.push(c);
            }
            let mut order: Vec<usize> = (0..q.len()).collect();
            if self.c.lpb_order == "hrrn" {
                let key = |i: usize| {
                    let so = cands[i].s0[cands[i].own].max(1e-6);
                    (t - self.reqs[q[i] as usize].lpb_arr + so) / so
                };
                order.sort_by(|&a, &b| key(b).total_cmp(&key(a)).then(a.cmp(&b)));
            }
            let mut ahead = vec![0.0f64; np];
            let mut pick: Option<(usize, usize, u8)> = None;
            for &i in &order {
                let c = &cands[i];
                let (o, cold, s, ps, dup, pt) = (c.own, c.cold, &c.s0, &c.ps, &c.dup, c.pt);
                let (recv, boot) = self.lpb_lead_parts(pt);
                let extra = self.c.lpb_lead_extra_s;
                let lead = |w: usize| recv + boot.max(ps[w]) + ti[w] + extra;
                let open = |w: usize| dlo[w] <= lead(w);
                let cw = |w: usize| d[w].max(recv + boot.max(ps[w])) + s[w];
                if cold {
                    let best = (0..np).filter(|&w| open(w)).min_by(|&a, &b| cw(a).total_cmp(&cw(b)).then(a.cmp(&b)));
                    if let Some(w) = best {
                        pick = Some((i, w, 2));
                        break;
                    }
                    continue;
                }
                if open(o) {
                    pick = Some((i, o, 1));
                    break;
                }
                if self.c.lpb_moves {
                    let co = (d[o] + ahead[o]).max(recv + boot) + s[o];
                    let bo = self.lpb_bound(d[o] + ahead[o] <= 1e-9, co);
                    let mut best: Option<(usize, f64)> = None;
                    for w in (0..np).filter(|&w| w != o && open(w)) {
                        let c = cw(w);
                        let bw = self.lpb_bound(d[w] <= 1e-9, c);
                        let g = (co - c - dup[w]) - (bo * bo + bw * bw).sqrt() - self.c.cs_pred_gmin_s;
                        if g > 0.0 && best.map_or(true, |x| g > x.1) {
                            best = Some((w, g));
                        }
                    }
                    if let Some((w, _)) = best {
                        pick = Some((i, w, 3));
                        break;
                    }
                }
                ahead[o] += s[o];
            }
            let Some((i, w, cls)) = pick else { break };
            let r = q[i];
            let pos = self.lpb_q.iter().position(|&x| x == r).unwrap();
            self.lpb_q.remove(pos);
            {
                let rq = &mut self.reqs[r as usize];
                rq.lpb_disp = t;
                rq.hold_s = t - rq.lpb_arr;
                rq.lpb_cls = cls;
            }
            self.lpb_cache.remove(&r);
            self.dispatch_to(t, r, w);
            // counters: owner, cold, move with a pull hint, move that recomputes (no holder hint on w)
            let k = if cls == 3 && self.reqs[r as usize].pull_src.is_none() { 4 } else { cls as usize };
            self.lpb_counts[0] += 1;
            self.lpb_counts[k] += 1;
        }
        if !self.lpb_q.is_empty() && !self.lpb_tick_on && t < self.c.duration_s {
            self.lpb_tick_on = true;
            self.at(t + self.c.lpb_tick_s, Ev::LpbTick);
        }
    }

    /// Output only (iww_metric): P idle while work waits in the router queue or in another P's waiting queue;
    /// per-P outstanding uncached tokens (waiting + admitted + bootstrap) max / mean.
    // ------------------------------------------------------------------------------ LAMBDA study
    /// Move the deletions logged by P p's tree into its ghost list and trim it to ghost_frac x host capacity.
    fn ghost_drain(&mut self, p: usize) {
        let cap = (self.c.ghost_frac * self.p[p].radix.host_cap as f64).max(1.0) as usize;
        let pw = &mut self.p[p];
        let log = std::mem::take(&mut pw.radix.ghost_log);
        for (s, c) in log {
            for i in 0..c as u64 {
                pw.ghost_seq += 1;
                pw.ghost_map.insert(s + i, pw.ghost_seq);
                pw.ghost_q.push_back((s + i, pw.ghost_seq));
            }
        }
        while pw.ghost_map.len() > cap {
            let Some((b, q)) = pw.ghost_q.pop_front() else { break };
            if pw.ghost_map.get(&b) == Some(&q) {
                pw.ghost_map.remove(&b);
            }
        }
        if pw.ghost_q.len() > 4 * cap + 1024 {
            let gm = &pw.ghost_map;
            pw.ghost_q.retain(|&(b, q)| gm.get(&b) == Some(&q));
        }
    }

    /// Ghost hit of request r on its serving P p (first chunk): blocks beyond the actual match, contiguous from the
    /// match end and within the engine match cap, that are in p's ghost. Counted blocks leave the ghost (recomputed).
    fn ghost_count(&mut self, t: f64, p: usize, r: u32) {
        self.ghost_drain(p);
        let bt = self.c.block_tokens;
        let (have, cap, runs) = {
            let rq = &self.reqs[r as usize];
            ((rq.dev + rq.host + rq.stor + rq.drd) / bt, rq.pt.saturating_sub(1) / bt, rq.runs.clone())
        };
        let pw = &mut self.p[p];
        let mut off = 0u64;
        let mut n = 0u64;
        'outer: for &(s, c) in &runs {
            let c = c as u64;
            if off + c <= have {
                off += c;
                continue;
            }
            for i in have.saturating_sub(off)..c {
                if off + i >= cap || pw.ghost_map.remove(&(s + i)).is_none() {
                    break 'outer;
                }
                n += 1;
            }
            off += c;
        }
        let tok = n * bt;
        pw.ghost_hit_all += tok;
        if t >= self.lo && t < self.hi {
            pw.ghost_hit_win += tok;
        }
        self.reqs[r as usize].ghost = tok;
    }

    /// Busy-period bookkeeping: called when P p becomes fully idle (nothing queued or running).
    fn bp_close(&mut self, t: f64, p: usize) {
        let tau = self.c.lam_tau_s.max(1e-3);
        let win = t >= self.lo && t < self.hi;
        let pw = &mut self.p[p];
        if !pw.bp_open {
            return;
        }
        pw.bp_open = false;
        let n = pw.bp_n as f64;
        pw.bp_n = 0;
        if n <= 0.0 {
            return;
        }
        let d = (-(t - pw.bp_t).max(0.0) / tau).exp();
        pw.bp_a = pw.bp_a * d + n * (n + 1.0) / 2.0;
        pw.bp_b = pw.bp_b * d + n;
        pw.bp_t = t;
        if win {
            pw.bp_win[0] += n * (n + 1.0) / 2.0;
            pw.bp_win[1] += n;
            pw.bp_win[2] += 1.0;
        }
    }

    /// lambda_auto parts over destination set w (None = all P) and source set o: [h, T_dup, q, R, lambda (s/token)].
    fn lam_parts(&self, w: Option<usize>, o: Option<usize>) -> [f64; 5] {
        let np = self.p.len();
        let iw: Vec<usize> = match w {
            Some(k) => vec![k],
            None => (0..np).collect(),
        };
        let io: Vec<usize> = match o {
            Some(k) => vec![k],
            None => (0..np).collect(),
        };
        let bt = self.c.block_tokens as f64;
        let sum = |ix: &[usize], k: usize| ix.iter().map(|&i| self.p[i].lam_e[k]).sum::<f64>();
        let gs = sum(&iw, 1);
        let gtarget: f64 = iw.iter().map(|&i| self.c.ghost_frac * self.p[i].radix.host_cap as f64 * bt).sum();
        let h = if gs > 0.01 * gtarget && gs > 0.0 { sum(&iw, 0) / gs } else { 0.0 };
        let (c0, e0, eall) = (sum(&io, 2), sum(&io, 3), sum(&io, 4));
        let cap: f64 = io.iter().map(|&i| self.p[i].radix.host_cap as f64 * bt).sum();
        let turn = cap / eall.max(1.0);
        let tdup = ((c0 + bt) / e0.max(1.0)).min(turn);
        let nw = iw.len() as f64;
        let busy = (sum(&iw, 5) / nw).clamp(0.0, 1.0);
        let q = match self.c.lam_q.as_str() {
            "sat" => self.c.concurrency as f64 / np as f64 * busy,
            "bpsat" => {
                let a: f64 = iw.iter().map(|&i| { let n = self.p[i].bp_n as f64; self.p[i].bp_a + n * (n + 1.0) / 2.0 }).sum();
                let b: f64 = iw.iter().map(|&i| self.p[i].bp_b + self.p[i].bp_n as f64).sum();
                let qb = if b > 0.0 { (a / b).max(1.0) } else { 1.0 };
                qb.min((self.c.concurrency as f64 / np as f64 * busy).max(1.0))
            }
            "bp" => {
                let a: f64 = iw.iter().map(|&i| self.p[i].bp_a).sum();
                let b: f64 = iw.iter().map(|&i| self.p[i].bp_b).sum();
                if b > 0.0 { (a / b).max(1.0) } else { 1.0 }
            }
            "L" => 1.0 + sum(&iw, 6) / nw,
            _ => 1.0,
        };
        let r = sum(&iw, 7) / sum(&iw, 5).max(1e-9);
        let mu = if self.c.lam_mu == "one" { 1.0 } else { h * tdup };
        let lam = if r > 0.0 { self.c.lam_scale * q * mu / r } else { 0.0 };
        [h, tdup, q, r, lam]
    }

    /// Copy price (s per pulled token) of lambda_auto for a pull to w from source o.
    fn lam_auto_s(&self, t: f64, w: usize, o: Option<usize>) -> f64 {
        if self.lam_t0 < 0.0 || t - self.lam_t0 < self.c.lam_min_s {
            return self.c.cs_eft_dup_per_token_s;
        }
        if self.c.lam_auto == "pair" {
            self.lam_parts(Some(w), o)[4]
        } else {
            self.lam_parts(None, None)[4]
        }
    }

    /// Online estimator tick: per P EWMAs of the lambda_auto inputs; window samples.
    fn lam_tick(&mut self, t: f64) {
        let np = self.p.len();
        let bt = self.c.block_tokens as f64;
        let dt = (t - self.lam_t).max(1e-9);
        self.lam_t = t;
        let a = 1.0 - (-dt / self.c.lam_tau_s.max(1e-3)).exp();
        for w in 0..np {
            if self.c.ghost_frac > 0.0 {
                self.ghost_drain(w);
            }
            let pw = &mut self.p[w];
            let raw = [
                pw.ghost_hit_all as f64,
                (pw.radix.stats.kv_evicted_blocks[0] + pw.radix.stats.kv_evicted_blocks[1]) as f64 * bt,
                pw.radix.stats.host_evicted_blocks as f64 * bt,
                pw.busy_s,
                pw.tok_done as f64,
            ];
            let c0 = pw.radix.kv_class0_host_blocks() as f64 * bt;
            let s = [
                (raw[0] - pw.lam_last[0]) / dt,
                pw.ghost_map.len() as f64 * bt,
                c0,
                (raw[1] - pw.lam_last[1]) / dt,
                (raw[2] - pw.lam_last[2]) / dt,
                ((raw[3] - pw.lam_last[3]) / dt).clamp(0.0, 1.0),
                (pw.bq.len() + pw.wq.len() + pw.pf_set.len()) as f64,
                (raw[4] - pw.lam_last[4]) / dt,
            ];
            if pw.lam_ok {
                for k in 0..8 {
                    pw.lam_e[k] += a * (s[k] - pw.lam_e[k]);
                }
            } else {
                pw.lam_e = s;
                pw.lam_ok = true;
            }
            pw.lam_last = raw;
        }
        if self.lam_t0 < 0.0 {
            self.lam_t0 = t;
        }
        if t >= self.lo && t < self.hi {
            let g = self.lam_parts(None, None);
            let sm = |k: usize| self.p.iter().map(|p| p.lam_e[k]).sum::<f64>();
            let a_: f64 = self.p.iter().map(|p| p.bp_a).sum();
            let b_: f64 = self.p.iter().map(|p| p.bp_b).sum();
            let qbp = if b_ > 0.0 { (a_ / b_).max(1.0) } else { 1.0 };
            self.lam_win.push([
                sm(0),
                sm(1),
                sm(2),
                sm(3),
                sm(4),
                sm(5) / np as f64,
                sm(6) / np as f64,
                sm(7),
                qbp,
                g[4],
                g[0],
                g[1],
                g[2],
                g[3],
            ]);
        }
        if t < self.c.duration_s {
            let dt = self.c.lam_tick_s.max(0.1);
            self.at(t + dt, Ev::LamTick);
        }
    }

    fn iww_tick(&mut self, t: f64) {
        if t >= self.hi {
            return;
        }
        let np = self.p.len();
        let router = !self.lpb_q.is_empty();
        let runnable: Vec<bool> = (0..np)
            .map(|k| self.p[k].wq.iter().any(|&r| !self.reqs[r as usize].pull_pending) || self.p[k].chunked.is_some())
            .collect();
        self.iww[0] += 1.0;
        for w in 0..np {
            let idle = self.p[w].run_items.is_none();
            let other = router || (0..np).any(|k| k != w && runnable[k]);
            if idle && other {
                self.iww[1] += 1.0;
            }
            self.iww[2] += 1.0;
        }
        let ld: Vec<f64> = (0..np)
            .map(|w| {
                let bq: u64 = self.p[w]
                    .bq
                    .iter()
                    .map(|&r| {
                        let rq = &self.reqs[r as usize];
                        rq.cs_cost.unwrap_or(rq.unc_est)
                    })
                    .sum();
                (self.true_backlog(w) + bq) as f64
            })
            .collect();
        let mean = ld.iter().sum::<f64>() / np as f64;
        if mean > 0.0 {
            self.lmm.push(ld.iter().cloned().fold(0.0, f64::max) / mean);
        }
        self.at(t + 0.1, Ev::IwwTick);
    }

    /// S5: backlog as sampled cs_state_age_s ago (0.1 s samples) + own charges routed since.
    fn cs_pend_stale(&self, t: f64) -> Vec<u64> {
        let due = t - self.c.cs_state_age_s;
        let Some(&(ts, ref v)) = self.st_hist.iter().rev().find(|x| x.0 <= due).or(self.st_hist.front()) else {
            return self.cs_pend_now();
        };
        let mut out = v.clone();
        if self.c.cs_state_own_fix {
            for (w, q) in self.st_routed.iter().enumerate() {
                out[w] += q.iter().filter(|x| x.0 > ts).map(|x| x.1).sum::<u64>();
            }
        }
        out
    }

    /// S5 FPI family on the EFT path. Returns (target, pull allowed for a move). Before the pilot ends
    /// the router is owner-always and counts arrivals per P.
    #[allow(clippy::too_many_arguments)]
    fn fpi_route(
        &mut self,
        t: f64,
        r: u32,
        own: usize,
        l: u64,
        local: &[u64],
        wait: &[f64],
        own_s: &[f64],
        pull_s: &[f64],
        pulled: &[u64],
        rec_s: &[f64],
    ) -> (usize, bool) {
        let np = local.len();
        let Some(lr) = self.fpi_lr.clone() else {
            if !self.reqs[r as usize].warm {
                self.fpi_n[own] += 1;
            }
            return (own, false);
        };
        self.fpi_counts[3] += 1;
        let cap = self.c.cs_eft_max_pull_tokens;
        let a = |w: usize, start: f64, x: f64| fpi_score(start, x, wait[w], lr[w].0, lr[w].1);
        // Best variant on each P: owner local; other P: pull (within the cap) or recompute.
        let mut best: Vec<(f64, bool)> = vec![(f64::INFINITY, false); np];
        for w in 0..np {
            if w == own {
                best[w] = (a(w, wait[w], own_s[w]), false);
                continue;
            }
            let rec = a(w, wait[w], rec_s[w]);
            best[w] = (rec, false);
            if pulled[w] > 0 && (cap == 0 || pulled[w] <= cap) {
                let pv = a(w, wait[w].max(pull_s[w]), own_s[w]);
                if pv < rec {
                    best[w] = (pv, true);
                }
            }
        }
        let fpi_pick = |best: &[(f64, bool)]| -> usize {
            let mut k = own;
            for w in 0..np {
                if w != own && best[w].0 < best[k].0 {
                    k = w;
                }
            }
            k
        };
        let mode = self.c.cs_fpi.clone();
        let (k, use_pull) = if mode == "fpi" {
            let k = fpi_pick(&best);
            (k, best[k].1)
        } else {
            // As-if-local FPI cost c(k) on every P (the chain's prefix L = owner hit already on k) and
            // the move cost M(k) = min(pull wait beyond the backlog + restore, recompute extra).
            let lmax = local[own];
            let ch = self.c.chunk_tokens;
            let mut c = vec![0.0; np];
            let mut m = vec![0.0; np];
            for w in 0..np {
                let b = self.p[w].beta;
                let x_loc = eft_secs(&b, ch, l - lmax, lmax, 0, 0);
                c[w] = a(w, wait[w], x_loc);
                if w == own {
                    continue;
                }
                let mv = lmax.saturating_sub(local[w]);
                let rec_extra = (rec_s[w] - x_loc).max(0.0);
                let mut mm = rec_extra;
                if mv > 0 && (cap == 0 || mv <= cap) {
                    let tp = self.c.cs_eft_pull_fixed_s + self.c.cs_eft_pull_s_per_100k * mv as f64 / 1e5;
                    let pull_cost = (tp - wait[w]).max(0.0) + b[4] * mv as f64 / 1e5;
                    mm = mm.min(pull_cost);
                }
                m[w] = if self.c.cs_fpi_move_cost == "fpi" { (best[w].0 - c[w]).max(0.0) } else { mm };
            }
            let key = self.chain_key(r);
            // Start: w = 0 at the holder, M at the other P (OR-MODELS S2).
            let w0: Vec<f64> = (0..np).map(|kk| if kk == own { 0.0 } else { m[kk] }).collect();
            let u0 = self.nrng.random();
            let mut st = self.wf.remove(&key).unwrap_or_else(|| WfState { w: w0.clone(), h: own, r: 0.0, u: u0, last: t });
            if st.h != own && !(st.h < np && local[st.h] + self.c.pull_min_tokens > lmax) {
                // The holder lost the prefix: reset, as a cold session on the current owner.
                self.fpi_counts[2] += 1;
                st = WfState { w: w0, h: own, r: 0.0, u: st.u, last: t };
            }
            let h = st.h;
            let k = if mode == "wfa" {
                let prev = st.w.clone();
                for kk in 0..np {
                    let via = (0..np).filter(|&y| y != kk).map(|y| prev[y] + m[kk]).fold(f64::INFINITY, f64::min);
                    st.w[kk] = c[kk] + prev[kk].min(via);
                }
                let mn = st.w.iter().cloned().fold(f64::INFINITY, f64::min);
                for x in st.w.iter_mut() {
                    *x -= mn;
                }
                let mut k = h;
                for kk in 0..np {
                    if kk != h && st.w[kk] + m[kk] < st.w[k] {
                        k = kk;
                    }
                }
                k
            } else {
                let dt = (t - st.last).max(0.0);
                let alt = (0..np).filter(|&kk| kk != h).min_by(|&x, &y| c[x].total_cmp(&c[y]).then(x.cmp(&y)));
                let mut k = h;
                if let Some(j) = alt {
                    st.r = st.r * (-dt / self.c.cs_ski_tau_s.max(1e-9)).exp() + (c[h] - c[j]).max(0.0);
                    let theta = if mode == "skir" {
                        m[j] * (1.0 + (std::f64::consts::E - 1.0) * st.u).ln()
                    } else {
                        m[j]
                    };
                    if st.r >= theta && c[j] < c[h] {
                        k = j;
                        st.r = 0.0;
                        st.u = self.nrng.random();
                    }
                }
                k
            };
            st.last = t;
            st.h = k;
            self.wf.insert(key, st);
            (k, best[k].1)
        };
        if k != own {
            self.fpi_counts[if use_pull { 0 } else { 1 }] += 1;
        }
        (k, use_pull)
    }

    /// S4: the idle timer of (session, version) expired; END unless a newer request was routed.
    fn kv_idle(&mut self, t: f64, sid: u32, ver: u32) {
        let Some(s) = self.kv_sess.get(&sid) else { return };
        if s.ended || s.ver != ver {
            return;
        }
        self.kv_idle_ends.push(t);
        self.kv_end_session(t, sid);
    }

    /// Whether `r` is its agent's last generate (trajectory END follows its completion).
    fn kv_is_last(&mut self, r: u32) -> bool {
        let (task, a, idx) = {
            let rq = &self.reqs[r as usize];
            (rq.task, rq.agent_state, rq.idx)
        };
        let row = self.tasks[task as usize].row;
        let agent = self.agents[a as usize].agent;
        let wl = self.wl;
        let last = *self.kv_lastgen.entry((row, agent)).or_insert_with(|| {
            wl.tasks[row as usize].agents[agent as usize]
                .events
                .iter()
                .rposition(|e| matches!(e, Event::Generate { .. }))
                .map(|x| x as u32)
                .unwrap_or(u32::MAX)
        });
        idx == last
    }

    /// Trajectory END (#25 KvSessions::end): clear pending retirements, send 'end' to every P the
    /// session used.
    fn kv_end_session(&mut self, t: f64, sid: u32) {
        let Some(s) = self.kv_sess.get_mut(&sid) else { return };
        if s.ended {
            return;
        }
        s.ended = true;
        let dropped = s.retire.len() as u64;
        s.retire.clear();
        let ws = s.workers.clone();
        self.kv_drop_end += dropped;
        let dt = self.c.kvret_ctl_s;
        for (wp, wv) in ws {
            self.at(t + dt, Ev::KvCtl(sid, wp, wv, 1));
            self.kv_end_t.push(t);
        }
    }

    /// Cross-P duplicate census: (blocks resident on >= 2 P, sum over blocks of copies - 1, union).
    fn dup_census(&self) -> (u64, u64, u64) {
        let mut ev: Vec<(u64, i32)> = Vec::new();
        let mut v: Vec<(u64, u64)> = Vec::new();
        for w in &self.p {
            v.clear();
            w.radix.resident_intervals(&mut v);
            v.sort_unstable();
            let mut cur: Option<(u64, u64)> = None;
            for &(a, b) in &v {
                cur = match cur {
                    Some((cs, ce)) if a <= ce => Some((cs, ce.max(b))),
                    Some((cs, ce)) => {
                        ev.push((cs, 1));
                        ev.push((ce, -1));
                        Some((a, b))
                    }
                    None => Some((a, b)),
                };
            }
            if let Some((cs, ce)) = cur {
                ev.push((cs, 1));
                ev.push((ce, -1));
            }
        }
        ev.sort_unstable();
        let (mut cov, mut last, mut dup, mut extra, mut uni) = (0i32, 0u64, 0u64, 0u64, 0u64);
        for (x, d) in ev {
            let len = x - last;
            if cov >= 2 {
                dup += len;
                extra += (cov - 1) as u64 * len;
            }
            if cov >= 1 {
                uni += len;
            }
            cov += d;
            last = x;
        }
        (dup, extra, uni)
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
        let actual = |r: &Req| (r.dev + r.host + r.stor + r.drd) as f64;
        let sa = sum(&|r| actual(r));
        let si = sum(&|r| r.ideal.min(r.pt) as f64);
        rep.hit_sum_a_over_i = if si > 0.0 { sa / si } else { 0.0 };
        rep.hit_actual_over_prompt = sa / pt;
        rep.ideal_over_prompt = si / pt;
        rep.dev_share = sum(&|r| r.dev as f64) / pt;
        rep.host_share = sum(&|r| r.host as f64) / pt;
        rep.pulled_share = sum(&|r| r.stor as f64) / pt;
        let n = cmp.len().max(1) as f64;
        if self.c.d_cache {
            let comp = sum(&|r| (r.pt as f64 - actual(r)).max(0.0));
            rep.d_prefix_share = sum(&|r| r.d_prefix as f64) / pt;
            rep.dsrv_tokens = sum(&|r| (r.ideal.min(r.pt).min(r.d_prefix) as f64 - actual(r)).max(0.0));
            rep.dsrv_over_ideal = if si > 0.0 { rep.dsrv_tokens / si } else { 0.0 };
            rep.dsrv_over_computed = if comp > 0.0 { rep.dsrv_tokens / comp } else { 0.0 };
            rep.d_prefix_gt_a_rows = cmp.iter().filter(|r| r.d_prefix as f64 > actual(r)).count() as f64 / n;
            rep.d_local_reuse = sum(&|r| (r.d_prefix as f64 - actual(r)).max(0.0)) / pt;
            rep.dreads_scored = cmp.iter().filter(|r| r.dread > 0).count() as u64;
            rep.dread_tokens_scored = cmp.iter().map(|r| r.dread).sum();
            rep.dread_hit_tokens_scored = cmp.iter().map(|r| r.drd).sum();
            rep.dreads_all = self.dreads_all;
            rep.dread_tokens_all = self.dread_tok_all;
        }
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
        if self.c.d_mode != "d0" {
            let mut h: Vec<f64> = cmp
                .iter()
                .filter_map(|r| match (r.t_alloc, r.ld_held_since) {
                    (Some(a), Some(b)) => Some((a - b).max(0.0)),
                    _ => None,
                })
                .collect();
            if !h.is_empty() {
                let m = h.iter().sum::<f64>() / h.len() as f64;
                let share = h.iter().filter(|&&x| x > 0.1).count() as f64 / h.len() as f64;
                h.sort_by(|a, b| a.total_cmp(b));
                let q = |v: &Vec<f64>, x: f64| {
                    let k = ((v.len() - 1) as f64 * x).round() as usize;
                    v[k.min(v.len() - 1)]
                };
                rep.ld_held = [m, q(&h, 0.50), q(&h, 0.95), q(&h, 0.99)];
                rep.ld_held_share = share;
            }
            let avg = |f: &dyn Fn(&Req) -> Option<f64>| {
                let v: Vec<f64> = cmp.iter().filter_map(|r| f(r)).collect();
                if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
            };
            rep.ld_waits = [
                avg(&|r| r.ld_held_since.map(|a| (a - r.d_recv).max(0.0))),
                avg(&|r| match (r.ld_bind_t, r.ld_bindable_t) {
                    (Some(b), Some(a)) => Some((b - a).max(0.0)),
                    _ => None,
                }),
                avg(&|r| match (r.ld_bound_t, r.ld_bind_t) {
                    (Some(b), Some(a)) => Some((b - a).max(0.0)),
                    _ => None,
                }),
            ];
        }
        rep.wall_s = wall_s;
        {
            // ---- session retention, duplicate census, moves per session (output only)
            let inw = |x: f64| x >= lo && x < hi;
            rep.kv_releases_sent = [self.kv_rel.len() as u64, self.kv_rel.iter().filter(|x| inw(x.0)).count() as u64];
            for kind in 0..2u8 {
                let mut ages: Vec<f64> = self.kv_rel.iter().filter(|x| inw(x.0) && x.2 == kind).map(|x| x.1).collect();
                if !ages.is_empty() {
                    let m = ages.iter().sum::<f64>() / ages.len() as f64;
                    let k = ages.len() as f64;
                    let v = [k, m, percentile_in_place(&mut ages, 50.0), percentile_in_place(&mut ages, 90.0)];
                    if kind == 0 {
                        rep.kv_ready_wait = v;
                    } else {
                        rep.kv_transfer_wait = v;
                    }
                }
            }
            rep.kv_transfer_releases = [
                self.kv_rel.iter().filter(|x| x.2 == 1).count() as u64,
                self.kv_rel.iter().filter(|x| x.2 == 1 && inw(x.0)).count() as u64,
            ];
            rep.cs_cap_kept = self.cs_cap_kept;
            rep.kv_end_ops = [self.kv_end_t.len() as u64, self.kv_end_t.iter().filter(|&&x| inw(x)).count() as u64];
            rep.kv_retire_cancelled = self.kv_cancel;
            rep.kv_retire_dropped_end = self.kv_drop_end;
            rep.kv_polls = self.kv_polls;
            for w in &self.p {
                if let Some(kv) = &w.radix.kv {
                    let s = &kv.stats;
                    let o = [s.release_applied, s.release_fence, s.release_stale, s.release_ended];
                    for k in 0..4 {
                        rep.kv_release_outcome[k] += o[k];
                    }
                    let tr = [s.admits, s.fenced, s.retries, s.ends, s.wakes, s.delete_moves, s.late_marks];
                    for k in 0..7 {
                        rep.kv_tracker[k] += tr[k];
                    }
                }
                for k in 0..4 {
                    rep.kv_evicted_tokens_all[k] += w.radix.stats.kv_evicted_blocks[k] * bt;
                }
                rep.kv_devonly_deleted_tokens += w.radix.stats.kv_devonly_deleted_blocks * bt;
                for e in w.radix.evictions.iter().filter(|e| inw(e.t) && e.class < 2) {
                    rep.kv_evicted_tokens_win[e.class as usize * 2 + e.dup as usize] += e.blocks as u64 * bt;
                }
                rep.kv_audit.push(w.radix.kv_audit());
            }
            let dc: Vec<_> = self.dupc.iter().filter(|x| inw(x.0)).collect();
            if !dc.is_empty() {
                let k = dc.len() as f64;
                let b = bt as f64;
                rep.dup_tokens_mean = dc.iter().map(|x| x.1 as f64 * b).sum::<f64>() / k;
                rep.dup_extra_tokens_mean = dc.iter().map(|x| x.2 as f64 * b).sum::<f64>() / k;
                rep.dup_token_s = rep.dup_tokens_mean * self.c.score_s;
                rep.resident_union_tokens_mean = dc.iter().map(|x| x.3 as f64 * b).sum::<f64>() / k;
                let pool = (self.c.n_p as f64 * self.c.p_host_tokens as f64).max(1.0);
                rep.kv_class0_host_frac = dc.iter().map(|x| x.4 as f64 * b / pool).sum::<f64>() / k;
            }
            let mut by: FxHashMap<u32, Vec<(f64, u16)>> = FxHashMap::default();
            for r in &self.reqs {
                by.entry(r.agent_state).or_default().push((r.admit, r.p));
            }
            let mut mv: Vec<f64> = Vec::new();
            let (mut moves, mut nreq) = (0u64, 0u64);
            for v in by.values_mut() {
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut m = 0u64;
                let mut n = 0u64;
                for i in 0..v.len() {
                    if !inw(v[i].0) {
                        continue;
                    }
                    n += 1;
                    if i > 0 && v[i].1 != v[i - 1].1 {
                        m += 1;
                    }
                }
                if n > 0 {
                    mv.push(m as f64);
                    moves += m;
                    nreq += n;
                }
            }
            if !mv.is_empty() {
                let k = mv.len() as f64;
                let mean = mv.iter().sum::<f64>() / k;
                let share = mv.iter().filter(|&&x| x >= 1.0).count() as f64 / k;
                rep.sess_moves = [k, mean, share, percentile_in_place(&mut mv, 90.0), 100.0 * moves as f64 / nreq.max(1) as f64];
            }
            rep.kv_idle = [
                self.kv_idle_ends.len() as u64,
                self.kv_idle_ends.iter().filter(|&&x| inw(x)).count() as u64,
                self.kv_idle_new,
            ];
            let mut g = self.agent_gaps.clone();
            if !g.is_empty() {
                let k = g.len() as f64;
                let mean = g.iter().sum::<f64>() / k;
                let mut v = [k, mean, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                for (j, q) in [10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0].iter().enumerate() {
                    v[2 + j] = percentile_in_place(&mut g, *q);
                }
                rep.agent_gap_s = v;
                for (j, th) in [30.0, 60.0, 120.0, 300.0, 600.0].iter().enumerate() {
                    rep.agent_gap_gt[j] = g.iter().filter(|&&x| x > *th).count() as f64 / k;
                }
            }
            if let Some(lr) = &self.fpi_lr {
                rep.fpi_lambda_rho = lr.iter().map(|&(a, b)| [a, b]).collect();
            }
            rep.fpi_counts = self.fpi_counts;
            {
                // ---- S6 diagnostics
                let mut rc = vec![[0.0f64; 3]; 4];
                for q in self.reqs.iter().filter(|q| !q.warm && q.start >= lo && q.start < hi) {
                    let c = &mut rc[q.rclass as usize];
                    c[0] += 1.0;
                    if q.lost > 0 {
                        c[1] += 1.0;
                        c[2] += q.lost as f64;
                    }
                }
                rep.route_class = rc;
                let ns = cmp.len().max(1) as f64;
                rep.lost_share = cmp.iter().filter(|q| q.lost > 0).count() as f64 / ns;
                let capm = |q: &Req| (q.pt.saturating_sub(1) / bt) * bt;
                let comp = |q: &Req| -> [f64; 8] {
                    let cm = capm(q);
                    let ideal = q.ideal.min(cm) as f64;
                    let b = q.route_best.min(cm) as f64;
                    let tg = q.route_tgt.min(cm) as f64;
                    let s = q.stor as f64;
                    let a = (q.dev + q.host + q.stor + q.drd) as f64;
                    [ideal, b, tg, s, a, ideal - b, b - tg - s, tg + s - a]
                };
                let mut d = [0.0f64; 8];
                for q in &cmp {
                    let v = comp(q);
                    for k in 0..8 {
                        d[k] += v[k] / ns;
                    }
                }
                rep.deficit = d;
                // Chain history: previous P and P used before, per agent state (non-warm requests).
                let mut hist: FxHashMap<u32, Vec<(f64, u32, u16)>> = FxHashMap::default();
                for q in self.reqs.iter().filter(|q| !q.warm) {
                    hist.entry(q.agent_state).or_default().push((q.admit, q.idx, q.p));
                }
                let mut grp: FxHashMap<(u32, u32), u8> = FxHashMap::default();
                for (a, v) in hist.iter_mut() {
                    v.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
                    let mut used: Vec<u16> = Vec::new();
                    for (i, &(_, idx, p)) in v.iter().enumerate() {
                        let g = if i == 0 {
                            0
                        } else if v[i - 1].2 == p {
                            1
                        } else if used.contains(&p) {
                            3
                        } else {
                            2
                        };
                        grp.insert((*a, idx), g);
                        if !used.contains(&p) {
                            used.push(p);
                        }
                    }
                }
                let mut dm = vec![[0.0f64; 6]; 4];
                for q in &cmp {
                    let g = *grp.get(&(q.agent_state, q.idx)).unwrap_or(&0) as usize;
                    let v = comp(q);
                    dm[g][0] += 1.0;
                    dm[g][1] += v[5];
                    dm[g][2] += v[6];
                    dm[g][3] += v[7];
                    dm[g][4] += v[0];
                    dm[g][5] += v[4];
                }
                for row in dm.iter_mut() {
                    let k = row[0].max(1.0);
                    for j in 1..6 {
                        row[j] /= k;
                    }
                    row[0] /= ns;
                }
                rep.deficit_by_move = dm;
                let mut em = vec![[0.0f64; 4]; 4];
                for q in &cmp {
                    let g = *grp.get(&(q.agent_state, q.idx)).unwrap_or(&0) as usize;
                    let k = match g {
                        0 => 0,
                        1 => 1,
                        _ if q.pull_src.is_some() => 3,
                        _ => 2,
                    };
                    em[k][0] += 1.0;
                    em[k][1] += q.done.unwrap() - q.start;
                    em[k][2] += q.first.unwrap() - q.start;
                    em[k][3] += q.pulled as f64;
                }
                for row in em.iter_mut() {
                    let k = row[0].max(1.0);
                    for j in 1..4 {
                        row[j] /= k;
                    }
                    row[0] /= ns;
                }
                rep.e2e_by_move = em;
                rep.pred_gate = self.pred_stats;
                rep.moved_share = cmp.iter().filter(|q| q.owner != q.p).count() as f64 / ns;
                if self.c.pred_metric {
                    let rows: Vec<(f64, f64, f64, bool)> = cmp
                        .iter()
                        .filter(|q| q.pred_t > 0.0 && q.t_pd.is_some_and(|x| x > q.pred_t))
                        .map(|q| (q.t_pd.unwrap() - q.pred_t, q.pred_c, q.pred_c0, q.pred_idle))
                        .collect();
                    let qs = |v: &mut Vec<f64>, p: f64| -> f64 {
                        if v.is_empty() {
                            return f64::NAN;
                        }
                        v.sort_by(|a, b| a.total_cmp(b));
                        let x = p * (v.len() - 1) as f64;
                        let i = x as usize;
                        let j = (i + 1).min(v.len() - 1);
                        v[i] + (v[j] - v[i]) * (x - i as f64)
                    };
                    for k in 0..2 {
                        let pc = |x: &(f64, f64, f64, bool)| if k == 0 { x.1 } else { x.2 };
                        let ape: Vec<(f64, bool)> = rows.iter().map(|x| ((pc(x) - x.0).abs() / x.0, x.3)).collect();
                        let mean = |f: &dyn Fn(&(f64, bool)) -> bool| -> f64 {
                            let v: Vec<f64> = ape.iter().filter(|x| f(x)).map(|x| x.0).collect();
                            if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
                        };
                        let mut all: Vec<f64> = ape.iter().map(|x| x.0).collect();
                        let bias = rows.iter().map(|x| pc(x) - x.0).sum::<f64>() / (rows.len().max(1) as f64);
                        rep.pred_err.push([
                            rows.len() as f64,
                            mean(&|_| true),
                            mean(&|x| x.1),
                            mean(&|x| !x.1),
                            qs(&mut all, 0.5),
                            qs(&mut all, 0.9),
                            bias,
                            rows.iter().filter(|x| x.3).count() as f64,
                        ]);
                    }
                    for k in 0..2 {
                        for idle in [true, false] {
                            let mut v: Vec<f64> = rows
                                .iter()
                                .filter(|x| x.3 == idle)
                                .map(|x| {
                                    let p = if k == 0 { x.1 } else { x.2 };
                                    (x.0 - p).abs() / p.max(1e-9)
                                })
                                .collect();
                            rep.pred_kq.push([qs(&mut v, 0.5), qs(&mut v, 0.6827), qs(&mut v, 0.8), qs(&mut v, 0.9)]);
                        }
                    }
                }
                if self.c.lpb {
                    let mut hw: Vec<f64> = cmp.iter().map(|q| q.hold_s).collect();
                    let m = hw.iter().sum::<f64>() / (hw.len().max(1) as f64);
                    let (p50, p95) = if hw.is_empty() {
                        (0.0, 0.0)
                    } else {
                        (percentile_in_place(&mut hw, 50.0), percentile_in_place(&mut hw, 95.0))
                    };
                    let c = |k: u8| cmp.iter().filter(|q| q.lpb_cls == k).count() as f64 / ns;
                    rep.lpb_stats = [m, p50, p95, c(1), c(2), c(3), self.lpb_rx[0], self.lpb_rx[2]];
                    rep.lpb_counts = self.lpb_counts;
                    rep.lpb_bypassed = self.lpb_bypassed;
                }
                rep.hol_shadow = self.hol_sh;
                if self.c.iww_metric {
                    let mut v = self.lmm.clone();
                    let (a, b) = if v.is_empty() {
                        (0.0, 0.0)
                    } else {
                        (percentile_in_place(&mut v, 50.0), percentile_in_place(&mut v, 90.0))
                    };
                    rep.iww = [if self.iww[2] > 0.0 { self.iww[1] / self.iww[2] } else { 0.0 }, self.iww[0], a, b];
                }
                if self.c.cs_hold_passes > 0.0 {
                    let h: Vec<&&Req> = cmp.iter().filter(|q| q.hold_s > 0.0).collect();
                    rep.held_share = h.len() as f64 / ns;
                    rep.stolen_share = cmp.iter().filter(|q| q.stolen).count() as f64 / ns;
                    rep.hold_s_mean = h.iter().map(|q| q.hold_s).sum::<f64>() / (h.len().max(1) as f64);
                    rep.hold_end.push(self.held.len() as f64);
                    for w in 0..self.p.len() {
                        rep.hold_end.push(self.backlog_secs(w, true));
                        rep.hold_end.push(self.p[w].transit_s);
                        rep.hold_end.push(self.p[w].wq.len() as f64);
                        rep.hold_end.push(self.p[w].bq.len() as f64);
                        rep.hold_end.push(self.p[w].pf_set.len() as f64);
                    }
                }
                if self.c.idle_overlap_metric {
                    // Union of idle intervals of the P other than p, per p (sorted, merged).
                    let mut uni: Vec<Vec<(f64, f64)>> = Vec::new();
                    for p in 0..self.p.len() {
                        let mut v: Vec<(f64, f64)> =
                            self.idle_log.iter().filter(|x| x.2 as usize != p).map(|x| (x.0, x.1)).collect();
                        v.sort_by(|a, b| a.0.total_cmp(&b.0));
                        let mut m: Vec<(f64, f64)> = Vec::new();
                        for (a, b) in v {
                            if let Some(l) = m.last_mut() {
                                if a <= l.1 {
                                    l.1 = l.1.max(b);
                                    continue;
                                }
                            }
                            m.push((a, b));
                        }
                        uni.push(m);
                    }
                    let (mut tot, mut ov) = (0.0, 0.0);
                    for q in &cmp {
                        let (Some(a), Some(b)) = (q.p_ready, q.t_fwd) else { continue };
                        if b <= a {
                            continue;
                        }
                        tot += b - a;
                        let u = &uni[q.p as usize];
                        let mut i = u.partition_point(|x| x.1 <= a);
                        while i < u.len() && u[i].0 < b {
                            ov += u[i].1.min(b) - u[i].0.max(a);
                            i += 1;
                        }
                    }
                    rep.idle_overlap_share = if tot > 0.0 { ov / tot } else { 0.0 };
                }
                rep.lp_mult_mean = if self.lp_stats.1 > 0 { self.lp_stats.0 / self.lp_stats.1 as f64 } else { 0.0 };
                rep.gate_fire_share =
                    if self.gate_stats.0 > 0 { self.gate_stats.1 as f64 / self.gate_stats.0 as f64 } else { 0.0 };
                let mut hc = vec![[0.0f64; 3]; 3];
                for w in &self.p {
                    for e in w.radix.evictions.iter().filter(|e| inw(e.t)) {
                        hc[(e.cause as usize).min(2)][(e.class as usize).min(2)] += e.blocks as f64 * bt as f64;
                    }
                }
                let span = (hi - lo).max(1e-9);
                for row in hc.iter_mut() {
                    for x in row.iter_mut() {
                        *x /= span;
                    }
                }
                rep.host_ev_cause_tok_s = hc;
                let dur_s = self.c.duration_s.max(1e-9);
                for w in &self.p {
                    for k in 0..3 {
                        rep.dev_ev_cause_tok_s[k] += w.radix.stats.dev_ev_cause[k] as f64 * bt as f64 / dur_s;
                        rep.devonly_ev_cause_tok_s[k] += w.radix.stats.devonly_ev_cause[k] as f64 * bt as f64 / dur_s;
                    }
                }
            }
            {
                let nb = (self.c.duration_s / 300.0).ceil().max(1.0) as usize;
                let mut s = vec![[0.0f64; 5]; nb];
                let mut cnt = vec![0.0f64; nb];
                let mut pt_ = vec![0.0f64; nb];
                let mut ev_win = 0u64;
                for w in &self.p {
                    for e in &w.radix.evictions {
                        let b = ((e.t / 300.0) as usize).min(nb - 1);
                        s[b][1] += e.blocks as f64 * bt as f64;
                        if inw(e.t) {
                            ev_win += e.blocks as u64 * bt;
                        }
                    }
                }
                for q in &self.reqs {
                    if q.warm || q.cancel {
                        continue;
                    }
                    let Some(d) = q.done else { continue };
                    let b = ((q.start / 300.0) as usize).min(nb - 1);
                    s[b][2] += d - q.start;
                    cnt[b] += 1.0;
                    s[b][3] += (q.dev + q.host + q.stor + q.drd) as f64;
                    pt_[b] += q.pt as f64;
                    let bd = ((d / 300.0) as usize).min(nb - 1);
                    s[bd][4] += 1.0 / 300.0;
                }
                for b in 0..nb {
                    s[b][0] = b as f64 * 300.0;
                    s[b][1] /= 300.0;
                    s[b][2] = if cnt[b] > 0.0 { s[b][2] / cnt[b] } else { f64::NAN };
                    s[b][3] = if pt_[b] > 0.0 { s[b][3] / pt_[b] } else { f64::NAN };
                }
                rep.series_300s = s;
                rep.host_evicted_tok_s_win = ev_win as f64 / (self.hi - self.lo).max(1e-9);
            }
            for w in &self.p {
                rep.pull_free_tokens[0] += w.radix.stats.pf_added_blocks * bt;
                rep.pull_free_tokens[1] += w.radix.stats.pf_removed_blocks * bt;
                rep.pull_free_tokens[2] += w.radix.pf_blocks * bt;
            }
        }
        if self.c.ghost_frac > 0.0 || self.c.lam_auto != "off" {
            let sc = (hi - lo).max(1e-9);
            let gh: f64 = self.p.iter().map(|p| p.ghost_hit_win as f64).sum::<f64>() / sc;
            let ns = self.lam_win.len().max(1) as f64;
            let m = |k: usize| self.lam_win.iter().map(|x| x[k]).sum::<f64>() / ns;
            rep.ghost = [gh, m(1), if m(1) > 0.0 { gh / m(1) } else { 0.0 }];
            let busy = m(5);
            let bw: [f64; 3] = self.p.iter().fold([0.0; 3], |acc, p| [acc[0] + p.bp_win[0], acc[1] + p.bp_win[1], acc[2] + p.bp_win[2]]);
            rep.q_bp_win = [
                if bw[1] > 0.0 { bw[0] / bw[1] } else { 0.0 },
                bw[2],
                if bw[2] > 0.0 { bw[1] / bw[2] } else { 0.0 },
            ];
            rep.lam_est = [
                if m(1) > 0.0 { m(0) / m(1) } else { 0.0 },
                m(11),
                self.c.concurrency as f64 / self.p.len() as f64 * busy,
                m(8),
                1.0 + m(6),
                m(13),
                m(9),
                m(2),
                m(3),
                m(4),
            ];
            let mut lv: Vec<f64> = self.lam_win.iter().map(|x| x[9]).collect();
            if !lv.is_empty() {
                rep.lam_auto_pct = [percentile_in_place(&mut lv, 10.0), percentile_in_place(&mut lv, 90.0)];
            }
        }
        rep.lam_used = [if self.lam_used.3 > 0 { self.lam_used.2 / self.lam_used.3 as f64 } else { 0.0 }, self.lam_used.3 as f64];
        let nf = self.fm_n[1].max(1) as f64;
        rep.fm = [self.fm_elig as f64, self.fm_n[0] as f64, self.fm_n[1] as f64, self.fm_tok_win as f64 / nf, self.fm_gain_win / nf];
        rep.inj = self.inj_n;
        rep.p_host_cap_end = self.p.iter().map(|p| p.radix.host_cap * self.c.block_tokens).collect();
        rep.cap_moved_tokens = self.cap_moved * self.c.block_tokens;
        rep.ghost_per_req = cmp.iter().map(|r| r.ghost as f64).sum::<f64>() / cmp.len().max(1) as f64;
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
                    actual: r.dev + r.host + r.stor + r.drd,
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
                    d_held: match (r.t_alloc, r.ld_held_since) {
                        (Some(a), Some(b)) => (a - b).max(0.0),
                        _ => -1.0,
                    },
                    d_prefix: r.d_prefix,
                    dread: r.drd,
                    pred_lab: match r.t_pd {
                        Some(d) if r.pred_t > 0.0 && d > r.pred_t => d - r.pred_t,
                        _ => -1.0,
                    },
                    pred_c: if r.pred_t > 0.0 { r.pred_c } else { -1.0 },
                    pred_c0: if r.pred_t > 0.0 { r.pred_c0 } else { -1.0 },
                    pred_b0: if r.pred_t > 0.0 { r.pred_b0 } else { -1.0 },
                    pred_s0: if r.pred_t > 0.0 { r.pred_s0 } else { -1.0 },
                    pred_idle: r.pred_idle,
                    recv_s: if r.pred_t > 0.0 { r.p_recv - r.pred_t } else { -1.0 },
                    ready_s: r.p_ready.map(|x| x - r.p_recv).unwrap_or(-1.0),
                    owner: r.owner,
                    forced: r.forced,
                    fm_gain: r.fm_gain,
                    fm_pulled: r.fm_pulled,
                    inj: r.inj,
                    ghost: r.ghost,
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
    fn table_draw_interpolates_quantiles() {
        let t = [0.0, 1.0, 3.0];
        assert_eq!(table_draw(&t, 0.0), 0.0);
        assert_eq!(table_draw(&t, 0.25), 0.5);
        assert_eq!(table_draw(&t, 0.75), 2.0);
        assert_eq!(table_draw(&t, 1.0), 3.0);
        assert_eq!(table_draw(&[], 0.3), 0.0);
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
    fn fpi_score_matches_the_worked_example() {
        // OR-MODELS section 2(a): lambda 1/s, rho 0.6; owner u 6 s, x 0.5 s; target u 1 s, pull 1.2 s.
        let owner = fpi_score(6.0, 0.5, 6.0, 1.0, 0.6);
        let pull = fpi_score(1.2, 0.5, 1.0, 1.0, 0.6);
        let rec = fpi_score(1.0, 4.0, 1.0, 1.0, 0.6);
        assert!((owner - 14.3125).abs() < 1e-9 && (pull - 3.2625).abs() < 1e-9 && (rec - 35.0).abs() < 1e-9);
        // lambda = 0 is greedy EFT.
        assert_eq!(fpi_score(1.0, 4.0, 1.0, 0.0, 0.6), 5.0);
    }

    #[test]
    fn s5_router_options_run_and_keep_counts_consistent() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        let base = SimConfig {
            cs_workload: "true".into(),
            p_dev_tokens: 131_072,
            p_host_tokens: 262_144,
            kv_retention: true,
            cs_margin_tokens: 0,
            ..kv_cfg("cosched")
        };
        let go = |c: SimConfig| {
            let mut s = Sim::new(c, &wl, &ct);
            s.run();
            for w in &s.p {
                w.radix.check().unwrap();
            }
            let (rep, _) = s.report(0.0);
            assert!(rep.n_scored > 100, "scored {}", rep.n_scored);
            for a in &rep.kv_audit {
                assert_eq!(a[..3], [0, 0, 0], "{:?}", rep.kv_audit);
            }
            // 300-s series cover the run (400 s -> 2 bins) and count every completion once.
            assert_eq!(rep.series_300s.len(), 2);
            let done: f64 = rep.series_300s.iter().map(|x| x[4] * 300.0).sum();
            let want = s.reqs.iter().filter(|q| !q.warm && !q.cancel && q.done.is_some()).count() as f64;
            assert!((done - want).abs() < 1e-6, "{done} vs {want}");
            assert!(rep.host_evicted_tok_s_win >= 0.0);
            rep
        };
        let r0 = go(base.clone());
        assert!(r0.pulls_all > 0, "co-schedule base never pulled");
        // Recompute moves: no pull at all.
        let rr = go(SimConfig { cs_move_mode: "recompute".into(), ..base.clone() });
        assert_eq!(rr.pulls_all, 0);
        // Noise and stale state run; zero settings are inert.
        let z = go(SimConfig { cs_noise_queue: 0.0, cs_noise_sigma: 0.0, cs_state_age_s: 0.0, ..base.clone() });
        assert_eq!(serde_json::to_string(&z).unwrap(), serde_json::to_string(&r0).unwrap());
        go(SimConfig { cs_noise_queue: 0.5, cs_noise_sigma: 0.25, ..base.clone() });
        go(SimConfig { cs_state_age_s: 1.0, ..base.clone() });
        go(SimConfig { cs_state_age_s: 2.0, cs_state_own_fix: false, ..base.clone() });
        // FPI family on the EFT path: the pilot measures lambda and rho, then moves are counted.
        for (mode, mc) in [("fpi", "seconds"), ("wfa", "seconds"), ("ski", "seconds"), ("skir", "seconds"), ("wfa", "fpi"), ("skir", "fpi")] {
            let rep = go(SimConfig {
                p_policy: "eft".into(),
                cs_eft_one_way: false,
                cs_eft_dup_rho: 0.0,
                cs_fpi: mode.into(),
                cs_fpi_move_cost: mc.into(),
                cs_fpi_pilot_s: 60.0,
                cs_eft_max_pull_tokens: 25_000,
                ..base.clone()
            });
            assert_eq!(rep.fpi_lambda_rho.len(), 3, "{mode}");
            assert!(rep.fpi_lambda_rho.iter().all(|x| x[0] > 0.0 && x[1] > 0.0 && x[1] <= 0.95), "{mode} {:?}", rep.fpi_lambda_rho);
            assert!(rep.fpi_counts[3] > 0, "{mode}");
        }
    }

    #[test]
    fn s6_trade_hybrids_and_diagnostics_are_consistent() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        let base = SimConfig {
            cs_workload: "true".into(),
            p_dev_tokens: 131_072,
            p_host_tokens: 262_144,
            kv_retention: true,
            pull_min_incremental_tokens: 512,
            ..kv_cfg("smgca")
        };
        let go = |c: SimConfig| {
            let mut s = Sim::new(c, &wl, &ct);
            s.run();
            for w in &s.p {
                w.radix.check().unwrap();
            }
            let (rep, _) = s.report(0.0);
            assert!(rep.n_scored > 50, "scored {}", rep.n_scored);
            // Deficit identity, class and eviction-cause totals.
            let d = rep.deficit;
            assert!((d[5] + d[6] + d[7] - (d[0] - d[4])).abs() < 1e-6, "{d:?}");
            let shares: f64 = rep.deficit_by_move.iter().map(|x| x[0]).sum();
            assert!((shares - 1.0).abs() < 1e-9, "{shares}");
            let want = s.reqs.iter().filter(|q| !q.warm && q.start >= s.lo && q.start < s.hi).count() as f64;
            let got: f64 = rep.route_class.iter().map(|x| x[0]).sum();
            assert_eq!(got, want);
            let ev: f64 = rep.host_ev_cause_tok_s.iter().flat_map(|x| x.iter()).sum();
            assert!((ev - rep.host_evicted_tok_s_win).abs() <= 1e-6 * ev.max(1.0), "{ev} vs {}", rep.host_evicted_tok_s_win);
            rep
        };
        let a = go(base.clone());
        // Hybrids and CA-U run; routing all classes by the P1 rule moves no request to a P with less cache
        // than its P1 no-pull candidates allow (no pull at all).
        go(SimConfig { smg_p1_classes: "cold,low".into(), ..base.clone() });
        let all = go(SimConfig { smg_p1_classes: "cold,low,multi,single".into(), ..base.clone() });
        assert_eq!(all.pulls_all, 0);
        go(SimConfig { smg_load: "uncached_tokens".into(), smg_abs_tokens: 65_536, ..base.clone() });
        assert_eq!(a.pulls_all, 0);
        // TRADE on the EFT path: lambda 0 pulls; a huge per-token charge stops every pull.
        let trade = |lam: f64| SimConfig {
            p_policy: "eft".into(),
            cs_eft_one_way: false,
            cs_eft_dup_rho: 0.0,
            cs_eft_dup_per_token_s: lam,
            ..base.clone()
        };
        let t0 = go(trade(0.0));
        assert!(t0.pulls_all > 0, "TRADE(0) never pulled");
        let t9 = go(trade(1.0));
        assert_eq!(t9.pulls_all, 0);
    }

    #[test]
    fn s7_pm_eft_options_run_and_keep_counts_consistent() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        let base = SimConfig {
            cs_workload: "true".into(),
            p_dev_tokens: 131_072,
            p_host_tokens: 262_144,
            kv_retention: true,
            kvret_mode: "transfer_end".into(),
            pull_min_incremental_tokens: 512,
            p_policy: "eft".into(),
            cs_eft_one_way: false,
            cs_eft_dup_rho: 0.0,
            cs_eft_wait_transit: true,
            ..kv_cfg("smgca")
        };
        let go = |c: SimConfig| {
            let mut s = Sim::new(c, &wl, &ct);
            s.run();
            for w in &s.p {
                w.radix.check().unwrap();
            }
            let (rep, _) = s.report(0.0);
            assert!(rep.n_scored > 50, "scored {}", rep.n_scored);
            let shares: f64 = rep.e2e_by_move.iter().map(|x| x[0]).sum();
            assert!((shares - 1.0).abs() < 1e-9, "{shares}");
            assert!(rep.e2e_by_move[3][0] == 0.0 || rep.pulls_all > 0);
            rep
        };
        let a = go(SimConfig {
            cs_eft_handoff_s: 0.32,
            cs_eft_margin_k: 0.2,
            cs_eft_owner_tie_c: true,
            cs_eft_noise_sigma: 0.25,
            cs_eft_noise_queue: 0.25,
            cs_eft_lp_ref: 1.0,
            cs_eft_lp_tau_s: 10.0,
            ..base.clone()
        });
        assert!(a.lp_mult_mean == 0.0 || a.lp_mult_mean >= 1.0, "{}", a.lp_mult_mean);
        go(SimConfig { cs_eft_state_age_s: 1.0, cs_eft_lp_ref: 1.0, cs_eft_lp_scope: "global".into(), ..base.clone() });
        let g = go(SimConfig { cs_eft_gate_big_tokens: 1, cs_eft_gate_theta_s: 0.0, cs_eft_owner_near_tokens: 512, ..base.clone() });
        assert!(g.gate_fire_share > 0.0 && g.gate_fire_share <= 1.0, "{}", g.gate_fire_share);
        // A gate that never fires keeps every request on its owner: no pulls.
        let n = go(SimConfig { cs_eft_gate_big_tokens: u64::MAX, ..base.clone() });
        assert_eq!(n.gate_fire_share, 0.0);
        assert_eq!(n.pulls_all, 0);
        // A huge noise-aware margin also stops every move with a pull.
        let m = go(SimConfig { cs_eft_margin_k: 1e9, cs_eft_margin_s: 1e9, ..base.clone() });
        assert_eq!(m.pulls_all, 0);
        // A copy price on recomputed duplicates too: with no pulls and a huge price nothing moves to a P
        // with less cache than the best.
        let dr = go(SimConfig {
            cs_eft_dup_on_recompute: true,
            cs_eft_dup_per_token_s: 1.0,
            pull_min_incremental_tokens: 1 << 50,
            ..base.clone()
        });
        assert_eq!(dr.pulls_all, 0);
        assert_eq!(dr.lost_share, 0.0);
        // S8 PRED-GATE: noise tables, gate counters and forecast scoring.
        let tab: Vec<f64> = (0..201).map(|i| -0.5 + i as f64 / 200.0).collect();
        let pg = |m: f64, g: f64| SimConfig {
            cs_eft_dup_on_recompute: true,
            cs_pred_noise: tab.clone(),
            cs_pred_gate: true,
            cs_pred_m: m,
            cs_pred_gmin_s: g,
            cs_pred_k_idle: 0.2,
            cs_pred_k_busy: 0.3,
            pred_metric: true,
            ..base.clone()
        };
        let g0 = go(pg(0.0, 0.0));
        assert!(g0.pred_gate[0] > 0 && g0.pred_gate[2] <= g0.pred_gate[1] && g0.pred_gate[1] <= g0.pred_gate[0]);
        assert_eq!(g0.pred_err.len(), 2);
        assert_eq!(g0.pred_kq.len(), 4);
        assert!(g0.pred_err[0][0] > 0.0 && g0.pred_err[0][1] > 0.0);
        // A huge minimum gain stops every move.
        let gb = go(pg(0.0, 1e9));
        assert_eq!(gb.pred_gate[2], 0);
        assert_eq!(gb.pulls_all, 0);
        // A larger margin never moves more requests than no margin (same noise stream per decision count
        // is not guaranteed, so only the bound check: moves <= cheaper-P decisions).
        let gm = go(pg(1.28, 0.5));
        assert!(gm.pred_gate[2] <= gm.pred_gate[1]);
        // B1 hold: runs with and without steals; every held request is dispatched in the end.
        let h = go(SimConfig { cs_hold_passes: 1.0, cs_eft_handoff_s: 0.32, cs_hold_margin_s: 0.0, idle_overlap_metric: true, ..base.clone() });
        assert!(h.held_share >= h.stolen_share && h.idle_overlap_share >= 0.0 && h.idle_overlap_share <= 1.0 + 1e-9);
        let h0 = go(SimConfig { cs_hold_passes: 1.0, cs_hold_steal: false, ..base.clone() });
        assert_eq!(h0.stolen_share, 0.0);
        // Slower think time lowers throughput.
        let s1 = go(base.clone());
        let s2 = go(SimConfig { wait_scale: 4.0, ..base.clone() });
        assert!(s2.rps <= s1.rps + 1e-9, "{} vs {}", s2.rps, s1.rps);
    }

    #[test]
    fn latep_dispatches_every_request_and_keeps_counts_consistent() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        let tab: Vec<f64> = (0..201).map(|i| -0.5 + i as f64 / 200.0).collect();
        let base = SimConfig {
            cs_workload: "true".into(),
            p_dev_tokens: 131_072,
            p_host_tokens: 262_144,
            kv_retention: true,
            kvret_mode: "transfer_end".into(),
            pull_min_incremental_tokens: 512,
            p_policy: "eft".into(),
            cs_eft_one_way: false,
            cs_eft_dup_rho: 0.0,
            cs_eft_wait_transit: true,
            cs_eft_dup_per_token_s: 6.8e-4,
            bnd_idle_k: vec![0.15, 0.3, 0.6],
            bnd_idle_b: vec![0.06, 0.18, 0.35],
            bnd_busy_k: vec![0.3, 0.7, 1.35, 2.8],
            bnd_busy_b: vec![0.5, 0.55, 0.86, 1.23],
            cs_pred_gmin_s: 0.1,
            iww_metric: true,
            lpb: true,
            ..kv_cfg("smgca")
        };
        let go = |c: SimConfig| {
            let on = c.lpb;
            let mut s = Sim::new(c, &wl, &ct);
            s.run();
            for w in &s.p {
                w.radix.check().unwrap();
            }
            let (rep, _) = s.report(0.0);
            assert!(rep.n_scored > 50, "scored {}", rep.n_scored);
            if !on {
                return rep;
            }
            let k = rep.lpb_counts;
            assert_eq!(k[0], k[1] + k[2] + k[3] + k[4], "{k:?}");
            assert!(rep.lpb_stats[0] >= 0.0 && rep.lpb_stats[2] >= rep.lpb_stats[1]);
            let sh = rep.lpb_stats[3] + rep.lpb_stats[4] + rep.lpb_stats[5];
            assert!((sh - 1.0).abs() < 1e-9, "{:?}", rep.lpb_stats);
            assert!(rep.iww[1] > 0.0 && rep.iww[0] >= 0.0 && rep.iww[0] <= 1.0, "{:?}", rep.iww);
            // Every request sent before the last 60 s left the router queue.
            assert!(s.lpb_q.iter().all(|&r| s.reqs[r as usize].start > s.c.duration_s - 60.0), "{}", s.lpb_q.len());
            rep
        };
        let f = go(base.clone());
        let h = go(SimConfig { lpb_order: "hrrn".into(), ..base.clone() });
        let n = go(SimConfig { cs_pred_noise: tab.clone(), lpb_moves: false, ..base.clone() });
        assert_eq!(n.lpb_counts[3] + n.lpb_counts[4], 0);
        assert!(f.rps > 0.0 && h.rps > 0.0);
        // Zero bound and zero receipt forecast still dispatch everything.
        go(SimConfig { lpb_bnd_scale: 0.0, lpb_rx_scale: 0.0, ..base.clone() });
        // PRED-GWQ: only free requests enter the router queue; bound ones take the TRADE path at arrival.
        {
            let c = SimConfig {
                lpb_order: "hrrn".into(),
                lpb_moves: false,
                lpb_hold_free: true,
                lpb_lead_extra_s: 0.6,
                ..base.clone()
            };
            let mut s = Sim::new(c, &wl, &ct);
            s.run();
            for w in &s.p {
                w.radix.check().unwrap();
            }
            let (rep, _) = s.report(0.0);
            assert!(rep.n_scored > 50, "scored {}", rep.n_scored);
            let k = rep.lpb_counts;
            assert_eq!(k[0], k[1] + k[2] + k[3] + k[4], "{k:?}");
            assert_eq!(k[3] + k[4], 0, "no moves");
            assert!(rep.lpb_bypassed + k[0] > 0);
            let sh = rep.lpb_stats[3] + rep.lpb_stats[4] + rep.lpb_stats[5];
            assert!(sh <= 1.0 + 1e-9, "{:?}", rep.lpb_stats);
            assert!(s.lpb_q.iter().all(|&r| s.reqs[r as usize].start > s.c.duration_s - 60.0), "{}", s.lpb_q.len());
        }
        // PRED-EFT+HOL with the r5 curve bound runs on the EFT path.
        let c = go(SimConfig {
            lpb: false,
            cs_hol_mu: 1.0,
            cs_pred_gate: true,
            cs_pred_curve: true,
            cs_pred_m: 1.0,
            cs_pred_noise: tab.clone(),
            ..base.clone()
        });
        assert!(c.pred_gate[2] <= c.pred_gate[1]);
        assert!(c.hol_shadow[0] > 0 && c.hol_shadow[1] <= c.hol_shadow[0] && c.hol_shadow[3] <= c.hol_shadow[2]);
        let c2 = go(SimConfig {
            lpb: false,
            cs_hol_mu: 1.0,
            cs_hol_form: "bp".into(),
            cs_pred_gate: true,
            cs_pred_curve: true,
            cs_pred_m: 1.0,
            ..base.clone()
        });
        assert!(c2.hol_shadow[0] > 0 && c2.hol_shadow[4] <= c2.hol_shadow[1]);
        assert_eq!(bound_curve(&[1.0, 2.0], &[0.5, 1.0], 0.5), 0.5);
        assert!((bound_curve(&[1.0, 2.0], &[0.5, 1.0], 1.5) - 0.75).abs() < 1e-12);
        assert!((bound_curve(&[1.0, 2.0], &[0.5, 1.0], 4.0) - 2.0).abs() < 1e-12);
        assert_eq!(bound_curve(&[], &[], 4.0), 0.0);
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

    fn ld_invariants(sim: &Sim) {
        for (k, d) in sim.d.iter().enumerate() {
            let want: u64 = sim.reqs.iter().filter(|r| r.pa_on && r.d as usize == k).map(|r| r.need).sum();
            assert_eq!(d.used, want, "D{k} used");
            let infl: u64 = sim.reqs.iter().filter(|r| r.d as usize == k).map(|r| r.ld_inflight).sum();
            assert_eq!(d.inflight_tok, infl, "D{k} in-flight tokens");
            let holders = sim.reqs.iter().filter(|r| r.pa_on && r.d as usize == k).count();
            assert!(d.used <= d.cap || holders == 1, "D{k} over capacity with {holders} holders");
        }
    }

    #[test]
    fn late_d_modes_complete_requests_and_keep_d_accounting() {
        let wl = synth_workload(12, 40);
        let ct = FxHashMap::default();
        for (mode, rule) in [("alloc", "po2"), ("choose", "po2"), ("choose", "d1")] {
            // Small D pools so that binds wait and claims matter.
            let c = SimConfig { d_mode: mode.into(), d_bind_rule: rule.into(), d_cap_tokens: 400_000, ..small_cfg() };
            let mut sim = Sim::new(c, &wl, &ct);
            sim.run();
            ld_invariants(&sim);
            let (rep, rows) = sim.report(0.0);
            assert!(rep.n_scored > 100, "{mode}/{rule}: scored {}", rep.n_scored);
            // Order of the late path: bindable -> bind -> ADMIT at D -> allocation -> bootstrap done
            // (waiting queue) -> first chunk -> KV at D.
            for r in sim.reqs.iter().filter(|r| r.t_kv.is_some()) {
                let seq = [
                    r.ld_bindable_t.unwrap(),
                    r.ld_bind_t.unwrap(),
                    r.ld_held_since.unwrap(),
                    r.t_alloc.unwrap(),
                    r.ld_bound_t.unwrap(),
                    r.p_ready.unwrap(),
                    r.t_fwd.unwrap(),
                    r.t_kv.unwrap(),
                ];
                assert!(seq.windows(2).all(|w| w[0] <= w[1] + 1e-9), "{mode}/{rule}: {seq:?}");
            }
            assert!(rows.iter().filter(|r| r.completed).all(|r| r.d_held >= 0.0));
            assert!(rep.ld_waits.iter().all(|&x| x >= 0.0));
            // Small pools: with the arrival D (alloc mode) some binds wait at D; the late choice may
            // avoid every wait.
            if mode == "alloc" {
                assert!(rep.ld_held_share > 0.0, "{mode}/{rule}: no held bind");
            }
        }
        // d0 keeps the early path: no bind, allocation before the waiting queue.
        let mut sim = Sim::new(SimConfig { d_cap_tokens: 400_000, ..small_cfg() }, &wl, &ct);
        sim.run();
        assert!(sim.reqs.iter().all(|r| r.ld_bind_t.is_none()));
        for r in sim.reqs.iter().filter(|r| r.p_ready.is_some()) {
            assert!(r.t_alloc.unwrap() <= r.p_ready.unwrap());
        }
        ld_invariants(&sim);
    }

    #[test]
    fn late_d_claim_reserves_freed_memory_for_the_oldest_held_entry() {
        let wl = synth_workload(1, 1);
        let ct = FxHashMap::default();
        let mut sim = Sim::new(SimConfig { d_mode: "alloc".into(), ..small_cfg() }, &wl, &ct);
        for need in [50u64, 30, 30] {
            sim.reqs.push(Req { need, d: 0, ..Default::default() });
        }
        sim.d[0].cap = 100;
        sim.d[0].used = 60;
        let t = 10.0;
        // Entry 0 held for 1 s (> 20 passes x 19.46 ms): it is the claim; it does not fit (free 40).
        sim.d[0].held.push_back((0, (t - 1.0f64).to_bits()));
        sim.d[0].held.push_back((1, t.to_bits()));
        sim.d[0].held.push_back((2, t.to_bits()));
        sim.ld_try_held(t, 0);
        // free 40 - claim 50 < 30: nobody passes the claim.
        assert_eq!(sim.d[0].held.len(), 3);
        // Without the claim (young head) a smaller entry passes.
        sim.d[0].held[0].1 = t.to_bits();
        sim.ld_try_held(t, 0);
        assert_eq!(sim.d[0].held.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(sim.d[0].used, 90);
        // Freed memory goes to the claim first, then to the next entry that fits.
        sim.d[0].used = 20;
        sim.d[0].held[0].1 = (t - 1.0f64).to_bits();
        sim.ld_try_held(t, 0);
        assert!(sim.d[0].held.is_empty());
        assert_eq!(sim.d[0].used, 100);
        assert!(sim.reqs[0].t_alloc.is_some() && sim.reqs[2].t_alloc.is_some());
    }

    #[test]
    fn late_d_binder_keeps_a_short_lead_and_binds_in_arrival_order() {
        let wl = synth_workload(24, 40);
        let ct = FxHashMap::default();
        // One slow P, so that rooms queue in the bootstrap queue.
        let c = SimConfig {
            d_mode: "alloc".into(),
            n_p: 1,
            n_d: 3,
            concurrency: 24,
            p_beta: vec![[0.08, 0.6, 0.0, 0.05, 0.05]],
            ld_max_outstanding_per_d: 1000,
            ..small_cfg()
        };
        let mut sim = Sim::new(c, &wl, &ct);
        sim.run();
        ld_invariants(&sim);
        // Binds follow arrival order on the P (qp_seq): no room binds while an earlier, bindable
        // room is still unbound.
        let mut v: Vec<(f64, u64, f64)> = sim
            .reqs
            .iter()
            .filter(|r| r.ld_bind_t.is_some())
            .map(|r| (r.ld_bindable_t.unwrap(), r.qp_seq, r.ld_bind_t.unwrap()))
            .collect();
        v.sort_by_key(|x| x.1);
        let mut out_of_order = 0;
        for w in v.windows(2) {
            if w[1].2 + 1e-9 < w[0].2 && w[0].0 <= w[1].2 {
                out_of_order += 1;
            }
        }
        assert_eq!(out_of_order, 0);
        // Some rooms waited in the bootstrap queue (deferred) because the lead window was full.
        let (rep, _) = sim.report(0.0);
        assert!(rep.ld_waits[1] > 0.0, "no deferral: {:?}", rep.ld_waits);
    }

    #[test]
    fn d1_bind_rule_picks_the_fewest_inflight_tokens_among_eligible() {
        let wl = synth_workload(1, 1);
        let ct = FxHashMap::default();
        let mut sim = Sim::new(SimConfig { d_mode: "choose".into(), d_bind_rule: "d1".into(), ..small_cfg() }, &wl, &ct);
        sim.d[0].inflight_tok = 500;
        sim.d[1].inflight_tok = 100;
        sim.d[2].inflight_tok = 100;
        assert_eq!(sim.ld_choose(&[0, 1, 2]), 1);
        sim.d[1].nreq = 3;
        assert_eq!(sim.ld_choose(&[0, 1, 2]), 2, "tie on tokens -> fewer charged requests");
        assert_eq!(sim.ld_choose(&[0, 1]), 1, "only eligible D are considered");
        // po2 draws two distinct eligible D and takes the lower polled load.
        sim.c.d_bind_rule = "po2".into();
        sim.d[0].load_snap = 10;
        sim.d[1].load_snap = 5;
        for _ in 0..20 {
            assert_eq!(sim.ld_choose(&[0, 1]), 1);
        }
        // Rule 4: a D with 4 outstanding binds from this P is not eligible.
        sim.p[0].ld_out[1] = 4;
        assert_eq!(sim.ld_eligible(0, 0), vec![0, 2]);
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

    /// S2 test config: a P cache that keeps one landed prefix but loses most prompts between turns,
    /// a large D cache; one D uses round robin (power of two needs two D).
    fn s2_cfg(n_d: usize, mode: &str) -> SimConfig {
        SimConfig {
            n_d,
            d_policy: if n_d == 1 { "rr".into() } else { "po2".into() },
            p_dev_tokens: 65_536,
            p_host_tokens: 131_072,
            d_cache: true,
            s2_read: mode.into(),
            ..small_cfg()
        }
    }

    #[test]
    fn d_cache_matches_at_prealloc_and_releases_every_lock() {
        let wl = synth_workload(6, 30);
        let ct = FxHashMap::default();
        // One D: every later turn finds its earlier prompt on the D.
        let c = SimConfig { n_d: 1, d_policy: "rr".into(), d_cache: true, ..small_cfg() };
        let mut sim = Sim::new(c, &wl, &ct);
        sim.run();
        let done: Vec<&Req> = sim.reqs.iter().filter(|r| r.done.is_some() && !r.cancel).collect();
        assert!(done.len() > 50);
        assert!(done.iter().all(|r| r.d_lock.is_none()), "a D prealloc lock was not released");
        let later = done.iter().filter(|r| r.ideal > 0).count();
        let covered = done.iter().filter(|r| r.ideal > 0 && r.d_prefix + 64 >= r.ideal.min(r.pt - 64)).count();
        assert!(covered * 10 >= later * 9, "D prefix covers {covered} of {later} later turns");
        let (rep, rows) = sim.report(0.0);
        assert!(rep.d_prefix_share > 0.5, "D prefix share {}", rep.d_prefix_share);
        assert!(rows.iter().any(|r| r.d_prefix > 0));
        // Off: no D prefix, no D-side numbers.
        let (rep0, _) = run(SimConfig { n_d: 1, d_policy: "rr".into(), ..small_cfg() }, &wl, &ct);
        assert_eq!(rep0.d_prefix_share, 0.0);
        assert_eq!(rep0.dsrv_tokens, 0.0);
    }

    #[test]
    fn s2b_reads_the_serving_d_copy_when_the_p_lost_the_prefix() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        let (base, _) = run(s2_cfg(1, "off"), &wl, &ct);
        let (rd, rows) = run(s2_cfg(1, "b"), &wl, &ct);
        // Base: the P lost part of the prefixes and the D held them (static bound > 0); no read.
        assert!(base.dsrv_over_ideal > 0.05, "Dsrv {} hit {}", base.dsrv_over_ideal, base.hit_sum_a_over_i);
        assert_eq!(base.dreads_all, 0);
        // S2b: reads from the serving D recover part of it.
        assert!(rd.dreads_scored > 0 && rd.dread_hit_tokens_scored > 0);
        assert!(rd.hit_sum_a_over_i > base.hit_sum_a_over_i + 0.02, "hit {} vs {}", rd.hit_sum_a_over_i, base.hit_sum_a_over_i);
        assert!(rd.uncached_per_req < base.uncached_per_req);
        assert!(rows.iter().filter(|r| r.completed).any(|r| r.dread > 0 && r.actual >= r.dread));
        // "a" reads only after a hinted pull; CA gives no hints: no read.
        let (ra, _) = run(s2_cfg(1, "a"), &wl, &ct);
        assert_eq!(ra.dreads_all, 0);
    }

    #[test]
    fn s2c_reads_from_the_previous_turn_d() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        // Three D chosen by power of two: the serving D often did not serve the previous turn.
        let mut sim = Sim::new(s2_cfg(3, "c"), &wl, &ct);
        sim.run();
        let reads: Vec<&Req> = sim.reqs.iter().filter(|r| r.dread > 0).collect();
        assert!(!reads.is_empty());
        assert!(reads.iter().any(|r| r.dread_src != Some(r.d)), "no read from another D");
        assert!(sim.reqs.iter().all(|r| r.dread_pending == false || r.done.is_none()));
        let (rc, _) = sim.report(0.0);
        assert!(rc.dread_hit_tokens_scored > 0);
    }

    fn kv_cfg(policy: &str) -> SimConfig {
        SimConfig { p_policy: policy.into(), kv_retention: true, split_on_match: true, ..small_cfg() }
    }

    #[test]
    fn kv_retention_off_is_inert() {
        let wl = synth_workload(12, 40);
        let ct = FxHashMap::default();
        let base = SimConfig {
            p_policy: "eft".into(),
            cs_workload: "true".into(),
            pull_start: "queue".into(),
            repull_on_admission: true,
            queued_prefix_protect: true,
            demote_source: true,
            split_on_match: true,
            ..small_cfg()
        };
        let go = |c: SimConfig| {
            let mut s = Sim::new(c, &wl, &ct);
            s.run();
            let (r, rows) = s.report(0.0);
            (serde_json::to_string(&r).unwrap(), rows.len())
        };
        let a = go(base.clone());
        let b = go(SimConfig {
            kvret_mode: "end_only".into(),
            kvret_poll_s: 0.37,
            kvret_ctl_s: 0.5,
            kvret_dup_reclaim: false,
            kvret_end: false,
            ..base.clone()
        });
        assert_eq!(a, b, "retention knobs must be inert while kv_retention is off");
        let mut s = Sim::new(base, &wl, &ct);
        s.run();
        assert!(s.p.iter().all(|w| w.radix.kv.is_none() && !w.radix.kvret));
        assert!(s.kv_log.is_empty() && s.kv_sess.is_empty() && s.kv_rel.is_empty());
    }

    #[test]
    fn kv_retention_releases_only_after_ready_and_ends_every_finished_session() {
        let wl = synth_workload(12, 40);
        let ct = FxHashMap::default();
        // Round robin moves almost every request to another P.
        let mut sim = Sim::new(kv_cfg("rr"), &wl, &ct);
        sim.run();
        let rel: Vec<_> = sim.kv_log.iter().filter(|x| x.4 == 0).copied().collect();
        let ends: Vec<_> = sim.kv_log.iter().filter(|x| x.4 == 1).copied().collect();
        assert!(!ends.is_empty());
        // Every completed move (the moving request finished its prefill well before the end) is
        // followed by a release of the old P's version.
        let mut last: FxHashMap<u32, u16> = FxHashMap::default();
        let mut moves_done = 0usize;
        let mut order: Vec<&Req> = sim.reqs.iter().collect();
        order.sort_by(|a, b| a.start.total_cmp(&b.start));
        for q in order {
            if let Some(&pp) = last.get(&q.agent_state) {
                if pp != q.p && q.t_pd.is_some_and(|x| x >= 0.0 && x + 1.0 < sim.c.duration_s) {
                    moves_done += 1;
                }
            }
            last.insert(q.agent_state, q.p);
        }
        assert!(moves_done >= 10, "moves {moves_done}");
        assert!(rel.len() >= moves_done, "releases {} < completed moves {moves_done}", rel.len());
        let ctl = sim.c.kvret_ctl_s;
        for &(t, sid, p, ver, _) in &rel {
            // READY before the release: a newer version of the session finished its prefill (the
            // write-through of the whole input) on another P before the release was sent.
            let ok = sim.reqs.iter().any(|q| {
                q.agent_state == sid
                    && q.kv_ver > ver
                    && q.p != p
                    && q.t_pd.is_some_and(|x| x >= 0.0 && x <= t - ctl + 1e-9)
            });
            assert!(ok, "release of session {sid} v{ver} on P{p} at {t} before READY");
        }
        // END reaches every P the session used, and that P then reports the session ended.
        for &(_, sid, p, ver, _) in &ends {
            assert_eq!(sim.p[p as usize].radix.kv_status(sid, ver).0, super::super::cache::KV_ENDED);
        }
        let ended: FxHashSet<u32> = ends.iter().map(|x| x.1).collect();
        for &sid in &ended {
            let used: FxHashSet<u16> = sim.reqs.iter().filter(|q| q.agent_state == sid).map(|q| q.p).collect();
            let got: FxHashSet<u16> = ends.iter().filter(|x| x.1 == sid).map(|x| x.2).collect();
            assert_eq!(used, got, "session {sid}");
        }
        let (rep, _) = sim.report(0.0);
        assert!(rep.kv_releases_sent[0] as usize >= rel.len());
        assert!(rep.kv_ready_wait[0] > 0.0 && rep.kv_ready_wait[1] > 0.0, "{:?}", rep.kv_ready_wait);
        for a in &rep.kv_audit {
            assert_eq!(a[..3], [0, 0, 0], "{:?}", rep.kv_audit);
        }
        for a in &rep.per_p_audit {
            assert_eq!(a[..3], [0, 0, 0], "{:?}", rep.per_p_audit);
        }
        // (Identical synthetic tasks run in lockstep, so the global round robin returns most sessions to
        // the same P: only some sessions move.)
        assert!(rep.sess_moves[0] > 0.0 && rep.sess_moves[2] > 0.0, "{:?}", rep.sess_moves);
        // Sticky never moves: no release, END only.
        let mut st = Sim::new(kv_cfg("sticky"), &wl, &ct);
        st.run();
        assert!(!st.kv_log.is_empty() && st.kv_log.iter().all(|x| x.4 == 1));
        // end_only: moves send no release; END still goes out.
        let mut eo = Sim::new(SimConfig { kvret_mode: "end_only".into(), ..kv_cfg("rr") }, &wl, &ct);
        eo.run();
        assert!(!eo.kv_log.is_empty() && eo.kv_log.iter().all(|x| x.4 == 1));
        // recency_only: no tags, no control traffic, every range stays class 1.
        let mut ro = Sim::new(SimConfig { kvret_mode: "recency_only".into(), ..kv_cfg("rr") }, &wl, &ct);
        ro.run();
        assert!(ro.kv_log.is_empty() && ro.kv_sess.is_empty());
        assert!(ro.p.iter().all(|w| w.radix.kvret && w.radix.kv_class0_host_blocks() == 0));
        let (rr, _) = ro.report(0.0);
        assert_eq!(rr.kv_evicted_tokens_all[0] + rr.kv_evicted_tokens_all[1], 0);
    }

    #[test]
    fn kv_transfer_end_releases_the_pull_source_at_the_landing() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        let c = SimConfig {
            cs_workload: "true".into(),
            pull_start: "queue".into(),
            repull_on_admission: true,
            queued_prefix_protect: true,
            p_dev_tokens: 131_072,
            p_host_tokens: 262_144,
            kvret_mode: "transfer_end".into(),
            ..kv_cfg("eft")
        };
        let mut sim = Sim::new(c, &wl, &ct);
        sim.run();
        let tr: Vec<_> = sim.kv_log.iter().filter(|x| x.4 == 2).copied().collect();
        assert!(!tr.is_empty(), "no transfer release");
        for &(t, sid, p, ver, _) in &tr {
            // The release happens at the landing of a newer version's pull from that P onto another P.
            // (A re-pull at admission can overwrite pull_end, so test the issue-to-admission window.)
            let ok = sim.reqs.iter().any(|q| {
                q.agent_state == sid
                    && q.kv_ver > ver
                    && q.p != p
                    && q.pull_src == Some(p)
                    && q.pull_first_issue_t.is_some_and(|x| x <= t)
                    && q.t_fwd.map_or(true, |f| t <= f + 1e-9)
            });
            assert!(ok, "transfer release of session {sid} v{ver} on P{p} at {t} without a landing");
        }
        let (rep, _) = sim.report(0.0);
        assert_eq!(rep.kv_transfer_releases[0] as usize, tr.len());
        for a in &rep.kv_audit {
            assert_eq!(a[..3], [0, 0, 0], "{:?}", rep.kv_audit);
        }
        for a in &rep.per_p_audit {
            assert_eq!(a[..3], [0, 0, 0], "{:?}", rep.per_p_audit);
        }
        // ready_end on the same config sends no transfer release.
        let mut re = Sim::new(SimConfig { kvret_mode: "ready_end".into(), ..sim.c.clone() }, &wl, &ct);
        re.run();
        assert!(re.kv_log.iter().all(|x| x.4 != 2));
    }

    #[test]
    fn cosched_pull_cap_keeps_large_pulls_on_the_owner() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        let base = SimConfig {
            p_policy: "cosched".into(),
            cs_workload: "true".into(),
            cs_margin_tokens: 0,
            pull_start: "queue".into(),
            split_on_match: true,
            ..small_cfg()
        };
        let cap = 16_384u64;
        let mut sim = Sim::new(SimConfig { cs_max_pull_tokens: cap, ..base.clone() }, &wl, &ct);
        sim.run();
        for q in sim.reqs.iter().filter(|q| q.p != q.owner) {
            assert!(q.pull_hint <= cap, "detour with pull {} > cap", q.pull_hint);
        }
        let (rc, _) = sim.report(0.0);
        let (r0, _) = run(base, &wl, &ct);
        assert!(rc.cs_cap_kept > 0 && r0.cs_cap_kept == 0);
    }

    #[test]
    fn pull_free_capacity_raises_the_capacity_only_by_resident_pulled_pages() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        for kv in [false, true] {
            let c = SimConfig {
                cs_workload: "true".into(),
                pull_start: "p_recv".into(),
                queued_prefix_protect: true,
                p_dev_tokens: 131_072,
                p_host_tokens: 262_144,
                pull_free_capacity: true,
                kv_retention: kv,
                ..kv_cfg("eft")
            };
            let mut sim = Sim::new(c, &wl, &ct);
            sim.run();
            for w in &sim.p {
                w.radix.check().unwrap();
                assert_eq!(w.radix.host_cap, 262_144 / 64 + w.radix.pf_blocks);
            }
            let (rep, _) = sim.report(0.0);
            assert!(rep.pulls_all > 0, "no pull");
            assert!(rep.pull_free_tokens[0] > 0, "{:?}", rep.pull_free_tokens);
            assert_eq!(rep.pull_free_tokens[0] - rep.pull_free_tokens[1], rep.pull_free_tokens[2]);
        }
    }

    #[test]
    fn kv_idle_end_ends_quiet_sessions_and_a_returning_agent_gets_a_new_identity() {
        let wl = synth_workload(12, 40);
        let ct = FxHashMap::default();
        // Waits are 3 s: a 2 s idle rule ends every session between two turns.
        let mut sim = Sim::new(SimConfig { kvret_end: false, kvret_idle_s: 2.0, ..kv_cfg("rr") }, &wl, &ct);
        sim.run();
        let (rep, _) = sim.report(0.0);
        assert!(rep.kv_idle[0] > 100 && rep.kv_idle[2] > 100, "{:?}", rep.kv_idle);
        assert!(rep.kv_idle[2] <= rep.kv_idle[0]);
        assert!(sim.kv_sess.iter().any(|(&s, k)| s >= 0x8000_0000 && k.ended));
        // A new identity starts at version 1 and holds one request per turn here.
        assert!(sim.kv_sess.iter().filter(|&(&s, _)| s >= 0x8000_0000).all(|(_, k)| k.ver == 1));
        for w in &sim.p {
            w.radix.check().unwrap();
        }
        for a in &rep.kv_audit {
            assert_eq!(a[..3], [0, 0, 0], "{:?}", rep.kv_audit);
        }
        assert!(rep.agent_gap_s[0] > 100.0 && rep.agent_gap_s[4] >= 2.99, "{:?}", rep.agent_gap_s);
        assert_eq!(rep.agent_gap_gt[0], 0.0);
        // A rule longer than every gap ends only finished trajectories and makes no new identity.
        let mut sim = Sim::new(SimConfig { kvret_end: false, kvret_idle_s: 30.0, ..kv_cfg("rr") }, &wl, &ct);
        sim.run();
        let (rep2, _) = sim.report(0.0);
        assert_eq!(rep2.kv_idle[2], 0, "{:?}", rep2.kv_idle);
        assert!(rep2.kv_idle[0] > 0);
        assert!(sim.kv_sess.keys().all(|&s| s < 0x8000_0000));
        // Idle gaps are measured in every arm (also without retention) and equal across the two rules.
        let mut sim = Sim::new(small_cfg(), &wl, &ct);
        sim.run();
        let (rep3, _) = sim.report(0.0);
        assert!(rep3.agent_gap_s[0] > 100.0 && rep3.kv_idle == [0, 0, 0]);
    }

    #[test]
    fn kv_retention_with_pulls_and_eviction_pressure_keeps_counts_consistent() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        let c = SimConfig {
            cs_workload: "true".into(),
            pull_start: "queue".into(),
            repull_on_admission: true,
            queued_prefix_protect: true,
            p_dev_tokens: 131_072,
            p_host_tokens: 262_144,
            ..kv_cfg("eft")
        };
        let mut sim = Sim::new(c, &wl, &ct);
        sim.run();
        for w in &sim.p {
            w.radix.check().unwrap();
        }
        let (rep, _) = sim.report(0.0);
        assert!(rep.n_scored > 100, "scored {}", rep.n_scored);
        for a in &rep.kv_audit {
            assert_eq!(a[..3], [0, 0, 0], "{:?}", rep.kv_audit);
        }
        for a in &rep.per_p_audit {
            assert_eq!(a[..3], [0, 0, 0], "{:?}", rep.per_p_audit);
        }
        assert!(rep.pulls_all > 0, "no pull");
        assert!(rep.guard_grants.iter().sum::<u64>() > 0, "guard never locked");
        assert!(rep.kv_evicted_tokens_all[0] + rep.kv_evicted_tokens_all[1] > 0, "{:?}", rep.kv_evicted_tokens_all);
        assert!(rep.kv_end_ops[0] > 0);
    }
    #[test]
    fn lambda_probes_and_auto_price_run_and_keep_counts_consistent() {
        let wl = synth_workload(12, 30);
        let ct = FxHashMap::default();
        let base = SimConfig {
            cs_workload: "true".into(),
            p_dev_tokens: 131_072,
            p_host_tokens: 262_144,
            kv_retention: true,
            kvret_mode: "transfer_end".into(),
            pull_min_incremental_tokens: 512,
            p_policy: "eft".into(),
            cs_eft_one_way: false,
            cs_eft_dup_rho: 0.0,
            cs_eft_wait_transit: true,
            cs_eft_dup_per_token_s: 6.8e-4,
            ..kv_cfg("smgca")
        };
        let go = |c: SimConfig| {
            let mut s = Sim::new(c, &wl, &ct);
            s.run();
            for w in &s.p {
                w.radix.check().unwrap();
                assert!(w.ghost_map.len() as f64 <= (s.c.ghost_frac * w.radix.host_cap as f64).max(1.0) + 1.0);
            }
            let (rep, rows) = s.report(0.0);
            assert!(rep.n_scored > 50, "scored {}", rep.n_scored);
            (rep, rows)
        };
        let (b, _) = go(base.clone());
        assert_eq!(b.ghost, [0.0; 3]);
        assert_eq!(b.fm[1], 0.0);
        let g = SimConfig { ghost_frac: 0.05, lam_tick_s: 5.0, lam_tau_s: 60.0, lam_min_s: 30.0, ..base.clone() };
        let (gr, _) = go(g.clone());
        // Ghost tracking alone does not change routing.
        assert_eq!(gr.rps, b.rps);
        assert_eq!(gr.e2e_mean, b.e2e_mean);
        assert!(gr.ghost.iter().all(|x| x.is_finite() && *x >= 0.0), "{:?}", gr.ghost);
        assert!(gr.lam_est.iter().all(|x| x.is_finite() && *x >= 0.0), "{:?}", gr.lam_est);
        assert!(gr.lam_est[5] > 0.0, "R {:?}", gr.lam_est);
        assert!(gr.q_bp_win[0] >= 1.0 || gr.q_bp_win[1] == 0.0, "{:?}", gr.q_bp_win);
        for (mode, q, mu) in [("global", "sat", "ghost"), ("pair", "bp", "ghost"), ("global", "L", "one"), ("pair", "one", "ghost"), ("global", "bpsat", "one")] {
            let (a, _) = go(SimConfig { lam_auto: mode.into(), lam_q: q.into(), lam_mu: mu.into(), record_rows: true, ..g.clone() });
            assert!(a.lam_used[0].is_finite() && a.lam_used[0] >= 0.0, "{:?}", a.lam_used);
            assert!(a.rps > 0.0);
        }
        let (f, rows) = go(SimConfig { fm_rate: 0.5, fm_min_tokens: 0, record_rows: true, ..g.clone() });
        assert!(f.fm[1] <= f.fm[0] && f.fm[2] <= f.fm[1], "{:?}", f.fm);
        let nf = rows.iter().filter(|r| r.forced).count() as f64;
        assert!(nf <= f.fm[2] + 1e-9, "{nf} {:?}", f.fm);
        assert!(rows.iter().filter(|r| r.forced).all(|r| r.fm_gain > 0.0 && r.fm_pulled > 0));
        let (i, rows) = go(SimConfig { inj_rate: 0.2, inj_s: 0.5, record_rows: true, ..g.clone() });
        assert!(i.inj[0] > 0 && i.inj[1] <= i.inj[0], "{:?}", i.inj);
        assert!(rows.iter().any(|r| r.inj));
    }
}
