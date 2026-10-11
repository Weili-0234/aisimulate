// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HiCache + P2P-pull P/D topology runtime (experimental, local branch).
//!
//! This is a separate topology runtime inside the AISimulate crate. It models a SGLang
//! P/D deployment with HiCache (GPU radix + host pool, write-through), peer-to-peer KV
//! pulls between prefill workers (source leases, demote-source), SMG `cache_aware`,
//! co-schedule and `power_of_two` routing, and the trie benchmark's closed-loop client.
//! It does NOT use AISimulate's token-level SGLang engine or `disagg.rs`; it has its own
//! single deterministic event loop and a calibrated per-pass timing model. See
//! `design-sim.txt` in the run bundle for the alignment table (exact / approximated /
//! not modelled) against the flex-pd engine and router code.

pub mod cache;
pub mod sim;
pub mod trie;

pub use cache::{DemoteScope, TwoTierRadix};
pub use sim::{SimConfig, SimReport, run};
pub use trie::{PyRandom, Workload};
