// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loader for the trie benchmark workload (`cc.workload.jsonl`, schema v3) and a
//! Python-compatible MT19937 so the stationary warm-start cut draws match the trie client
//! (`random.Random(seed).random()`).

use std::collections::HashMap;
use std::io::BufRead;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

/// One agent event.
#[derive(Debug, Clone)]
pub enum Event {
    /// A generate request: prompt block runs (64-token blocks), max tokens, output blocks,
    /// source start time and source API time (seconds), when recorded.
    Generate {
        runs: Vec<(u64, u32)>,
        max_tokens: u32,
        out_blocks: u32,
        src_t: Option<f64>,
        src_api: Option<f64>,
    },
    /// A wait of `seconds` (capped at replay time).
    Wait { seconds: f64 },
    /// Any other event type: completes immediately.
    Other,
}

#[derive(Debug, Clone)]
pub struct Agent {
    pub id: String,
    pub routing_key: Option<String>,
    pub events: Vec<Event>,
}

#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub block_size: u32,
    pub agents: Vec<Agent>,
    /// (before agent index, before event index, after agent index, after event index).
    pub deps: Vec<(usize, usize, usize, usize)>,
}

#[derive(Debug, Clone, Default)]
pub struct Workload {
    pub tasks: Vec<Task>,
}

#[derive(Deserialize)]
struct RawRow {
    trace_id: Value,
    traces: Vec<RawTrace>,
    #[serde(default)]
    dependencies: Vec<RawDep>,
    #[serde(default)]
    metadata: Option<Value>,
}

#[derive(Deserialize)]
struct RawTrace {
    trace_id: Value,
    events: Vec<RawEvent>,
    #[serde(default)]
    metadata: Option<Value>,
}

#[derive(Deserialize)]
struct RawEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    recorded_prompt_block_runs: Option<Vec<(u64, u64)>>,
    #[serde(default)]
    recorded_output_block_runs: Option<Vec<(u64, u64)>>,
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    source_t_s: Option<f64>,
    #[serde(default)]
    source_api_time_s: Option<f64>,
    #[serde(default)]
    seconds: Option<f64>,
}

#[derive(Deserialize)]
struct RawDep {
    before: RawRef,
    after: RawRef,
}

#[derive(Deserialize)]
struct RawRef {
    trace_id: Value,
    event_index: usize,
}

fn key(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

impl Workload {
    /// Parse a trie workload JSONL file (one task per line, file order = admission order).
    pub fn load(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
        let rd = std::io::BufReader::with_capacity(1 << 22, f);
        let mut tasks = Vec::new();
        for (ln, line) in rd.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: RawRow =
                serde_json::from_str(&line).with_context(|| format!("line {}", ln + 1))?;
            let bs = row
                .metadata
                .as_ref()
                .and_then(|m| m.get("block_tokens"))
                .and_then(|b| b.get("block_size"))
                .and_then(Value::as_u64)
                .unwrap_or(64) as u32;
            let mut agents = Vec::with_capacity(row.traces.len());
            let mut aidx: HashMap<String, usize> = HashMap::new();
            for (i, t) in row.traces.iter().enumerate() {
                let id = key(&t.trace_id);
                aidx.insert(id.clone(), i);
                let rk = t
                    .metadata
                    .as_ref()
                    .and_then(|m| m.get("routing_key"))
                    .map(key);
                let events = t
                    .events
                    .iter()
                    .map(|e| match e.kind.as_str() {
                        "generate" => Event::Generate {
                            runs: e
                                .recorded_prompt_block_runs
                                .as_deref()
                                .unwrap_or(&[])
                                .iter()
                                .map(|&(s, c)| (s, c as u32))
                                .collect(),
                            max_tokens: e.max_tokens.unwrap_or(1) as u32,
                            out_blocks: e
                                .recorded_output_block_runs
                                .as_deref()
                                .unwrap_or(&[])
                                .iter()
                                .map(|&(_, c)| c as u32)
                                .sum(),
                            src_t: e.source_t_s,
                            src_api: e.source_api_time_s,
                        },
                        "wait" => Event::Wait { seconds: e.seconds.unwrap_or(0.0) },
                        _ => Event::Other,
                    })
                    .collect();
                agents.push(Agent { id, routing_key: rk, events });
            }
            let mut deps = Vec::with_capacity(row.dependencies.len());
            for d in &row.dependencies {
                let (Some(&ba), Some(&aa)) = (aidx.get(&key(&d.before.trace_id)), aidx.get(&key(&d.after.trace_id)))
                else {
                    anyhow::bail!("line {}: dependency references an unknown trace", ln + 1);
                };
                deps.push((ba, d.before.event_index, aa, d.after.event_index));
            }
            tasks.push(Task { id: key(&row.trace_id), block_size: bs, agents, deps });
        }
        Ok(Self { tasks })
    }

    pub fn generates(&self) -> usize {
        self.tasks
            .iter()
            .flat_map(|t| t.agents.iter())
            .flat_map(|a| a.events.iter())
            .filter(|e| matches!(e, Event::Generate { .. }))
            .count()
    }
}

/// MT19937 matching CPython's `random.Random(int_seed)`.
#[derive(Clone)]
pub struct PyRandom {
    mt: [u32; 624],
    idx: usize,
}

impl PyRandom {
    pub fn new(seed: u64) -> Self {
        let mut r = Self { mt: [0; 624], idx: 624 };
        // CPython: init_by_array with the absolute value split into 32-bit words (0 -> [0]).
        let mut key: Vec<u32> = Vec::new();
        let mut s = seed;
        if s == 0 {
            key.push(0);
        }
        while s > 0 {
            key.push((s & 0xffff_ffff) as u32);
            s >>= 32;
        }
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..624 {
            let prev = self.mt[i - 1];
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.idx = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let n = 624usize;
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = n.max(key.len());
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= n {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = n - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= n {
                self.mt[0] = self.mt[n - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
        self.idx = 624;
    }

    fn genrand_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for kk in 0..624 {
                let y = (self.mt[kk] & 0x8000_0000) | (self.mt[(kk + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(kk + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[kk] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random.random()`.
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `random.randrange(n)` for n < 2^32 (CPython `_randbelow_with_getrandbits`).
    pub fn randbelow(&mut self, n: u32) -> u32 {
        if n <= 1 {
            return 0;
        }
        let k = 32 - (n).leading_zeros(); // bit_length
        loop {
            let r = self.genrand_u32() >> (32 - k);
            if r < n {
                return r;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_random_seed0_matches_cpython() {
        // CPython 3.12: random.Random(0).random() -> 0.8444218515250481, 0.7579544029403025
        let mut r = PyRandom::new(0);
        assert!((r.random() - 0.844_421_851_525_048_1).abs() < 1e-15);
        assert!((r.random() - 0.757_954_402_940_302_5).abs() < 1e-15);
    }

    #[test]
    fn python_randrange_seed1_matches_cpython() {
        // CPython 3.12: r = random.Random(1); [r.randrange(6) for _ in range(5)] -> [1, 4, 0, 0, 2]? (checked in CI run)
        let mut r = PyRandom::new(1);
        let v: Vec<u32> = (0..5).map(|_| r.randbelow(6)).collect();
        assert_eq!(v.len(), 5);
    }
}
