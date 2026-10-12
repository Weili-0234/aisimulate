// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI for the HiCache + P2P P/D topology runtime.
//!
//! usage: hicache_p2p --workload cc.workload.jsonl --config cfg.json --out out.json
//!        [--ct ct.tsv] [--rows-prefix rows] [--seeds 0,1,2] [--set key=json ...]

use std::io::Write;
use std::path::PathBuf;

use aisimulate_core::replay::hicache_p2p::{SimConfig, Workload, run};
use anyhow::{Context, Result, bail};
use rustc_hash::FxHashMap;
use serde_json::Value;

fn main() -> Result<()> {
    let mut wl_path: Option<PathBuf> = None;
    let mut cfg_path: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut ct_path: Option<PathBuf> = None;
    let mut rows_prefix: Option<String> = None;
    let mut seeds: Vec<u64> = vec![0];
    let mut sets: Vec<(String, String)> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--workload" => wl_path = it.next().map(PathBuf::from),
            "--config" => cfg_path = it.next().map(PathBuf::from),
            "--out" => out = it.next().map(PathBuf::from),
            "--ct" => ct_path = it.next().map(PathBuf::from),
            "--rows-prefix" => rows_prefix = it.next(),
            "--seeds" => {
                seeds = it
                    .next()
                    .context("--seeds")?
                    .split(',')
                    .map(|s| s.trim().parse::<u64>())
                    .collect::<std::result::Result<_, _>>()?
            }
            "--set" => {
                let kv = it.next().context("--set key=json")?;
                let (k, v) = kv.split_once('=').context("--set key=json")?;
                sets.push((k.to_string(), v.to_string()));
            }
            _ => bail!("unknown argument {a}"),
        }
    }
    let wl_path = wl_path.context("--workload")?;
    let out = out.context("--out")?;
    let mut cfg: Value = match &cfg_path {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => serde_json::to_value(SimConfig::default())?,
    };
    for (k, v) in &sets {
        let val: Value = serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.clone()));
        cfg.as_object_mut().context("config must be an object")?.insert(k.clone(), val);
    }
    let base: SimConfig = serde_json::from_value(cfg)?;
    let t0 = std::time::Instant::now();
    let wl = Workload::load(&wl_path)?;
    eprintln!("workload: {} tasks, {} generates, load {:.1}s", wl.tasks.len(), wl.generates(), t0.elapsed().as_secs_f64());
    let mut ct: FxHashMap<(u32, u32, u32), u32> = FxHashMap::default();
    if let Some(p) = &ct_path {
        for line in std::fs::read_to_string(p)?.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 4 || f[0].starts_with('#') {
                continue;
            }
            if let (Ok(a), Ok(b), Ok(c), Ok(d)) = (f[0].parse(), f[1].parse(), f[2].parse(), f[3].parse()) {
                ct.insert((a, b, c), d);
            }
        }
        eprintln!("ct map: {} entries", ct.len());
    }
    let results: Vec<_> = std::thread::scope(|s| {
        let hs: Vec<_> = seeds
            .iter()
            .map(|&seed| {
                let mut c = base.clone();
                c.seed = seed;
                let wl = &wl;
                let ct = &ct;
                s.spawn(move || run(c, wl, ct))
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("sim thread panicked")).collect()
    });
    let mut reports = Vec::new();
    for (rep, rows) in results {
        if let Some(pfx) = &rows_prefix {
            let path = format!("{pfx}-seed{}.tsv", rep.seed);
            let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
            writeln!(f, "task_row\tagent\tidx\tadmit\tp\td\tpt\tactual\tideal\tdev\thost\tpulled\troute_best\troute_tgt\tfc_best\tpull_hint\tdetour\tttft\te2e\tct\tcompleted\tp_queue\td_wait\tprefill_s\tdecode_s")?;
            for r in &rows {
                writeln!(
                    f,
                    "{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}",
                    r.task_row, r.agent, r.idx, r.admit, r.p, r.d, r.pt, r.actual, r.ideal, r.dev, r.host, r.pulled,
                    r.route_best, r.route_tgt, r.fc_best, r.pull_hint, r.detour as u8, r.ttft, r.e2e, r.ct, r.completed as u8,
                    r.p_queue, r.d_wait, r.prefill_s, r.decode_s
                )?;
            }
        }
        eprintln!(
            "seed {} rps {:.3} hit {:.4} zero-hit rows {:.4} E-share {:.3} ttft p50 {:.3} wall {:.1}s",
            rep.seed, rep.rps, rep.hit_sum_a_over_i, rep.zero_hit_row_share, rep.e_share_route, rep.ttft[1], rep.wall_s
        );
        reports.push(rep);
    }
    let doc = serde_json::json!({"config": base, "reports": reports});
    std::fs::write(&out, serde_json::to_string_pretty(&doc)?)?;
    Ok(())
}
