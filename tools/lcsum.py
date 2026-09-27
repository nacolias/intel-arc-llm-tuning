#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarise lcbench.py / accprobe.py JSON files.

For each file: number of follow-up turns, MTP (multi-token prediction) acceptance, milliseconds per
MTP step (draft_n / 3 steps when a draft ran, else one step per token), overall decode tok/s and the
median per-turn prompt latency. Turn 0 (the cold read) is left out.

Usage:
  python3 lcsum.py a.json b.json
"""
import json, sys

for p in sys.argv[1:]:
    r = json.load(open(p))[1:]
    acc = sum(x["timings"].get("draft_n_accepted", 0) for x in r)
    dr = sum(x["timings"].get("draft_n", 0) for x in r)
    ms = sum(x["timings"]["predicted_ms"] for x in r)
    n = sum(x["timings"]["predicted_n"] for x in r)
    steps = dr / 3 if dr else n
    pm = sorted(x["timings"]["prompt_ms"] for x in r)
    print(f"{p}: turns {len(r)} | acceptance {acc}/{dr} = {acc / max(dr, 1):.1%} | {ms / max(steps, 1):.1f} ms/step | "
          f"decode {1000 * n / max(ms, 1):.1f} tok/s overall | prompt median {pm[len(pm) // 2] / 1000:.2f}s")
