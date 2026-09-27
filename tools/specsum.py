#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summary of a specreplay.py JSON: per draft setting, decode tok/s over all turns, acceptance, tokens
per verification step, milliseconds per step, drafted tokens per step, and the tok/s ratio against
the baseline setting with a 90% bootstrap interval over turns (turns resampled with replacement;
the ratio is total tokens over total decode time, both settings on the same resampled turns).

Usage: python3 specsum.py specreplay.json|replay.csv [bootstrap_samples]

The CSV form is the one in models/qwen3.8-flash-next/benchmarks/raw/ (for example
2026-09-26-mtp-draft-replay-768.csv).
"""
import csv, json, random, sys

B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
if sys.argv[1].endswith(".csv"):   # the raw CSV form kept in this repository
    rows, arms = [], []
    for r in csv.DictReader(open(sys.argv[1])):
        rows.append(dict(turn=int(r["turn"]), ctx=int(r["context_tokens"]), arm=r["arm_nmax_pmin"],
                         n=int(r["decode_n"]), ms=float(r["decode_ms"]), acc=int(r["draft_accepted"]),
                         drafted=int(r["draft_generated"])))
        if r["arm_nmax_pmin"] not in arms:
            arms.append(r["arm_nmax_pmin"])
    arms.insert(0, arms.pop(arms.index(rows[0]["arm"])))   # turn 0's first arm is the baseline
else:
    d = json.load(open(sys.argv[1]))
    rows, arms = d["rows"], d["arms"]
base = arms[0]
turns = sorted({r["turn"] for r in rows})
by = {(r["turn"], r["arm"]): r for r in rows}
done = [t for t in turns if all((t, a) in by for a in arms)]


def tot(arm, ts):
    n = sum(by[(t, arm)]["n"] for t in ts)
    ms = sum(by[(t, arm)]["ms"] for t in ts)
    return n, ms


rng = random.Random(1)
print(f"{len(done)} complete turns; baseline {base}; context {by[(done[0], base)]['ctx']}-{by[(done[-1], base)]['ctx']} tokens")
print("arm     tok/s   accept  tok/step  ms/step  drafted/step  vs base (90% CI over turns)")
for arm in arms:
    rs = [by[(t, arm)] for t in done]
    n, ms = tot(arm, done)
    acc = sum(r["acc"] for r in rs)
    dr = sum(r["drafted"] for r in rs)
    steps = n - acc
    ratios = []
    for _ in range(B):
        ts = [rng.choice(done) for _ in done]
        na, ma = tot(arm, ts)
        nb, mb = tot(base, ts)
        ratios.append((na / ma) / (nb / mb))
    ratios.sort()
    lo, hi = ratios[int(0.05 * B)], ratios[int(0.95 * B) - 1]
    nb, mb = tot(base, done)
    print(f"{arm:6s} {1000 * n / ms:6.2f}  {100 * acc / max(dr, 1):5.1f}%  {n / steps:7.2f}  {ms / steps:7.1f}  "
          f"{dr / steps:12.2f}  x{(n / ms) / (nb / mb):.3f} ({lo:.3f}-{hi:.3f})")
