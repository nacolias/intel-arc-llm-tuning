#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-request table from a llama-server journal (llama.cpp print_timing / release / slot-selection lines).

Reads `journalctl -o short-iso` text of the server's unit on stdin, writes one CSV row per finished task to stdout and a summary
to stderr. Columns: time, invocation (server start time), task, prompt_new (tokens evaluated for the prompt),
prompt_ms, decode_n, decode_ms, decode_tps, draft_acc, draft_gen, n_ctx_after (tokens in the slot after the
task), f_sim, f_keep (prompt-cache similarity and kept fraction when the slot was picked), bench (1 when the
request looks like one of our benchmarks: fixed decode lengths 1/16/128/192/256/512 or a ~134.8K cold read).
Only timings and counts are read; no prompt or output text is in these lines.
"""
import os, re, sys, statistics as st

rx_ts = re.compile(r"^(\S+)\s")
# a line that marks a server (re)start; set TURNLOG_START to match your unit, e.g. "Started llama-server"
rx_start = re.compile(os.environ.get("TURNLOG_START", r"Started .*llama"))
rx_sel = re.compile(r"selected slot by LCP similarity, f_sim_best = ([\d.]+).*f_keep = ([\d.]+)")
rx_pe = re.compile(r"task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
rx_ev = re.compile(r"task (\d+) \|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
rx_da = re.compile(r"task (\d+) \| draft acceptance = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)")
rx_rel = re.compile(r"task (\d+) \| stop processing: n_tokens = (\d+)")

BENCH_DECODE = {1, 16, 128, 192, 256, 512}
rows, cur, inv, sel = [], {}, "", (None, None)
for line in sys.stdin:
    m = rx_ts.match(line)
    ts = m.group(1) if m else ""
    if rx_start.search(line):
        inv = ts
        continue
    m = rx_sel.search(line)
    if m:
        sel = (float(m.group(1)), float(m.group(2)))
        continue
    m = rx_pe.search(line)
    if m:
        t = m.group(1)
        cur[t] = dict(time=ts, invocation=inv, task=t, prompt_ms=float(m.group(2)), prompt_new=int(m.group(3)),
                      f_sim=sel[0], f_keep=sel[1])
        sel = (None, None)
        continue
    m = rx_ev.search(line)
    if m and m.group(1) in cur:
        cur[m.group(1)].update(decode_ms=float(m.group(2)), decode_n=int(m.group(3)))
        continue
    m = rx_da.search(line)
    if m and m.group(1) in cur:
        cur[m.group(1)].update(draft_acc=int(m.group(2)), draft_gen=int(m.group(3)))
        continue
    m = rx_rel.search(line)
    if m and m.group(1) in cur:
        r = cur.pop(m.group(1))
        r["n_ctx_after"] = int(m.group(2))
        rows.append(r)

cols = ["time", "invocation", "task", "prompt_new", "prompt_ms", "decode_n", "decode_ms", "decode_tps", "draft_acc",
        "draft_gen", "n_ctx_after", "f_sim", "f_keep", "bench"]
print(",".join(cols))
for r in rows:
    dn = r.get("decode_n", 0)
    r["decode_tps"] = round(1000 * dn / r["decode_ms"], 2) if r.get("decode_ms") else ""
    r["bench"] = int(dn in BENCH_DECODE or 134700 <= r["prompt_new"] <= 135000)
    print(",".join("" if r.get(c) is None else str(r.get(c, "")) for c in cols))

def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")

real = [r for r in rows if not r["bench"]]
deep = [r for r in real if r["n_ctx_after"] >= 100000]
out = sys.stderr
print(f"requests parsed: {len(rows)}; benchmark-like: {len(rows) - len(real)}; real: {len(real)}; real at >=100K context: {len(deep)}", file=out)
for name, rs in (("real, all depths", real), ("real, >=100K", deep)):
    if not rs:
        continue
    dn = [r.get("decode_n", 0) for r in rs]
    dms = [r.get("decode_ms", 0) / 1000 for r in rs]
    pms = [r["prompt_ms"] / 1000 for r in rs]
    pn = [r["prompt_new"] for r in rs]
    tot_d, tot_p = sum(dms), sum(pms)
    acc = sum(r.get("draft_acc", 0) for r in rs); gen = sum(r.get("draft_gen", 0) for r in rs)
    tps = [r["decode_tps"] for r in rs if r["decode_tps"] and r.get("decode_n", 0) >= 64]
    print(f"\n== {name}: {len(rs)} requests", file=out)
    print(f"  decode tokens per request: median {q(dn, .5)}, p75 {q(dn, .75)}, p90 {q(dn, .9)}, max {max(dn)}", file=out)
    print(f"  decode seconds per request: median {q(dms, .5):.1f}, p90 {q(dms, .9):.1f}; total {tot_d:.0f} s", file=out)
    print(f"  prompt seconds per request: median {q(pms, .5):.2f}, p90 {q(pms, .9):.2f}; total {tot_p:.0f} s", file=out)
    print(f"  new prompt tokens per request: median {q(pn, .5)}, p90 {q(pn, .9)}, max {max(pn)}", file=out)
    print(f"  share of server time in decode: {100 * tot_d / (tot_d + tot_p):.1f}%", file=out)
    print(f"  draft acceptance (production sampler): {acc}/{gen} = {100 * acc / max(gen, 1):.1f}%", file=out)
    print(f"  decode tok/s (requests with >=64 tokens): median {q(tps, .5):.1f}, p10 {q(tps, .1):.1f}, p90 {q(tps, .9):.1f}", file=out)
    big = [r for r in rs if r["prompt_new"] > 5000]
    print(f"  requests re-reading >5000 prompt tokens: {len(big)} ({sum(r['prompt_ms'] for r in big) / 1000:.0f} s total); f_keep of those: {sorted(round(r['f_keep'] or 0, 3) for r in big)[:12]}", file=out)
    fk = [r["f_keep"] for r in rs if r["f_keep"] is not None]
    print(f"  requests with f_keep < 0.9 (cache largely not reused): {sum(1 for x in fk if x < 0.9)}", file=out)
