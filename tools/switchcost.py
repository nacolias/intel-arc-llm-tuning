#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Time llama-server spends between picking the slot and starting a task (prompt-cache save/load when the
agent switches between conversations). Reads `journalctl -o short-iso` on stdin; timings only, no text."""
import re, sys, statistics as st
rx_t = re.compile(r": (\d+)\.(\d+)\.(\d+)\.(\d+) [IWE] ")
rx_sel = re.compile(r"selected slot by (LCP similarity|LRU)(?:, f_sim_best = ([\d.]+).*f_keep = ([\d.]+))?")
rx_launch = re.compile(r"launch_slot_: id +\d+ \| task (\d+)")
rx_pe = re.compile(r"task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
rx_ev = re.compile(r"task (\d+) \|\s+eval time =\s*([\d.]+) ms")
rx_rel = re.compile(r"task (\d+) \| stop processing: n_tokens = (\d+)")
rows, sel, tasks, prev_n = [], None, {}, None
for line in sys.stdin:
    m = rx_t.search(line)
    if not m: continue
    t = int(m[1]) * 60 + int(m[2]) + int(m[3]) / 1e3 + int(m[4]) / 1e6
    if (m2 := rx_sel.search(line)):
        sel = (t, m2[1], float(m2[3]) if m2[3] else None)
    elif (m2 := rx_launch.search(line)) and sel:
        tasks[m2[1]] = dict(gap=t - sel[0], kind=sel[1], f_keep=sel[2], prev_n=prev_n)
        sel = None
    elif (m2 := rx_pe.search(line)) and m2[1] in tasks:
        tasks[m2[1]].update(pe=float(m2[2]) / 1e3, pn=int(m2[3]))
    elif (m2 := rx_ev.search(line)) and m2[1] in tasks:
        tasks[m2[1]]["ev"] = float(m2[2]) / 1e3
    elif (m2 := rx_rel.search(line)):
        if m2[1] in tasks:
            tasks[m2[1]]["n"] = int(m2[2]); rows.append(tasks.pop(m2[1]))
        prev_n = int(m2[2])
def q(v, p): v = sorted(v); return v[min(len(v)-1, int(p*len(v)))] if v else float("nan")
sw = [r for r in rows if r["kind"] == "LRU" or (r["f_keep"] is not None and r["f_keep"] < 0.5)]
same = [r for r in rows if r not in sw]
tot = sum(r["gap"] + r.get("pe", 0) + r.get("ev", 0) for r in rows)
print(f"tasks {len(rows)}; server time {tot:.0f} s")
for name, rs in (("switch (LRU or f_keep<0.5)", sw), ("same conversation", same)):
    g = [r["gap"] for r in rs]
    if not g: continue
    print(f"{name}: {len(rs)} tasks; slot-pick-to-start gap median {q(g,.5):.2f} s, p90 {q(g,.9):.2f} s, max {max(g):.2f} s, total {sum(g):.0f} s ({100*sum(g)/tot:.1f}% of server time)")
big = [r for r in sw if (r["prev_n"] or 0) >= 50000 or r.get("n", 0) >= 50000]
if big:
    g = [r["gap"] for r in big]
    print(f"  switches touching a >=50K conversation: {len(big)}; gap median {q(g,.5):.2f} s, total {sum(g):.0f} s")
