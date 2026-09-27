#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sum the [SYCL-OP-DEVPROF] blocks of a llama-server log (patch 0020, GGML_SYCL_OP_DEVPROF) into one
table: per op key, total device ms, calls and share, plus per device the summed op time, host submit
time and graph count. The target model's graphs and the small (MTP draft head) graphs are listed
apart. Ops are grouped into coarse classes by name so the table can be read per step.

Usage: grep SYCL-OP-DEVPROF server.log | python3 devprofsum.py [steps]
  steps: number of decode verify steps in the window, to print ms per step (optional)
"""
import re, sys
from collections import defaultdict

steps = float(sys.argv[1]) if len(sys.argv) > 1 else 0
rx_head = re.compile(r"\[SYCL-OP-DEVPROF\] (\d+) ms of device op time;(.*)$")
rx_dev = re.compile(r"d(\d+): ops (\d+) ms, host submit (\d+) ms, (\d+) graphs")
rx_row = re.compile(r"\[SYCL-OP-DEVPROF\]\s+([\d.]+) ms\s+[\d.]+%\s+(\d+)\s+[\d.]+ us avg  (.*)$")
ops = defaultdict(lambda: [0.0, 0])
dev = defaultdict(lambda: [0.0, 0.0, 0])
blocks = 0
for line in sys.stdin:
    m = rx_head.search(line)
    if m:
        blocks += 1
        for d, o, h, g in rx_dev.findall(m.group(2)):
            dev[int(d)][0] += float(o); dev[int(d)][1] += float(h); dev[int(d)][2] += int(g)
        continue
    m = rx_row.search(line)
    if m:
        ops[m.group(3).strip()][0] += float(m.group(1))
        ops[m.group(3).strip()][1] += int(m.group(2))

total = sum(v[0] for v in ops.values())
print(f"{blocks} print blocks; {total:.0f} ms of device op time in the listed rows")
for d in sorted(dev):
    o, h, g = dev[d]
    print(f"  device {d}: ops {o:.0f} ms, host submit {h:.0f} ms, {g} graphs")


def cls(key):
    k = key.lower()
    small = k.startswith("[small graph]")
    for name, pat in (("MoE matmul (MUL_MAT_ID)", "mul_mat_id"), ("flash attention", "flash_attn"),
                      ("gated delta net", "gated_delta"), ("matmul (dense)", "mul_mat"),
                      ("get_rows", "get_rows"), ("copy / cont / cpy", r"\bcpy\b|\bcont\b|\bdup\b"),
                      ("top-k / argsort", "argsort|top_k|topk"), ("norm", "norm"), ("ssm conv", "ssm_conv"),
                      ("softmax", "soft_max"), ("rope", "rope")):
        if re.search(pat, k):
            return ("[draft] " if small else "") + name
    return ("[draft] " if small else "") + "other"


by_cls = defaultdict(lambda: [0.0, 0])
for k, (ms, n) in ops.items():
    c = cls(k)
    by_cls[c][0] += ms; by_cls[c][1] += n
print("\nby class:")
for c, (ms, n) in sorted(by_cls.items(), key=lambda kv: -kv[1][0]):
    per = f", {ms / steps:.2f} ms/step" if steps else ""
    print(f"  {100 * ms / total:5.1f}%  {ms:9.1f} ms  {n:8d} calls  {1000 * ms / max(n, 1):8.1f} us avg{per}  {c}")
print("\ntop ops:")
for k, (ms, n) in sorted(ops.items(), key=lambda kv: -kv[1][0])[:40]:
    per = f", {ms / steps:.2f} ms/step" if steps else ""
    print(f"  {100 * ms / total:5.1f}%  {ms:9.1f} ms  {n:8d} calls  {1000 * ms / max(n, 1):8.1f} us avg{per}  {k}")
