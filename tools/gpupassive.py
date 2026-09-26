#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Passive version of gpuprof.py: sample llama-server's per-GPU compute-engine busy % (xe fdinfo
drm-cycles), the process's CPU use and its busiest threads over WINDOW seconds, while something else
drives the server. Sends no requests and opens no GPU. Prints one block per window.

Usage: python3 gpupassive.py [windows] [seconds]   (as root, or as the user running llama-server)

GPU indices follow PCI address order, which may differ from the SYCL device order.
"""
import glob, os, re, sys, time

NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3
WINDOW = float(sys.argv[2]) if len(sys.argv) > 2 else 10
PID = int(os.popen("pgrep -x llama-server").read().split()[0])


def intel_gpus():
    """Map PCI address -> index for Intel display-class devices, in PCI address order (as gpuprof.py)."""
    found = []
    for dev in glob.glob("/sys/bus/pci/devices/*"):
        try:
            vendor = open(dev + "/vendor").read().strip()
            cls = open(dev + "/class").read().strip()
        except OSError:
            continue
        if vendor == "0x8086" and cls[:6] in ("0x0300", "0x0380"):
            found.append(os.path.basename(dev))
    return {bdf: i for i, bdf in enumerate(sorted(found))}


BDF = intel_gpus()
HZ = os.sysconf("SC_CLK_TCK")


def cycles():
    acc, seen = {}, set()
    for f in glob.glob(f"/proc/{PID}/fdinfo/*"):
        try:
            txt = open(f).read()
        except OSError:
            continue
        m = re.search(r"drm-pdev:\s*(\S+)", txt)
        cid = re.search(r"drm-client-id:\s*(\d+)", txt)
        if not m or not cid or (m.group(1), cid.group(1)) in seen:
            continue
        seen.add((m.group(1), cid.group(1)))
        d = acc.setdefault(BDF.get(m.group(1), m.group(1)), {})
        for k, v in re.findall(r"drm-(total-cycles-\w+|cycles-\w+):\s*(\d+)", txt):
            d[k] = max(d.get(k, 0), int(v)) if k.startswith("total-") else d.get(k, 0) + int(v)
    return acc


def thread_ticks():
    out = {}
    for t in glob.glob(f"/proc/{PID}/task/*/stat"):
        try:
            st = open(t).read()
        except OSError:
            continue
        f = st.split(")")[1].split()
        out[(t.split("/")[4], st.split("(")[1].split(")")[0])] = int(f[11]) + int(f[12])
    return out


for w in range(NWIN):
    c0, t0, th0 = cycles(), time.time(), thread_ticks()
    time.sleep(WINDOW)
    c1, t1, th1 = cycles(), time.time(), thread_ticks()
    dt_s = t1 - t0
    cpu = sum(th1.values()) - sum(th0.get(k, 0) for k in th1)
    top = sorted(((th1[k] - th0.get(k, 0), k) for k in th1), reverse=True)[:3]
    parts, ccs_sum = [], 0.0
    for g in sorted(k for k in c1 if isinstance(k, int)):
        dt = c1[g].get("total-cycles-ccs", 0) - c0.get(g, {}).get("total-cycles-ccs", 0)
        dc = c1[g].get("cycles-ccs", 0) - c0.get(g, {}).get("cycles-ccs", 0)
        b = 100 * dc / dt if dt > 0 else 0.0
        ccs_sum += b
        parts.append(f"GPU{g} {b:.0f}%")
    print(f"window {w} ({dt_s:.0f} s): ccs busy {', '.join(parts)}; summed {ccs_sum:.0f}% | llama-server CPU "
          f"{100 * cpu / HZ / dt_s:.0f}% of one core; busiest threads: "
          + ", ".join(f"{k[1]} {100 * v / HZ / dt_s:.0f}%" for v, k in top), flush=True)
