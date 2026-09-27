#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Per-GPU busy %, clock and host CPU use while llama-server decodes one request.

What it measures: it sends one /completion request (n_predict tokens, greedy, ignore_eos, no prompt
cache) to a running llama.cpp llama-server, waits past the prefill, and over a fixed window reads
  - per-GPU, per-engine busy % from the xe driver's fdinfo counters (drm-cycles-<engine> over
    drm-total-cycles-<engine>) summed over llama-server's DRM clients;
  - each GPU's actual GT0 clock (tile0/gt0/freq0/act_freq), sampled every 250 ms;
  - llama-server's CPU use and its three busiest threads.
With a layer split (-sm layer) the cards run one after another, so the compute-engine (ccs) busy %
summed over all GPUs shows how much of the time any card is working: 100% means one card is always
busy. A low sum with a busy main thread means decode is host-bound.

GPUs are discovered generically: Intel (vendor 0x8086) PCI display-class devices (class 0x0300 VGA
or 0x0380 other display), numbered in PCI address order. On a host whose only Intel display devices
are the Arc cards this matches the SYCL / xpu-smi order when those also follow PCI order; check
with sycl-ls or xpu-smi discovery. An Intel integrated GPU would also be picked up.

Run it on the inference host as root: reading another user's /proc/<pid>/fdinfo needs it.

Environment:
  BASE          server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY  API key sent as a bearer token; leave unset for a server without --api-key
  DELAY         seconds to wait before measuring (default 3 + prompt_tokens_approx / 500, skips prefill)
  WINDOW        measurement window in seconds (default 8)

Usage:
  sudo -E python3 gpuprof.py [n_predict] [prompt_tokens_approx]
  sudo -E DELAY=5 WINDOW=10 python3 gpuprof.py 64 30000    # profile prefill instead of decode
"""
import glob, json, os, re, sys, threading, time, urllib.request

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
PID = int(os.popen("pgrep -x llama-server").read().split()[0])


def intel_gpus():
    """Map PCI address -> index for Intel display-class devices, in PCI address order."""
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


def cycles():
    """Sum drm-cycles-<engine> and drm-total-cycles-<engine> per GPU over llama-server's DRM fds."""
    acc = {}
    seen = set()
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
            # cycles add up across the process's DRM clients; total-cycles is the same GPU clock in each, so take the max
            d[k] = max(d.get(k, 0), int(v)) if k.startswith("total-") else d.get(k, 0) + int(v)
    return acc


def gpu_freq_power():
    out = {}
    for card in glob.glob("/sys/class/drm/card*/device"):
        bdf = os.path.basename(os.path.realpath(card))
        if bdf not in BDF:
            continue
        fs = glob.glob(card + "/tile0/gt0/freq0/act_freq")
        out[BDF[bdf]] = int(open(fs[0]).read()) if fs else None
    return out


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


def cpu_ticks():
    s = open(f"/proc/{PID}/stat").read().split(")")[1].split()
    return int(s[11]) + int(s[12])


samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        samples.append(gpu_freq_power())
        time.sleep(0.25)


prompt = "Write a detailed technical explanation of how a PCIe switch routes peer-to-peer transactions."
if CTX:
    prompt = ("Background notes: the PEX88096 is a 98-lane PCIe Gen4 switch with five x16 downstream ports. " * (CTX // 20)) + prompt
body = json.dumps({"prompt": prompt, "n_predict": N, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False}).encode()
headers = {"Content-Type": "application/json"}
if KEY:
    headers["Authorization"] = "Bearer " + KEY
req = urllib.request.Request(BASE + "/completion", data=body, headers=headers)
result = {}
th = threading.Thread(target=lambda: result.update(json.load(urllib.request.urlopen(req, timeout=1800))))
th.start()
# DELAY/WINDOW override when to measure; e.g. DELAY=5 WINDOW=10 with a large CTX profiles prefill
time.sleep(float(os.environ.get("DELAY", 3 + CTX / 500)))   # default: skip prefill
c0, t0, cpu0, th0 = cycles(), time.time(), cpu_ticks(), thread_ticks()
threading.Thread(target=sampler, daemon=True).start()
time.sleep(float(os.environ.get("WINDOW", 8)))
c1, t1, cpu1, th1 = cycles(), time.time(), cpu_ticks(), thread_ticks()
stop.set()
th.join()
tm = result["timings"]
print(f"decode {tm['predicted_per_second']:.1f} tok/s; draft {tm.get('draft_n_accepted')}/{tm.get('draft_n')}")
print(f"llama-server CPU: {100 * (cpu1 - cpu0) / os.sysconf('SC_CLK_TCK') / (t1 - t0):.0f}% of one core")
hz = os.sysconf('SC_CLK_TCK')
top = sorted(((th1[k] - th0.get(k, 0), k) for k in th1), reverse=True)[:3]
print("busiest threads: " + ", ".join(f"{k[1]}/{k[0]} {100 * v / hz / (t1 - t0):.0f}%" for v, k in top))
ccs_sum = 0.0
for g in sorted(k for k in c1 if isinstance(k, int)):
    busy = []
    for k in c1[g]:
        if k.startswith("cycles-"):
            eng = k[len("cycles-"):]
            dt = c1[g].get("total-cycles-" + eng, 0) - c0.get(g, {}).get("total-cycles-" + eng, 0)
            dc = c1[g][k] - c0.get(g, {}).get(k, 0)
            if dt > 0 and dc > 0:
                busy.append(f"{eng} {100 * dc / dt:.0f}%")
                if eng == "ccs":
                    ccs_sum += 100 * dc / dt
    fr = [s.get(g) for s in samples if s.get(g)]
    print(f"GPU{g}: busy {', '.join(busy) or '0%'}; act_freq MHz min/avg/max {min(fr)}/{sum(fr)//len(fr)}/{max(fr)}")
print(f"compute-engine busy summed over GPUs: {ccs_sum:.0f}% (layer split runs the cards in turn: 100% = one card always busy)")
