#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Teacher-forced next-token check at long context for a llama.cpp llama-server.

What it measures: whether two builds (or two settings) predict the same next-token distribution
after a long context, without the drift that free-running greedy decode adds. It reads the same
fixed ~LC_TOKENS context as lcbench.py, then for KL_N positions of a fixed continuation asks for the
next-token top-KL_TOP log-probabilities (n_predict 1, prompt cache reused), and writes them to KL_OUT.
Comparing two files prints top-1 agreement and the KL divergence over the union of both top-k lists
(tokens missing from one list get a floor 1 nat below its lowest entry).

Environment (probe mode):
  BASE          server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY  API key sent as a bearer token; leave unset for a server without --api-key
  LC_CORPUS     corpus text file (default: lc-corpus.txt next to this script); see lcbench.py for how
                to build it from a public llama.cpp checkout
  LC_TOKENS     context size in tokens (default 135000)
  KL_N          positions to probe (default 128)
  KL_TOP        top-k log-probabilities per position (default 20)
  KL_OUT        output JSON file (required)

Usage:
  KL_OUT=a.json python3 klprobe.py            # run against build A
  KL_OUT=b.json python3 klprobe.py            # run against build B
  python3 klprobe.py --compare a.json b.json  # top-1 agreement and KL mean / median / p95 / max
"""
import json, math, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    kls, top1 = [], 0
    for pa, pb in zip(a, b):
        da = {t: lp for t, lp in pa}
        db = {t: lp for t, lp in pb}
        floor = min(min(da.values()), min(db.values())) - 1.0   # tokens outside a top-k list
        keys = set(da) | set(db)
        qa = {k: math.exp(da.get(k, floor)) for k in keys}
        qb = {k: math.exp(db.get(k, floor)) for k in keys}
        sa, sb = sum(qa.values()), sum(qb.values())
        kls.append(sum((qa[k] / sa) * math.log((qa[k] / sa) / (qb[k] / sb)) for k in keys))
        top1 += max(da, key=da.get) == max(db, key=db.get)
    kls.sort()
    n = len(kls)
    print(f"{os.path.basename(a_path)} vs {os.path.basename(b_path)}: {n} positions | top-1 agree {top1}/{n} | "
          f"KL mean {sum(kls) / n:.5f} median {kls[n // 2]:.5f} p95 {kls[int(n * 0.95)]:.5f} max {kls[-1]:.5f}")


if len(sys.argv) == 4 and sys.argv[1] == "--compare":
    compare(sys.argv[2], sys.argv[3])
    sys.exit(0)

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
CORPUS = os.environ.get("LC_CORPUS", os.path.join(HERE, "lc-corpus.txt"))
TARGET = int(os.environ.get("LC_TOKENS", "135000"))
N = int(os.environ.get("KL_N", "128"))
TOP = int(os.environ.get("KL_TOP", "20"))
OUT = os.environ["KL_OUT"]


def post(path, body, timeout=7200):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def tok(text):
    return post("/tokenize", {"content": text})["tokens"]


corpus = open(CORPUS).read()
lo, hi = 0, min(len(corpus), int(TARGET * 4.5))           # same context as lcbench.py
while hi - lo > 2000:
    mid = (lo + hi) // 2
    if len(tok(corpus[:mid])) < TARGET:
        lo = mid
    else:
        hi = mid
prefix = tok("You are a senior engineer reviewing a C++ codebase. The full source is below.\n\n" + corpus[:lo])
cont = tok(corpus[lo:lo + 4000])[:N]
rows = []
for i in range(N):
    r = post("/completion", {"prompt": prefix + cont[:i], "n_predict": 1, "n_probs": TOP, "temperature": 0.0,
                             "cache_prompt": True, "post_sampling_probs": False})
    cp = r["completion_probabilities"][0]
    tops = cp.get("top_logprobs") or cp.get("probs") or []
    row = []
    for t in tops:
        lp = t["logprob"] if "logprob" in t else math.log(max(t["prob"], 1e-30))
        row.append([t.get("id", t.get("tok_str", t.get("token"))), lp])
    rows.append(row)
json.dump(rows, open(OUT, "w"))
print(f"wrote {OUT}: {N} positions after a {len(prefix)}-token prefix", flush=True)
