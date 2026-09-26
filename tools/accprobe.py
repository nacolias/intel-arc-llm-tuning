#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Greedy MTP acceptance probe for a llama.cpp llama-server at long context.

What it measures: MTP (multi-token prediction) draft acceptance, step time and decode speed over
ACC_TURNS greedy turns on top of a long context. Two sessions of builds that differ only in the
draft head produce nearly the same text, so their acceptance compares like for like; sessions that
differ in the target model do not (see the "identical outputs" line pairprobe.py prints).

It starts from the final prompt of an lcbench.py run (LC_SAVE_PROMPT). Turn 0 reads that context
cold; later turns append a "tool output" snippet from the corpus plus a fixed task line, decode
ACC_PREDICT tokens greedily, and continue with the answer. Writes ACC_OUT with per-turn timings
and the generated text.

Environment:
  BASE          server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY  API key sent as a bearer token; leave unset for a server without --api-key
  LC_CORPUS     corpus text file (default: lc-corpus.txt next to this script); see lcbench.py
  ACC_PROMPT    JSON written by lcbench.py with LC_SAVE_PROMPT (default: lc-turnprof-prompt.json here)
  ACC_TURNS     number of turns including the cold read (default 12)
  ACC_PREDICT   tokens decoded per turn (default 192)
  ACC_OUT       output JSON (default acc.json)

Usage:
  ACC_OUT=acc-A.json python3 accprobe.py      # build or setting A
  ACC_OUT=acc-B.json python3 accprobe.py      # build or setting B, same context
"""
import json, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
CORPUS = os.environ.get("LC_CORPUS", os.path.join(HERE, "lc-corpus.txt"))
PROMPT_FILE = os.environ.get("ACC_PROMPT", os.path.join(HERE, "lc-turnprof-prompt.json"))
TURNS = int(os.environ.get("ACC_TURNS", "12"))
PREDICT = int(os.environ.get("ACC_PREDICT", "192"))
OUT = os.environ.get("ACC_OUT", "acc.json")


def post(body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=3600))


prompt = json.load(open(PROMPT_FILE))["prompt"]
tail = open(CORPUS).read()[-120000:]
res = []
a = d = 0
ms = n = 0.0
for k in range(TURNS):
    prompt += "\n\n### Tool output\n" + tail[k * 5000: k * 5000 + (1500 if k % 3 == 2 else 160)] + "\n\n### Task\nExplain what this code does.\n\n### Answer\n"
    r = post({"prompt": prompt, "n_predict": PREDICT, "temperature": 0.0, "cache_prompt": True, "ignore_eos": True})
    tm = r["timings"]
    res.append({"timings": tm, "content": r["content"]})
    print(f"turn {k}: prompt {tm['prompt_n']} tok in {tm['prompt_ms'] / 1000:.2f}s | draft {tm.get('draft_n_accepted')}/{tm.get('draft_n')} | {tm['predicted_per_second']:.1f} tok/s", flush=True)
    if k > 0:
        a += tm.get("draft_n_accepted", 0)
        d += tm.get("draft_n", 0)
        ms += tm["predicted_ms"]
        n += tm["predicted_n"]
    prompt += r["content"]
json.dump(res, open(OUT, "w"))
print(f"turns 1-{TURNS - 1}: acceptance {a}/{d} = {a / max(d, 1):.1%} | {ms / max(d / 3, 1):.1f} ms/step | {1000 * n / max(ms, 1):.1f} tok/s", flush=True)
