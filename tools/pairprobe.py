#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paired A/B of a runtime switch on a llama.cpp llama-server at long context.

What it measures: MTP (multi-token prediction) draft acceptance, step time and decode speed of one
build with a runtime switch on and off, on the same context and (greedy) mostly the same text.
Run-to-run acceptance noise is about 3 points on this workload, so two separate sessions cannot
resolve a small change; decoding every turn twice in one session, once per setting, can.

It starts from the final prompt of an lcbench.py run (LC_SAVE_PROMPT), so run lcbench.py first with
the same server. Each turn appends a "tool output" snippet from the corpus plus a fixed task line,
decodes PAIR_PREDICT tokens greedily with the switch off, then again with it on, and continues the
conversation with the first answer. The switch is a flag file that the patched build polls
(PAIR_FLAG; see configs/patches/README.md for the files each patch honours, for example
/tmp/llama-mtp-qsa-off, which turns the MTP draft's QSA off). The file is created in the server's
own /tmp, so run this on the server host.

Environment:
  BASE            server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY    API key sent as a bearer token; leave unset for a server without --api-key
  LC_CORPUS       corpus text file (default: lc-corpus.txt next to this script); see lcbench.py
  PAIR_PROMPT     JSON written by lcbench.py with LC_SAVE_PROMPT (default: lc-turnprof-prompt.json here)
  PAIR_FLAG       flag file whose presence turns the feature OFF (default /tmp/llama-mtp-qsa-off)
  PAIR_TURNS      number of turns (default 12)
  PAIR_PREDICT    tokens decoded per turn and setting (default 192)

Usage:
  LC_SAVE_PROMPT=lc-turnprof-prompt.json LC_TURNS=1 python3 lcbench.py
  PAIR_TURNS=12 python3 pairprobe.py
"""
import json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
CORPUS = os.environ.get("LC_CORPUS", os.path.join(HERE, "lc-corpus.txt"))
PROMPT_FILE = os.environ.get("PAIR_PROMPT", os.path.join(HERE, "lc-turnprof-prompt.json"))
OFF = os.environ.get("PAIR_FLAG", "/tmp/llama-mtp-qsa-off")
TURNS = int(os.environ.get("PAIR_TURNS", "12"))
PREDICT = int(os.environ.get("PAIR_PREDICT", "192"))


def post(body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=3600))


prompt = json.load(open(PROMPT_FILE))["prompt"]
tail = open(CORPUS).read()[-120000:]
tot = {"on": [0, 0, 0.0, 0], "off": [0, 0, 0.0, 0]}
same = 0
for k in range(TURNS):
    prompt += "\n\n### Tool output\n" + tail[k * 5000: k * 5000 + (1500 if k % 3 == 2 else 160)] + "\n\n### Task\nExplain what this code does.\n\n### Answer\n"
    outs = {}
    for mode in ("on", "off"):
        if mode == "off":
            open(OFF, "w").close()
        elif os.path.exists(OFF):
            os.remove(OFF)
        time.sleep(1.2)  # the server polls the flag file about twice a second
        r = post({"prompt": prompt, "n_predict": PREDICT, "temperature": 0.0, "cache_prompt": True, "ignore_eos": True})
        tm = r["timings"]
        outs[mode] = r["content"]
        t = tot[mode]
        t[0] += tm.get("draft_n_accepted", 0)
        t[1] += tm.get("draft_n", 0)
        t[2] += tm["predicted_ms"]
        t[3] += tm["predicted_n"]
        print(f"turn {k} {mode:3s}: draft {tm.get('draft_n_accepted')}/{tm.get('draft_n')} | {tm['predicted_per_second']:.1f} tok/s | prompt {tm['prompt_ms'] / 1000:.2f}s", flush=True)
    same += outs["on"] == outs["off"]
    prompt += outs["on"]
if os.path.exists(OFF):
    os.remove(OFF)
for mode, (a, d, ms, n) in tot.items():
    steps = d / 3 if d else n
    print(f"{mode}: acceptance {a}/{d} = {a / max(d, 1):.1%} | {ms / max(steps, 1):.1f} ms/step | {1000 * n / ms:.1f} tok/s", flush=True)
print(f"identical outputs in {same}/{TURNS} turns", flush=True)
