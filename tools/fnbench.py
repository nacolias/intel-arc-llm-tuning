#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Quick speed and sanity bench for a llama.cpp llama-server (single slot).

What it measures, in order:
  1. Sanity: one chat question with a known answer (a trip from 14:35 to 17:10 is 155 minutes),
     through /v1/chat/completions with max_tokens 2048, so a reasoning model can think first.
     It prints the last 40 characters of the answer.
  2. Short-context decode: SHORT_REPS (default 5) runs of a one-line prompt, 512 tokens each;
     prints each run and the median decode tok/s.
  3. Prefill and decode on two prompts of a repeated paragraph, 256 tokens decoded each. The printed
     labels "~8000" and "~32000" are nominal; with the Qwen3.8-Flash-Next tokenizer the prompts are
     9,749 and 39,119 tokens.
  4. Optional: one long prompt sized by LONG_TOKENS, made of varied records (so MTP cannot coast
     on repetition), 128 tokens decoded, to prove the full context fills and still decodes.
All completion runs use temperature 0, ignore_eos and cache_prompt false, so every prompt is a cold
prefill. Prompt and decode speeds come from the server's own timings; MTP (multi-token prediction)
draft acceptance is printed when the server reports it.

Environment:
  BASE          server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY  API key sent as a bearer token; leave unset for a server without --api-key
  SHORT_REPS    number of short-context runs (default 5)
  LONG_TOKENS   rough size of the optional long prompt (default 0 = skip); the record count is
                LONG_TOKENS // 26, and LONG_TOKENS=200000 gave a 219,217-token prompt with the
                Qwen3.8-Flash-Next tokenizer

Usage:
  python3 fnbench.py
  LONG_TOKENS=200000 python3 fnbench.py
"""
import json, os, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
LONG = int(os.environ.get("LONG_TOKENS", "0"))


def post(path, body, timeout=3600):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def line(label, tm):
    extra = f" | draft accepted {tm['draft_n_accepted']}/{tm['draft_n']}" if tm.get("draft_n") else ""
    print(f"{label}: prompt {tm['prompt_n']} tok at {tm['prompt_per_second']:.1f} tok/s ({tm['prompt_ms'] / 1000:.1f}s) | "
          f"decode {tm['predicted_per_second']:.1f} tok/s{extra}", flush=True)


r = post("/v1/chat/completions", {
    "messages": [{"role": "user", "content": "A train leaves at 14:35 and arrives at 17:10 the same day. How many minutes is the trip? Answer with just the number."}],
    "max_tokens": 2048})
print("SANITY answer:", repr((r["choices"][0]["message"].get("content") or "").strip()[-40:]), flush=True)

short = []
for i in range(int(os.environ.get("SHORT_REPS", "5"))):
    r = post("/completion", {"prompt": "Write a detailed technical explanation of how a PCIe switch routes peer-to-peer transactions between two endpoints.",
                             "n_predict": 512, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False})
    line(f"short run {i + 1}", r["timings"])
    short.append(r["timings"]["predicted_per_second"])
print(f"SHORT median {sorted(short)[len(short) // 2]:.1f} tok/s (min {min(short):.1f}, max {max(short):.1f})", flush=True)

para = ("The PEX88096 is a 98-lane PCIe Gen4 switch. Each downstream port can run at 16 GT/s across 16 lanes, "
        "and peer-to-peer traffic between downstream ports is routed inside the switch without touching the host. ")
for target in (8000, 32000):
    prompt = "Summarise the following notes in one sentence.\n\n" + para * (target // 45) + "\n\nSummary:"
    r = post("/completion", {"prompt": prompt, "n_predict": 256, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False})
    line(f"~{target} ctx", r["timings"])

if LONG:
    recs = []
    for i in range(LONG // 26):
        recs.append(f"Record {i}: sensor {i % 97} reported {(i * 7919) % 1000} units at step {i}; status {'check' if i % 5 == 0 else 'ok'}.")
    prompt = "\n".join(recs) + "\n\nQuestion: which sensor number appears in record 12345? Answer:"
    t = time.perf_counter()
    r = post("/completion", {"prompt": prompt, "n_predict": 128, "ignore_eos": True, "temperature": 0.0, "cache_prompt": False})
    line(f"LONG ~{LONG} ctx ({time.perf_counter() - t:.0f}s wall)", r["timings"])
    print("LONG answer start:", repr(r.get("content", "")[:80]), flush=True)
