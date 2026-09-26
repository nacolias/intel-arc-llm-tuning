#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Decode load at long context for timing and profiling, without a cold read.

Continues the conversation held in the server's slot (restore it from DRIVE_SLOT first, for example
with llama-server-slot-cache.sh restore) DRIVE_REQS times, DRIVE_PREDICT tokens each, under the
server's own sampler with a fixed seed per request. Each request opens a new assistant turn at the
end of the previous one (prompt, reply, end of turn, "<|im_start|>assistant\\n"), so nothing is
re-read. Decodes with ignore_eos, so text after an end-of-turn token may repeat and raise acceptance:
use it for step times and profiles, not for acceptance. Prints decode tok/s, acceptance, tokens per
verify step and milliseconds per step; no text.

Environment:
  BASE            server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY    API key sent as a bearer token; leave unset for a server without --api-key
  DRIVE_SLOT      the slot file the slot was restored from (required: supplies the token IDs)
  DRIVE_REQS      requests (default 3)
  DRIVE_PREDICT   tokens per request (default 512)
The chat-template token IDs below are the Qwen3.8 vocabulary's.
"""
import json, os, struct, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
REQS = int(os.environ.get("DRIVE_REQS", "3"))
PREDICT = int(os.environ.get("DRIVE_PREDICT", "512"))


def slot_tokens(path):
    """Token IDs from a llama-server slot file (see specreplay.py)."""
    with open(path, "rb") as f:
        _magic, _ver, n = struct.unpack("<III", f.read(12))
        packed = struct.unpack("<%di" % n, f.read(4 * n))
    return list(packed) if packed[0] != -1 else list(packed[3:3 + packed[2]])


prompt = slot_tokens(os.environ["DRIVE_SLOT"]) + [248045, 74455, 198]  # <|im_start|>assistant\n
headers = {"Content-Type": "application/json"}
if KEY:
    headers["Authorization"] = "Bearer " + KEY
tot_n = tot_ms = tot_acc = tot_dr = 0
for k in range(REQS):
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(
        {"prompt": prompt, "n_predict": PREDICT, "seed": 7 + k, "cache_prompt": True, "return_tokens": True,
         "ignore_eos": True}).encode(), headers=headers)
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    tm = r["timings"]
    n, ms, acc, dr = tm["predicted_n"], tm["predicted_ms"], tm.get("draft_n_accepted", 0), tm.get("draft_n", 0)
    tot_n += n; tot_ms += ms; tot_acc += acc; tot_dr += dr
    print(f"req {k} ctx {len(prompt)}: {n} tok {1000 * n / ms:.2f} tok/s | draft {acc}/{dr} | "
          f"{n / max(1, n - acc):.2f} tok/step, {ms / max(1, n - acc):.1f} ms/step | prompt {tm['prompt_n']} in "
          f"{tm['prompt_ms'] / 1000:.2f} s", flush=True)
    prompt = prompt + list(r.get("tokens", [])) + [248046, 198, 248045, 74455, 198]
print(f"total: {1000 * tot_n / tot_ms:.2f} tok/s over {tot_n} tokens, acceptance {100 * tot_acc / max(1, tot_dr):.1f}%, "
      f"{tot_n / max(1, tot_n - tot_acc):.2f} tok/step, {tot_ms / max(1, tot_n - tot_acc):.1f} ms/step", flush=True)
