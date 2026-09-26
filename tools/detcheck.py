#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Is greedy decoding reproducible from an identical state?

Restores the slot file DET_SLOT_NAME (a file name inside the server's --slot-save-path), decodes 32
greedy tokens (ignore_eos) of a new assistant turn, restores the same file again and repeats the
request, DET_RUNS times; prints how many leading tokens each run shares with the first. On our
Qwen3.8-Flash-Next build they shared 32, 19 and 1 (findings/llama-cpp-sycl-greedy-not-reproducible.md).
No text is printed.

Environment:
  BASE            server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY    API key sent as a bearer token; leave unset for a server without --api-key
  DET_SLOT        path of that slot file, to read the token IDs (required)
  DET_SLOT_NAME   its file name for the restore action (default: the base name of DET_SLOT)
  DET_RUNS        runs (default 3)
"""
import json, os, struct, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
RUNS = int(os.environ.get("DET_RUNS", "3"))
SLOT = os.environ["DET_SLOT"]
NAME = os.environ.get("DET_SLOT_NAME", os.path.basename(SLOT))


def slot_tokens(path):
    with open(path, "rb") as f:
        _magic, _ver, n = struct.unpack("<III", f.read(12))
        packed = struct.unpack("<%di" % n, f.read(4 * n))
    return list(packed) if packed[0] != -1 else list(packed[3:3 + packed[2]])


def call(path, body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=3600))


prompt = slot_tokens(SLOT) + [248045, 74455, 198]  # <|im_start|>assistant\n (Qwen3.8 vocabulary)
outs = []
for k in range(RUNS):
    call("/slots/0?action=restore", {"filename": NAME})
    r = call("/completion", {"prompt": prompt, "n_predict": 32, "temperature": 0.0, "ignore_eos": True,
                             "cache_prompt": True, "return_tokens": True})
    outs.append(r.get("tokens", []))
    m = next((i for i, (a, b) in enumerate(zip(outs[0], outs[-1])) if a != b), min(len(outs[0]), len(outs[-1])))
    print(f"run {k}: {len(outs[-1])} tokens, {m} leading tokens shared with run 0", flush=True)
