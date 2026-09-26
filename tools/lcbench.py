#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Long-context, agent-like benchmark for a llama.cpp llama-server.

What it measures: prompt latency and decode speed of short follow-up turns on top of a long cached
context, which is what a coding agent does. It builds a fixed ~LC_TOKENS-token context from a C++
source corpus (so the context never changes between runs), reads it once with cache_prompt, then runs
LC_TURNS short turns. Each turn appends the previous answer plus a small "tool output" snippet
(120 or 1500 characters, about 60 or 400 new tokens) and decodes LC_PREDICT tokens greedily
(temperature 0, ignore_eos). It prints per-turn prompt latency, decode tok/s and MTP (multi-token
prediction) draft acceptance, and optionally writes the generated texts to LC_OUT (JSON) for parity
checks between builds. Uses only the llama-server native endpoints /tokenize and /completion.

Environment:
  BASE           server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY   API key sent as a bearer token; leave unset for a server without --api-key
  LC_CORPUS      path to the corpus text file (default: lc-corpus.txt next to this script)
  LC_TOKENS      context size in tokens (default 135000)
  LC_TURNS       number of follow-up turns (default 6)
  LC_PREDICT     tokens decoded per turn (default 256)
  LC_OUT         write per-turn timings and generated text to this JSON file
  LC_SALT        text prepended to the first line, so a cached copy of the context cannot be reused
  LC_SAVE_PROMPT write the final prompt (JSON) to this path

Usage:
  LC_CORPUS=lc-corpus.txt LC_TOKENS=135000 LC_TURNS=6 LC_PREDICT=256 LC_OUT=out.json python3 lcbench.py

Building the corpus. Our runs used the llama.cpp sources at commit 6fcaa16 (the head of
https://github.com/ggml-org/llama.cpp/pull/28243): every .c, .cpp, .h and .hpp file under common/,
ggml/src/ggml-cpu/, ggml/src/ggml-sycl/, src/ and tools/server/, in sorted path order (en_US.UTF-8
collation), each preceded by a newline and a "// ===== FILE: <path> =====" line. That is 575 files,
12,732,437 bytes. This rebuilds it byte for byte on Linux with glibc (checked by md5 against the
original; macOS collation sorts some names differently):

  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
  git fetch origin pull/28243/head && git checkout 6fcaa16f4b360649933a54d1f91ad40ed35c0e11
  git ls-tree -r --name-only HEAD -- common ggml/src/ggml-cpu ggml/src/ggml-sycl src tools/server \\
    | grep -E '\\.(c|cpp|h|hpp)$' | LC_ALL=en_US.UTF-8 sort \\
    | while read -r p; do printf '\\n// ===== FILE: %s =====\\n' "$p"; git show HEAD:"$p"; done > lc-corpus.txt

klprobe.py reads the same corpus and builds the same context.
"""
import json, os, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.environ.get("LC_CORPUS", os.path.join(HERE, "lc-corpus.txt"))
TARGET = int(os.environ.get("LC_TOKENS", "135000"))
TURNS = int(os.environ.get("LC_TURNS", "6"))
PREDICT = int(os.environ.get("LC_PREDICT", "256"))
OUT = os.environ.get("LC_OUT", "")


def post(path, body, timeout=7200):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def n_tokens(text):
    return len(post("/tokenize", {"content": text})["tokens"])


corpus = open(CORPUS).read()
# ~3.3 chars per token for C++; take a slice, then trim to the target by bisection on characters
lo, hi = 0, min(len(corpus), int(TARGET * 4.5))
while hi - lo > 2000:
    mid = (lo + hi) // 2
    if n_tokens(corpus[:mid]) < TARGET:
        lo = mid
    else:
        hi = mid
context = corpus[:lo]
# snippets for the turns come from the end of the corpus, which the context never includes
tail = corpus[-60000:]
snippets = [tail[i * 7000: i * 7000 + (1500 if i % 3 == 2 else 120)] for i in range(TURNS)]

print(f"context: {n_tokens(context)} tokens", flush=True)
# LC_SALT: a different first line, so a cached copy of this conversation cannot be reused (cold read)
prompt = (os.environ.get("LC_SALT", "") + "You are a senior engineer reviewing a C++ codebase. The full source is below.\n\n"
          + context + "\n\n### Task\nSummarise the purpose of the last file above in two sentences.\n\n### Answer\n")
results = []


def run(label, prompt, n_predict):
    t = time.perf_counter()
    r = post("/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": True,
                             "ignore_eos": True})
    wall = time.perf_counter() - t
    tm = r["timings"]
    acc = f" | draft {tm['draft_n_accepted']}/{tm['draft_n']}" if tm.get("draft_n") else ""
    print(f"{label}: prompt {tm['prompt_n']} new tok in {tm['prompt_ms'] / 1000:.2f}s | decode {tm['predicted_n']} tok at "
          f"{tm['predicted_per_second']:.1f} tok/s{acc} | wall {wall:.1f}s", flush=True)
    results.append({"label": label, "timings": tm, "content": r["content"]})
    return r["content"]


answer = run("turn 0 (cold read)", prompt, 16)
for k in range(1, TURNS + 1):
    prompt += answer + "\n\n### Tool output\n" + snippets[k - 1] + "\n\n### Task\nExplain what the code in the tool output does and point out one possible bug.\n\n### Answer\n"
    answer = run(f"turn {k}", prompt, PREDICT)

dec = [r["timings"]["predicted_per_second"] for r in results[1:]]
pro = [r["timings"]["prompt_ms"] / 1000 for r in results[1:]]
print(f"SUMMARY: decode median {sorted(dec)[len(dec) // 2]:.1f} tok/s (min {min(dec):.1f}, max {max(dec):.1f}); "
      f"per-turn prompt median {sorted(pro)[len(pro) // 2]:.2f}s (max {max(pro):.2f}s)", flush=True)
if OUT:
    json.dump(results, open(OUT, "w"), indent=1)
if os.environ.get("LC_SAVE_PROMPT"):
    json.dump({"prompt": prompt + answer}, open(os.environ["LC_SAVE_PROMPT"], "w"))
