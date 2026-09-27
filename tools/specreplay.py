#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paired A/B of MTP draft settings on a real agent conversation, under the server's own sampler.

What it measures: decode tok/s, MTP (multi-token prediction) draft acceptance and tokens per
verification step for several draft settings (n_max:p_min), on the same real conversation and the
server's production sampling settings (not greedy). Greedy benchmarks overstate acceptance: on our
agent traffic at temperature 1.0 it was about 59% against 64-72% greedy.

Input is a saved llama-server slot (the file written by POST /slots/0?action=save), which holds the
conversation's token IDs. The replay takes the last REPLAY_TURNS assistant turns: each turn's prompt
is the conversation up to that turn's "<|im_start|>assistant\\n" (REPLAY_MARK), so the model writes
the reply it would have written there, reasoning included. Every setting decodes every turn with
the same seed, one after another, in an order rotated per turn. The draft settings change between
requests through a file that the patched build re-reads on every draft call (LLAMA_SPEC_OVERRIDE,
configs/patches/0018); start the server with --spec-draft-n-max at the largest n_max you test.

With a fixed seed, speculative sampling in llama.cpp would emit the same tokens whatever the draft
does, but the verify batch size changes the logits slightly, so outputs diverge within a few hundred
tokens (checked by hash: see the "same output" column). Compare settings on totals over many turns,
not per turn.

The script prints numbers and output hashes only; no conversation text is printed or saved. The
slot file is the conversation: keep it and the extracted tokens private.

Environment:
  BASE            server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY    API key sent as a bearer token; leave unset for a server without --api-key
  REPLAY_SLOT     slot file to read the conversation from (required)
  REPLAY_MARK     comma-separated token IDs that start an assistant turn
                  (default 248045,74455,198: "<|im_start|>assistant\\n" in the Qwen3.8 vocabulary)
  REPLAY_TURNS    number of turns, taken from the end of the conversation (default 16)
  REPLAY_PREDICT  n_predict per request; replies may end earlier (default 768)
  REPLAY_ARMS     settings as n_max:p_min, the first is the baseline (default "3:0 2:0 4:0 3:0.5 4:0.5 5:0.5")
  SPEC_OVR        the override file named by the server's LLAMA_SPEC_OVERRIDE (default /tmp/spec-ovr);
                  written here, so run this on the server host
  REPLAY_OUT      JSON output (default specreplay.json)
  REPLAY_TOUCH    optional file to create after the warm-up read (e.g. a profiler switch)
"""
import hashlib, json, math, os, struct, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
OVR = os.environ.get("SPEC_OVR", "/tmp/spec-ovr")
N_TURNS = int(os.environ.get("REPLAY_TURNS", "16"))
N_PREDICT = int(os.environ.get("REPLAY_PREDICT", "768"))
ARMS = os.environ.get("REPLAY_ARMS", "3:0 2:0 4:0 3:0.5 4:0.5 5:0.5").split()
MARK = [int(x) for x in os.environ.get("REPLAY_MARK", "248045,74455,198").split(",")]
OUT = os.environ.get("REPLAY_OUT", "specreplay.json")


def slot_tokens(path):
    """Token IDs from a llama-server slot file: magic, version, packed length (u32 words), then the
    packed server_tokens (a -1 marker, the format version, the token count, the tokens)."""
    with open(path, "rb") as f:
        _magic, _ver, n = struct.unpack("<III", f.read(12))
        packed = struct.unpack("<%di" % n, f.read(4 * n))
    if packed[0] != -1:          # plain token list, as older servers wrote it
        return list(packed)
    return list(packed[3:3 + packed[2]])


toks = slot_tokens(os.environ["REPLAY_SLOT"])
starts = [i for i in range(len(toks) - len(MARK) + 1) if toks[i:i + len(MARK)] == MARK]
turns = starts[-N_TURNS:]
print(f"conversation: {len(toks)} tokens, {len(starts)} assistant turns; replaying {len(turns)} "
      f"from position {turns[0]}", flush=True)


def post(body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=3600))


def set_arm(arm):
    n, p = arm.split(":")
    with open(OVR + ".tmp", "w") as f:
        f.write(f"{n} {p}\n")
    os.replace(OVR + ".tmp", OVR)


rows = []
set_arm(ARMS[0])
t0 = time.time()
r = post({"prompt": toks[:turns[0] + len(MARK)], "n_predict": 1, "cache_prompt": True})
print(f"warm-up: read {r['timings']['prompt_n']} tokens in {r['timings']['prompt_ms'] / 1000:.1f} s", flush=True)
if os.environ.get("REPLAY_TOUCH"):
    open(os.environ["REPLAY_TOUCH"], "w").close()

for k, pos in enumerate(turns):
    prompt = toks[:pos + len(MARK)]
    order = ARMS[k % len(ARMS):] + ARMS[:k % len(ARMS)]
    for arm in order:
        set_arm(arm)
        time.sleep(0.2)
        r = post({"prompt": prompt, "n_predict": N_PREDICT, "seed": 1000 + k, "cache_prompt": True})
        tm = r["timings"]
        row = dict(turn=k, ctx=len(prompt), arm=arm, prompt_n=tm["prompt_n"], prompt_ms=round(tm["prompt_ms"], 1),
                   n=tm["predicted_n"], ms=round(tm["predicted_ms"], 1), tps=round(tm["predicted_per_second"], 2),
                   acc=tm.get("draft_n_accepted", 0), drafted=tm.get("draft_n", 0),
                   out=hashlib.sha1(r["content"].encode()).hexdigest()[:12])
        rows.append(row)
        print(f"turn {k:2d} ctx {row['ctx']:6d} arm {arm:6s} | {row['n']:4d} tok {row['tps']:6.2f} tok/s | "
              f"draft {row['acc']}/{row['drafted']} | prompt {row['prompt_n']} in {row['prompt_ms'] / 1000:.2f} s | {row['out']}",
              flush=True)
        json.dump(dict(arms=ARMS, n_predict=N_PREDICT, rows=rows), open(OUT, "w"), indent=1)

base = ARMS[0]
print(f"\nwall {time.time() - t0:.0f} s; baseline arm {base}")
for arm in ARMS:
    rs = [x for x in rows if x["arm"] == arm]
    n, ms = sum(x["n"] for x in rs), sum(x["ms"] for x in rs)
    acc, dr = sum(x["acc"] for x in rs), sum(x["drafted"] for x in rs)
    ratios, same = [], 0
    for x in rs:
        b = next(y for y in rows if y["turn"] == x["turn"] and y["arm"] == base)
        ratios.append(x["tps"] / b["tps"])
        same += x["out"] == b["out"]
    gm = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
    print(f"arm {arm:6s}: {1000 * n / ms:6.2f} tok/s over {n} tokens | acceptance {acc}/{dr} = {100 * acc / max(dr, 1):.1f}% "
          f"| {n / max(1, n - acc):.2f} tok/step | vs {base}: x{gm:.3f} (turn range {min(ratios):.2f}-{max(ratios):.2f}) "
          f"| same output as {base}: {same}/{len(rs)}")
