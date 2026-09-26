#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cost of swapping the agent's conversation in and out of llama-server's host-RAM prompt cache
(--cache-ram), which happens every time an agent's side request or subagent takes the single slot.

Assumes the slot holds the conversation saved in SWITCH_SLOT (restored from that file).
Alternates an unrelated ~2K-token prompt ("away": the conversation is copied to host RAM) with the
conversation plus "<|im_start|>user\\n" ("back": it is copied back to the GPUs), SWITCH_ROUNDS times,
each with n_predict 1. The time outside prompt eval and decode is the swap (client wall minus the
server's timings; the journal's slot-pick-to-launch gap is printed by switchcost.py). Run
slot-cache.sh restore afterwards to put the exact saved state back. Prints numbers only.

Last, a correctness check: 32 greedy tokens (ignore_eos) of a new assistant turn on the conversation,
repeated once (a checkpoint rollback, no swap), then an unrelated prompt (swap out) and the same
request again (swap back plus the same rollback): the last two must produce the same token IDs.

Environment:
  BASE            server URL (default http://127.0.0.1:8080)
  VLLM_API_KEY    API key sent as a bearer token; leave unset for a server without --api-key
  SWITCH_SLOT     slot file holding the conversation the slot was restored from (required)
  SWITCH_ROUNDS   away/back rounds (default 2)
  SWITCH_OUT      JSON output (default switchprobe.json)
The "<|im_start|>user\\n" and "<|im_end|>\\n" token IDs below are the Qwen3.8 vocabulary's.
"""
import json, os, struct, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8080")
KEY = os.environ.get("VLLM_API_KEY", "")
ROUNDS = int(os.environ.get("SWITCH_ROUNDS", "2"))


def slot_tokens(path):
    """Token IDs from a llama-server slot file (see specreplay.py)."""
    with open(path, "rb") as f:
        _magic, _ver, n = struct.unpack("<III", f.read(12))
        packed = struct.unpack("<%di" % n, f.read(4 * n))
    return list(packed) if packed[0] != -1 else list(packed[3:3 + packed[2]])


conv = slot_tokens(os.environ["SWITCH_SLOT"])
back_prompt = conv + [248045, 846, 198]  # <|im_start|>user\n
away_prompt = ("The quick brown fox jumps over the lazy dog. " * 200) + "\nCount the foxes."


def post(body):
    headers = {"Content-Type": "application/json"}
    if KEY:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(body).encode(), headers=headers)
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r, time.time() - t


rows = []
for k in range(ROUNDS):
    for name in ("away", "back"):
        # "away" differs per round so it never hits the cache; "back" grows by the token generated last
        # time plus a new user marker, so the conversation only ever extends (no rollback needed)
        prompt = away_prompt + f" Round {k}." if name == "away" else back_prompt
        r, wall = post({"prompt": prompt, "n_predict": 1, "cache_prompt": True, "seed": 1, "return_tokens": True})
        if name == "back":
            if not r.get("tokens"):
                raise SystemExit("server did not return the generated token; later rounds would need a rollback")
            back_prompt = back_prompt + list(r["tokens"]) + [248046, 198, 248045, 846, 198]
        tm = r["timings"]
        other = wall - (tm["prompt_ms"] + tm["predicted_ms"]) / 1000
        rows.append(dict(round=k, dir=name, wall=round(wall, 2), prompt_n=tm["prompt_n"],
                         prompt_s=round(tm["prompt_ms"] / 1000, 2), outside_s=round(other, 2)))
        print(f"round {k} {name}: wall {wall:.2f} s | prompt {tm['prompt_n']} tokens in {tm['prompt_ms'] / 1000:.2f} s "
              f"| outside eval {other:.2f} s", flush=True)
        time.sleep(1)
# correctness: greedy tokens after a swap out and back must match the same request before the swap
check_prompt = back_prompt[:-3] + [248045, 74455, 198]  # the pending user marker becomes an assistant turn
body = {"prompt": check_prompt, "n_predict": 32, "temperature": 0.0, "ignore_eos": True, "cache_prompt": True,
        "return_tokens": True}
# first request extends the slot; the repeat rolls back to a checkpoint and re-evaluates the last
# tokens in a smaller batch (which can flip a near-tie); the third does the same after a swap out and
# back. The swap is lossless if the third matches the second.
ref, _ = post(body)
ctl, _ = post(body)
post({"prompt": away_prompt + " Check.", "n_predict": 1, "cache_prompt": True, "seed": 1})
chk, wall = post(body)


def match(x, y):
    return next((i for i, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))


r_t, c_t, k_t = ref.get("tokens", []), ctl.get("tokens", []), chk.get("tokens", [])
print(f"check: after swap out and back vs the same rollback without a swap: "
      f"{'identical' if k_t == c_t else 'differ'} ({match(c_t, k_t)} of {len(c_t)} match); "
      f"rollback vs first request: {match(r_t, c_t)} of {len(r_t)} match; back-swap request wall {wall:.2f} s, "
      f"prompt {chk['timings']['prompt_n']} tokens in {chk['timings']['prompt_ms'] / 1000:.2f} s", flush=True)
rows.append(dict(check_swap_identical=k_t == c_t, check_swap_match=match(c_t, k_t),
                 check_rollback_match=match(r_t, c_t), check_tokens=len(c_t)))
json.dump(rows, open(os.environ.get("SWITCH_OUT", "switchprobe.json"), "w"), indent=1)
