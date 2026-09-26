# Production build `20260926-2b84213a4`: real coding-agent traffic, from the server journal

| | |
|---|---|
| **Date** | 2026-09-26 (one server instance, 116 requests over 10.4 hours) |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Config** | [llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh) (`-ub 1536 -ts 12,13,13,11`, `-b 2048`, `--cache-ram 24576`, one slot), GPU clock floor 2800 MHz |
| **Experiment** | none: production as deployed. Follow-ups: [MTP draft settings under the production sampler](../experiments/2026-09-26-mtp-draft-settings-sampled-replay.md), [decode device profile at 133K](2026-09-26-decode-device-profile-133k.md), [idle slot autosave](../experiments/2026-09-26-slot-autosave.md) |
| **Raw data** | [raw/2026-09-26-real-agent-traffic.csv](raw/2026-09-26-real-agent-traffic.csv): one row per request, times relative to the first request, no text |

## Result

On real single-user coding-agent traffic, decode is 88.5% of the server's busy time, and 93.0% for requests whose context ended at 100K tokens or more. Prompt processing is small because the prefix cache hits: the median request evaluated 567 new prompt tokens in 1.92 s. The median reply was 410 tokens, but the long tail carries the time: 18 replies of more than 2,048 tokens took 60% of all decode time. MTP (multi-token prediction) draft acceptance under the production sampler (temperature 1.0) was 59.3%, below the 64-72% of the greedy benchmarks. Decode ran at a median 46.1 tok/s over requests of 64 or more tokens, and 45.0 tok/s at 100K or more.

A second cost the per-request timings do not show: the agent alternates between its main conversation and side conversations (subagents), and every switch moves the displaced conversation into the host-RAM prompt cache (`--cache-ram`) and the returning one back. 42 of the 116 requests started with such a switch, and each spent a median 5.07 s between llama-server picking the slot and starting the task: 200 s in all, 5.7% of the server's busy time.

So for this workload the lever is decode speed at long context, then the conversation switch; cold prompt reads matter only after a restart.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Frozen build `20260926-2b84213a4`: PR #28243 at `6fcaa16` plus patches 0002-0017 ([configs/patches/](../configs/patches/README.md)) |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0; oneAPI 2026.1 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| KV cache dtype / block size | f16 K and V, 262,144 cells |
| Speculative decoding | MTP draft, `--spec-draft-n-max 3`, draft on `SYCL3`, draft QSA (Qwen sparse attention) on |
| Graph / compile mode | eager; `GGML_SYCL_SPARSE_FA=1`; pooled QSA key cache on |

## Workload

| | |
|---|---|
| Workload set | real traffic: one user's coding-agent sessions (a main conversation plus subagent conversations), not a benchmark; the table below excludes nothing, and no request matched the benchmark patterns [`tools/turnlog.py`](../../../tools/turnlog.py) flags (decode lengths of 1/16/128/192/256/512 tokens or a ~134.8K cold read) |
| Input length | contexts of 4,916 to 135,226 tokens at the end of a request |
| Output length | as the agent asked: median 410, p75 1,239, p90 2,956, max 10,058 tokens |
| Sampling | the server's defaults, which the agent did not override: temperature 1.0, top-k 20, top-p 0.95, min-p 0.05 (read from `/slots`) |
| Warm-up | none; the first request followed a restore of the saved slot |
| Repetitions | not applicable |

Command, on the server host:

```bash
journalctl -u <unit> --since "<instance start>" -o short-iso | python3 tools/turnlog.py > turns.csv
journalctl -u <unit> --since "<instance start>" -o short-iso | python3 tools/switchcost.py
```

`turnlog.py` reads llama.cpp's per-task timing lines, draft acceptance lines, slot release lines and slot-selection lines; `switchcost.py` measures the time from a "selected slot" line to the next "launch_slot_" line. Neither reads prompt or output text, which these log lines do not contain.

## Results

| Measure | All 116 requests | 36 requests ending at >=100K tokens | Source |
|---|---|---|---|
| Share of server time in decode (decode / (decode + prompt)) | 88.5% | 93.0% | raw CSV `decode_ms`, `prompt_ms` |
| Decode tokens per request: median / p75 / p90 / max | 410 / 1,239 / 2,956 / 10,058 | 778 / 1,411 / 3,811 / 10,058 | `decode_n` |
| Decode seconds per request: median / p90; total | 8.8 / 59.1; 2,936 s | 17.1 / 74.1; 1,186 s | `decode_ms` |
| Prompt seconds per request: median / p90; total | 1.92 / 7.53; 380 s | 1.42 / 4.16; 90 s | `prompt_ms` |
| New prompt tokens per request: median / p90 / max | 567 / 3,322 / 12,792 | 260 / 1,221 / 10,094 | `prompt_new` |
| MTP draft acceptance, production sampler | 81,250 / 136,980 = 59.3% | 32,647 / 54,381 = 60.0% | `draft_acc`, `draft_gen` |
| Decode tok/s over requests of >=64 tokens: median (p10-p90) | 46.1 (38.9-55.4) | 45.0 (38.9-52.6) | `decode_tps` |
| Requests that re-read more than 5,000 prompt tokens | 5 (101 s in all) | 1 (29 s) | `prompt_new`, `prompt_ms` |

Where the decode time goes, by reply length:

| Reply length | Requests | Share of decode time | Acceptance | Decode tok/s |
|---|---|---|---|---|
| under 256 tokens | 37 | 4% | 69.4% | 48.4 |
| 256-767 | 32 | 9% | 63.7% | 45.8 |
| 768-2,047 | 29 | 27% | 59.7% | 43.5 |
| 2,048 and more | 18 | 60% | 57.8% | 42.4 |

Short replies (tool calls) draft well; long replies, which are mostly reasoning, draft worse and dominate the time.

Conversation switches (source: `slot_pick`, `f_keep` and `pick_to_launch_s` columns):

| | Value |
|---|---|
| Requests that started with a switch (slot picked by LRU, or kept under half of the cached prompt) | 42 of 116 (33 by LRU) |
| Of those, switches into or out of a conversation of 50K tokens or more | 32 |
| Time from slot pick to task start on a switch: median / p90 / max | 5.07 / 6.20 / 6.34 s |
| The same without a switch | 0.00 s (74 requests) |
| Total switch time | 200 s, 5.7% of the server's busy time (3,516 s, prompt plus decode plus switches) |

In the busiest stretch the agent alternated between the main conversation (105K-119K tokens) and a subagent conversation (5K-60K tokens) roughly every 10-30 s. Both came back from the host-RAM cache with only their new tokens to evaluate (32-12,792 per switching request, median 567), so the cache worked; the cost is the copy itself. Where those 5 s go is measured in [decode device profile at 133K](2026-09-26-decode-device-profile-133k.md) (swap probe section).

## Observations

- **Acceptance under the production sampler is lower than in greedy benchmarks.** 59.3% here, against 63.8% in a greedy [lcbench](../../../tools/lcbench.py) session of the same code ([`-b 3072` experiment](../experiments/2026-09-26-batch-3072.md), baseline row) and 64-72% in earlier greedy sessions. The server samples every verified position at temperature 1.0 and accepts a draft token only if the sample equals it, so acceptance falls with the target's entropy. Draft-length choices tuned on greedy runs (the 2026-09-24 sweep in [raw/2026-09-24-speed-work.csv](raw/2026-09-24-speed-work.csv)) need re-checking under sampling; see the [draft-settings replay](../experiments/2026-09-26-mtp-draft-settings-sampled-replay.md).
- **The 2026-09-25 cancelled re-read was stopped by the client, not a proxy.** The server journal of that instance (not archived) shows the 166K-token re-read starting at 22:47:31 UTC, progress 0.88 (146,100 tokens) at 500.66 s, and `stop: cancel task` at 22:55:58, 507 s after the start. That is not a round timeout. Both proxies in the path are ruled out: the reverse proxy's read and send timeouts are 1,800 s, and the forwarder behind it copies bytes over plain TCP with no timeout. A second request 4 s later was cancelled after 0.5 s. That pattern looks like the user giving up by hand (estimate). The partial prefill survived the cancel: the slot was released holding 148,148 tokens, and the next request found it by prefix similarity (`f_sim` 0.998). The slot save/restore added on 2026-09-25 removes the restart-induced cold read; the remaining exposure is a crash or kill between saves, which the [idle autosave](../experiments/2026-09-26-slot-autosave.md) covers.
- A turn's prompt time includes rolling the hybrid model's recurrent state back to a context checkpoint when the new prompt diverges inside the cached tokens (for example when the chat template re-renders the previous reply). In the [draft-settings replay](../experiments/2026-09-26-mtp-draft-settings-sampled-replay.md), re-running a 95K-134K prompt after a reply of up to 768 tokens took 0.16-0.19 s for the 4 re-evaluated tokens, so the rollback is cheap.
