# MTP draft length and confidence cut under the production sampler, on a replayed agent conversation

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept (production since 2026-09-26 22:10 UTC) |
| **Baseline** | production draft settings `--spec-draft-n-max 3`, `--spec-draft-p-min` unset (0), as measured on [real agent traffic](../benchmarks/2026-09-26-real-agent-traffic.md) |
| **Result** | [raw/2026-09-26-mtp-draft-replay-768.csv](../benchmarks/raw/2026-09-26-mtp-draft-replay-768.csv) (6 settings, 16 turns, 768-token cap); [raw/2026-09-26-mtp-draft-replay-3072.csv](../benchmarks/raw/2026-09-26-mtp-draft-replay-3072.csv) (3 settings, 8 turns, 3,072-token cap) |
| **Related** | the greedy draft-length sweep of 2026-09-24 in [raw/2026-09-24-speed-work.csv](../benchmarks/raw/2026-09-24-speed-work.csv); [decode device profile at 133K](../benchmarks/2026-09-26-decode-device-profile-133k.md) |

## Result

Stopping the draft where the draft head is unsure made decode faster on real agent turns, under the sampler production uses. Pooled over 24 replayed turns at 94.8K-134.2K context, decode ran:

- **+5.4% with `--spec-draft-n-max 4 --spec-draft-p-min 0.5`** (90% bootstrap interval +2.2 to +9.3%);
- **+4.0% with `--spec-draft-n-max 3 --spec-draft-p-min 0.5`** (+1.3 to +6.2%).

Both are measured against the production setting of 3 drafted tokens with no cut. Changing only the draft length did nothing measurable: 2 or 4 tokens without a cut stayed within ±2%.

The mechanism is the same in both runs. The 0.5 cut drops about a quarter of the drafted tokens, which cost about 8 ms per step each. It loses few accepted ones: tokens per step fell 2-6% with n-max 3, and rose with n-max 4. Production now runs n-max 4 with p-min 0.5.

## Hypothesis

The production draft settings (up to 3 drafted tokens, no confidence cut) were chosen from a greedy sweep at short context on 2026-09-24. Real traffic samples at temperature 1.0, and there the draft is accepted 59% of the time, not the 64-72% of greedy runs ([real agent traffic](../benchmarks/2026-09-26-real-agent-traffic.md)). Each drafted token costs a draft-head pass and a wider verify batch whether it is accepted or not. With lower acceptance, a shorter draft, or a cut that stops drafting when the draft head is unsure (`p_min`: stop when its top candidate's probability, renormalized over its top 10, is below the threshold), should pay off. Expected: a few percent either way; the draft is not most of a step.

## Change

No production change yet. A test hook (patch 0018, [configs/patches/](../configs/patches/README.md)) makes the MTP draft re-read `n_max` and `p_min` from a file on every draft call when `LLAMA_SPEC_OVERRIDE=<file>` is set, capped at the server's `--spec-draft-n-max`. That lets every setting run on the same server session, context and prompt cache.

Settings tested, as `n_max:p_min`: `3:0` (production), `2:0`, `4:0`, `3:0.5`, `4:0.5`, `5:0.5`.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container: production tree `2b84213a4` plus patch 0018 (test hook), built as `2be97a7d9` on 2026-09-26; nothing else differs from the production build |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0; oneAPI 2026.1 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, production flags ([llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh)) plus `--spec-draft-n-max 5` so the hook can go up to 5; MTP draft on `SYCL3`, draft QSA on, `GGML_SYCL_SPARSE_FA=1`, pooled QSA key cache on; GPU clock floor 2800 MHz |

## Procedure

[`tools/specreplay.py`](../../../tools/specreplay.py) on the saved production slot. That slot is a real coding-agent conversation of 135,226 tokens with 40 assistant turns; its text is private and is not in this repository. For each of the last 16 assistant turns, the prompt is the conversation up to that turn's `<|im_start|>assistant\n`, 94,838 to 134,211 tokens. The model then writes the reply it would have written there, reasoning first. Every setting decodes every turn with the same seed, `n_predict` 768, under the server's sampler (temperature 1.0, top-k 20, top-p 0.95, min-p 0.05). The order of settings rotates per turn. Replies may stop early; 41 of 96 did. The first request of turns 1-15 evaluates the turn's new tokens (321 to 10,094); every other request re-evaluates 4 tokens after a checkpoint rollback, in 0.16-0.19 s.

```bash
REPLAY_SLOT=/path/to/slots/slot0.bin SPEC_OVR=/tmp/spec-ovr \
  REPLAY_TURNS=16 REPLAY_PREDICT=768 REPLAY_ARMS="3:0 2:0 4:0 3:0.5 4:0.5 5:0.5" \
  python3 tools/specreplay.py
python3 tools/specsum.py models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-mtp-draft-replay-768.csv 4000
```

The fixed seed was meant to make every setting emit the same tokens, because llama.cpp samples the target at each drafted position and stops at the first mismatch, one random draw per emitted token. It did not hold: only 1 of 16 outputs per setting matched the baseline's hash. The verify batch width changes the logits slightly, and at temperature 1.0 that flips a sample within a few hundred tokens. So the settings decode different text of the same kind, and the comparison rests on totals over 16 turns with a bootstrap interval over turns, not on per-turn pairs.

## Results

Replay capped at 768 tokens per reply, 16 turns, 94.8K-134.2K context ([raw CSV](../benchmarks/raw/2026-09-26-mtp-draft-replay-768.csv); `tools/specsum.py` output):

| Setting `n_max:p_min` | Decode tok/s | Acceptance | Tokens per verify step | ms per step | Drafted per step | vs `3:0` (90% CI over turns) |
|---|---|---|---|---|---|---|
| `3:0` (production) | 40.85 | 55.3% | 2.65 | 65.0 | 2.99 | 1.000 |
| `2:0` | 40.94 | 66.9% | 2.33 | 57.0 | 1.99 | 1.002 (0.967-1.041) |
| `4:0` | 40.35 | 48.9% | 2.95 | 73.0 | 3.98 | 0.988 (0.954-1.022) |
| `3:0.5` | 41.76 | 68.8% | 2.50 | 59.9 | 2.18 | 1.022 (0.993-1.058) |
| `4:0.5` | 42.83 | 64.7% | 2.74 | 64.0 | 2.69 | 1.049 (1.010-1.091) |
| `5:0.5` | 40.94 | 57.7% | 2.79 | 68.1 | 3.10 | 1.002 (0.972-1.033) |

What the step times say (from the table, without a cut):

- A verify step with 2, 3 and 4 drafted tokens took 57.0, 65.0 and 73.0 ms. Each drafted token costs about 8 ms per step, for its draft-head pass plus the wider verify. By extrapolation, a step with no draft would take about 41 ms at this depth (estimate).
- The third and fourth draft positions added 0.32 and 0.30 tokens per step (2.33 to 2.65 to 2.95). At about 8 ms each, that is 25-27 ms per extra token, against 24.5 ms per token for the whole `3:0` step (65.0 / 2.65). So without a cut, the third position just breaks even and the fourth loses.
- A 0.5 cut stops drafting at the draft head's unsure positions. `4:0.5` drafts 2.69 tokens per step instead of 3.98 and keeps most of the accepted ones: 2.74 tokens per step at 64.0 ms, the best tokens per millisecond of the six.

Replay capped at 3,072 tokens per reply, 8 turns, 111.1K-134.2K context, the production setting and the two cut settings ([raw CSV](../benchmarks/raw/2026-09-26-mtp-draft-replay-3072.csv)):

| Setting `n_max:p_min` | Decode tok/s | Acceptance | Tokens per verify step | ms per step | Drafted per step | vs `3:0` (90% CI over turns) |
|---|---|---|---|---|---|---|
| `3:0` (production) | 40.81 | 55.7% | 2.67 | 65.4 | 3.00 | 1.000 |
| `4:0.5` | 43.32 | 66.1% | 2.86 | 66.0 | 2.81 | 1.062 (1.008-1.173) |
| `3:0.5` | 43.04 | 70.7% | 2.62 | 60.9 | 2.29 | 1.055 (1.008-1.074) |

Replies still ended early in most turns. Only 2, 0 and 3 replies per setting reached the 3,072-token cap, so the three settings decoded 9,082, 5,563 and 11,433 tokens. The `4:0.5` total leans on shorter replies, which run faster, so its long-reply figure is the less certain one.

Both replays together (24 turns; bootstrap over turns, each setting against `3:0` on the same resampled turns):

| Setting | Short replay (16 turns) | Long replay (8 turns) | Pooled (24 turns) |
|---|---|---|---|
| `4:0.5` | 1.049 (1.010-1.091) | 1.062 (1.008-1.173) | **1.054 (1.022-1.093)** |
| `3:0.5` | 1.022 (0.993-1.058) | 1.055 (1.008-1.074) | 1.040 (1.013-1.062) |

Tokens per millisecond of step time, which depends less on which text each setting happened to decode, rank the same way in both replays: `4:0.5` 0.0428 and 0.0433, `3:0.5` 0.0417 and 0.0430, `3:0` 0.0408 and 0.0408.

After the change, the saved production slot restored into a server with the new flags (135,226 tokens) and decoded normally. `--spec-draft-n-max` also sets how many recurrent-state snapshots the target keeps for rollback, so this was checked before deploying ([decode driver runs](../benchmarks/raw/2026-09-26-decode-driver-runs.csv), part C).

## Correctness

Draft settings cannot change what the target model samples, only how many verify passes it takes: every emitted token is sampled from the target's own distribution. They do change the batch width of the verify pass, and so the logits by rounding, which is why the outputs diverged; the same happens between any two draft settings. The greedy correctness check of [benchmarks/README.md](../../../benchmarks/README.md) was not re-run for a flag change that leaves the kernels and graphs untouched.

## Decision

Kept: `--spec-draft-n-max 4 --spec-draft-p-min 0.5` in production since 2026-09-26 22:10 UTC, deployed together with the [prompt-cache swap fix](2026-09-26-prompt-cache-swap-speedup.md) as build `20260926-5c258f538`. Rollback is `--spec-draft-n-max 3` without `--spec-draft-p-min`. The expected gain on real traffic is about 5% of decode time, which is 88-93% of the server's busy time on that workload (estimate).

## Follow-ups

- [ ] After a change, check per-position acceptance on real traffic: production has had `--metrics` on since 2026-09-26, which counts accepted tokens per draft position (`llamacpp:` counters on `/metrics`).
- [ ] A cut between 0.3 and 0.7, and `p_min` with `n_max` 6, were not tested.
