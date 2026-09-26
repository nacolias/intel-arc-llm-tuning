# Greedy decoding of Qwen3.8-Flash-Next on llama.cpp SYCL is not reproducible, even from an identical restored state

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Confidence** | verified-here |
| **Applies to** | llama.cpp PR #28243 at `6fcaa16` plus our patch stack (builds `2b84213a4` and `5c258f538`), SYCL backend on 4 Arc Pro B70, Qwen3.8-Flash-Next with QSA (Qwen sparse attention) sparse flash attention and the MTP draft |
| **Area** | engine |

## Finding

Two greedy (temperature 0) requests, each starting from the same saved slot restored from the same file, can differ as early as the second generated token. So "the output is token-identical" cannot serve as a correctness check for changes on this stack. That includes changes that provably copy state byte for byte. Comparisons need teacher-forced next-token distributions (KL divergence against a same-build noise floor) or byte-level state checks.

## Evidence

- 2026-09-26, production build `2b84213a4`. The saved 135,226-token conversation was restored from disk three times, each followed by the same request: a new assistant turn, 32 greedy tokens, `ignore_eos`. The runs shared 32, 19 and 1 leading tokens with the first run ([raw/2026-09-26-slot-restore-and-start.csv](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-slot-restore-and-start.csv), `greedy_repeat` rows).
- Each run evaluates the same 3 prompt tokens after the restore and produces the same first token. The second token comes from the first draft-and-verify step. If every kernel were deterministic, that step would see identical inputs, propose the same draft and verify it in the same batch shape. The third run still diverged there, so something in the forward pass is not deterministic run to run. Candidates, unconfirmed: the QSA indexer's top-k tie order, and kernels whose reduction order depends on scheduling.
- A restore then save reproduces the slot file byte for byte (target and draft state), so the restored state itself is identical ([prompt-cache swap experiment](../models/qwen3.8-flash-next/experiments/2026-09-26-prompt-cache-swap-speedup.md)).
- Earlier experiments already saw greedy outputs differ between sessions and attributed it to MTP batching and top-k tie order, for example [fused MUL_MAT_ID for MTP verify](../models/qwen3.8-flash-next/experiments/2026-09-24-fused-mul-mat-id-mtp-verify.md) and [pooled QSA key cache](../models/qwen3.8-flash-next/experiments/2026-09-25-pooled-qsa-key-cache.md). This measurement removes batching as the only cause.
- With sampling at temperature 1.0 and a fixed seed, settings that only change the draft also produced different text: 1 of 16 turns matched per setting ([draft-settings replay](../models/qwen3.8-flash-next/experiments/2026-09-26-mtp-draft-settings-sampled-replay.md)).

## Impact

- Paired A/B comparisons decode different text in each arm. They need totals over many turns, with an interval, rather than per-turn pairs.
- Token-match checks give false alarms: an unchanged build fails them. They can also give false passes when a divergence happens to fall late.

## What to do

- For kernel, graph or cache changes, compare teacher-forced next-token distributions against a same-build noise floor ([`tools/klprobe.py`](../tools/klprobe.py)).
- For state save, restore or copy changes, compare the saved bytes: restore, save back, `cmp`.
- For speed A/B tests, rotate the arm order per turn and report a bootstrap interval over turns ([`tools/specsum.py`](../tools/specsum.py)).

## Still open

- Which kernel is nondeterministic. A run with the MTP draft off and dense attention (`GGML_SYCL_SPARSE_FA=0`) would narrow it down; not done.
- Whether upstream llama.cpp SYCL without our patches shows the same on a smaller model.
