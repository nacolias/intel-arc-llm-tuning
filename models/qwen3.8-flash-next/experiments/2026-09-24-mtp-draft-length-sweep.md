# MTP draft length sweep after the verify fix

| | |
|---|---|
| **Date** | 2026-09-24 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | reverted (no consistent gain; n-max 3 kept) |
| **Baseline** | n-max 3 run on the same build in the same session |
| **Result** | [raw/2026-09-24-speed-work.csv](../benchmarks/raw/2026-09-24-speed-work.csv) |
| **Related** | [fused MUL_MAT_ID](2026-09-24-fused-mul-mat-id-mtp-verify.md), [research: decode bottleneck](../research/2026-09-24-decode-bottleneck-and-speed-plan.md) |

## Hypothesis

Before the verify fix, a 4-token verify cost about 2.5x a single-token pass, so deeper drafts lost. With verify now cheap, longer drafts could pay off. A community report on 2x B70 measured +25% over MTP off with n-max 7 and p-min 0.75 (a comment on PR #28243; UNVERIFIED here). `--spec-draft-p-min` stops drafting when the draft's top-1 probability drops below the threshold. Expected: 0% to +10%.

## Change

```diff
- --spec-draft-n-max 3
+ --spec-draft-n-max 4
```

```diff
- --spec-draft-n-max 3
+ --spec-draft-n-max 6 --spec-draft-p-min 0.75
```

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Build `20260924-7e5cb8f13` (PR #28243 at `6fcaa16` + #28931 + 0002 + 0003) |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1024 -ts 13,13,13,10 -fit off`, draft on `SYCL3` |
| GPU clock floor | none (the sweep ran before the floor was added) |

## Procedure

For each arm: set the flags through a drop-in, restart the service (about 77-78 s to load from the page cache; a cold load from the Gen3 NVMe takes about 2.5 minutes, [baseline benchmark](../benchmarks/2026-09-24-baseline-262k-mtp.md)), then run [`tools/fnbench.py`](../../../tools/fnbench.py): 5 short runs of 512 tokens (median reported), a 9,749-token and a 39,119-token prompt with 256 tokens decoded. Greedy, `ignore_eos`, no prompt cache. One pass per arm.

## Results

| Arm | Short (median of 5, min-max) | ~10k | ~39k | Acceptance ~10k | Acceptance ~39k |
|---|---|---|---|---|---|
| n-max 3 (reference) | 43.6 (42.3-45.6) | 53.6 | 44.0 | 186/206 (90%) | 180/223 (81%) |
| n-max 4 | 43.6 (40.1-45.8) | 52.7 | 45.0 | 192/252 (76%) | 190/259 (73%) |
| n-max 6, p-min 0.75 | 46.2 (44.9-50.1) | 57.1 | 35.7 | 189/229 (83%) | 187/239 (78%) |

| Metric | n-max 3 | n-max 6 + p-min 0.75 | Delta |
|---|---|---|---|
| Single-stream tok/s, short | 43.6 | 46.2 | +6% |
| Single-stream tok/s, ~10k | 53.6 | 57.1 | +7% |
| Single-stream tok/s, ~39k | 44.0 | 35.7 | -19% |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

- n-max 4 is within noise of n-max 3 everywhere; it accepts the same number of tokens from more drafts.
- n-max 6 with p-min 0.75 gains 6-7% at short and ~10k context, inside the ~5% run-to-run spread plus one run, and loses 19% at ~39k.
- The ~10k and ~39k prompts repeat one paragraph, which flatters long drafts.

Earlier data, before the verify fix, from the same day's service journal with the config inferred from the mean accepted length (UNVERIFIED): n-max 3 gave 33.5-36.4 / 44.0 / 29.2 tok/s (short / 10k / 39k), n-max 2 gave 32.5-37.8 / 42.1 / 34.0, and n-max 4 or more gave 31-38 / 38.0 / 29.5. Rows `draft-sweep-before-fix` in [raw/2026-09-24-speed-work.csv](../benchmarks/raw/2026-09-24-speed-work.csv). n-max 2 was not rerun after the fix.

## Correctness

Draft length changes only what is proposed. The target model verifies every draft token, so greedy output should not depend on it. Outputs were not compared, and greedy output on this model is not token-identical run to run in any case (MTP batching changes summation order).

## Decision

Reverted to n-max 3, without p-min. No arm gained consistently, and the best short-context arm regressed 19% at ~39k.

## Follow-ups

- [ ] Rerun n-max 2 and a p-min sweep (0.5, 0.7, 0.85) on the current build, with at least 3 passes per arm and the agent-session workload ([`tools/lcbench.py`](../../../tools/lcbench.py)), where acceptance is lower than on repeated paragraphs.
