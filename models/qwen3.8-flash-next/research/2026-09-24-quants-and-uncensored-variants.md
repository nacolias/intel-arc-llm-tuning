# Flash-Next GGUF quants and uncensored variants that fit 262K context on four B70s

| | |
|---|---|
| **Date read** | 2026-09-24 |
| **Source** | Hugging Face model cards and GGUF headers (links in the table), llama.cpp source at PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) `6fcaa16`, and a VRAM model fitted to our own `xpu-smi` logs |
| **Author / org** | this repository (AI-assisted survey); model authors as linked |
| **Type** | community model survey |
| **Applies to** | Qwen3.8-Flash-Next GGUF files on llama.cpp SYCL, `-sm layer` across 4x Arc Pro B70 32 GB, host [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |

## Summary

The quant was not what stopped 262K context plus MTP (multi-token prediction) from fitting; the micro-batch size was. At 262K with `-ub 2048`, the QSA (Qwen sparse attention) indexer took about 7.1 GiB of compute buffer on every card. Halving `-ub` and splitting layers explicitly with `-ts` freed more memory than any IQ4-class quant would. The huihui-ai abliterated UD-Q4_K_XL has the same tensor layout as unsloth's UD-Q4_K_XL, so it is a drop-in swap with identical VRAM and speed.

Every VRAM figure for an untested file is an estimate from the fitted model, which reproduced the measured configurations to within about 120 MiB per card.

## Measured on our hardware

**262K plus MTP fits at `-ub 1024`.** Measured 2026-09-24, one run each ([baseline benchmark](../benchmarks/2026-09-24-baseline-262k-mtp.md); per-card peaks in the [raw memory log](../benchmarks/raw/2026-09-24-baseline-memory.csv)):

| Config | Busiest card | Short | ~10K | ~39K | 219K | Prompt tok/s |
|---|---|---|---|---|---|---|
| `-ub 1024 -ts 13,13,13,10` + MTP | 27,420 MiB | 34-37 | 43.3 | 33.4 | 16.1 | about 525 |
| `-ub 1024 -ts 12,12,12,13`, no MTP | 26,115 MiB | 33.4 | 31.4 | 23.4 | 7.2 | about 570 |
| `-ub 2048`, no `-ts`, no MTP | 30,949 MiB | 33.4 | 31.8 | 23.1 | – | about 700 |
| `-ub 2048` + MTP | overflowed | 0.6 tok/s: GPU3 ran out of VRAM and driver-held host RAM climbed to 57.8 GiB ([raw memory log](../benchmarks/raw/2026-09-24-baseline-memory.csv)) | | | | |

Decode figures are tok/s. The `-ub 2048` + MTP failure happened because llama.cpp's memory fitter left the draft head out of its sizing. See [xe VRAM overcommit spills into host RAM](../../../findings/xe-vram-overcommit-spills-into-host-ram.md).

**Refusal smoke test, stock versus abliterated.** A 10-prompt mild probe (profanity, dark fiction, adult humour, harm-reduction facts, security training, persuasion), one sample each, plus a 5-question capability check. It is a smoke test, not a quality benchmark. Generated text is not reproduced here.

| File | Refused | Answered with a disclaimer | Capability | Decode short / 10K / 39K |
|---|---|---|---|---|
| unsloth UD-Q4_K_XL (stock) | 2/10 | 2/10 | 5/5 | 34-37 / 43 / 33 |
| huihui-ai abliterated UD-Q4_K_XL | 0/10 | 0/10 | 5/5 | 38-40 / 47 / 38 |

The two files have identical tensor layouts (1,224 tensors, same types) and identical VRAM, so the decode gap is within run-to-run noise (31-38 tok/s for one config on the same day). SHA-256 hashes of both downloads were checked. The abliterated file is the one used in every later benchmark.

## Candidates

Assumptions for every row: 262,144 context, `-ub 1024`, f16 KV cache, `-ts` split, MTP with unsloth's shared-Q8_0 head on the last card. "Max card" is estimated GiB on the fullest card without MTP / with MTP (upper bound in brackets). Quality is KLD (Kullback-Leibler divergence) against the reference each source names. All values in this table are UNVERIFIED unless marked.

| Option | Download | Max card GiB | Quality evidence (from the source) | SYCL decode risk | Uncensoring method and evidence |
|---|---|---|---|---|---|
| [unsloth UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (stock) | 111.3 GB | measured: 27,420 MiB with MTP | KLD 0.047 vs BF16 (unsloth docs) | measured fast: Q4_K experts use the reordered MoE kernel | none |
| [huihui-ai abliterated UD-Q4_K_XL](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) | 111.33 GB, 4 shards | measured: same as stock | same quant; abliteration damage not measured by the author | measured: same as stock | Rank-1 abliteration applied only to the 101 Q8_0 tensors; routed experts untouched. Third-party probe: 0/10 refusals vs 3/10 stock ([discussion #7](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/discussions/7)) |
| [spiritfather heretic-2 i1 IQ4_XS](https://huggingface.co/spiritfather/Qwen3.8-Flash-Next-heretic-2-i1-GGUF) | 125.31 GB, 5 shards | 22.8 / 23.7 (25.5) | KLD 0.047 vs its own Q8_0 (not comparable with unsloth's BF16 figure) | IQ4_XS / IQ4_NL experts miss the reordered kernel | Heretic, per-layer direction ([source model](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2)): 0/100 refusals vs 99/100 base, KL 0.082. **Needs the `compress_ratios` fix below** |
| [spiritfather heretic-2 static Q4_K_M](https://huggingface.co/spiritfather/Qwen3.8-Flash-Next-heretic-2-GGUF) | 140.41 GB | about 26.4 / 27.8 (28.6) | KLD 0.036 vs its own Q8_0, no imatrix | Q4_K experts: fast kernel | same as above; **needs the fix** |
| [cygnal Heretic2-IQ4XS-NGQ4](https://huggingface.co/cygnal/Qwen3.8-Flash-Next-Heretic2-IQ4XS-NGQ4-GGUF) | 98.40 GB, 1 file | about 22.6 / 24.2 (25.1) | no KLD; author's HumanEval+ 84.1/79.3 vs base IQ4_XS 82.3/78.0 | IQ types in trunk and experts: highest risk | Heretic-2 weights |
| [apetersson abliteration LoRA](https://huggingface.co/apetersson/Qwen3.8-Flash-Next-Abliterated-Adapter) + unsloth UD-IQ4_XS | about 94.0 GB | about 21.6 / 22.7 (24.8) | base quant KLD 0.084 vs BF16 | IQ experts plus an F32 LoRA on all 48 `ffn_down_exps`: unmeasured | Abliteration as a GGUF LoRA, switchable per request |
| [orcarouter Uncensored IQ4_XS](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) (gated) or the ungated [davenetdev copy](https://huggingface.co/davenetdev/halogen-qwen3.8-flash-next-uncensored) | 97.47 GB | 22.2 / 23.3 (25.1) | no KLD; source reports MMLU −2.3 | IQ types in trunk and experts | Single-direction abliteration; source reports 1-3% refusals, 50-68% of answers carry a caveat |
| [windowsxp811203 Abliterated Q4_K_M](https://huggingface.co/windowsxp811203/Qwen3.8-Flash-Next-Abliterated-GGUF) | 119.15 GB | 26.4 / 27.6 (28.5) | stock recipe, no imatrix; source: MMLU 86.0 to 84.4 | Q4_K experts: fast kernel | Abliteration; source: AdvBench refusals 99.4% to 0.96% |
| [Navin-Models AD-4.27](https://huggingface.co/Navin-Models/Qwen3.8-Flash-Next-Uncensored-AD-4.27-GGUF) (gated) | 94.53 GB | about 19 / 20 | IQ2_S experts in 36 of 48 layers | IQ2_S / IQ3_S | orcarouter weights; `compress_ratios` unknown |

No uncensored Flash-Next had been published by DavidAU, HauhauCS, mlabonne, bartowski or unsloth (Hugging Face search, 2026-09-24).

## Claims to verify

| Claim | Reported number | Their setup | Status |
|---|---|---|---|
| The QSA indexer compute buffer scales with context x ubatch | 7.1 GiB per card at 262K, `-ub 2048` | our logs | verified-here (measured at `-ub 2048`; `-ub 1024` fit with MTP, table above) |
| `-ub 1024` costs prefill speed | 570-573 tok/s at `-ub 1024` (64K context) vs 700-712 at `-ub 2048` (262K context), no MTP | our runs, 2026-09-24 | verified-here ([baseline benchmark](../benchmarks/2026-09-24-baseline-262k-mtp.md)); later `-ub 1536` recovered 4-6% ([experiment](../experiments/2026-09-25-ubatch-1536.md)) |
| MTP helps at depth | 41.3 vs 32.1 tok/s at ~10K, 39.6 vs 23.3 at ~39K (64K context, `-ub 1024`, MTP on vs off) | our runs, 2026-09-24 | verified-here; the long prompts are repetitive, which flatters MTP |
| Abliterated MTP acceptance is lower | 75% vs 81% stock, n-max 2 | third party on ik_llama.cpp | UNVERIFIED |
| q8_0 KV slows decode at depth | – | community posts | UNVERIFIED; no source found. On a B70, q8_0 was faster at depth in [#26689](https://github.com/ggml-org/llama.cpp/pull/26689). We kept f16 |
| Sparse FA helps long context | 6.0 to 14.6 tok/s at 88K | one B70, [discussion #28695](https://github.com/ggml-org/llama.cpp/discussions/28695) | reproduced-community; our own result is in the [sparse FA experiment](../experiments/2026-09-25-sparse-fa-multi-token.md) |
| IQ quants decode slower on SYCL | IQ4_XS 17.5 vs Q4_K_M 20.6 tok/s; IQ4_NL 5.85 | dense model on a B70, [#21517](https://github.com/ggml-org/llama.cpp/issues/21517) | UNVERIFIED for this model |

## Converter bug: all-zero `compress_ratios`

Many community GGUFs ship `qwen4exp.attention.compress_ratios` as all zeros. The cause is a converter bug: transformers renamed `full_attention` to `qwen_sparse_attention`, and the converter missed it. With zeros, the 12 sparse-attention layers run dense beyond about 2K tokens (`qwen4exp.cpp:1029`), which is not how the model was trained. The quality effect is unmeasured.

Affected (per the survey): all mradermacher files, all spiritfather heretic-2 files, and several smaller uploads. unsloth and huihui-ai files are correct.

Check shard 1 before loading any other file. It should print `[0,0,0,4,0,0,0,4,…]`:

```bash
PYTHONPATH=/path/to/llama.cpp/gguf-py python3 -c "import sys; from gguf import GGUFReader; \
f=GGUFReader(sys.argv[1]).fields['qwen4exp.attention.compress_ratios']; \
print([int(f.parts[i][0]) for i in f.data])" /path/to/models/<shard-00001>.gguf
```

A third-party script in the windowsxp811203 repo rewrites the 48-entry uint32 array in place. Read it before running it and keep a copy of shard 1. `--override-kv` cannot set arrays.

## MTP head

- unsloth's shared-Q8_0 MTP head (2.8 GB) borrows `token_embd` and `output` from the main model. Its layout matches every trunk above.
- huihui-ai's `token_embd` and `output` are byte-identical to unsloth's, so the shared head works unchanged.
- spiritfather's self-contained heretic-2 MTP file contains the stock MTP layer and costs about 0.6 GiB more on the draft card for no gain (UNVERIFIED).
- The fitter cannot size the qwen4exp draft ("failed to measure the memory of the extra model"), so run with `-fit off` and size with `-ts` by hand.

## Relevance

- Q8_0 trunk tensors account for about 4.5 of the 6.33 GB read per token (GGUF headers; see the [roofline](2026-09-24-decode-bottleneck-and-speed-plan.md#roofline-under-layer-split)), so smaller IQ experts save few bytes and risk a slower kernel.
- The heretic-2 files keep the PLE (per-layer embedding) n-gram table at Q8_0: 54.4 GB versus 28.8 GB for the others. It is read lazily from host memory, so resident size is likely lower (UNVERIFIED).
- The base model license is the Qwen Community License 1.0. Two community cards claim Apache-2.0, which conflicts with it; assume the Qwen terms apply.

## Actions

- [x] Test 262K + MTP at `-ub 1024 -ts 13,13,13,10` on the stock file
- [x] Swap in huihui-ai abliterated UD-Q4_K_XL and run the refusal smoke test
- [ ] Benchmark one IQ4_XS file for decode speed on SYCL before choosing an IQ trunk
- [ ] Run a larger refusal and quality battery (KLD against BF16 logits) on any candidate before switching
