# Baseline 262K + MTP (unpatched llama.cpp): fnbench benchmark

| | |
|---|---|
| **Date** | 2026-09-24 |
| **Model / checkpoint** | unsloth `UD-Q4_K_XL` (stock, 4 shards) + unsloth `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Config** | [llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh) with `UB=1024 TS=13,13,13,10 MMPROJ=` (no vision projector), unpatched build (no sparse FA), no clock floor |
| **Experiment** | [2026-09-24-baseline-llamacpp-mtp-262k.md](../experiments/2026-09-24-baseline-llamacpp-mtp-262k.md) |
| **Raw data** | [raw/2026-09-24-baseline-fnbench.csv](raw/2026-09-24-baseline-fnbench.csv), [raw/2026-09-24-baseline-memory.csv](raw/2026-09-24-baseline-memory.csv) |

## Result

The unpatched build serves 262,144 tokens with MTP at 34-37 tok/s on short prompts, 43 at ~10k, 33 at ~39k and 16 at 219K. The busiest card peaks at 27,420 MiB. MTP beats MTP off by 38% to 124% at depth on these prompts.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. [PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) at `6fcaa16`, unpatched. SYCL, Level Zero allocator, FP16 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0; oneAPI 2026.1 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| KV cache dtype / block size | f16 K and V, allocated at load for 262,144 tokens |
| Speculative decoding | MTP draft, `--spec-draft-n-max 3`, draft on `SYCL3` |
| Graph / compile mode | eager (SYCL graphs never engage with more than one device) |

Launch flags: `-dev SYCL0,SYCL1,SYCL2,SYCL3 -sm layer -ngl 99 -ts 13,13,13,10 -ot per_layer_token_embd=CPU -c 262144 -np 1 -fa on -ctk f16 -ctv f16 -b 2048 -ub 1024 -t 8 --jinja -fit off --spec-type draft-mtp --spec-draft-model <mtp> --spec-draft-n-max 3 -devd SYCL3`. PCIe uplink at Gen4 x16.

## Workload

| | |
|---|---|
| Workload set | [`tools/fnbench.py`](../../../tools/fnbench.py); not yet one of the shared [workloads](../../../benchmarks/workloads/) |
| Input length | short: a 19-token one-line technical prose prompt; ~10k: 9,749 tokens; ~39k: 39,119 tokens (one two-sentence paragraph repeated, then "summarise"); long: 219,217 tokens of structured records with a lookup question |
| Output length | 512 (short), 256 (~10k, ~39k), 128 (long); `ignore_eos` |
| Sampling | greedy (temperature 0) |
| Warm-up | one sanity chat question before measuring |
| Repetitions | short: 3 runs; others: 1 run |

Command:

```bash
BASE=http://127.0.0.1:8080 VLLM_API_KEY="$(cat "$API_KEY_FILE")" SHORT_REPS=3 LONG_TOKENS=200000 python3 tools/fnbench.py
```

The repeated-paragraph prompts and the record list are easy to predict, so they flatter MTP acceptance. Short-context acceptance is the more realistic figure.

## Results

| Concurrency | Aggregate tok/s | Per-stream tok/s | TTFT p50 | TPOT p50 | Acceptance |
|---|---|---|---|---|---|
| 1 (short, 3 runs) | 33.8-36.9 | 33.8-36.9 | 0.4-0.5 s | not recorded | 58-65% |
| 2 and up | not applicable: one slot (`-np 1`) | | | | |

| Context length | Prompt time (cold) | Prompt tok/s | Single-stream decode tok/s | Acceptance |
|---|---|---|---|---|
| 9,749 | 18.3 s | 531.6 | 43.3 | 185/210 (88%) |
| 39,119 | 74.8 s | 523.3 | 33.4 | 183/214 (86%) |
| 219,217 | 861.3 s | 254.5 | 16.1 | 87/119 (73%) |

MTP-off reference, same build and context, `-ts 12,12,12,13`: 33.1-33.5 tok/s short, 31.4 at ~10k, 23.4 at ~39k, 7.2 at 219K; prompt 568.8 / 570.6 / 274.8 tok/s.

## Observations

- Peak VRAM per card at 219K: 27,420 / 26,671 / 26,938 / 25,360 MiB. VRAM does not grow as the context fills, because the KV cache is allocated at load.
- Driver-held host RAM peaked at 17.5 GiB; MemAvailable stayed at or above 99.8 GiB of 121 GiB.
- With `-ub 2048`, the memory fitter and MTP, the last card overflowed (32,636 MiB), driver-held host RAM climbed to 57.8 GiB and decode fell to 0.6 tok/s. See [findings/xe-vram-overcommit-spills-into-host-ram.md](../../../findings/xe-vram-overcommit-spills-into-host-ram.md).
- A profile of the same config (huihui checkpoint, same tensor layout) showed the server's main thread at 96% of one core and each card's compute engine busy 16-18% during decode ([raw/2026-09-24-decode-profile.csv](raw/2026-09-24-decode-profile.csv)).
- Loading takes about 2.5 minutes, mostly reading 111 GB from a PCIe Gen3 NVMe drive.
