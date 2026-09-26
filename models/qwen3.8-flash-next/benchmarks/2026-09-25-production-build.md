# Production build 20260925-f47a6a5f3: fnbench benchmark

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Config** | [llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh) (defaults), GPU clock floor 2800 MHz |
| **Experiment** | [pooled QSA key cache](../experiments/2026-09-25-pooled-qsa-key-cache.md) |
| **Raw data** | [raw/2026-09-25-production-fnbench.csv](raw/2026-09-25-production-fnbench.csv) |

## Result

The production build decodes at 48.6-50.5 tok/s on short prompts, 60.6-67.2 at ~10k, 53.6-55.2 at ~39k and 40.6 at 219K, across three runs on 2026-09-25. It reads prompts at 568-645 tok/s at 10k-39k and 268 tok/s at 219K. The busiest card peaks at 28,957 MiB with the context filled to 219K.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 + 0004 + #28796 + 0006 + 0007 (build `20260925-f47a6a5f3`); see [configs/patches/](../configs/patches/README.md) |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0; oneAPI 2026.1 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| KV cache dtype / block size | f16 K and V, 262,144 cells allocated at load |
| Speculative decoding | MTP draft, `--spec-draft-n-max 3`, draft on `SYCL3` |
| Graph / compile mode | eager; `GGML_SYCL_SPARSE_FA=1`; pooled QSA key cache on |

Launch flags: `-dev SYCL0,SYCL1,SYCL2,SYCL3 -sm layer -ngl 99 -ts 12,13,13,11 -ot per_layer_token_embd=CPU -c 262144 -np 1 -fa on -ctk f16 -ctv f16 -b 2048 -ub 1536 -t 8 --jinja --temp 1.0 --top-p 0.95 --top-k 20 -fit off --spec-type draft-mtp --spec-draft-model <mtp> --spec-draft-n-max 3 -devd SYCL3 --mmproj <mmproj>`; environment `ZES_ENABLE_SYSMAN=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 GGML_SYCL_SPARSE_FA=1`.

## Workload

| | |
|---|---|
| Workload set | [`tools/fnbench.py`](../../../tools/fnbench.py); not yet one of the shared [workloads](../../../benchmarks/workloads/) |
| Input length | short: a 19-token one-line technical prose prompt; ~10k: 9,749 tokens; ~39k: 39,119 tokens (a two-sentence paragraph repeated, then "summarise"); long: 219,217 tokens of structured records with a lookup question |
| Output length | 512 (short), 256 (~10k, ~39k), 128 (long); `ignore_eos` |
| Sampling | greedy (temperature 0), overriding the server's defaults per request |
| Warm-up | one sanity chat question (answered correctly, 155) |
| Repetitions | short: 5 runs, median reported; others: 1 run per session; 3 sessions |

Command:

```bash
BASE=http://127.0.0.1:8080 VLLM_API_KEY="$(cat "$API_KEY_FILE")" LONG_TOKENS=200000 python3 tools/fnbench.py
```

Repeated paragraphs and record lists are easy to predict and flatter MTP; see [the agent-session benchmark](2026-09-25-agent-session-135k.md) for a code workload at 135K.

## Results

| Concurrency | Aggregate tok/s | Per-stream tok/s | TTFT p50 | TPOT p50 | Acceptance |
|---|---|---|---|---|---|
| 1 (short, median of 5; three sessions) | 48.6 / 49.2 / 50.5 | same | not recorded | about 20 ms | not recorded |
| 2 and up | not applicable: one slot (`-np 1`) | | | | |

| Session | Short (median, min-max) | ~10k prompt / decode | ~39k prompt / decode | 219K prompt / decode |
|---|---|---|---|---|
| deploy (after the build was frozen) | 48.6 (46.4-49.9) | 589.8 / 60.6 (182/219) | 577.2 / 53.6 | 268.3 (816.9 s) / 40.6 (89/112) |
| before the ASPM trial | 49.2 (46.9-52.0) | 644.7 / 67.2 (186/206) | 573.7 / 55.2 (180/225) | not run |
| after enabling ASPM L1 | 50.5 (47.6-51.6) | 641.2 / 66.4 (186/206) | 567.6 / 53.7 (178/230) | not run |

| Context length | TTFT (cold) | Single-stream decode tok/s |
|---|---|---|
| 19 (short) | not recorded | 48.6-50.5 |
| 9,749 | 15.1-16.5 s | 60.6-67.2 |
| 39,119 | 67.8-68.9 s | 53.6-55.2 |
| 219,217 | 816.9 s | 40.6 |
| 262,144 (maximum) | not measured | not measured |

The ~10k decode varies with acceptance: 182/219 in the deploy run against 186/206 in the two later runs.

Earlier builds on the same workload, for comparison (the `20260924-7e5cb8f13` short figure is the median of 3 runs, not 5; source: session log, not archived):

| Build | Short | ~10k | ~39k | 219K | Prompt 10k / 39k / 219K |
|---|---|---|---|---|---|
| `20260924-7e5cb8f13`, `-ub 1536` (dense attention) | 50.3 | 63.6 | 52.6 | 18.4 | 589.9 / 582.8 / 270.0 |
| `20260925-60a598ed8` (sparse FA, no pooled cache) | 49.3 (45.7-51.1) | 63.1 | 49.7 | not run | 590.8 / 582.4 / - |

## Observations

- Peak VRAM during the 219K run: 28,541 / 28,280 / 28,530 / 28,957 MiB. Host MemAvailable never went below 91.5 GiB.
- The ASPM trial changed only PCIe link power management. Decode moved by -2.7% to +2.6% and prefill by -1.1% to -0.5%, which is inside run-to-run noise. Idle power of the four cards fell from 185 W to 27 W. See [findings/bmg-aspm-l1-idle-power.md](../../../findings/bmg-aspm-l1-idle-power.md).
- No new PCIe correctable or uncorrectable error bits and no AER or `xe` errors appeared in the kernel log after enabling ASPM L1.
- Short-context decode is no faster than on the dense build. Sparse FA and the pooled cache only engage at depth; short-context decode is host-bound.
- Provenance of the [raw CSV](raw/2026-09-25-production-fnbench.csv): the short, ~10k and ~39k values of the `sparse` row (`20260925-60a598ed8`) and of the deploy session (`pooled` row), and the per-card VRAM of the `pp-test` row, were read from the service journal, which is not archived. The deploy session's 219K values and the two ASPM sessions come from saved fnbench output.
