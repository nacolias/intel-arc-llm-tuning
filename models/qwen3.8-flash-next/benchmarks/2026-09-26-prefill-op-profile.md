# Prefill op profile during a 135K cold read: builds `993baf141` and `85b26781b`

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Config** | [llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh) defaults (`-b 2048 -ub 1536`), GPU clock floor 2800 MHz, plus `GGML_SYCL_OP_PROFILE=<seconds>` with the flag file `/tmp/ggml-sycl-op-profile` ([patch 0004](../configs/patches/0004-sycl-op-profile.diff)) |
| **Experiment** | [MTP `eh_proj` as one 2D product](../experiments/2026-09-26-mtp-eh-proj-2d-product.md); QSA score expansion by broadcast ([patch 0011](../configs/patches/0011-qwen4exp-qsa-score-expansion-by-broadcast.diff)); grouped MoE (mixture-of-experts) XMX (Xe Matrix Extensions) GEMM (general matrix multiply) ([patches 0012-0015](../configs/patches/)) |
| **Raw data** | [raw/2026-09-26-prefill-op-profile.csv](raw/2026-09-26-prefill-op-profile.csv); window timing in [raw/2026-09-26-cold-read-windows.csv](raw/2026-09-26-cold-read-windows.csv) rows `bgzher66i` and `begg31572`; progress lines in [raw/2026-09-26-cold-read-progress.csv](raw/2026-09-26-cold-read-progress.csv) |

## Result

On build `85b26781b` (grouped MoE GEMM with in-place I/O, before the `eh_proj` fix), dense prompt attention is the largest op: `FLASH_ATTN_EXT` over the 12 QSA (Qwen sparse attention) layers took 32.8% of listed op time at 38.8 ms per call. The three MoE matmuls took 35.5% together at 3.4-3.6 ms per call, down from 50.2% at 7.3-8.8 ms per call on `993baf141`. `MUL_MAT mtp_eh_proj` took 115 ms per call on both builds (4.7% and 8.4%); [patch 0016](../configs/patches/0016-qwen4exp-mtp-eh-proj-2d-product.diff) removed it afterwards. On `993baf141` the QSA indexer's `CONT (permuted)` copies took 13.3% at 13.2 ms per call; patch 0011 removed them.

Shares are of serialized op time, not of wall time; see the caveats.

## Environment

| Component | `993baf141` | `85b26781b` |
|---|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. PR #28243 at `6fcaa16` + #28931 + 0002-0004 + #28796 + 0006-0010 (frozen build `20260925-993baf141`) | same + 0011 + 0012 (upstream #29245) + 0013-0015 (sandbox build) |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 | same |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 | same |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 | same |
| oneCCL | not used | |
| PyTorch XPU | not used | |
| Power cap per card | 230 W | |
| KV cache dtype / block size | f16 K and V, 262,144 cells allocated at load | |
| Speculative decoding | MTP (multi-token prediction) draft, `--spec-draft-n-max 3`, draft on `SYCL3`, draft QSA on | |
| Graph / compile mode | eager; `GGML_SYCL_SPARSE_FA=1` (does not engage on prompt batches over 32 rows); pooled QSA key cache on | |
| MoE prompt path | per-expert loop (host-synced) | grouped XMX GEMM, `GGML_SYCL_XMX_GATHER_TYPES` bits 9-12 |

## Workload

| | |
|---|---|
| Workload set | one cold read of the fixed 134,862-token C++ context of [`tools/lcbench.py`](../../../tools/lcbench.py); not yet one of the shared [workloads](../../../benchmarks/workloads/) |
| Input length | 134,862 tokens, read in micro-batches (ubatches) that alternate 1,536 and 512 tokens (`-b 2048 -ub 1536`) |
| Output length | 16 tokens after the read (not profiled) |
| Sampling | greedy |
| Warm-up | none; the profile windows fall in the second half of the read |
| Repetitions | one profiled read per build; 3 and 4 consecutive print windows summed (the CSV's `blocks` column) |

The profiler (patch 0004) times every op on every device, with a queue wait after each op, and prints per-op totals, call counts and shares every `<seconds>` while the flag file exists. The CSV sums the windows of one read and merges the per-layer nodes of one op type (`FLASH_ATTN_EXT node_*`, `TOP_K node_*`, `ADD node_*`, `GATED_DELTA_NET+ node_*`) into one row.

Coverage, from the call counts: `mtp_eh_proj` runs once per ubatch (one draft pass), so the windows cover 10 ubatches on `993baf141` and 22 on `85b26781b`; `FLASH_ATTN_EXT` at 12 calls per ubatch (123 and 255 calls) and `MUL_MAT_ID` at 48 per ubatch (493 and 1,021 calls) agree. With alternating 1,536- and 512-token ubatches that is about 10K and 22K tokens of the read. The profiler ran for 62 s in each read ([raw/2026-09-26-cold-read-windows.csv](raw/2026-09-26-cold-read-windows.csv)): about 297-359 s into the 447.75 s read of `993baf141` (run 23:27:50 UTC) and about 188-250 s into the 295.68 s read of `85b26781b` (run 01:26:45 UTC). Interpolating the server's progress lines ([raw/2026-09-26-cold-read-progress.csv](raw/2026-09-26-cold-read-progress.csv): 92,160 tokens at 265.74 s and 122,880 at 389.93 s in the first read; 92,160 at 173.42 s and 122,880 at 257.63 s in the second) puts both windows at about 100K-120K tokens of context (estimate).

## Results

Top 15 ops per build. `share` is the CSV's `share_pct`: the op's share of the 30 listed ops (see caveat 2). `ms/call` is the mean over 1,536- and 512-token ubatches and, for `node_*` rows, over the 12 layers.

### `993baf141` (per-expert MoE loop, gather-based QSA score expansion; 3 windows, 24.4 s of listed op time)

| Op | Total ms | Share | Calls | ms/call |
|---|---|---|---|---|
| `MUL_MAT_ID ffn_moe_gate` | 4,357.3 | 17.8% | 493 | 8.84 |
| `MUL_MAT_ID ffn_moe_down` | 4,287.8 | 17.5% | 493 | 8.70 |
| `FLASH_ATTN_EXT` (12 QSA layers, dense) | 4,190.3 | 17.1% | 123 | 34.07 |
| `MUL_MAT_ID ffn_moe_up` | 3,600.5 | 14.7% | 493 | 7.30 |
| `CONT (permuted)` (QSA score expansion) | 3,256.5 | 13.3% | 246 | 13.24 |
| `MUL_MAT mtp_eh_proj` | 1,152.1 | 4.7% | 10 | 115.21 |
| `CPY attn_inp_kq_mask` | 485.6 | 2.0% | 123 | 3.95 |
| `ADD indexer_score_tokens` | 310.9 | 1.3% | 123 | 2.53 |
| `MUL_MAT linear_attn_out` | 251.6 | 1.0% | 370 | 0.68 |
| `MUL_MAT z` | 221.5 | 0.9% | 370 | 0.60 |
| `MUL ffn_moe_weighted` | 213.6 | 0.9% | 493 | 0.43 |
| `DSV4_HC_PRE hc_mixed` | 212.8 | 0.9% | 986 | 0.22 |
| `TOP_K` (12 layers) | 204.4 | 0.8% | 63 | 3.24 |
| `MUL_MAT hc_gate` | 172.5 | 0.7% | 996 | 0.17 |
| `DSV4_HC_POST hc_combine` | 170.1 | 0.7% | 493 | 0.35 |

MoE matmuls together: 50.2%. Attention: 17.1%.

### `85b26781b` (grouped MoE GEMM with in-place I/O, broadcast score expansion; before the `eh_proj` fix; 4 windows, 30.1 s of listed op time)

| Op | Total ms | Share | Calls | ms/call |
|---|---|---|---|---|
| `FLASH_ATTN_EXT` (12 QSA layers, dense) | 9,885.4 | 32.8% | 255 | 38.77 |
| `MUL_MAT_ID ffn_moe_down` | 3,641.4 | 12.1% | 1,021 | 3.57 |
| `MUL_MAT_ID ffn_moe_gate` | 3,569.2 | 11.8% | 1,021 | 3.50 |
| `MUL_MAT_ID ffn_moe_up` | 3,490.5 | 11.6% | 1,021 | 3.42 |
| `MUL_MAT mtp_eh_proj` | 2,531.8 | 8.4% | 22 | 115.08 |
| `CPY attn_inp_kq_mask` | 1,119.1 | 3.7% | 255 | 4.39 |
| `MUL_MAT linear_attn_out` | 510.7 | 1.7% | 766 | 0.67 |
| `MUL_MAT z` | 468.7 | 1.6% | 766 | 0.61 |
| `MUL ffn_moe_weighted` | 464.1 | 1.5% | 1,021 | 0.45 |
| `TOP_K` (12 layers) | 459.4 | 1.5% | 135 | 3.40 |
| `DSV4_HC_PRE hc_mixed` | 439.7 | 1.5% | 2,042 | 0.22 |
| `DSV4_HC_POST hc_combine` | 361.2 | 1.2% | 1,021 | 0.35 |
| `MUL_MAT hc_gate` | 358.5 | 1.2% | 2,064 | 0.17 |
| `DSV4_HC_POST l_last` | 346.7 | 1.2% | 1,000 | 0.35 |
| `CONT (view)` | 293.1 | 1.0% | 1,042 | 0.28 |

MoE matmuls together: 35.5%. Attention: 32.8%. `CONT (permuted)` is gone.

### What changed between the builds

| Op | `993baf141` | `85b26781b` | Cause |
|---|---|---|---|
| `MUL_MAT_ID` gate / down / up, ms per call | 8.84 / 8.70 / 7.30 | 3.50 / 3.57 / 3.42 | grouped XMX GEMM for Q4_K, Q5_K, Q5_1 and Q8_0 experts (patches 0012-0015). The kernel microbenchmark for one 1,536-token ubatch, 10 of 512 experts used, went q4_K 11.31 to 3.93 ms, q5_K 11.94 to 4.44, q5_1 11.02 to 3.78, q8_0 10.94 to 7.40 ([raw/2026-09-26-mul-mat-id-kernel-perf.csv](raw/2026-09-26-mul-mat-id-kernel-perf.csv)); the profile's means are lower because half the profiled ubatches have 512 tokens |
| `CONT (permuted)`, 13.24 ms per call, 2 per QSA layer per ubatch | 13.3% | absent | patch 0011: score expansion by broadcast instead of a row gather over a transposed copy plus a transpose back |
| `MUL_MAT mtp_eh_proj`, ms per call | 115.21 | 115.08 | unchanged here; fixed by patch 0016 after this profile ([experiment](../experiments/2026-09-26-mtp-eh-proj-2d-product.md)) |
| `FLASH_ATTN_EXT`, ms per call | 34.07 | 38.77 | kernel unchanged (dense TILE kernel; the sparse path stops at 32 query rows). Both windows sit at about 100-120K of context, the second a little later in the read, which is consistent with a 14% higher per-call time (estimate) |
| `CPY attn_inp_kq_mask`, ms per call | 3.95 | 4.39 | the `[n_kv x n_q]` f16 mask copied once per QSA layer per ubatch; scales with context |

## Observations

1. **Serialized op time.** The profiler waits on the queue after every op, so overlap between ops, between devices and between the host and the GPUs is removed. Use the shares, not the totals, and expect the real wall-time split to weight small ops less. In the earlier decode profile the profiler slowed decode from 22.3 to 20.1 tok/s ([agent-session benchmark](2026-09-25-agent-session-135k.md)). The durations of the profiled reads are not used as timings anywhere in this repository.
2. **Denominator.** The CSV's shares are of the 30 listed ops, which sum to 24.4 s (`993baf141`) and 30.1 s (`85b26781b`). The profiler's own totals for the same windows were 28.7 s and 39.0 s (source: server journal excerpt, not archived), so the listed ops cover about 85% and 77% of profiled op time. Against the profiler's total, every share above is lower by that factor: for example attention about 25% and `mtp_eh_proj` about 6.5% of the `85b26781b` windows.
3. **The `85b26781b` profile predates the `eh_proj` fix.** With `mtp_eh_proj` removed from the listed ops, attention is 35.8% and the MoE matmuls 38.8% of what remains (derived from the CSV). The production README's "about a third of its GPU time in attention and a third in the MoE matmuls" at ~100K refers to this state.
4. **Attention scales with context; MoE does not.** The two profiles sit at similar contexts (about 100-120K, see Workload), but the attention/MoE ratio is not a fixed property of a build: it depends on where in the read a window falls. Over a whole cold read from zero, attention's share is smaller than at the profiled point; the [tile-union note](../research/2026-09-26-sparse-prompt-attention-tile-union.md) discusses how much smaller and why that is unresolved.
5. **What is left after both fixes** (from the `85b26781b` list without `eh_proj`): dense prompt attention, the three MoE GEMMs, the mask copy (3.7%), `TOP_K` (1.5%), the linear-attention products `linear_attn_out` and `z` (1.6-1.7% each) and the hyper-connection ops (`DSV4_HC_*`, `hc_gate`, `hc_inject`: about 1-1.5% each).
6. **No profile of the fixed build (`a890bf8b0` / `2b84213a4`) was taken.** The expectation that `mtp_eh_proj` drops to a few milliseconds per call is UNVERIFIED.
7. **The profiled reads were not slow.** The profiled `85b26781b` read took 295.68 s, the fastest of that build's three reads (312.71 s without the profiler, 314.11 s with the profiler on for the read's last 16 s; [windows CSV](raw/2026-09-26-cold-read-windows.csv)), so cold-read times vary by about 6% run to run on this host and a 62 s profiler window does not dominate them.
8. **Raw file format.** The `build` field of the `85b26781b` rows contains a comma, so it is quoted; every row has seven fields.
