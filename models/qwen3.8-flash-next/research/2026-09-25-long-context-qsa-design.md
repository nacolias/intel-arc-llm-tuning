# Long-context decode for qwen4exp: QSA reference semantics, cost model and four designs

| | |
|---|---|
| **Date read** | 2026-09-25 |
| **Source** | reference implementations: Hugging Face transformers [`modeling_qwen4_exp.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) (main, read 2026-09-25) and vLLM v0.30.0 [`qwen4_exp/common/qsa_cache.py`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/qwen4_exp/common/qsa_cache.py) and [`qwen4_exp/amd/mtp.py`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/qwen4_exp/amd/mtp.py); llama.cpp PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.cpp`, `ggml/src/ggml-sycl/topk-radix.cpp`); our op profiles and benchmarks |
| **Author / org** | this repository (AI-assisted design, then implementation of two of the four designs) |
| **Type** | source-code review, cost model and design |
| **Applies to** | Qwen3.8-Flash-Next on llama.cpp SYCL with MTP, 4x Arc Pro B70 layer split, 135K-262K context, host [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |

## Summary

At 135K context, decode is GPU-bound in attention and in the QSA (Qwen sparse attention) indexer, not in the host. llama.cpp's upstream sparse flash attention covers single-token passes only, and with MTP (multi-token prediction) every target pass verifies 4 tokens, so it never engaged. We built two of four designed changes: sparse flash attention for multi-token batches, and a pooled indexer-key cache. Together they took the 135K agent-session decode median from 22-23 tok/s to 39-49 tok/s (the spread follows MTP acceptance), and 219K decode from 18.4 to 40.6 tok/s. Block-level top-k and a sparse MTP draft layer are designed but not built.

## Reference QSA semantics

The 12 sparse-attention layers attend a selected subset of past tokens. For a query at position L, with compression ratio r = 4 and a budget of 2048 tokens (512 blocks):

```
n_c   = (L+1) // 4                                        # complete blocks
k_b   = RoPE(RMSNorm_k(mean_f32(k_raw[4b .. 4b+3])), pos = 4b)   # pooled key, rope at the first member
s_b   = sum_h relu(q_h . k_b) / sqrt(128)                 # for b < n_c
B     = topk(s, min(512, n_c))
cells = {4b .. 4b+3 : b in B}  U  [4*n_c, L]              # tail of at most 3 cells; width <= 2051
```

- Selection of whole blocks plus the tail is in HF `modeling_qwen4_exp.py` around lines 757-765 (`scores.topk(min(self.block_topk, num_complete_blocks))`, then `tail = local_visible_indices[num_complete_blocks * self.compress_ratio:]`).
- vLLM caches one pooled key per complete group of 4 positions (`compressed_qsa_slot_mapping` in `common/qsa_cache.py:161-190`: a slot is valid only when `(pos + 1) % compress_ratio == 0`).
- The pooled key is the fp32 mean of the 4 raw indexer keys (before norm and RoPE), then the key layer norm, then RoPE at the first member's position. The same holds in vLLM's AMD and NVIDIA paths and in sglang.
- The model's MTP layer is also a QSA layer with its own indexer in vLLM (`amd/mtp.py:224-229`).

## Where llama.cpp PR #28243 differs

| Difference | Effect | Changed by us? |
|---|---|---|
| Recomputes every pooled block for every micro-batch | about 7.4 ms/step of CONT slices plus about 5.6 ms of GET_ROWS at 135K (profile below) | yes: pooled key cache (patch 0007) |
| Takes top-k over expanded cells with width 2051 instead of over blocks | when `(q+1) % 4 != 3` and more than 512 blocks are visible, it attends up to 3 cells of a 513th block; at 135K that is 3 of every 4 positions (code reading plus simulation). Effect on logits not measured | no |
| Radix top-k resolves ties at the cut in an unspecified order when tied cells span work-groups (`topk-radix.cpp`) | two runs of the same build can select different cells; every correctness gate needs a same-build noise floor | no |
| MTP draft layer (blk.48) runs dense attention over its own 262,144-cell KV; the draft GGUF has `compress_ratios[48] = 0` but ships `blk.48.indexer.*` weights | about 7.1 ms/step at 135K | no (design below) |

## Cost model versus measurement

**Unit of cost:** one MTP verify step. Every step verifies T = 4 tokens and yields about 2.9-3.2 tokens. At 135K a step took 124-137 ms (median about 135 ms). A fit over 4K-106K has an intercept of about 54 ms, so about 75-81 ms per step is context-dependent. The one measured point at 219K took 153.8 ms per step.

**Host-synced op profile at 134.8K, dense attention** (the profiler adds about 15 ms/step):

| Term | ms/step |
|---|---|
| 12 trunk FLASH_ATTN_EXT calls | about 24 (a T=4 call takes 2.0 ms, a T=1 call 1.76 ms: occupancy- or re-read-bound, not compute-bound) |
| MTP draft flash attention (1 call at T=4, 3 at T=1) | about 7.1 |
| Pooling slice CONTs | about 7.4 |
| Indexer member GET_ROWS | about 5.6 |

**Share profile at 135K** (`GGML_SYCL_OP_PROFILE`, patch 0004, decode turns), before and after sparse flash attention:

| Op | Dense attention | Sparse attention (build `60a598ed8`) |
|---|---|---|
| MUL_MAT_ID `ffn_moe_down` | 9.6% | 11.7% |
| CONT, indexer pooling slices | 8.2% | 10.5% |
| FLASH_ATTN_EXT, MTP draft | 7.3% | 9.4% (1.87 ms/call) |
| MUL_MAT_ID gate / up | 6.7% / 6.4% | 8.0% / 7.7% |
| `result_output` | 4.8% | 6.0% |
| `hc_inject` (hyper-connections) | 4.7% | 5.8% |
| 12 QSA-layer FLASH_ATTN_EXT | about 2.1% each (about 2.0 ms/call), about 25% total | gone from the top list |
| TOP_K | < 0.3% | – |

Profiling barely slowed decode (20.1 vs 22.3 tok/s), so decode at 135K is GPU-bound.

## Designs

| # | Change | Predicted saving per step at 135K (estimate) | Output vs current | Measured | Status |
|---|---|---|---|---|---|
| 1 | Pooled indexer-key cache | 17-26 ms | bitwise identical selection (expected) | about 72-73 ms/step on vs about 94 ms off: about 21 ms | **built, kept** (patch 0007) |
| 2 | Sparse flash attention for multi-token batches | 18-20 ms | numerically equivalent | decode median 22.2 to 34.5 tok/s | **built, kept** (patch 0006) |
| 3 | Top-k over blocks instead of cells | 1-3 ms | differs only at the 513th block and on ties, where it matches the reference | – | designed, not built |
| 4 | QSA for the MTP draft layer | 5-6 ms (about 12 at 262K) | greedy target output unchanged; acceptance may move | – | designed, not built |

### 1. Pooled indexer-key cache

- **Storage.** One F32 tensor per QSA layer, `[128, kv_size/4 + 1, n_stream]`, about 96 MiB per GPU. Row = pos / 4.
- **Watermark per stream.** Lowered in `apply()` (to the lowest block in the micro-batch) and in `seq_rm` (covers MTP rollback). Reset on clear, `seq_cp`, `seq_keep`, `seq_add`/`seq_div`, full `state_read` and `state_drop`. Raised only in `next()` after a successful micro-batch.
- **Update.** Dirty complete blocks are recomputed with the existing op chain and written with `set_rows`. That is 0-1 blocks per verify step.
- **Fast path only** for one stream, one sequence, unique positions and the per-block bias path. Anything else, including image turns with repeated M-RoPE positions, falls back to the old path.
- **Switch.** `LLAMA_QSA_POOLED=0` disables it.
- **Review.** An adversarial review (three lenses plus verification) found no invalidation or equivalence defects and one minor defect (pool size reported as 0 under a no-alloc dry run), which was fixed.

Measured A/B at 135K with sparse attention on in all runs ([experiment](../experiments/2026-09-25-pooled-qsa-key-cache.md)):

| Run | Decode per turn (tok/s) | Median | MTP acceptance |
|---|---|---|---|
| A: pooled on | 47.1 / 51.0 / 46.7 / 50.7 / 48.5 / 45.2 | 48.5 | about 81% |
| B: pooled off | 29.9 / 31.7 / 29.4 / 29.8 / 27.9 / 30.0 | 29.9 | about 60% |
| C: pooled on again | 38.5 / 38.5 / 36.8 / 43.8 / 39.1 / 40.7 | 39.1 | about 60% |

Run A had luckier acceptance. The fair comparison is C against B at the same acceptance: +31%. Per-step time (72-73 vs 94 ms) does not depend on acceptance. VRAM rose by 96 MiB per GPU.

### 2. Sparse flash attention for multi-token batches

Upstream [#28796](https://github.com/ggml-org/llama.cpp/pull/28796) (merged as `cd74ef627`) added SYCL sparse flash attention for single-token queries only (`Q->ne[1] == 1`). CUDA has qwen4 sparse flash attention ([#28770](https://github.com/ggml-org/llama.cpp/pull/28770)) and a multi-token prefill variant for another model ([#29298](https://github.com/ggml-org/llama.cpp/pull/29298)).

Patch 0006 extends SYCL to up to `GGML_SYCL_SPARSE_FA_MAX_Q` = 32 query rows:

- per-row atomic compaction of the finite mask cells;
- per-row gather of K, V and mask;
- rows re-dispatched as sequences (`ne[3] = T`) to the unchanged TILE kernel;
- used only when `T * n_kv_g * 2 <= n_kv`, with `n_kv_g = pad(n_kv_max + 256, 256)` = 2560.

`test-backend-ops` (SYCL0 against CPU): 8/8 qwen4-shaped sparse cases pass, 5 of them new multi-row cases.

Kernel time (`test-backend-ops perf`, head size 256, 2 KV heads, GQA 12, f16 KV, `n_kv_max` 2051):

| KV cells | Query rows | Dense µs/run | Sparse µs/run |
|---|---|---|---|
| 32,768 | 1 | 451.97 | 53.67 |
| 32,768 | 4 | 510.49 | 200.88 |
| 131,072 | 1 | 1728.39 | 54.47 |
| 131,072 | 4 | 2004.42 | 206.05 |
| 262,144 | 1 | 3421.66 | 54.48 |
| 262,144 | 4 | 3969.57 | 209.18 |

Model A/B at 135K in one server session ([experiment](../experiments/2026-09-25-sparse-fa-multi-token.md)): sparse median 34.5, dense 22.2, sparse again 31.3 tok/s.

### 3. Top-k over blocks (not built)

Score whole blocks with a 0 / −inf bias (tail, dead, future and incomplete blocks get −inf), take `top_k(min(512, n_blocks))`, build a block mask with the picks plus the row's tail block, then expand to cells with `get_rows`. It matches the reference exactly at the 513th block and makes selection deterministic. Saving about 1-3 ms (estimate); the main value is conformance.

### 4. QSA for the MTP draft layer (not built)

The reference MTP layer is a full QSA layer with its own indexer, raw-key cache, pooled-key cache and top-k, with the same r = 4 and 2048-token budget. llama.cpp runs it dense. The design makes the draft memory a hybrid indexer cache in an attention-only mode and reuses the trunk's QSA graph builder, gated to batches of 8 tokens or fewer. About 300-350 lines; 64 MiB of indexer cache plus a 32 MiB pool on the draft card. Gate on MTP acceptance only, because greedy target output cannot change.

## Correctness

Each A/B was checked with `tools/klprobe.py`: teacher-forced next-token distributions at 128 positions after the same 134,840-token prefix, compared by top-1 agreement and KL (Kullback-Leibler divergence) over the union of the top-20.

| Pair | Top-1 agreement | KL mean | KL median |
|---|---|---|---|
| sparse A vs sparse C (same build, noise floor) | 115/128 | 0.213 | 0.00196 |
| sparse A vs dense | 112/128 | 0.108 | 0.00178 |
| sparse C vs dense | 113/128 | 0.145 | 0.00242 |
| pooled A vs pooled C (same build, noise floor) | 116/128 | 0.088 | 0.00247 |
| pooled A vs pooled off | 116/128 | 0.136 | 0.00224 |

Both changes sit inside the same-build noise floor.

## Results

| Build | 135K agent-session decode median | 219K decode | Source |
|---|---|---|---|
| `20260924-7e5cb8f13` (dense attention) | 23.3 tok/s | 18.4 (`-ub 1536`; the unpatched build at `-ub 1024` gave 16.1) | [agent-session benchmark](../benchmarks/2026-09-25-agent-session-135k.md) |
| `20260925-60a598ed8` (+ sparse multi-token) | 34.5 (A/B session) | – | [experiment](../experiments/2026-09-25-sparse-fa-multi-token.md) |
| `20260925-f47a6a5f3` (+ pooled key cache) | 48.5 (A/B session) | 40.6 | [production benchmark](../benchmarks/2026-09-25-production-build.md) |

## Claims to verify

| Claim | Reported number | Their setup | Status |
|---|---|---|---|
| Pooled cache saves 17-26 ms/step at 135K | – | our estimate | verified-here: about 21 ms/step |
| Multi-token sparse FA saves 18-20 ms/step at 135K | – | our estimate | verified-here: decode 22.2 to 34.5 tok/s |
| An exact sparse gather on CUDA with MTP on cut verify time only 6.8% | 6.8% | comment on [#28213](https://github.com/ggml-org/llama.cpp/pull/28213) | UNVERIFIED (measured on CUDA, not on a B70) |
| Block-level top-k saves 1-3 ms/step | – | our estimate | UNVERIFIED |
| Draft-layer QSA saves 5-6 ms/step at 135K | – | our estimate | UNVERIFIED |

## Relevance

- The decode-step time jumps by 20-28 ms between about 106K and 112K and reproduces on a fresh server. The cause is unknown; VRAM residency is one candidate (UNVERIFIED).
- MTP draft attention is now the largest attention term (9.4% of decode at 135K).
- ~60-token prompt turns exceed the 32-row sparse limit and still run dense attention; see the [prefill note](2026-09-25-prefill-bottleneck.md).

## Actions

- [x] Op profiler with node names (patch 0004)
- [x] Sparse flash attention for multi-token batches (patch 0006)
- [x] Pooled indexer-key cache (patch 0007)
- [ ] Block-level top-k (reference conformance at the 513th block, deterministic ties)
- [ ] QSA for the MTP draft layer
- [ ] Explain the step-time jump near 106K
