# Commit messages of the patch series

The `.diff` files in this folder are plain diffs. This file keeps the commit message of each commit in the series, in apply order, so that numbers the experiment and benchmark files quote from a commit message can be checked. The messages are copied from the development branch as committed (2026-09-23 to 2026-09-26), with two changes: the host's private name is replaced by `quad-b70-5800x-pex88096`, and `Co-Authored-By` trailers are dropped. Numbers in these messages were measured on that host; the experiment files give their raw sources and caveats.

| Order | Commit | File | Kind |
|---|---|---|---|
| 1 | `9fe305c67` (2026-09-23) | not in this folder (upstream commit, see [README.md](README.md)) | upstream cherry-pick (#28931) |
| 2 | `4fb54754f` (2026-09-24) | [0002-sycl-gettid-cache.diff](0002-sycl-gettid-cache.diff) | ours |
| 3 | `7e5cb8f13` (2026-09-24) | [0003-sycl-fused-mul-mat-id-2-8-tokens.diff](0003-sycl-fused-mul-mat-id-2-8-tokens.diff) | ours |
| 4 | `127f3f0e9` (2026-09-25) | [0004-sycl-op-profile.diff](0004-sycl-op-profile.diff) | ours |
| 5 | `6505cf2be` (2026-09-25) | not in this folder (upstream commit, see [README.md](README.md)) | upstream cherry-pick (#28796) |
| 6 | `60a598ed8` (2026-09-25) | [0006-sycl-sparse-fa-multi-token.diff](0006-sycl-sparse-fa-multi-token.diff) | ours |
| 7 | `f47a6a5f3` (2026-09-25) | [0007-qwen4exp-pooled-qsa-key-cache.diff](0007-qwen4exp-pooled-qsa-key-cache.diff) | ours |
| 8 | `9f41d810e` (2026-09-25) | [0008-qwen4exp-mtp-draft-output-rows-only-attention.diff](0008-qwen4exp-mtp-draft-output-rows-only-attention.diff) | ours |
| 9 | `0f23f62c8` (2026-09-25) | [0009-qwen4exp-mtp-draft-qsa.diff](0009-qwen4exp-mtp-draft-qsa.diff) | ours |
| 10 | `993baf141` (2026-09-25) | [0010-server-slot-save-restore-draft-context.diff](0010-server-slot-save-restore-draft-context.diff) | ours |
| 11 | `2c88bdf19` (2026-09-26) | [0011-qwen4exp-qsa-score-expansion-by-broadcast.diff](0011-qwen4exp-qsa-score-expansion-by-broadcast.diff) | ours |
| 12 | `020bf5940` (2026-09-23) | [0012-upstream-pr29245-sycl-grouped-moe-xmx-gemm.diff](0012-upstream-pr29245-sycl-grouped-moe-xmx-gemm.diff) | upstream PR #29245, copied |
| 13 | `8793154f8` (2026-09-26) | [0013-sycl-grouped-moe-gemm-q4k-q5k-q51-q80.diff](0013-sycl-grouped-moe-gemm-q4k-q5k-q51-q80.diff) | ours |
| 14 | `517eb833d` (2026-09-26) | [0014-sycl-grouped-moe-gemm-vector-loads.diff](0014-sycl-grouped-moe-gemm-vector-loads.diff) | ours |
| 15 | `85b26781b` (2026-09-26) | [0015-sycl-grouped-moe-gemm-in-place-io.diff](0015-sycl-grouped-moe-gemm-in-place-io.diff) | ours |
| 16 | `a890bf8b0` (2026-09-26) | [0016-qwen4exp-mtp-eh-proj-2d-product.diff](0016-qwen4exp-mtp-eh-proj-2d-product.diff) | ours |
| 17 | `2b84213a4` (2026-09-26) | [0017-sycl-sparse-fa-union-stats-debug.diff](0017-sycl-sparse-fa-union-stats-debug.diff) | ours |

## 1. `9fe305c67` (2026-09-23), upstream cherry-pick (#28931)

```text
sycl: extend MMVQ GLU fusion, add rms_norm+scale and ssm_conv+silu fusions (#28931)

* sycl : extend MMVQ GLU fusion to mixed quant types; add rms_norm+scale and ssm_conv+silu fusions

* fixing spacing issue and macro converted to template function
```

## 2. `4fb54754f` (2026-09-24), ours

```text
sycl: cache gettid per thread in dpct get_tid()
```

## 3. `7e5cb8f13` (2026-09-24), ours

```text
sycl: fused MUL_MAT_ID for 2-8 token batches (per-token MMVQ, no host sync)
```

## 4. `127f3f0e9` (2026-09-25), ours

```text
sycl: GGML_SYCL_OP_PROFILE per-op timing (flag file /tmp/ggml-sycl-op-profile), GGML_SYCL_MMID_MULTITOKEN_MAX
```

## 5. `6505cf2be` (2026-09-25), upstream cherry-pick (#28796)

```text
[SYCL] support sparse FA (#28796)

* fix conflict

* fix format issue

* rm unused code

(cherry picked from commit cd74ef6274e55b1591373637cc35324af000a5ee)
```

## 6. `60a598ed8` (2026-09-25), ours

```text
sycl: sparse FA for small multi-token batches (per-row gather, rows as sequences); qwen4 sparse FA tests; A/B off-file
```

## 7. `f47a6a5f3` (2026-09-25), ours

```text
qwen4exp: pooled QSA indexer-key cache (position-block rows, watermark, dirty-block recompute); LLAMA_QSA_POOLED
```

## 8. `9f41d810e` (2026-09-25), ours

```text
qwen4exp MTP draft: output-rows-only attention, optional LLAMA_MTP_WINDOW; LLAMA_HOST_PROF host timing
```

## 9. `0f23f62c8` (2026-09-25), ours

```text
qwen4exp MTP draft: QSA with its own indexer (attention-only hybrid_idx draft memory, ratio override, sparse FA); /tmp/llama-mtp-qsa-off A/B switch
```

## 10. `993baf141` (2026-09-25), ours

```text
server: slot save/restore also saves and restores the speculative draft context (<file>.dft)
```

## 11. `2c88bdf19` (2026-09-26), ours

```text
QSA: expand block scores by broadcast when cells sit in position order

build_qsa_top_k expanded per-block indexer scores to cells with get_rows over a
transposed copy, then transposed the [n_kv, n_tokens] result back: two
cont(permute) per QSA layer and ubatch, ~13% of a cold 110K prefill. When the
pooled plan finds every cached cell at its own position (the single-conversation
case), cell j is in block j/ratio, so the expansion is a broadcast add of the
[1, n_blocks, n_tokens] scores onto the [ratio, n_blocks, n_tokens] view of the
mask. Same values, no gather, no transposes.

The broadcast leaves the cell_blk input out of the graph, so it gets no buffer;
set_input_qsa_fast skips writing it then.

quad-b70-5800x-pex88096, 135K cold read: 447-461 s -> 382 s. Decode and per-turn prompt time
unchanged; next-token KL vs the previous build within run-to-run noise.
```

## 12. `020bf5940` (2026-09-23), upstream PR #29245, copied

```text
sycl: add a grouped MoE XMX GEMM, gathering IQ weights into the tiles

A MUL_MAT_ID over a MoE layer ran one library GEMM per expert. With 128 experts
and 8 active, each is a sliver, so launches dominate and the XMX units idle.
This groups every expert of a node into one launch, laying out the work-groups
on the host from the row offsets it already has. Weights are dequantized inside
the GEMM, gathered into the XMX A tiles instead of being written out to f16 and
read back; activations are packed once into VNNI f16.

Nine weight formats have an A stage: iq4_nl, iq3_s, iq4_xs, iq3_xxs, iq2_xxs,
iq2_xs, iq2_s, iq1_s, iq1_m. Which of them are covered decides almost the whole
result, so GGML_SYCL_XMX_GATHER_TYPES selects them as a bitmask, one bit per
format, all set by default; 0 disables the paths and is the baseline to measure
against. Quant names do not give coverage away: unsloth's UD quants mix per
tensor, so a file named IQ3_XXS holds IQ3_XXS, IQ3_S and IQ4_XS experts, and a
Q4_K_M model has no IQ tensors at all and cannot benefit.

Both paths compute in f16 on XMX. The grouped MUL_MAT_ID path therefore trades
some precision against the per-expert library GEMM it replaces, which in a
default build (GGML_SYCL_F16=OFF) computes in f32. The plain MUL_MAT half only
engages when built with GGML_SYCL_F16=ON, as it sits inside that f16 branch.

One Arc Pro B60, unsloth/Qwen3-30B-A3B-GGUF UD-IQ3_XXS, -fa on, 4 interleaved
passes after a warm-up:

    GGML_SYCL_XMX_GATHER_TYPES   pp512    pp2048   tg128
    0 (off)                      671.1     677.2    49.3
    iq4_nl|iq3_s   (42/144)      720.1     725.3    49.4    +7.3%
    all formats   (144/144)     1016.9    1006.9    49.3   +51.5%

Token generation is unchanged, as expected for a prefill path.

test-backend-ops -o MUL_MAT_ID and -o MUL_MAT against the CPU backend: 125 and
191 iq* cases, zero failures with every bit set, identical to the disabled arm.

Assisted-by: Claude Opus 5
```

## 13. `8793154f8` (2026-09-26), ours

```text
sycl: grouped MoE XMX GEMM for Q4_K/Q5_K (plain and reordered), Q5_1 and Q8_0 experts

PR #29245 decodes IQ formats only. Qwen3.8-Flash-Next UD-Q4_K_XL keeps its
experts in Q4_K/Q5_K (gate, up) and Q5_1/Q8_0 (down). Adds A stages for those,
row views for the per-expert SoA layout reorder_qw_q*_k_moe leaves Q4_K/Q5_K
experts in, and reorders on the grouped path too so it meets one layout. The
new formats are for the grouped MUL_MAT_ID path only (bits 9-12 of
GGML_SYCL_XMX_GATHER_TYPES); plain MUL_MAT keeps the IQ list.

test-backend-ops: 60/60 new MUL_MAT_ID cases pass with and without the reorder.
One -ub 1536 ubatch (512 experts, 10 used) on a B70, per-expert loop -> grouped:
q4_K 11.3 -> 5.6 ms, q5_K 11.9 -> 7.8, q5_1 11.0 -> 4.7, q8_0 10.9 -> 8.0.
```

## 14. `517eb833d` (2026-09-26), ours

```text
sycl grouped GEMM: vector loads in the q4_K/q5_K/q5_1/q8_0 A stages

A lane decodes one weight row, so every A-stage load is a gather across 16 rows,
and patch-16 times ranked by load count. q4_K/q5_K read the step's 32 qs (and qh)
bytes as two 16-byte loads, q5_1 as two 8-byte loads plus a dword of qh, q8_0 in
pairs. One -ub 1536 ubatch: q4_K 5.6 -> 4.9 ms, q5_K 7.8 -> 5.3, q5_1 4.7 -> 4.4,
q8_0 8.0 -> 8.0. test-backend-ops 60/60 with and without the reorder.
```

## 15. `85b26781b` (2026-09-26), ours

```text
sycl grouped GEMM: read src1 and write dst in place through the row mapping

The B pack reads each routed row straight from src1 and moves an FG_BN x 64
block through local memory, so row reads and packed writes are both contiguous;
the GEMM stores each routed row straight into dst. The contiguous buffers and
their copy kernels (157 MB written for the gate/up gather, 157 MB read and
written for the down scatter) are only set up for the per-expert loop.

One -ub 1536 ubatch: q4_K 4.9 -> 3.9 ms, q5_K 5.3 -> 4.4, q5_1 4.4 -> 3.8,
q8_0 8.0 -> 7.4. test-backend-ops 60/60 with and without the reorder.
```

## 16. `a890bf8b0` (2026-09-26), ours

```text
qwen4exp MTP: eh_proj as one 2D product

concat is [2*n_embd, hc, n_tokens]; a 3D src1 makes the SYCL backend run one
hc-column product per token, each reading the whole 14 MB Q8_0 weight: 115 ms per
1536-token prompt ubatch on quad-b70-5800x-pex88096 (8.5% of a cold read at ~100K). The weight is 2D,
so [2*n_embd, hc*n_tokens] is the same product. 135K cold read 313 -> 284 s.
```

## 17. `2b84213a4` (2026-09-26), ours

```text
sycl sparse FA: GGML_SYCL_SPARSE_FA_DEBUG=2 logs selection-union sizes of prompt batches

For a masked-sparse FA call too large for the per-row gather, count for tiles of R
consecutive query rows how many cells any row keeps (R = 1, 16, 32, 64, 128); one
call in 12 is logged. At n_kv 34K a 16-row tile keeps ~7.6K cells, a 64-row tile
~15.8K, against 2051 per row.
```
