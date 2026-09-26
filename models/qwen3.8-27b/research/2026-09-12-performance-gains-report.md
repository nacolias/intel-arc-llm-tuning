# Dual Arc Pro B70 + Qwen3.8-27B: Performance Gains Report

**Date:** 2026-09-12
**Target:** `dual-b70-5800x` (Ryzen 7 5800X, 128 GB DDR4, 2x Intel Arc Pro B70 32 GB, 10 GbE)
**Baseline:** `vllm/vllm-openai-xpu:v0.28.0`, TP2, GPTQ-INT4 sym G128 + BF16 MTP head, MTP2, XPU graphs, FP8 KV, 262K context: **78 to 84 tok/s single stream**, 405 to 442 tok/s at 16 streams, 150K cold prefill about 183 s.
**Goal:** find every lever that gets closer to the 140 tok/s reported for dual B70 + DFlash2, and explain how to implement each.

Everything below was checked against source code (vLLM v0.28.0, v0.29.0 and main; vllm-xpu-kernels main; oneCCL 2022.1.0 `deps/libccl`; intel/compute-runtime master and tags 26.27/26.31; Linux 6.18 `xe`) or against published, reproducible measurements on the same GPU and model family. Anything I could not confirm is marked **UNVERIFIED**.

---

## 0. Executive summary

| # | Lever | Expected single-stream effect | Effort | Risk | Section |
|---|---|---|---|---|---|
| 1 | MTP depth 2 -> 4 (`num_speculative_tokens: 4`) | +12% to +27% (65.8 -> 83.7 on one card; 76.7 MTP1 -> 91.0 MTP4 on TP2 graphs-on) | 5 min | low | 3.1 |
| 2 | INT4 draft LM head, lab implementation (`VLLM_XPU_DRAFT_LM_HEAD_INT4=1`) | +24% on top of MTP4 (91.0 -> 112.9 tok/s on dual B70 TP2) | 1 hr (prebuilt image) or half a day (port) | medium | 3.2 |
| 3 | oneCCL low-latency allreduce protocol (`CCL_SYCL_ALLREDUCE_LL=twoshots`) | 130 collectives per token; 32% lower per-collective latency measured on 4 ranks | 1 min | low | 3.4 |
| 4 | Power cap 150 W -> 230 W per card | +18% to +30% on dense decode if you are still at the 150 W default | 1 min | thermal | 3.6 |
| 5 | Whole-graph compile (`"splitting_ops": []`) + `FULL_DECODE_ONLY` capture sizes to 320 + `--block-size 64` | avoids an MTP phantom-token bug, unlocks concurrency scaling (+22% at c2, +10% at c4) | 10 min | medium | 3.3 |
| 6 | Sync-free grouped GDN speculative branch (lab patch R276) | lets verify batches above 16 sequences be graph-captured; +22%/+10% at c2/c4 | 30 min | low | 3.5 |
| 7 | vLLM 0.29.0 upgrade (mamba pointer fix, oneCCL warm-up fix, N-contiguous FP16 linears, xpu-kernels 0.1.14.1, W4A8 prefill path) | small decode gain, removes 1 to 2 of your 7 patches | half a day | medium | 4.1 |
| 8 | compute-runtime 26.27 -> 26.31.39395.13 in the container, and master-only BMG commits (midthread preemption timer, OOQ sync skip, IPC offset fix) | 0% to a few %; the IPC fixes matter for P2P stability | 1 hr | low | 4.3 |
| 9 | DFlash2 on XPU | not viable today on any released vLLM; documented root causes and a test plan | days | high | 5 |
| 10 | Prefix-cache + MTP correctness trio (silent corruption on hybrid models) | 0% speed, but you are running the exact configuration that corrupts | 30 min | n/a | 7 |

Realistic stacked target on your box: **105 to 120 tok/s single stream** (levers 1, 2, 3, 5) with the same GPTQ checkpoint. 140 tok/s has only been shown on code-heavy prompts at short context (183 tok/s on Python, 111 on prose, same server) and the poster's claim could not be retrieved (Reddit is blocked from this environment; see section 2).

---

## 1. Where the time goes at 80 tok/s

Qwen3.8-27B is 64 layers, hidden 5120, 48 Gated DeltaNet layers + 16 full-attention layers (3:1), vocab 248,320, plus a 1-layer MTP head. On INT4 the target body is about 15 GB of weights.

**Bandwidth roofline.** One B70 has 608 GB/s. A no-speculation decode step reads all 15 GB, so one card is bounded at ~40 tok/s (the community measures 32.9 to 36). TP2 halves the per-card read (~7.5 GB, ~12 ms at realistic efficiency), so the two-card no-spec ceiling is ~80 tok/s and measured no-spec TP2 is 49.8 tok/s (graphs on) / 34 to 36 (graphs off).

**Measured step decomposition on dual B70 TP2, MTP4, INT4 draft head, 112 tok/s** (steveseguin/b70-optimization-lab, R261-R268): a ~27 ms step is ~12 ms of weight reads for the 5-token verify, ~6 ms for the four draft passes, and **~8 ms of two-rank host round trip** (allreduces plus host waits). That last term is the part your host controls.

**Collective count.** Each token step issues ~130 all-reduces (2 per layer for 64 layers, plus MTP/draft collectives). Message size per decode token per allreduce is hidden x fp16 = 10 KB; MTP2 verify rows make it ~30 KB; MTP4 ~50 KB. This is latency-bound, not bandwidth-bound. Measured per-collective latency at two rows on two B70s:

| Host | PCIe | async launch | 2-card allreduce |
|---|---|---|---|
| EPYC 9015 (Zen 5) | Gen5 | 3.1 us | 13 us |
| Threadripper PRO 5955WX (Zen 3) | Gen4 | 5.2 to 6.2 us | 48 to 51 us |

Your 5800X is a Zen 3 desktop part on PCIe 4.0, so expect the 48 us class. 130 x 48 us = 6.2 ms per token with the graph off, which is why the same recipe ran 1.8x slower on the Threadripper host than on the EPYC host until XPU graph capture was enabled. Your affinity patch already captures the SYCL collectives inside the graph, which is the single most important thing you did; section 3.4 covers what is left.

**Your 40 MB allreduce = 2.56 GB/s** is not a link problem. It is the oneCCL algorithm and staging (section 3.4). PCIe 4.0 x8 is ~13 GB/s practical, and the lab measured 13.79 GB/s for a 256 MiB XCCL allreduce on a Gen4 host and 27.88 GB/s on Gen5. Decode never sends messages that big, so this number does not affect decode; it does affect prefill.

---

## 2. The 140 tok/s claim and the "custom Intel driver"

I could not read the Reddit thread (reddit.com, its mirrors, archive.org and most blogs are blocked by this environment's egress proxy) and none of the search indexes carried the post text. What I could verify from public, reproducible sources on the same hardware:

| Source | Config | Single-stream tok/s |
|---|---|---|
| SergiioB cookbook, 1x B70, your exact GPTQ checkpoint, fp8 KV, 230 W | MTP2 / MTP4 | 65.8 / 83.7 |
| same, + draft-INT4 overlay (single card only) | MTP4 | 112.65 |
| steveseguin lab, 2x B70 TP2, AutoRound INT4 via the same GPTQ `XPUwNa16` path, FP16 KV, XPU graphs | MTP0 / MTP1 / MTP4 / MTP5 | 49.8 / 76.7 / 91.0 / 88.8 |
| same, + INT4 draft LM head | MTP4 | **112.9** (R276), 120.6 at 2K context, 100.3 at 32K |
| same, per prompt class at 2K | MTP4 + INT4 head | prose 110.9, **Python 183.0**, docs 120.6 |
| steveseguin lab, 2x B70 TP2, official FP8 W8A16 + Xe2 small-M kernel, graphs off, whole-graph compile | MTP2 / MTP3 / MTP4 / MTP5 / MTP6 | 70.1 / 79.2 / 82.4 / 86.2 / 87.1 |
| SergiioB cookbook, 2x B70 TP2, FP8 W8A16, MTP8, graphs off | greedy random tokens | 53 to 60 (collapses to 12 to 22 with sampling) |
| SergiioB cookbook, 1x B70, DFlash2 draft | n=7 | 0% accepted, 19 tok/s; 24.7% / 22 tok/s after a mask fix |

So 112 to 125 tok/s on dual B70 is reproducible with the INT4 path you already run, and 140+ is plausible only for code-heavy prompts at short context. Nobody has published a working DFlash2 number on Battlemage; every published fast dual-B70 number uses native MTP.

**What "custom Intel driver" most likely means** (three candidates, all verifiable):

1. **Intel's own platform bundle.** llm-scaler's install path is the Intel RDC "multi-arc-bmg-offline-installer" (currently 26.18.8.2), which ships a Linux kernel, GuC firmware, compute-runtime, oneCCL, `xpu-smi`, `ze_peer` and platform scripts as one package. Anyone who says "Intel's custom driver" for B60/B70 usually means this.
2. **Rebuilt user-space runtime.** The lab's published images are the stock vLLM XPU image with `_xpu_C.abi3.so` and the GDN device library rebuilt from `steveseguin/vllm-xpu-kernels` (AOT `bmg-g31-a0`) plus a rebuilt public oneCCL (`b52f40c`/`4ceafd1`) because the pinned `libccl` in the image failed a graph-replay correctness test (replay 1 mismatched on all ranks at the default LL threshold; the rebuilt one passed 100/100).
3. **compute-runtime from master.** The BMG-specific performance and IPC commits in section 4.3 are not in any tagged release yet.

---

## 3. Tier 1: software levers with measured gains on this GPU and model

### 3.1 MTP depth 4

Your MTP2 gives ~3 tokens per verify with ~97% acceptance. Depth 4 gives ~4.8 tokens per verify at ~94% acceptance on this checkpoint. Measured on one B70: 65.8 -> 83.7 tok/s (+27%). On TP2 with graphs the lab saw MTP4 91.0 vs MTP5 88.8 vs MTP6 84.1, so **4 is the sweet spot for 1 to 4 users**; depth 1 wins from ~8 users, and no speculation from ~32 users (their c64 aggregate: 991 tok/s no-spec vs 591 MTP4).

```yaml
- SPEC={"method":"mtp","num_speculative_tokens":4}
```

Notes:
- Drop `--gpu-memory-utilization` from 0.90 to 0.88 if warm-up OOMs; MTP4 needs more draft buffers.
- Add capture sizes for the verify batch. With `max-num-seqs 32` and 5 tokens per spec row, the decode batch can reach 160 tokens; use the `cudagraph_capture_sizes` list in 3.3.
- Re-measure acceptance on **your** uncensored fine-tune with `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total` from `/metrics`. Your doc says "high"; if it is under ~85% at depth 4, use depth 3.
- For 8 to 16 concurrent users switch to depth 1 or 2; the lab's multi-user table is in section 6.

### 3.2 INT4 draft LM head (the +24% that gets you to ~112)

The draft passes run the 248,320 x 5120 FP16 LM head four times per step. Quantizing only the draft-side copy of the head to INT4 g128 (the target verifier keeps FP16) cuts each draft pass's dominant read. Two implementations exist:

| Implementation | Where | TP2 | Measured |
|---|---|---|---|
| SergiioB `patch_draft_lmhead_int4.py` + `patch_draft_mtp_int4.py` (`B70_DRAFT_LMHEAD_INT4=1`, `B70_DRAFT_MTP_INT4=1`) | cookbook `patches/` | **refuses TP>1** (fails closed, issue #9: output corruption under TP2) | 81.2 -> 112.65 on one card |
| steveseguin lab `VLLM_XPU_DRAFT_LM_HEAD_INT4=1` (group 128, bf16 scales, `..._CHUNK_ROWS=2048`) plus `docker/r256-draft-int4-head-fallback.py` for GPTQ-labelled checkpoints | lab images `ghcr.io/steveseguin/vllm-openai-xpu-qwen38-int4@sha256:521eb277...` (R276) | **lossless under TP2**: 12/12 token arrays identical to the MTP0 oracle, identity-exact through 16 users | 91.0 -> 112.9 on two cards |

Use the lab implementation. Fastest route: pull the R276 image and run their standalone `docker run` (copied into [`configs/lab-mtp4-int4-draft-head.sh`](../configs/lab-mtp4-int4-draft-head.sh)), swapping in your model directory and adding your `--max-model-len`, `--kv-cache-dtype fp8`, prefix caching and API flags. Their image is vLLM `0.27.2rc1.dev77` with rebuilt kernels, so treat it as an A/B box first, not production.

If you want it on your 0.28.0 image instead: the head quantization lives in `vllm/v1/spec_decode/llm_base_proposer.py` (`make_xpu_int4_draft_copy`) in the lab fork; port it as your patch 8. **UNVERIFIED** whether the lab's `_xpu_ops.py` changes are required for the head path alone; the R256 fallback patch only touches the proposer.

### 3.3 Graph mode, compile mode, block size, KV dtype

The lab's headline configuration:

```
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY",
  "cudagraph_capture_sizes":[1,2,3,4,5,6,8,10,15,16,20,25,30,32,40,50,60,64,80,100,120,160,200,240,320],
  "max_cudagraph_capture_size":320,
  "splitting_ops":[],
  "inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false,"deterministic":true,
    "benchmark_epilogue_fusion":false,"split_reductions":false,"triton.autotune_pointwise":false}}'
--block-size 64
```

Why each part matters:
- **`"splitting_ops": []` (whole-graph Inductor compile).** On the default piecewise compile, MTP depth 2 emitted a phantom first token (the prompt's last token re-emitted) on 1 request in 64, every run, on both the lab image and the unmodified vLLM XPU image (R192/R194). The whole-graph compile avoided it in six passes. This is an upstream, unfixed bug on the path you are on. Your `PIECEWISE 14/14` capture log means you are running the affected mode.
- **`FULL_DECODE_ONLY`.** Your log shows `FULL 8/8`, so you already capture full decode graphs; extend the size list so spec batches above 8 tokens are captured (c1 MTP4 = 5 tokens; c8 MTP4 = 40; c32 = 160). Without the R276 GDN patch (3.5) the grouped GDN branch does a `.tolist()` sync above 16 sequences and capture fails at 320.
- **`--block-size 64`.** The XPU flash-attention backend's preferred block size is 64 (`FlashAttentionBackend.get_preferred_block_size` returns max(default, 64) on XPU), and the GDN backend requires a multiple of 64. Your `--block-size 32` is being silently padded or is off the fast path. Change it.
- **KV dtype.** Every published fast dual-B70 number uses `--kv-cache-dtype auto` (FP16 KV). FP8 KV halves KV bytes, which you need for 262K context, but on XPU the FA kernel dequantizes K/V per step. **UNVERIFIED** how much it costs on BMG; A/B it at your real context distribution. If most requests are under 64K, consider `--max-model-len 131072` + FP16 KV as the fast profile and keep the 262K/FP8 profile as a second served model (you already use sleep mode for swapping).
- **`--max-num-batched-tokens`.** 16384 is a prefill budget; it does not slow decode by itself, but with `max-num-seqs 32` and MTP4 the verify batch can reach 160 rows, and the W4A16 verify GEMM is dequant-bound and scales linearly above ~32 rows (55 ms at M=160 on the lab box). For a single-user fast profile use `--max-num-seqs 4`.

### 3.4 oneCCL: you are already on the Arc LL256 path; pick the faster protocol

From `oneCCL/deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_sycl.cpp` (2022.1.0):

```
if (is_arc_card(family) &&
    (total_size <= env.sycl_allreduce_simple_threshold || !has_p2p_access())) {
    if (!env.sycl_enable_arc_allreduce) {          // CCL_SYCL_ALLREDUCE_ARC, default 0
        // chunked LL256 ring; protocol from CCL_SYCL_ALLREDUCE_LL:
        //   "ring" (default) | "twoshots" | "oneshot" (only <= CCL_SYCL_ALLREDUCE_ONESHOT_THRESHOLD, 4096 B)
        allreduce_ll<RingTransmit | TwoShotsTransmit | OneShotTransmit>(...)
    }
}
```

Implications for your environment:
- Battlemage is an "arc card" in oneCCL, and your `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296` makes the first condition always true, so **every allreduce goes through the LL256 kernels**, never the small/medium/large XeLink-style kernels. That is why they capture into SYCL graphs.
- The protocol knob is `CCL_SYCL_ALLREDUCE_LL`. Default `ring`. The lab measured on 4 B70s, BF16 [1,4096]: **ring 128.7 us vs twoshots 87.4 us** per call (-32%), which is why `CCL_SYCL_ALLREDUCE_LL=twoshots` appears 796 times in their launchers. For two ranks the gap should be smaller but nonzero. `oneshot` only applies at or below 4 KB and falls back to ring above it; your messages are 10 to 50 KB, so it is useless here.
- **Never set `CCL_SYCL_ALLREDUCE_ARC=1`**: the lab found it corrupted all 64 measured epochs (max abs error 342).
- `CCL_SYCL_ALLREDUCE_LL_THRESHOLD` (default 4096) selects `Rt64_PCIE` at or below and `Rt64_128_PCIE` above; raising it to 8192 saved 0.17 ms over 87 calls in their test, i.e. nothing. Leave it.
- Keep `CCL_TOPO_P2P_ACCESS=1` (your P2P probe is right; llm-scaler documents ~15% higher throughput at large batch with P2P on). Note that the second condition (`!has_p2p_access()`) also routes to LL256, so P2P only changes what the LL kernels do underneath (direct peer reads vs host staging), not which kernel runs.
- `CCL_SEND=direct CCL_RECV=direct` are set by every lab launcher on top of yours; harmless, and they avoid the naive send/recv schedulers for the odd point-to-point call.
- Remove `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`; the lab measured it as neutral (19.85 -> 19.89) and it disables topology validation.
- oneCCL versions: torch 2.13 XPU wheels bundle `oneccl==2022.0.0`; torch main pins `2022.1.1`. The public 2022.1.0 source is what I read; the LL path exists in both.

Measure before and after with the lab's public-domain census script (copied to [`tools/xccl_allreduce_census.py`](../../../tools/xccl_allreduce_census.py)):

```bash
docker exec -it vllm-a bash -lc 'python -m torch.distributed.run --standalone --nproc_per_node=2 /tools/xccl_allreduce_census.py /tmp/ar.json'
# prints allreduce timing us: {rows2, rows32, rows64, rows900} and exactness checks
```

Run it once with your current env and once with `CCL_SYCL_ALLREDUCE_LL=twoshots`. Also run it with 100 changing inputs under graph capture if you can (the lab's libccl passed eager and replay 0 but mismatched on replay 1 at the default LL threshold; the rebuilt public build passed 100/100). Your production stack replays captured collectives thousands of times per request, so this is worth an hour.

### 3.5 GDN (Gated DeltaNet) kernel path

48 of 64 layers are GDN. On XPU they run `vllm_xpu_kernels` `gdn_attn` (`csrc/xpu/gdn_attn`, `causal_conv1d_spec/non_spec` + `chunk_gated_delta_rule_xe2`), not Triton, which is good. Two things still cost you:

1. **Mixed spec + non-spec batches** need your `patch_gdn_mixed_split_v5.py` (upstream `vllm-xpu-kernels#510`). Keep it.
2. **Grouped speculative branch syncs.** The lab's R228 grouping (`VLLM_XPU_GDN_SPEC_GROUP=16`) computed sequence boundaries with `.to("cpu").tolist()`, which XPU graph capture rejects, so no verify batch above 16 sequences was ever captured. `docker/r276-gdn-spec-group-sync-free.py` computes them arithmetically (every spec row has k+1 tokens) and made capture succeed to 320 tokens: **+22% at 2 users, +10% at 4 users**, lossless. Their `VLLM_XPU_GDN_*` switches only exist in their fork; the equivalent on your stack is to make sure nothing in your v5 split path does a device-to-host sync inside the captured region (grep for `.tolist()` / `.cpu()` in the patched `gdn_attn.py`).
3. Upstream main (not in 0.29.0) added `forward_xpu` for `Mixer2RMSNormGated`/`FusedRMSNormGated` (commit `5b6cf93e`, 2026-09-11) and routed activation custom ops to SYCL kernels (`4c21d417`); both touch every GDN layer and are worth cherry-picking when you rebase.

### 3.6 Power cap and clocks

The B70 ships with a **150 W** cap (its TBP is 230 W). Dense decode scales with the cap because higher clocks raise effective bandwidth: the cookbook measured +18% to +30% from 150 W to 230 W on dense 27B, with package temperature 79 C at 230 W sustained, and recommends 180 W as the sustained sweet spot (0.155 tok/s/W). MoE models do not benefit. Your doc never mentions the cap, so check it first:

```bash
for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = xe ] && echo "$h $(cat $h/power1_cap) uW  temp $(cat $h/temp2_input 2>/dev/null)"; done
# set per card (microwatts); the driver rejects values above the 230 W ceiling
echo 230000000 | sudo tee /sys/class/hwmon/hwmonN/power1_cap
# or: sudo xpu-smi config -d 0 --powerlimit 230
```

There is no user clock pinning on `xe` (no `gt_min/gt_max`); `xpu-smi config --frequencyrange` is the sysman path and **UNVERIFIED** on B70. Watch `xpu-smi dump -d 0 -m 0,1,2,18,19,20` for frequency and throttle reasons while decoding; if you see throttling at 230 W, use 180 W.

---

## 4. Tier 2: upgrade paths (vLLM, kernels, runtime, driver)

### 4.1 vLLM 0.29.0 (released 2026-09-08) versus your 0.28.0

XPU-relevant commits present in v0.29.0 and not in v0.28.0 (verified with `git merge-base` on the tags):

| Commit | What | Effect on you |
|---|---|---|
| `f936a267` #48109 | Fix Mamba state pointer overflow on XPU | **replaces your patch 5** (`patch_mtp_ptr_wrap.py`) |
| `f94666b6` #52389 | Skip oneCCL warm-up all_reduce at world size 1 | n/a for TP2 |
| `2a61f060` #53536 | Ensure unquantized linear weights are N-contiguous on XPU | faster FP16 lm_head / `mtp.fc` (the lab found the vocab projection was 3 to 8% of their step) |
| `cdb8545a` #52539 | Support Qwen head ratios in fused GDN MTP kernel | draft pass speed (CUDA path; XPU **UNVERIFIED**) |
| `94a54f58` #54203 | `vllm_xpu_kernels` 0.1.13.2 -> 0.1.14.1 | includes the Muse paged-decode tuple and the fused grouped_topk kernel |
| `9fd750f0` #50501 | INC int4 W4A8 (dynamic int8 activations) backend | prefill only, INC checkpoints only (see 4.2) |
| `cbe3966f` #52066 | Sparse-MLA metadata sync fix | n/a |

Still **not** upstream (keep your patches): BF16 MTP draft with a GPTQ base (`vllm#47828` open), the 131,072 boundary fix, the GDN mixed split, per-rank `ZE_AFFINITY_MASK` at spawn, sleep-mode graph release, and the three prefix-cache correctness fixes (`vllm#53919`, `#48375`, `#53505` all open as of 2026-09-05).

Main-only, worth cherry-picking when you rebase: `4c21d417` (activation ops to SYCL), `ed29dfae` (fused input norm), `a0d3e5c1` (fused GemmaRMSNorm eager), `5b6cf93e` (GDN gated RMSNorm `forward_xpu`), `ffe3bb3c` (batch-invariant XPU matmul kernels), `978c22f5` (triton-xpu 3.8.0 shim), `d61b6e18` (DFlash draft RoPE layout from its own config, see section 5).

### 4.2 Quantization path: what is and is not faster

- Your GPTQ checkpoint goes through `AutoGPTQLinearMethod` -> `choose_mp_linear_kernel` -> XPU list `[XPUW4A8IntLinearKernel, XPUwNa16LinearKernel]`. GPTQ's quant type is `uint4b8`, and the W4A8 kernel requires `scalar_types.int4`, so you land on **`XPUwNa16LinearKernel`** (oneDNN `int4_gemm_w4a16`). That is the fastest decode path on Xe2; nothing to change.
- The new `VLLM_XPU_INC_WNA16_BACKEND=w4a8` (0.29.0) only applies to `--quantization inc` / AutoRound checkpoints, only kicks in at **>= 512 tokens per call** (`_MIN_TOKENS_FOR_INT8 = 512`), and was measured faster on B70 because ARK cannot use its XMX int8 path there. It is a **prefill** lever. If you ever re-quantize the uncensored base with AutoRound (`auto_round:auto_gptq` packing) you get both the same W4A16 decode path and W4A8 prefill; the lab and the cookbook both found AutoRound INT4 equal or better than GPTQ in logprob parity. **UNVERIFIED** gain on your 150K prefill; expect it to help the GEMM share but not the GDN chunked prefill share.
- FP8 (W8A16 + Xe2 small-M kernel) on TP2 tops out at 86 to 87 tok/s with MTP5/6 and needs a rebuilt kernel package. INT4 with the INT4 draft head is faster on this silicon. Do not switch.
- Do not switch to compressed-tensors/AWQ/GGUF for this model; the cookbook documents them as unproven or MTP-stripping on this stack.

### 4.3 Runtime and driver

What your container actually runs (from `docker/Dockerfile.xpu` at v0.28.0): compute-runtime **26.27.39122.11** (`libze-intel-gpu1`, `intel-opencl-icd`, `ocloc`), IGC 2.38.2, GmmLib 22.10.0, Level Zero loader 1.32.0, `xpu-smi` 2.1.0. The oneCCL and SYCL runtime come from the torch 2.13 XPU wheels: `oneccl 2022.0.0`, `intel-sycl-rt 2026.0.0`, `intel-cmplr-lib-ur 2026.0.0`, `umf 1.1.0`.

Latest tagged compute-runtime: **26.31.39395.13** (2026-08-13; IGC 2.40.13, GmmLib 22.10.0, Level Zero API 1.17). Upgrading inside the container is four `.deb` swaps in the Dockerfile. The BMG-relevant commits that are **only on master** (tags fetched shallow; none of these subjects appear in the 26.27 or 26.31 histories):

| Commit | Date | What | Why it matters for you |
|---|---|---|---|
| `96bec218` | 2026-08-27 | Program a 150 us midthread preemption delay timer on BMG (`ScmMidthreadPreemptionDelayTimerOverride`) | fewer mid-thread state saves on short decode kernels |
| `54028dd3` | 2026-09-11 | Skip redundant synchronization on out-of-order immediate lists | host-side per-launch cost |
| `e759e3dd` | 2026-09-11 | Increase usable local-memory ring buffer capacity (ULLS 256 KB -> 2 MiB) | direct-submission ring |
| `8aabb883` | 2026-08-14 | Correctly pass the offset for mapped memory in IPC exchange | **oneCCL IPC of peer buffers (your P2P path)** |
| `83384f5d` | 2026-08-20 | P2P: prefer FD over reservedHandleData when available | same |
| `b4826721` | 2026-09-10 | Close temporary FDs after Linux opaque IPC imports | FD leak under long TP2 uptime |
| `2339f9e1` | 2026-09-01 | Add P2P checks for multi-device madvise | correctness with two devices in one process |
| `4c292b0b` | 2026-08-26 | Enable standby on xe only for BMG and CRI | idle power |
| `4e0b3e24` | 2026-07-21 | Enable prefetch for Xe2 and later | after the 26.27 tag date; **UNVERIFIED** whether 26.31 has it |
| `f50d9842` / `178d8204` | 2026-09-04 | "Disable TLB invalidation since Xe2 on Linux" then reverted | not stable yet |

If you want these, build compute-runtime from `master` (or `releases/26.35` when it tags) with IGC 2.40.x and install into the container as the "custom driver"; the debug key `NEOReadDebugKeys=1 ScmMidthreadPreemptionDelayTimerOverride=...` exists only in those builds. Do it as an A/B on the census script and the single-stream benchmark; expect small gains and better P2P stability rather than a step change.

Level Zero / Unified Runtime: BMG defaults to the L0 **V2** adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=1`, immediate command lists only). Keep it. `SYCL_UR_USE_LEVEL_ZERO_V2=0` plus `SYCL_PI_LEVEL_ZERO_USM_RESIDENT=0` is a documented recovery combination for the 2025.3 multi-device context regression (`llm-scaler#463`), not a performance setting. Useful V2 knob: `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` if copy-engine offload adds latency for the tiny per-step H2D copies (**UNVERIFIED**, A/B it). Do **not** set `SYCL_CACHE_PERSISTENT=1`: the chriswagner-ai four-B70 field notes record cross-restart kernel-cache poisoning on Battlemage (corrupt output after a restart); accept the ~30 s JIT warm-up. Your vLLM compile cache volume (`/root/.cache/vllm`) is a different cache and is fine. `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` (you set it) is right for a 32 GB card with >4 GB single allocations.

Kernel side: the 6.18 `xe` driver has `allow_peer2peer = true` and uses `pci_p2pdma_distance()` in `xe_dma_buf.c`, so PCIe P2P DMA-BUF between the two cards is supported when the cards sit under the same root complex; that is why your `zeDeviceCanAccessPeer` probe returned 1. Module params worth knowing: `xe.vram_bar_size` (0 = max needed, requires Resizable BAR in firmware), `xe.probe_display=0` on a headless box, `xe.wedged_mode` (see section 7). GuC firmware must be >= 70.54 for B70.

### 4.4 Intel llm-scaler as an alternative image

`intel/llm-scaler-vllm:0.26.0-b2` (2026-09) is vLLM 0.26 with Intel's patches: "fix prefix caching with MTP, correct `sym_int4` output, improve MTP decoding performance, fix block-FP8 errors during concurrent decoding". It supports `method: qwen3_5_mtp`, DFlash v1 for Qwen3.6-27B/35B via `VLLM_USE_V2_MODEL_RUNNER=1` and `--enforce-eager`, and ships Intel's oneCCL with P2P/USM fallback. It is older than your 0.28 base and their DFlash recipe is eager-only, so it is not a step forward for single-stream decode, but it is the only place Intel ships a **prefix-cache + MTP fix** today. Worth reading their patch set (`vllm/` in the repo) when you address section 7.

---

## 5. DFlash2 on Intel XPU: why it failed and what it would take

### 5.1 What DFlash2 is

A block-diffusion drafter (`DFlash2DraftModel`, `qwen3_dflash2.py`): a few Qwen3 layers with a grouped local convolution (`DFlashGroupedConv` / `attention_conv`) and a learned `CandidateSelector` (top-k per position, low-rank scorer), conditioned on **target hidden states** (`use_aux_hidden_state`, EAGLE-3 style) and on mask-token embeddings, drafting a whole block in one non-causal forward. Merged in vLLM PR #52816 on 2026-08-21; on H200 it reaches 3.1x to 3.4x at concurrency 1 and 1.0x to 1.45x at concurrency 32. Registry entries and `decoder_layer_cls` are intact in v0.28.0 and v0.29.0 (the `#53428` load regression only hit `main` briefly), so the model loads on your image.

### 5.2 Why you saw 0% acceptance (ranked)

1. **The XPU attention path handles the draft's non-causal mask wrongly.** DFlash needs bidirectional attention over the block (`dflash_has_any_non_causal` -> `attention_config.use_non_causal=True`; `DFlashProposer` asserts `attn_metadata.causal is False` for every layer). `FlashAttentionBackend.supports_non_causal()` returns True generically, and on XPU `flash_attn_varlen_func` forwards `causal` to `vllm_xpu_kernels.flash_attn_interface`, which passes it into the FA2 varlen kernel. But the XPU wrapper **drops `dynamic_causal`** (accepted, never forwarded), and the `causal` tensor form that `flash_attn.py` builds (`causal.to(torch.int32)` at line ~720 in v0.29.0) is not what the XPU op expects. SergiioB reproduced exactly this on the base GPTQ model: 0/574 accepted unpatched, 68/434 (15.7%) after a "mixed-dtype + causal-fallback" overlay, 24.7% with a labelled n=5 run, and concluded that 24.7% "is the wrong causal mask for block diffusion, not a DFlash2 ceiling"; the remaining bugs were "conv/RMS scale and packing". So the primary cause is the XPU stack, not your fine-tune.
2. **RoPE layout bug in the DFlash draft**, fixed in `d61b6e18` (#54373, "Take the DFlash draft's RoPE layout from its own config") on 2026-08-31. It is in **neither** v0.28.0 nor v0.29.0. A wrong RoPE layout in the draft produces garbage drafts and near-zero acceptance. Any retest must be on `main` after this commit.
3. **FP8 KV cache.** On CUDA, DFlash is rejected with every quantized KV dtype (`vllm#41559`; FlashInfer later added support). On XPU `flash_attn_supports_kv_cache_dtype()` returns True unconditionally, so the draft silently runs with fp8 K/V and non-causal attention on a kernel combination nobody has validated. Retest with `--kv-cache-dtype auto`.
4. **Distribution shift from the uncensored fine-tune.** Real but secondary. The draft conditions on target hidden states, and your native MTP head, which conditions on the same states, accepts ~95% at depth 2, so the hidden-state distribution has not moved far. Expect a fine-tune to cost some acceptance length (z-lab reports that 1.6K samples of fine-tuning recover long-context acceptance for DFlash), not to zero it.

### 5.3 Test plan if you want to try again

1. Build a test container from vLLM `main` at or after `d61b6e18`, with the same xpu-kernels pin (0.1.14.1) and your affinity/MTP patches removed (the draft is external; only the affinity patch is needed for TP2).
2. Serve with `--kv-cache-dtype auto`, `--max-model-len 16384`, `--max-num-seqs 1`, `VLLM_USE_V2_MODEL_RUNNER=1`, `--enforce-eager` first, and `--speculative-config '{"method":"dflash","model":"/models/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'` (the official n is 7; the MLX docs say quantized targets prefer block size <= 5).
3. Probe acceptance with 5 greedy prompts and read `/metrics`. Anything above ~40% means the mask path works and the rest is fine-tune shift; 0% again means the XPU kernel path, and the fix is in `vllm/_xpu_ops.py` (forward `dynamic_causal`, honour a per-token causal tensor) plus `vllm_xpu_kernels/flash_attn_interface.py` (do not route non-causal spec batches to the causal split-K fast path; it already checks `causal` so this part is fine).
4. Only then compare against MTP4 + INT4 head. On this hardware the draft is 3.85 GB of extra weight reads per step; z-lab's own numbers (acceptance length 5.46 at 3.4x on H200) are against a 27B **BF16** target, where the target step is 2.7x heavier than your INT4 target step. On INT4 the relative gain is smaller, and native MTP4 at ~4.8 accepted tokens is already close. My estimate is that a working DFlash2 would land within +/-15% of MTP4 + INT4 head here.

**Recommendation:** keep native MTP as the production speculator; treat DFlash2 as a research item until upstream has an XPU CI test for `test_dflash2` (there is one for CUDA: `c01b50e3`).

---

## 6. Concurrency and prefill

**Multi-user guidance measured on dual B70 TP2, INT4, XPU graphs, 128 tokens per request (lab R284-R288):**

| users | no speculation | MTP1 | MTP2 | MTP4 |
|---|---|---|---|---|
| 1 | 49 | 77 | - | 113 |
| 2 | 95 | 147 | - | 191 |
| 4 | 179 | 269 | - | 295 |
| 8 | 327 | 456 | - | 423 |
| 16 | 534 | 711 | 591 | 574 |
| 32 | 815 | 854 | 723 | 641 |
| 64 | 991 | 842 | 815 | 591 |

Every speculative depth plateaus once the verify batch exceeds ~32 rows because the W4A16 verify GEMM is dequant-bound above 32 rows. Your 16-stream number (420 to 442 at MTP2) matches their MTP2 row within noise. If you serve mixed loads, run **two profiles behind the proxy**: MTP4 + INT4 head with `--max-num-seqs 4` for chat, and MTP1 with `--max-num-seqs 32` for batch/agent fan-out. You already have sleep mode; alternatively split the cards (one engine per card with `ZE_AFFINITY_MASK=0` / `=1`) when aggregate throughput matters more than single-stream latency.

**Prefill.** 150K cold in 183 s is 820 tok/s; the single-card cookbook gets 1,774 tok/s at 8K and the lab's TP2 FP8 lane 1,206 to 1,736 tok/s at 512 to 1K. Long-context prefill on this model is dominated by the GDN chunked scan and the 16 full-attention layers over the whole context, so it degrades with length by design. Levers, in order of confidence:
1. Prefix caching is already on; make sure it actually hits (see section 7 for the mamba prefix-reuse defect `vllm#52047` that disables reuse on >= 0.28.1 nightlies when the drafter's KV groups cannot be identified; measured 0% hits at 32K on nightly vs 91% on the pinned image).
2. `--max-num-batched-tokens`: 16384 chunks are fine for GEMM efficiency but the W4A16 kernel is dequant-bound; the W4A8 INC path (4.2) is the only kernel-level prefill lever and needs an AutoRound checkpoint. **UNVERIFIED** magnitude.
3. Check that the chunked-prefill collectives (40 MB class) are not being host-staged: with `CCL_TOPO_P2P_ACCESS=1` and the LL256 path, large messages are chunked (`CCL_SYCL_ALLREDUCE_CHUNKING_THRESHOLD`, default 0 = no chunking). Your 2.56 GB/s at 40 MB suggests a single huge LL256 transfer; try `CCL_SYCL_ALLREDUCE_CHUNKING_THRESHOLD=4194304` (4 MiB chunks) and re-run the census at 900 to 2048 rows. **UNVERIFIED** direction; it is a one-line A/B.
4. Lowering `--max-model-len` to what you actually use reduces KV block-table overhead and lets you switch to FP16 KV.

---

## 7. Correctness guardrails you are currently missing

1. **Prefix caching + MTP + hybrid model = silent corruption.** With `--enable-prefix-caching` (which forces `mamba_cache_mode="align"` on Qwen3.5-class hybrids), MTP, and async scheduling, three open upstream bugs combine: the accepted-token D2H copy lands in step-N row order while step N+1 permutes the same pinned buffer (`vllm#53919`); `MambaManager.find_longest_cache_hit` ignores `drop_eagle_block` so a snapshot over rejected draft positions stays reachable through the cache (`vllm#48375`); the align boundary state copy has no guard against a backward (`dest < src`) copy (`vllm#53505`). Symptoms reported in production: short duplicated keys, fragments of another context, empty `content` with `finish_reason=stop`. The cookbook ships fail-closed ports as `patch_fix_accepted_sync.py`, `patch_fix_backward_copy.py`, `patch_fix_eagle_drop.py` with a GPU-free dry run (`scripts/verify-mtp-apc-fixes.sh`). Bug 1 does not exist on the V2 model runner (GPU-resident counters); 2 and 3 do. Add them as patches 8 to 10, or run `--no-enable-prefix-caching` on the fast profile and keep caching only where you need warm TTFT. Intel's `llm-scaler-vllm:0.26.0-b2` claims a fix for the same interaction.
2. **Graph-replay exactness.** Add a 100-replay changing-input check for the captured collectives (the lab's protocol: eager hash series must equal the graph hash series on both ranks) before trusting a new oneCCL, driver, or capture-size change. A wrong reduction under graph replay produces plausible text.
3. **Wedge recovery.** Sustained TP2 load has produced permanent `ccs`/`bcs` engine resets on kernel 7.x + GuC 70.58 + compute-runtime 26.05 (`compute-runtime#948`, `vllm#41663`), recoverable on 6.17 + GuC 70.44. Your 6.18 kernel is in the untested middle. Keep `xe.wedged_mode` at its default, log `dmesg` for `Engine reset`, and keep the container's `restart: unless-stopped`.
4. **ASPM on AMD.** `pcie_aspm.policy=powersupersave` bricks the B70 at boot on AMD platforms (L1.1 wake failure; LKML quirk submitted May 2026, unmerged). Use the default policy or `pcie_aspm=off`.

---

## 8. Host and platform checklist (5800X / AM4)

The lab measured CPU governor, EPP, CCD pinning, `iommu=pt`, ACS override, GuC 70.44 vs 70.72 and ECC as "none to 2%" once XPU graphs were on. Do these once, in this order, and stop:

1. **Verify the PCIe topology.** The 5800X has 24 PCIe 4.0 lanes; two GPUs are either x8/x8 from the CPU (X570/B550 boards with two CPU-attached x16 slots, bifurcated) or x16 + a chipset-attached slot behind the x4 chipset uplink. The second layout puts every allreduce through the chipset and shares it with NVMe and the 10 GbE NIC.
   ```bash
   lspci -t; for d in /sys/bus/pci/devices/0000:*; do [ -e $d/current_link_width ] && grep -q 0x8086 $d/vendor && echo "$d $(cat $d/current_link_speed) x$(cat $d/current_link_width) (max $(cat $d/max_link_speed) x$(cat $d/max_link_width))"; done
   xpu-smi topology -m
   ```
   Expect `16.0 GT/s x8` on both cards for the good layout. The card advertises Gen5 x16 but the 5800X negotiates Gen4. (The internal bridge may print "Gen1 x1"; that is a known reporting artifact, read the root-facing link.)
2. **Resizable BAR / Above 4G decoding on**, else `xe` cannot map the 32 GB aperture and `vram_bar_size` will fail; check `lspci -vvv -s <bdf> | grep -A2 'Physical Resizable BAR'`.
3. **IOMMU:** `amd_iommu=on iommu=pt` (passthrough keeps DMA-BUF P2P off the IOMMU page-walk path). No `pcie_acs_override` is needed on AM4 CPU root ports and the cookbook confirms it does nothing for the L0 IPC failure.
4. **Power:** 230 W cap per card (3.6). Check the PSU: two cards at 230 W plus the 5800X is ~650 W sustained.
5. **ECC:** `xpu-smi config -d 0 --memoryecc 0` (reboot) returns ~4 GB of VRAM per card (28 -> 32 GB visible). It does not change bandwidth; use it only if you want more KV headroom.
6. **Docker:** `--cap-add SYS_PTRACE --ipc=host` is sufficient; `--privileged` is not needed (cookbook). Keep `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
7. **Known Battlemage multi-GPU gotchas** (chriswagner-ai/intel-arc-b70-vllm-multi-gpu, four B70s, TP2/TP4): the multi-root L0 USM host-pool defect that OOMs TP at `init_device` ("OOM 2 MiB") is fixed in compute-runtime >= 26.14 (`028e23e576`), so your 26.27 image is past it; `TRITON_INTEL_DEVICE_ARCH=bmg` must be exported for spawned workers (you have it); after a `kill -9`, orphaned `VLLM::EngineCore` / `VLLM::Worker` processes hold ~30 GB per card, so `pkill -9 -f 'VLLM::(EngineCore|Worker)'` before relaunch; serialize model loads when starting several engines to avoid host RSS spikes.
8. **Direct measurement of P2P:** build `ze_peer` from `oneapi-src/level-zero-tests` (`perf_tests/ze_peer`) and run `ze_peer -t transfer_bw` and `-t transfer_latency` between devices 0 and 1; that gives the hardware ceiling your oneCCL numbers should be compared to.

---

## 9. Implementation plan

### Phase 1 (one evening, no rebuild): expected 80 -> 90 to 95 tok/s
1. Set both cards to 230 W; confirm no throttling.
2. Change `--block-size 32` -> `64`.
3. `SPEC={"method":"mtp","num_speculative_tokens":4}`; `--gpu-memory-utilization 0.88`.
4. Add `CCL_SYCL_ALLREDUCE_LL=twoshots`, `CCL_SEND=direct`, `CCL_RECV=direct`; remove `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`. Run `tools/xccl_allreduce_census.py` before and after.
5. Extend `--compilation-config` with the capture-size list and `"splitting_ops": []` from 3.3; watch for the phantom-token symptom at MTP depth 2 if you ever go back to piecewise.
6. Add the three prefix-cache correctness patches (7.1) or disable prefix caching on the fast profile.
7. Benchmark: `bench_vllm.py ... 1 200` and `... 16 200`, plus a fixed-prompt exactness check (same prompt, temperature 0, 5 runs, compare token ids).

### Phase 2 (a weekend, image rebuild): expected -> 105 to 120 tok/s
1. Pull `ghcr.io/steveseguin/vllm-openai-xpu-qwen38-int4@sha256:521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad` and run [`configs/lab-mtp4-int4-draft-head.sh`](../configs/lab-mtp4-int4-draft-head.sh) with your model to confirm the INT4-draft-head number on your host. That isolates "host vs stack".
2. Port `VLLM_XPU_DRAFT_LM_HEAD_INT4` (+ R256 fallback) and the sync-free GDN grouping onto a v0.29.0 base; drop your `patch_mtp_ptr_wrap.py`; keep the other six.
3. Rebuild the container with compute-runtime 26.31.39395.13 + IGC 2.40.13.
4. A/B FP16 KV at 131K vs FP8 KV at 262K as two served profiles.

### Phase 3 (optional, research)
1. compute-runtime `master` build with the BMG preemption timer and OOQ sync changes; A/B on the census script.
2. AutoRound re-quantization of the uncensored base to unlock W4A8 prefill.
3. DFlash2 retest per 5.3 on vLLM `main` >= `d61b6e18`.

---

## 10. Sources

Additional community field notes read: `chriswagner-ai/intel-arc-b70-vllm-multi-gpu` (4x B70 bare-metal TP2/TP4, symptom-to-fix table), `kkornas/intel-arc-pro-b70-ubuntu-guide` (Ubuntu driver bring-up, compute-runtime 26.18 via the kobuk-team PPA).

Source trees read directly: `vllm-project/vllm` (tags v0.28.0, v0.29.0, main @ 2026-09-12), `vllm-project/vllm-xpu-kernels` (main @ 2026-09-11), `uxlfoundation/oneCCL` 2022.1.0 incl. `deps/libccl`, `intel/compute-runtime` (master @ 2026-09-12, tags 26.27.39122.11, 26.31.39395.13), `intel/llm-scaler` (main), `intel/intel-xpu-backend-for-triton`, `pytorch/pytorch` (v2.13.0, main), `intel/torch-xpu-ops`, Linux `drivers/gpu/drm/xe` (v6.16 to v6.19), `SergiioB/intel-arc-pro-b70-inference-cookbook`, `steveseguin/b70-optimization-lab` (public domain; two probe scripts copied into `tools/`), `PMZFX/intel-arc-pro-b70-benchmarks`, `z-lab/dflash`.

Web references (search snippets only; most pages are blocked from this environment):
- vLLM PR #52816 (DFlash2), issue #53428 (`decoder_layer_cls` regression), PR #54373 (DFlash RoPE layout), issue #41559 (DFlash vs quantized KV), PR #39995 (FlashInfer DFlash FP8 KV), PR #50501 (XPU INC W4A8), issue #48109 (Mamba pointer overflow), PRs #53919 / #48375 / #53505 (hybrid prefix-cache correctness), issue #52047 (mamba prefix reuse disabled), issue #41663 (dual B70 TP2 engine resets).
- intel/compute-runtime issues #942 (quad-B70 peer memcpy corruption), #948 (permanent wedge after engine reset); intel/llm-scaler issues #463 (L0 V2 multi-device context), #594 (dual B60 oneCCL thresholds).
- Inco AI, "DFlash 2: Keep Drafting Parallel" (2026-08); z-lab DFlash paper arXiv:2602.06036.
- Intel Arc Pro B70 specs: 32 Xe2 cores, 256 XMX, 32 GB GDDR6 256-bit, 608 GB/s, PCIe 5.0 x16, 230 W TBP (160 to 290 W partner range).
