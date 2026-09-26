# Configs

Launch scripts for Qwen3.8-Flash-Next, one per named configuration. llama.cpp patches are in [`patches/`](patches/).

Replace model paths, the bind address and the API key file with your own before running. The script never reads a key from an environment file; it takes a path to a 0600 key file.

| File | Description | Status |
|---|---|---|
| [llama-server-262k-mtp.sh](llama-server-262k-mtp.sh) | llama.cpp SYCL, 4 cards, `-sm layer -ts 12,13,13,11`, 262,144 context, f16 KV, `-ub 1536`, MTP draft (n-max 3) on the last card, sparse FA on, vision projector | production since 2026-09-25 (build `20260925-f47a6a5f3`) |

## Settings that matter

| Setting | Value | Why | Evidence |
|---|---|---|---|
| Split | `-sm layer`, `-ts 12,13,13,11`, `-fit off` | `-sm tensor` excludes qwen4exp and the SYCL all-reduce supports only 2 devices; the fitter cannot size the MTP draft | [baseline](../experiments/2026-09-24-baseline-llamacpp-mtp-262k.md) |
| Ubatch | `-ub 1536`, `-b 2048` | +4.5% prompt speed at 39k over 1024; 2048 does not fit with MTP at 262k | [ubatch 1536](../experiments/2026-09-25-ubatch-1536.md) |
| KV cache | f16 | KV is only about 6 GiB at 262k (12 attention layers, 2 KV heads, head dim 256); quantized KV was not tested here | [model card](../README.md) |
| MTP | shared-Q8_0 draft, `--spec-draft-n-max 3`, `-devd SYCL3` | n-max 4 and n-max 6 with p-min 0.75 gave no consistent gain | [draft sweep](../experiments/2026-09-24-mtp-draft-length-sweep.md) |
| Sparse FA | `GGML_SYCL_SPARSE_FA=1` | +41% to +55% decode at 135K with patch 0006 | [sparse FA](../experiments/2026-09-25-sparse-fa-multi-token.md) |
| Pooled QSA cache | on by default with patch 0007 | MTP step at 135K about 94 ms to 72-73 ms | [pooled cache](../experiments/2026-09-25-pooled-qsa-key-cache.md) |
| GPU clock floor | `min_freq = rp0` (2800 MHz) while serving, via [tools/gpu-clock-floor.sh](../../../tools/gpu-clock-floor.sh) | layer split leaves each card idle between bursts | [clock floor](../experiments/2026-09-24-gpu-clock-floor.md) |
| Sampling defaults | `--temp 1.0 --top-p 0.95 --top-k 20` | production sampling for clients; every benchmark here overrides it to temperature 0 per request | |
| Slots | `-np 1` | one user gets the full 262,144-token context | |

## Not included

The production service also saves and restores the prompt-cache slot across restarts and keeps a host-RAM prompt cache. Those flags are host-specific and are left out of this script.
