# Startup patches

Applied in order by the production compose file before `vllm serve`. The patch sources are not committed yet; add them here so the config is reproducible. Upstream status is tracked in [engines/vllm-xpu](../../../../engines/vllm-xpu/README.md#patch-catalog).

| Order | Patch | Purpose |
|---|---|---|
| 1 | `patch_mtp_nightly.py` | BF16 MTP draft head with a quantized base (`B70_MTP_BF16_DRAFT=1`) |
| 2 | `patch_mtp_boundary.py` | off-by-one in speculative verification at context boundaries |
| 3 | `patch_gdn_mixed_split_v5.py` | Gated DeltaNet mixed split on Battlemage |
| 4 | `patch_worker_affinity.py` | per-rank `ZE_AFFINITY_MASK` at worker spawn |
| 5 | `patch_mtp_ptr_wrap.py` | Mamba state pointer overflow; fixed upstream in vLLM 0.29.0 |
| 6 | `patch_sleep_graphs_v5.py` | release and recapture graphs around sleep mode |
| 7 | `patch_mem_report2.py` | memory diagnostics RPC endpoints |
