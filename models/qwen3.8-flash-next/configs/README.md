# Configs

Launch scripts for Qwen3.8-Flash-Next, one per named configuration. llama.cpp patches are in [`patches/`](patches/).

Replace model paths, the bind address and the API key file with your own before running. The script never reads a key from an environment file; it takes a path to a 0600 key file.

| File | Description | Status |
|---|---|---|
| [llama-server-262k-mtp.sh](llama-server-262k-mtp.sh) | llama.cpp SYCL, 4 cards, `-sm layer -ts 12,13,13,11`, 262,144 context, f16 KV, `-ub 1536`, MTP draft (n-max 3) on the last card, sparse FA on, vision projector; optional `SLOT_DIR`, `CACHE_RAM` and `SIGNATURE` for the prompt cache across restarts | production since 2026-09-25; build `20260926-2b84213a4` (patches 0002-0017, [list](patches/README.md)) since 2026-09-26, with the draft QSA and the grouped MoE GEMM on by default |
| [llama-server-slot-cache.sh](llama-server-slot-cache.sh) | saves slot 0 to `SLOT_DIR` on stop and restores it on start, only into an identical setup (signature file); `hold` / `release` protect the save during benchmarks; needs `--slot-save-path` (set `SLOT_DIR` in the launch script) and, with a draft, [patch 0010](patches/0010-server-slot-save-restore-draft-context.diff) | production since 2026-09-25 (build `20260925-993baf141`) |

## Settings that matter

| Setting | Value | Why | Evidence |
|---|---|---|---|
| Split | `-sm layer`, `-ts 12,13,13,11`, `-fit off` | `-sm tensor` excludes qwen4exp and the SYCL all-reduce supports only 2 devices; the fitter cannot size the MTP draft | [baseline](../experiments/2026-09-24-baseline-llamacpp-mtp-262k.md) |
| Ubatch | `-ub 1536`, `-b 2048` | +4.5% prompt speed at 39k over 1024; 2048 does not fit with MTP at 262k | [ubatch 1536](../experiments/2026-09-25-ubatch-1536.md) |
| KV cache | f16 | KV is only about 6 GiB at 262k (12 attention layers, 2 KV heads, head dim 256); quantized KV was not tested here | [model card](../README.md) |
| MTP | shared-Q8_0 draft, `--spec-draft-n-max 3`, `-devd SYCL3` | n-max 4 and n-max 6 with p-min 0.75 gave no consistent gain | [draft sweep](../experiments/2026-09-24-mtp-draft-length-sweep.md) |
| Sparse FA | `GGML_SYCL_SPARSE_FA=1` | +41% to +55% decode at 135K with patch 0006 | [sparse FA](../experiments/2026-09-25-sparse-fa-multi-token.md) |
| Pooled QSA cache | on by default with patch 0007 | MTP step at 135K about 95-96 ms to 72-73 ms | [pooled cache](../experiments/2026-09-25-pooled-qsa-key-cache.md) |
| Grouped MoE (mixture-of-experts) GEMM | on by default with patches 0012-0015; `GGML_SYCL_XMX_GATHER_TYPES=0` restores the per-expert loop | one 1536-token expert matmul (512 experts, 10 used): q4_K 11.31 to 3.93 ms, q5_K 11.94 to 4.44, q5_1 11.02 to 3.78, q8_0 10.94 to 7.40; cold 135K read 382 s to 312-313 s (N=3) | [grouped GEMM](../experiments/2026-09-26-grouped-moe-xmx-gemm.md), [raw kernel](../benchmarks/raw/2026-09-26-mul-mat-id-kernel-perf.csv), [raw cold reads](../benchmarks/raw/2026-09-26-cold-read-progress.csv) |
| Draft QSA | on by default with patch 0009; `LLAMA_MTP_QSA=0` or the file `/tmp/llama-mtp-qsa-off` gives the dense draft | 10-turn session at 135K: 70.5 ms per MTP step and 0.92 s prompt median, against 72.7-73.4 ms and 1.24-1.27 s in the two dense-draft sessions of the same day; the paired A/B is in the experiment | [draft QSA](../experiments/2026-09-25-mtp-draft-qsa.md), [raw](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv) (`lc-draftqsa`, `lc-mtpwin-0-allrows`, `lc-mtpwin-0-rowsonly`) |
| GPU clock floor | `min_freq = rp0` (2800 MHz) while serving, via [tools/gpu-clock-floor.sh](../../../tools/gpu-clock-floor.sh) | layer split leaves each card idle between bursts | [clock floor](../experiments/2026-09-24-gpu-clock-floor.md) |
| Sampling defaults | `--temp 1.0 --top-p 0.95 --top-k 20` | production sampling for clients; every benchmark here overrides it to temperature 0 per request | |
| Slots | `-np 1` | one user gets the full 262,144-token context | |

## Prompt cache across restarts

A restart empties the server's KV (key/value) cache, and re-reading a long agent conversation costs minutes: 403-472 s for a 135K-token code context on the 2026-09-25 builds ([raw summary CSV](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv)). Since 2026-09-25 the production service saves slot 0 on every stop and restores it on every start. Measured at 135,760 tokens: save 1.5 s (3.7 GB + 303 MB for the MTP draft), restart 88 s, first follow-up turn 3.92 s, later turns 0.78-0.80 s ([raw slot-cache CSV](../benchmarks/raw/2026-09-26-slot-cache-timings.csv), [experiment](../experiments/2026-09-25-slot-save-restore-and-ram-cache.md), [finding](../../../findings/llama-server-restart-drops-prefix-cache.md)).

Three optional variables in [`llama-server-262k-mtp.sh`](llama-server-262k-mtp.sh); empty values leave the flags out:

| Variable | Flag or file | What it does | Production value |
|---|---|---|---|
| `SLOT_DIR` | `--slot-save-path "$SLOT_DIR"` | directory for the saved slot (`slot0.bin`, `slot0.bin.dft`, `slot0.sig`, `slot0.meta`, `hold`). The script creates it with mode 0700: the file holds the conversation text | set (a local directory) |
| `CACHE_RAM` | `--cache-ram "$CACHE_RAM"` | host-RAM prompt cache in MiB. When a second request takes the single slot, the displaced conversation is kept in RAM and loaded back instead of re-read. Default 8192 MiB ([upstream server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), checked 2026-09-26); a 135K conversation is about 4.0 GB of state with the draft | `24576` |
| `SIGNATURE` | a file, one `key=value` per line | written before `exec`: `model`, `mtp`, `bin`, `ctx`, `kv=f16/f16`, `np=1`, `mtp_qsa`, `mtp_window`. `llama-server-slot-cache.sh restore` loads a save only when this file is byte-identical to the `slot0.sig` saved with it. A new build (`bin`) therefore starts cold; for a kernel-only build, copy the running signature over `slot0.sig` and run `restore` (0.89 s to the first turn at 41K, same CSV) | a file under the unit's `RuntimeDirectory` |

The systemd pattern: save from `ExecStop=`, restore from `ExecStartPost=` after the server answers `/health`, and leave time for the write.

```ini
[Service]
RuntimeDirectory=llama-server
RuntimeDirectoryMode=0700
Environment=SLOT_DIR=/path/to/slots CACHE_RAM=24576 SIGNATURE=/run/llama-server/slot-signature
ExecStart=/path/to/configs/llama-server-262k-mtp.sh
# bring back the prompt cache saved at the last stop; never fails the unit
ExecStartPost=/path/to/configs/llama-server-slot-cache.sh restore
# save the prompt cache before stopping, so a long conversation survives the restart
ExecStop=/path/to/configs/llama-server-slot-cache.sh save
# room for the save (about 4 GB at 135K tokens)
TimeoutStopSec=240
```

Rules the script enforces and rules we follow by hand:

- Saves under `MIN_TOKENS` (2,048) are discarded, so a restart with an empty slot keeps the previous save.
- Before benchmarks or tests that fill the slot with their own prompts: `llama-server-slot-cache.sh hold` saves now and creates `$SLOT_DIR/hold`; while that file exists, automatic saves do nothing. `release` removes it. `status` shows what is saved and whether its signature matches the running server.
- Restart only when the current instance has served no requests. A crash cannot save, and conversations with images cannot be saved (the server refuses); in both cases the next start restores the last good save.
- Patch 0010 is needed with a draft: upstream saves only the target model's context, and a slot restored without the draft's `.dft` file drafts with no prefix behind it.
