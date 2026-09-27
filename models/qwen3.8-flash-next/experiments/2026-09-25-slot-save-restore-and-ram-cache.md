# Slot save/restore across restarts, with the MTP draft, plus a 24 GiB host-RAM prompt cache

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept (production since 2026-09-25) |
| **Baseline** | a cold re-read of the conversation after every restart: 403-472 s for the 134,862-token [`tools/lcbench.py`](../../../tools/lcbench.py) context (N=10 sessions, [raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv)); 455.35 s in this test window ([raw/2026-09-26-slot-cache-timings.csv](../benchmarks/raw/2026-09-26-slot-cache-timings.csv)) |
| **Result** | [raw/2026-09-26-slot-cache-timings.csv](../benchmarks/raw/2026-09-26-slot-cache-timings.csv) |
| **Related** | [patch 0010](../configs/patches/0010-server-slot-save-restore-draft-context.diff), [`configs/llama-server-slot-cache.sh`](../configs/llama-server-slot-cache.sh), [configs README: prompt cache across restarts](../configs/README.md#prompt-cache-across-restarts), [finding: a restart empties the prompt cache](../../../findings/llama-server-restart-drops-prefix-cache.md), [pooled QSA key cache](2026-09-25-pooled-qsa-key-cache.md) |

A service restart no longer costs a long-context agent its conversation. Saving the single slot on stop and restoring it on start brought a 135,760-token conversation back in one 88 s restart; the first follow-up turn took 3.92 s and later turns 0.78-0.80 s, instead of a cold re-read of 403-472 s. The MTP (multi-token prediction) draft's copy of the context is saved too ([patch 0010](../configs/patches/0010-server-slot-save-restore-draft-context.diff)); upstream saves only the target model's context. A saved slot is restored only into an identical setup, checked by a signature file, and re-signing a save lets a kernel-only build change keep the conversation warm.

## Hypothesis

llama-server keeps the KV (key/value) cache in GPU memory, so every restart empties it. On 2026-09-25 that produced an outage as seen from a coding-agent client: a restart for a new build emptied the cache, benchmark prompts then filled the single slot, and when the client returned its conversation of about 166K tokens had to be re-read cold. The read ran at about 290 tok/s, about 9.5 minutes to the first token, and the client's own timeout cancelled it at 88% (source: session log, not archived). The rate matches the archived cold reads of that build: 301.7 tok/s over 134,864 tokens, 447.01 s ([raw/2026-09-26-cold-read-progress.csv](../benchmarks/raw/2026-09-26-cold-read-progress.csv), run `2026-09-25T23:27:50`).

llama-server can write a slot's state to disk and read it back (`--slot-save-path`, `POST /slots/0?action=save|restore`; [upstream server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), checked 2026-09-26). Expected: a restart costs seconds of disk I/O for about 4-5 GB of state at 135K-166K tokens (estimate, before measurement) instead of a 7-10 minute re-read. Two conditions had to hold. The draft context must come back as well, or the draft would propose tokens with no prefix behind them. And a save must never be loaded into a server with a different model, build or KV layout.

## Change

Four pieces, all shipped on 2026-09-25 and in production since build `20260925-993baf141`.

**1. Patch 0010: save and restore the draft context.** [0010-server-slot-save-restore-draft-context.diff](../configs/patches/0010-server-slot-save-restore-draft-context.diff), commit message: "server: slot save/restore also saves and restores the speculative draft context (<file>.dft)". The save writes `llama_state_seq_save_file` of the draft context to `<file>.dft` next to the slot file and reports both sizes in `n_bytes`. The restore loads `<file>.dft` if it exists; if it is missing or unreadable, the draft's sequence is cleared (`llama_memory_seq_rm`) so the draft starts empty rather than on whatever it held before.

**2. [`llama-server-slot-cache.sh`](../configs/llama-server-slot-cache.sh)** (scrubbed copy of the production script): `save`, `restore`, `hold`, `release`, `status`. Rules built into it:

- **Signature rule.** The launch script writes one `key=value` per line to the `SIGNATURE` file before it execs llama-server: `model`, `mtp`, `bin` (the build directory), `ctx`, `kv=f16/f16`, `np=1`, `mtp_qsa`, `mtp_window`. `save` copies that file to `slot0.sig` next to the slot. `restore` runs only when the running server's signature file is byte-identical to `slot0.sig` (`cmp -s`); otherwise it logs "saved slot is from a different setup; not restoring" and exits 0. Because `bin` is in the signature, every new build starts cold by default.
- **Re-signing.** When a build changes kernels only and not the state layout, copy the running server's signature file over `slot0.sig` and run `restore`; the conversation stays warm. The 2026-09-26 builds ([patches 0011-0017](../configs/patches/)) change SYCL kernels, graph construction and diagnostics; a re-signed restore across them worked (0.89 s to the first turn, raw CSV), which is the evidence that they left the saved state's layout alone. That they never could change it is UNVERIFIED: check the restore log and the first turn's acceptance after every re-sign. The signature exists so that nobody has to reason about that at 2 a.m.; re-signing is the deliberate override.
- **Minimum size.** Saves under `MIN_TOKENS` (2048) are discarded, so a restart with an idle or nearly empty slot keeps the previous save.
- **Hold file.** `hold` saves now and creates `$SLOT_DIR/hold`; while it exists, `save` does nothing, so benchmarks that fill the slot with their own prompts cannot overwrite the agent's saved conversation. `release` removes it.
- The script never fails its caller (always exits 0). The API key, if any, goes to `curl` on stdin, never on the command line. The slot file holds the conversation text, so `SLOT_DIR` is mode 0700.

**3. Launch script and unit.** [`llama-server-262k-mtp.sh`](../configs/llama-server-262k-mtp.sh) gained three optional variables; empty values leave the flags out.

```diff
+#   SLOT_DIR      directory for saving the prompt-cache slot across restarts (0700); empty disables --slot-save-path
+#   CACHE_RAM     host-RAM prompt cache in MiB for displaced conversations (--cache-ram); empty keeps the default
+#   SIGNATURE     file to write the setup signature to, for llama-server-slot-cache.sh; empty skips it
...
+if [ -n "$SLOT_DIR" ]; then
+  mkdir -p "$SLOT_DIR" && chmod 700 "$SLOT_DIR"
+  SLOTS+=(--slot-save-path "$SLOT_DIR")
+fi
+[ -n "$CACHE_RAM" ] && SLOTS+=(--cache-ram "$CACHE_RAM")
+if [ -n "$SIGNATURE" ]; then
+  printf '%s\n' "model=$MODEL" "mtp=$MTP" "bin=$BIN" "ctx=$CTX" "kv=f16/f16" "np=1" \
+      "mtp_qsa=${LLAMA_MTP_QSA-}" "mtp_window=${LLAMA_MTP_WINDOW-}" > "$SIGNATURE"
+fi
```

The systemd unit runs `llama-server-slot-cache.sh save` from `ExecStop=` and `llama-server-slot-cache.sh restore` from `ExecStartPost=`, after the server answers `/health`. `TimeoutStopSec` was raised to 240 s to leave room for the save. The pattern is in the [configs README](../configs/README.md#prompt-cache-across-restarts).

**4. `--cache-ram 24576`.** llama-server keeps the state of a displaced slot in host RAM (its prompt cache) and loads it back when that prompt returns; the default budget is 8192 MiB ([upstream server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), checked 2026-09-26). With one slot, any side request from the client (a title, a compaction, a sub-agent) displaces the main conversation. A 135,760-token conversation is 3.7 GiB plus 303 MiB for the draft on disk (raw CSV; `du -h` sizes), so the 8 GiB default holds about two such conversations and 24 GiB about six, or one and four at 166K (estimate from the save size; the in-RAM state is the same data). Host RAM allows it: MemAvailable stayed at or above 91.5 GiB during a 219K run ([pooled cache experiment](2026-09-25-pooled-qsa-key-cache.md)).

**5. Operating rule.** Restart only when the current instance has served no requests. A restart still evicts whatever is not saved, and a crash cannot save.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Build `20260925-993baf141`: PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 + 0004 + #28796 + 0006 + 0007 + 0008 + 0009 + 0010. The 2026-09-26 re-signed restore ran on a later build with the same state layout |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1536 -ts 12,13,13,11 -fit off`, MTP n-max 3 on `SYCL3`, f16 KV, `GGML_SYCL_SPARSE_FA=1`, `-np 1`, `--slot-save-path`, `--cache-ram 24576` |
| GPU clock floor | 2800 MHz |
| Slot storage | local NVMe, directory mode 0700 |

## Procedure

1. Start the new build with `--slot-save-path`; the first `restore` finds nothing ("nothing saved"). On the previous build the `save` had failed with HTTP 501, which is what a server started without `--slot-save-path` returns.
2. Build a conversation with [`tools/lcbench.py`](../../../tools/lcbench.py): one cold read of the 134,862-token C++ context, then follow-up turns of about 60 new tokens, up to 135,760 tokens in the slot.
3. Restart the service. `ExecStop` saves; `ExecStartPost` restores. Time the whole restart.
4. Continue the same conversation with three more turns and record per-turn prompt time, decode speed and draft acceptance.
5. Repeat the cycle with a second conversation of 136,528 tokens.
6. Over the 2026-09-26 test windows, keep a 41,364-token conversation saved with `hold` before each window and `release` after it; the production build restores it after every window.
7. After a kernel-only build change (`bin` differs, layout does not), copy the running signature over `slot0.sig`, run `restore`, and time the first follow-up turn.

Nothing here used the vision projector. Conversations with images cannot be saved: the server refuses, and the previous save is kept.

## Results

All numbers from [raw/2026-09-26-slot-cache-timings.csv](../benchmarks/raw/2026-09-26-slot-cache-timings.csv) unless stated.

| Metric | Cold re-read (before) | Slot save/restore (this change) |
|---|---|---|
| Time from restart to the first follow-up turn's answer, 135K conversation | 403-472 s cold read (N=10 sessions, 2026-09-25 builds, [summary CSV](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv)); 455.35 s in this window | 88 s restart (model load included) + 3.92 s first turn |
| Follow-up turns after that, ~60 new tokens | 0.92 s median on the same code without the slot patch (`lc-draftqsa`, [summary CSV](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv)) | 0.78-0.80 s (2 turns) |
| Save, 135,760 tokens | | 1.5 s; 3.7 GiB + 303 MiB draft |
| Save, 136,528 tokens | | 1.5 s; 3.7 GiB + 305 MiB draft |
| Save, 41,364 tokens | | 0.4 s; 1.2 GiB + 93 MiB draft, the same on all 14 saves |
| Restore, 41,364 tokens, after re-signing for a new build with the same layout (2026-09-26) | | 0.89 s to the first turn |
| Restores refused by the signature rule, 2026-09-25T23:49 to 2026-09-26T02:10 | | 11 starts of a differently signed build, each logged "saved slot is from a different setup; not restoring"; no cross-build restore happened |
| Draft acceptance after a restore | | normal |
| RAM prompt cache: a displaced conversation of about 40K tokens brought back | cold re-read | 0.58 s (source: session log, not archived) |
| Single-stream tok/s, aggregate at c8 / c16 | unchanged: no kernel or graph change | not applicable (one slot) |

State size is about 29 KB per token for the target model and about 2.3 KB per token for the draft (3.7 GiB and 303 MiB at 135,760 tokens, as `du -h` reports them), so a 166K conversation is about 4.9 GiB (estimate, linear scaling).

The 3.92 s first turn is a one-time cost. The pooled QSA (Qwen sparse attention) indexer-key cache of [patch 0007](../configs/patches/0007-qwen4exp-pooled-qsa-key-cache.diff) resets its watermark on a full `state_read`, so the first ubatch (micro-batch) after a restore re-pools every complete 4-token block of the context before it can score ([pooled cache experiment](2026-09-25-pooled-qsa-key-cache.md), "Watermark"). The next turns are back at the cached-prefix speed.

## Correctness

- Draft acceptance on the three turns after the restore was normal (raw CSV note), which shows the draft's context came back with the target's. Without the `.dft` file the draft starts empty (patch 0010 clears its sequence); how far acceptance would fall in that case was not measured (UNVERIFIED).
- The server's own restore check passed: `n_restored` equalled the saved token count on every logged restore (135,760, 136,528 and 41,364 tokens).
- The signature rule refused all 11 cross-build starts, and the two re-signed restores on 2026-09-26 were only done between builds that differ in kernels (source for the count: session log, not archived).
- No teacher-forced comparison of a restored slot against a continuous session was run. Whether the restored state reproduces the continuous session's next-token distribution exactly is UNVERIFIED; greedy output is not token-identical run to run on this model anyway ([pooled cache experiment](2026-09-25-pooled-qsa-key-cache.md), "Correctness").
- Not a kernel, graph or sampling change, so the greedy correctness check of [`benchmarks/README.md`](../../../benchmarks/README.md) was not repeated for it.

## Decision

Kept, in production since 2026-09-25. A restart now costs about 90 s plus a 4 s first turn instead of a 7-8 minute re-read, saves cost 0.4-1.5 s at stop, and `--cache-ram 24576` keeps a displaced conversation resident. The operating rule stays: restart only when the running instance has served no requests, because a crash or an unsaved slot still starts cold.

## Follow-ups

- [ ] Put a state-layout key in the signature instead of the build path, so kernel-only builds restore without manual re-signing.
- [ ] Measure the RAM-cache reload of a 135K conversation; only a ~40K case was timed and it is not archived.
- [ ] Conversations with images cannot be saved. Find out whether the refusal is in the server or in the multimodal state and whether it can be lifted.
- [ ] Upstream the draft save/restore (patch 0010); the upstream server saves only the target context.
- [ ] Add a size check: the save is refused or truncated if the disk fills; `TimeoutStopSec=240` assumes about 5 GB writes in well under a minute.
