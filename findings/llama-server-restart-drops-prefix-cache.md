# A llama-server restart empties the prompt cache, and a long-context agent then waits minutes for a cold re-read

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Confidence** | verified-here |
| **Applies to** | llama-server (llama.cpp) with any model; measured on Qwen3.8-Flash-Next with builds based on PR #28243 on 4x Arc Pro B70, single slot, MTP draft. Any engine whose KV cache lives only in process memory behaves the same in principle (UNVERIFIED here for vLLM) |
| **Area** | engine / host (operations) |

## Finding

llama-server holds the KV (key/value) cache of every slot in GPU memory, so a restart, for a new build, a changed flag or a crash, discards it. The next request from a client with a long conversation is a cold read of the whole conversation. On four B70s that is 403-472 s for a 135K-token code context and about 9.5 minutes for 166K, long enough for a coding-agent client's own timeout to cancel the request, so the user sees a server that does not answer. Saving the slot to disk on stop and restoring it on start, with the speculative draft's context included, cuts the cost of a planned restart to the restart itself plus a 3.92 s first turn.

## Evidence

**The incident (2026-09-25).** A service restart for a new build emptied the prompt cache; benchmark prompts then filled the single slot. When the coding-agent client returned, its conversation of about 166K tokens had to be re-read cold at about 290 tok/s, about 9.5 minutes to the first token, and the client's timeout cancelled the request at 88% (source: session log, not archived). The archived cold-read rate of that build agrees: 301.7 tok/s over 134,864 tokens, 447.01 s ([raw cold-read progress CSV](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-cold-read-progress.csv), run `2026-09-25T23:27:50`).

**What a cold read costs** ([raw lcbench summary CSV](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv), 134,862-token C++ context, [`tools/lcbench.py`](../tools/lcbench.py)):

| Build date | Cold read of 134,862 tokens | N |
|---|---|---|
| 2026-09-24 / 2026-09-25 builds | 403-472 s | 10 sessions |
| 2026-09-26 build `2b84213a4` | 284-286 s (283.66 s at the default flags on tree `a890bf8b0`, the same code without the diagnostic patch 0017; 286.44 s on `2b84213a4` with `-b 3072`; [raw cold-read windows CSV](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-cold-read-windows.csv) rows `bcp5hgsyz` and `bsw2jblow`; the second is also `lc-test` in the summary CSV) | 2 |

Even the fastest build reads a 135K conversation for close to five minutes. Follow-up turns on a cached prefix take about 0.9-1.4 s on the 2026-09-24 and 2026-09-25 builds (same CSV, `prompt_median_s` of the 135K sessions) and 0.76-0.77 s on the 2026-09-26 code with the in-place grouped MoE GEMM ([windows CSV](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-cold-read-windows.csv), rows `b2pfxuo2i` and `bcp5hgsyz`), which is what the client normally sees. After a slot restore on the 2026-09-25 build they took 0.78-0.80 s ([slot-cache CSV](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-slot-cache-timings.csv)).

**What save/restore costs** ([raw slot-cache timings CSV](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-slot-cache-timings.csv), [experiment](../models/qwen3.8-flash-next/experiments/2026-09-25-slot-save-restore-and-ram-cache.md)):

| Step | 135,760 tokens | 41,364 tokens |
|---|---|---|
| Save on stop | 1.5 s; 3.7 GiB + 303 MiB draft | 0.4 s; 1.2 GiB + 93 MiB draft (14 saves, all the same) |
| Restart wall time, model load included | 88 s | not timed |
| First follow-up turn after the restore | 3.92 s | 0.89 s (after re-signing for a new build with the same layout) |
| Later turns | 0.78-0.80 s | |
| Draft acceptance | normal | |

The 3.92 s first turn is the pooled QSA (Qwen sparse attention) key cache rebuilding after a full state read ([pooled cache experiment](../models/qwen3.8-flash-next/experiments/2026-09-25-pooled-qsa-key-cache.md)); it is one turn, not a re-read.

**Upstream saves only the target context.** With a speculative draft (`--spec-draft-model`, here the MTP (multi-token prediction) head) the draft keeps its own cache of the sequence. The upstream slot save writes the target's state only, so a restored slot would draft with no prefix behind it. [Patch 0010](../models/qwen3.8-flash-next/configs/patches/0010-server-slot-save-restore-draft-context.diff), commit message "server: slot save/restore also saves and restores the speculative draft context (<file>.dft)", writes the draft state next to the slot file and clears the draft's sequence when no usable `.dft` file exists.

**The signature rule worked.** Between 2026-09-25T23:49 and 2026-09-26T02:10, 11 starts of builds with a different signature logged "saved slot is from a different setup; not restoring" (same CSV); no save was loaded into a mismatched server.

## Impact

- A planned restart of a server that holds a 135K conversation costs the operator's restart time plus about 4 s, instead of 7-8 minutes of dead air on the 2026-09-25 builds (about 5 minutes on the 2026-09-26 build) and a possibly cancelled request. At 166K the re-read would be about 9.5 minutes.
- Without the draft context in the save, the restored slot drafts with no prefix behind it (patch 0010's comment); how far acceptance falls in that case was not measured (UNVERIFIED). With patch 0010 acceptance was normal on the first restored turn.
- `--cache-ram` covers a different case: one slot, and a side request (a title, a compaction, a sub-agent) displaces the main conversation. llama-server keeps the displaced state in host RAM up to `--cache-ram` MiB, default 8192 ([upstream server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), checked 2026-09-26). A 135K conversation is about 4.0 GiB of state with the draft (save size as `du -h` reports it, same CSV), so the default holds about two and 24,576 MiB about six; at 166K (about 4.9 GiB) about one and four (estimate, assuming the in-RAM state is the size of the save). A displaced conversation of about 40K tokens came back from RAM in 0.58 s (source: session log, not archived).

## What to do

1. **Start llama-server with `--slot-save-path <dir>`** (mode 0700: the file holds the conversation) and save slot 0 on every stop, restore on every start. [`configs/llama-server-slot-cache.sh`](../models/qwen3.8-flash-next/configs/llama-server-slot-cache.sh) does both and never fails the unit: `ExecStop=... save`, `ExecStartPost=... restore` after `/health` answers. Give the stop enough time for the write (`TimeoutStopSec=240` here). The unit pattern is in the [configs README](../models/qwen3.8-flash-next/configs/README.md#prompt-cache-across-restarts).
2. **Restore only into an identical setup.** Have the launch script write a signature (model, draft, build, context, KV types, slot count, draft mode) and let `restore` compare it with the copy saved next to the slot. A mismatch must be a no-op, not an error. When a build changes kernels only, re-sign the save (copy the running signature over the saved one) and restore; that kept a conversation warm across the 2026-09-26 kernel builds.
3. **Save the draft context too** when a speculative draft is loaded ([patch 0010](../models/qwen3.8-flash-next/configs/patches/0010-server-slot-save-restore-draft-context.diff)), or the restored slot drafts without its prefix.
4. **Size `--cache-ram` for the conversations you serve**: about 29 KB per token for the target plus 2.3 KB for the draft on this model (3.7 GiB + 303 MiB at 135,760 tokens), times the number of conversations that should survive a displacement. The production value is 24,576 MiB.
5. **Restart only when the current instance has served no requests**, and before benchmarks put a hold on the saved slot (`slot-cache.sh hold` / `release`) so test prompts cannot overwrite it. Discard saves below a threshold (2,048 tokens here) so an idle restart keeps the previous save.

## Still open

- A crash cannot save; the next start restores the last good save, which may be stale. Periodic saves while idle would close that gap and were not built.
- Conversations that contain images cannot be saved (the server refuses); the previous save is kept.
- The saved state depends on llama.cpp's session-file format and the KV layout. Whether a save survives an upstream format change is UNVERIFIED; the signature includes the build directory so that it is never tried by accident.
- The RAM-cache reload was timed once at about 40K tokens and not archived; the 135K case is unmeasured.
- Not tested with more than one slot, with unified KV, or on vLLM.

## Update 2026-09-26

- **Who cancelled the 166K re-read.** The server journal shows the re-read starting at 22:47:31 UTC on 2026-09-25 and `stop: cancel task` at 22:55:58, 507 s in, at progress 0.88. That is not a round timeout. The reverse proxy's read and send timeouts are 1,800 s, and the forwarder behind it copies bytes with no timeout. A second request 4 s later was cancelled after 0.5 s. That points to the client being stopped by hand (estimate). The partial prefill survived the cancel: the slot was released with 148,148 tokens, and the next request found them. Details in [real agent traffic](../models/qwen3.8-flash-next/benchmarks/2026-09-26-real-agent-traffic.md).
- **Periodic saves while idle now exist.** A 5-minute timer saves after an idle period ([idle autosave](../models/qwen3.8-flash-next/experiments/2026-09-26-slot-autosave.md)), so a crash loses only the turns since the last idle spell.
- **The 135K RAM-cache reload is now measured.** Before our patches 0021-0022 a swap-in took about 2.95 s and a swap-out about 2.5 s; with them, about 0.6 s each. A slot restore from disk went from 3.2 s to 1.0 s ([swap speedup](../models/qwen3.8-flash-next/experiments/2026-09-26-prompt-cache-swap-speedup.md)).
