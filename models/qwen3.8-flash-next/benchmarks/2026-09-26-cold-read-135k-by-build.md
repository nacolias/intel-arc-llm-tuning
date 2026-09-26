# Cold 135K read by build: lcbench cold-read progression

| | |
|---|---|
| **Date** | 2026-09-26 (runs from 2026-09-25 20:41 to 2026-09-26 02:06 UTC) |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Config** | [llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh) (`-ub 1536 -ts 12,13,13,11`, `-b 2048` except one run), GPU clock floor 2800 MHz |
| **Experiment** | [QSA score expansion by broadcast](../experiments/2026-09-26-qsa-score-expansion-broadcast.md), [slot save/restore](../experiments/2026-09-25-slot-save-restore-and-ram-cache.md); [grouped MoE XMX GEMM](../experiments/2026-09-26-grouped-moe-xmx-gemm.md) (patches 0012-0015), [`eh_proj` as one 2D product](../experiments/2026-09-26-mtp-eh-proj-2d-product.md) (0016), [`-b 3072`](../experiments/2026-09-26-batch-3072.md); op shares in [prefill op profile](2026-09-26-prefill-op-profile.md) |
| **Raw data** | [raw/2026-09-26-cold-read-progress.csv](raw/2026-09-26-cold-read-progress.csv) (server progress lines per run), [raw/2026-09-26-cold-read-windows.csv](raw/2026-09-26-cold-read-windows.csv) (client-side totals and per-turn summaries per test window), [raw/2026-09-26-lcbench-sessions-summary.csv](raw/2026-09-26-lcbench-sessions-summary.csv), [raw/2026-09-26-lcbench-sessions-per-turn.csv](raw/2026-09-26-lcbench-sessions-per-turn.csv), [raw/2026-09-26-klprobe-comparisons.csv](raw/2026-09-26-klprobe-comparisons.csv) |

## Result

A cold read of the 134,862-token [`tools/lcbench.py`](../../../tools/lcbench.py) code context went from a median of 455 s (N=6, 436.8-462.0 s, all builds with the score expansion by gather) to 382 s with the expansion by broadcast (`2c88bdf19`), 313 s with the grouped MoE (mixture of experts) XMX (Xe Matrix Extensions) GEMM (general matrix multiply) for prompt batches (`517eb833d` and `85b26781b`: 312.58 and 312.71 s), and 284 s with the MTP (multi-token prediction) head's `eh_proj` as one 2D product (`a890bf8b0`): -38% over the night. Every figure after the baseline is one run per build. The broadcast step is N=1 (382.49 s client-side, 381.74 s server end line) against a baseline whose archived reads span 403.26-472.24 s over 14 sessions with differing builds and settings, so its size is somewhere between about 5% and 17% (estimate, against the same night's nine reads of 403.26-461.98 s). The later steps are repeated and tight: server end lines of 311.84, 311.98 and 313.34 s with the grouped GEMM (two builds and one profiling run; a second profiling run ended at 294.94 s), then 283.10 and 285.88 s with the 2D `eh_proj` (the second with `-b 3072`). Prompt turns of about 60 new tokens went from 0.92-0.96 s to 0.76 s, almost all of it from the in-place grouped GEMM. Decode medians stayed at 39-49 tok/s and moved with draft acceptance, not with the build. A `-b 3072` run on the final tree read in 286.44 s, no gain.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. PR #28243 at `6fcaa16` plus the patch stack in [configs/patches/](../configs/patches/README.md); the tree under test is given per run below. Frozen builds: `20260925-993baf141`, `20260926-2c88bdf19`, `20260926-2b84213a4` |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0; oneAPI 2026.1 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| KV cache dtype / block size | f16 K and V, 262,144 cells allocated at load |
| Speculative decoding | MTP draft, `--spec-draft-n-max 3`, draft on `SYCL3`; draft QSA (Qwen sparse attention) on from `0f23f62c8` |
| Graph / compile mode | eager; `GGML_SYCL_SPARSE_FA=1`; pooled QSA key cache on |

Production during the runs (source: service journal, not archived): `20260925-f47a6a5f3` until about 22:32 UTC on 2026-09-25, `20260925-0f23f62c8` until about 23:06, `20260925-993baf141` until 00:04 on 2026-09-26, `20260926-2c88bdf19` until about 02:10, then `20260926-2b84213a4`. Every dev-tree run started the service from the development build directory through a systemd drop-in, with the saved production slot held; production was restarted at the end of each window.

## Workload

| | |
|---|---|
| Workload set | [`tools/lcbench.py`](../../../tools/lcbench.py): a fixed 134,862-token C++ context read once, then follow-up turns of 57-62 or 358-416 new tokens; not yet one of the shared [workloads](../../../benchmarks/workloads/) |
| Input length | 134,862 tokens; the test windows from 23:50 UTC on salt the prompt (`LC_SALT`), which adds 4 tokens (134,866; the profile runs 134,868), so a saved or cached prefix cannot match |
| Output length | 16 tokens on the cold read, 256 per turn, `ignore_eos` |
| Sampling | greedy (temperature 0) |
| Warm-up | none: every 135K read counted below is the first request of a freshly started server instance (server task 0). The KL (Kullback-Leibler divergence) probe reads are the second 135K read of an instance and are listed but not counted |
| Repetitions | 1 per build after the baseline; the baseline pools 6 runs of the same expansion code on three trees |

Command (test-window pattern; the candidate build is started by the service with the production slot held):

```bash
LC_SALT="[lc test] " LC_TURNS=6 LC_OUT=lc-test.json python3 tools/lcbench.py | grep -E "context|cold read|SUMMARY"
python3 tools/lcsum.py lc-test.json
KL_OUT=kl-<name>.json python3 tools/klprobe.py            # teacher-forced probe, 128 positions
python3 tools/klprobe.py --compare kl-<name>.json kl-poolA.json
```

The server logs a progress line every 30,720 tokens (20 ubatches of 1,536) and one at the end of the prompt (`progress = 1.00`). The journal excerpt in [raw/2026-09-26-cold-read-progress.csv](raw/2026-09-26-cold-read-progress.csv) has the end line only for the runs from 23:27 UTC on and never for the KL probe reads, so the client-side "cold read" seconds from each window's stdout ([raw/2026-09-26-cold-read-windows.csv](raw/2026-09-26-cold-read-windows.csv)) are the comparable total; they run 0.6-0.8 s above the server's end line. `run_start_utc` in the CSV is the time of the first progress line, 40-80 s after the request began.

## Results

### Per build

Clean runs only: no profiler, no changed launch flags, no changed draft configuration. Median and range are over the client-side cold read seconds. Prompt turn and decode are the lcbench medians over the follow-up turns of the same session; decode min-max in brackets.

| Tree or build | Score expansion | MoE prompt path | `eh_proj` | N | Cold read (s) | Server line at 122,880 tokens (s) | Prompt turn, ~60 tokens | Decode tok/s | Acceptance | ms per step |
|---|---|---|---|---|---|---|---|---|---|---|
| dev tree before `9f41d810e` (dense draft; `LLAMA_MTP_WINDOW` tests) | gather | per-expert loop | 3D | 3 | 458.60 (454.37-461.98) | 396.27-402.95 | 0.98-1.27 s | 41.4-43.0 | 61.7-65.6% | 68.4-72.7 |
| dev tree before `0f23f62c8` (draft QSA) | gather | loop | 3D | 2 | 437.7 (436.80-438.65) | 380.48-383.10 | 0.92 s (10 turns) | 41.1 (39.6-44.7) | 63.1% | 70.5 |
| production `20260925-993baf141` | gather | loop | 3D | 1 | 455.35 | 397.21 | 0.96 s (2 turns) | 39.1 (37.7-39.1) | not summarised | not summarised |
| **all gather runs pooled (baseline)** | gather | loop | 3D | **6** | **454.9 (436.80-461.98)** | 380.48-402.95 | 0.92-1.27 s | 39.1-43.0 | | |
| dev tree before `2c88bdf19` (broadcast) | broadcast | loop | 3D | 1 | 382.49 | 337.12 | 0.91 s | 43.9 (39.9-54.2) | 66.0% | 67.1 |
| dev tree at `517eb833d` (grouped GEMM, vector loads) | broadcast | grouped XMX GEMM | 3D | 1 | 312.58 | 274.07 | 0.90 s | 48.3 (40.8-52.7) | 72.2% | 68.4 |
| dev tree at `85b26781b` (grouped GEMM, in-place I/O) | broadcast | grouped, in place | 3D | 1 | 312.71 | 275.10 | 0.77 s | 48.7 (36.3-51.5) | 70.7% | 68.1 |
| dev tree before `a890bf8b0` (`eh_proj` 2D) | broadcast | grouped, in place | 2D | 1 | 283.66 | 247.51 | 0.76 s (10 turns) | 43.7 (37.6-50.2) | 63.8% | 68.0 |
| `2b84213a4` with `-b 3072` (different configuration) | broadcast | grouped, in place | 2D | 1 | 286.44 | 251.13 | 1.53 s, not comparable | 62.4 (47.1-62.6), not comparable | 93.1%, inflated | 64.4 |

Against the pooled baseline median of 454.9 s: -16% (broadcast), -31% (grouped GEMM), -38% (`eh_proj`). The -16% is one read against that median; against the same night's nine archived pre-broadcast reads (403.26-461.98 s, including the excluded ones) the broadcast step is about -5% to -17%, and -19% against the slowest archived read (472.24 s, `lc-mmid512` on an earlier 2026-09-25 build) (estimate; [raw/2026-09-26-cold-read-windows.csv](raw/2026-09-26-cold-read-windows.csv), [raw/2026-09-26-lcbench-sessions-summary.csv](raw/2026-09-26-lcbench-sessions-summary.csv)). Step to step: -18% for the grouped GEMM, -9% for `eh_proj`, +1% for `-b 3072`. Whole-read prompt speed (134,862 tokens over the client-side seconds): 296 tok/s (`993baf141`), 353, 431, 431, 475 tok/s (`a890bf8b0`). The server's own end-of-prompt figures are 353.28, 432.47, 432.27 and 476.37 tok/s.

Sources: [raw/2026-09-26-cold-read-windows.csv](raw/2026-09-26-cold-read-windows.csv) (cold read, prompt turn, decode, acceptance, step), [raw/2026-09-26-cold-read-progress.csv](raw/2026-09-26-cold-read-progress.csv) (server lines), [raw/2026-09-26-lcbench-sessions-summary.csv](raw/2026-09-26-lcbench-sessions-summary.csv) (the `LLAMA_MTP_WINDOW`, draft-QSA and `-b 3072` sessions).

Where the read spends its time, one clean run per build, from the server progress lines (30,720-token segments, tok/s derived as tokens over the segment's seconds):

| Tree | 0-30,720 | 30,720-61,440 | 61,440-92,160 | 92,160-122,880 | 122,880-end |
|---|---|---|---|---|---|
| `993baf141` (23:07 UTC) | 389 | 353 | 286 | 248 | not logged |
| `2c88bdf19` (23:50) | 455 | 391 | 345 | 302 | 269 |
| `517eb833d` (00:33) | 523 | 511 | 435 | 364 | 317 |
| `85b26781b` (01:04) | 539 | 521 | 399 | 374 | 325 |
| `a890bf8b0` (01:36) | 661 | 545 | 467 | 390 | 337 |
| `2b84213a4`, `-b 3072` (02:02) | 760 | 569 | 395 | 389 | 345 |

The last logged segment runs at 0.45-0.64 of the first segment's speed on every build (0.51 on `a890bf8b0`, 0.59-0.64 on the earlier builds). The context-dependent part (attention, indexer) is the likely cause; it was not profiled per segment. `-b 3072` gained only on the first segment.

### Every run in the journal

All 25 runs in [raw/2026-09-26-cold-read-progress.csv](raw/2026-09-26-cold-read-progress.csv), in order. "Start" is the first progress line. Builds are attributed by time from the production timeline above, the test-window scripts and the patch dates; "wip" means the tree was later committed as that hash. Client-side totals are from [raw/2026-09-26-cold-read-windows.csv](raw/2026-09-26-cold-read-windows.csv).

| Start (UTC) | Tree or build | What ran in this window | Server line at 122,880 tokens | Server end line | Client cold read | Counted |
|---|---|---|---|---|---|---|
| 2026-09-25 20:41:20 | dev, wip `9f41d810e` | lcbench, 10 turns, `LLAMA_MTP_WINDOW=2048` | 400.04 s | not logged | 458.60 s | yes |
| 20:51:36 | same | lcbench, 10 turns, `LLAMA_MTP_WINDOW=8192` | 402.95 | not logged | 461.98 | yes |
| 21:01:39 | same | lcbench, 10 turns, `LLAMA_MTP_WINDOW=0` with the draft attending on all rows (`lc-mtpwin-0-allrows`) | 348.89 | not logged | 403.26 | no: different draft configuration; see below |
| 21:20:06 | same | lcbench, 10 turns, `LLAMA_MTP_WINDOW=0`, output rows only (`lc-mtpwin-0-rowsonly`) | 396.27 | not logged | 454.37 | yes |
| 21:33:10 | same | host-timing run: `LLAMA_HOST_PROF=20`, 2 turns, then `turnprof.py` | 393.59 | not logged | 447.52 | no: profiling run |
| 21:51:42 | dev, wip `0f23f62c8` | fnbench2's 39,119-token prompt (progress 0.79 at 30,720 tokens), draft-QSA window | 50.47 at 30,720 | | 67.5 s for 39,119 tokens | not a 135K read |
| 21:53:19 | same | lcbench, 10 turns (`lc-draftqsa`), same instance after fnbench2 | 380.48 | not logged | 436.80 | yes |
| 22:16:58 | same | pairprobe, 12 paired turns, draft QSA against dense | 383.10 | not logged | 438.65 | yes |
| 22:31:30 | same | fnbench2's 39,119-token prompt (progress 0.79), by time; stdout not archived | 50.16 at 30,720 | | | not a 135K read |
| 23:07:49 | production `993baf141` | slot save/restore test: cold read, 2 turns, restart with a save at 23:14:28 | 397.21 | not logged | 455.35 | yes |
| 23:23:27 | production `993baf141` | a ~40K-token prompt (progress 0.77 at 30,720 tokens): the RAM-cache test conversation, by time, UNVERIFIED | 80.12 at 30,720 | | | not a 135K read |
| 23:27:50 | production `993baf141` | prefill profile run: `GGML_SYCL_OP_PROFILE=20 LLAMA_HOST_PROF=20`, op profiler on 23:31:30 to 23:32:32 (about 297-359 s into the read), 1 turn | 389.93 | 447.01 | 447.75 | no: profiling run |
| 23:50:17 | dev, wip `2c88bdf19` | lctest: salted read, 6 turns, then the KL probe (`kl-ident`) | 337.12 | 381.74 | 382.49 | yes |
| 23:57:20 | same instance | KL probe context read (raw 134,840-token prefix, server task 608) | 317.26 | not logged | | second read, not counted |
| 2026-09-26 00:33:05 | dev at `517eb833d` | lctest, 6 turns, then the KL probe (`kl-mmid`) | 274.07 | 311.84 | 312.58 | yes |
| 00:38:45 | same instance | KL probe context read (task 578) | 242.56 | not logged | | second read |
| 01:04:07 | dev at `85b26781b` | lctest, 6 turns, then the KL probe (`kl-mmid18`) | 275.10 | 311.98 | 312.71 | yes |
| 01:09:46 | same instance | KL probe context read (task 586) | 233.31 | not logged | | second read |
| 01:17:44 | dev at `85b26781b` | prefill profile run; op profiler on 01:21:58 to 01:23:00, which covered only the last 16 s of the read and the decode turn; host timing throughout; 1 turn | 271.65 | 313.34 | 314.11 | no: profiling run |
| 01:26:45 | dev at `85b26781b` | prefill profile run; op profiler on 01:29:04 to 01:30:06, about 188-250 s into the read (roughly 97K-120K tokens of context, interpolated; estimate); 1 turn | 257.63 | 294.94 | 295.68 | no: profiling run |
| 01:36:43 | dev, wip `a890bf8b0` | lctest, 10 turns, then the KL probe (`kl-eh`) | 247.51 | 283.10 | 283.66 | yes |
| 01:42:40 | same instance | KL probe context read (task 985) | 238.12 | not logged | | second read |
| 01:52:41 | `eh_proj` 2D (`acc-A`) | accprobe, 41,419-token context (progress 0.74 at 30,720 tokens), 11 turns | 42.95 at 30,720 | | 60.41 s for 41,419 tokens | not a 135K read |
| 01:55:56 | old 3D `eh_proj` (`acc-B`) | accprobe, 41,419-token context, 11 turns | 43.61 at 30,720 | | 62.07 s for 41,419 tokens | not a 135K read |
| 02:02:06 | dev at `2b84213a4`, `-b 3072` | lctest, 6 turns | 251.13 | 285.88 | 286.44 | no: different configuration |

Runs that are not comparable:

- **21:01 UTC, 403.26 s.** The `lc-mtpwin-0-allrows` session: `LLAMA_MTP_WINDOW=0` with the draft attending on all rows, on the work-in-progress tree between two other draft configurations (the rows-only variant ran at 21:20, and patch 0008 was committed at 21:47 UTC). It is a different draft configuration from every other run, and why its read was about 50 s faster than the runs before and after it is not recorded (UNVERIFIED). Its per-turn numbers are in [raw/2026-09-26-lcbench-sessions-summary.csv](raw/2026-09-26-lcbench-sessions-summary.csv) as `lc-mtpwin-0-allrows.json`.
- **21:51 and 22:31 UTC, partial.** Progress 0.79 at 30,720 tokens is fnbench2's 39,119-token prompt, not a 135K read; the draft-QSA window's stdout gives 39,119 tokens at 579.8 tok/s for the first. The 23:23 partial (progress 0.77, about 40K tokens) and the 01:52 and 01:55 partials (progress 0.74, the 41,419-token accprobe context) are not 135K reads either.
- **Profiling runs.** 23:27, 01:17 and 01:26 UTC ran with `GGML_SYCL_OP_PROFILE=20` and `LLAMA_HOST_PROF=20` and the op-profiler flag file on for 62 s; 21:33 ran with `LLAMA_HOST_PROF=20`. All four are excluded from the medians. Their op shares are in [raw/2026-09-26-prefill-op-profile.csv](raw/2026-09-26-prefill-op-profile.csv) and [2026-09-26-prefill-op-profile.md](2026-09-26-prefill-op-profile.md).
- **02:02 UTC, `-b 3072`.** 286.44 s against 283.66 s with the default `-b 2048` on the same tree: no gain. Its decode figures (62.4 tok/s median, 93.1% acceptance, 1.53 s prompt median) are not comparable: the model's output collapsed into the same answer from turn 2 on, which inflates MTP acceptance (source: session log, not archived); the archived per-turn rows for that session show 189-191 of 191-196 drafted tokens accepted in turns 3-6, against 152-185 of 208-306 in every other 256-token session ([raw/2026-09-26-lcbench-sessions-per-turn.csv](raw/2026-09-26-lcbench-sessions-per-turn.csv), `lc-test.json` rows). This degenerate session (identical outputs in turns 2-6; source: session log, not archived) was never investigated further.

The CSV has no rows between 22:31 and 23:07 UTC, the period of the cancelled 166K re-read described in [slot save/restore](../experiments/2026-09-25-slot-save-restore-and-ram-cache.md); whether the excerpt covered that request is unknown.

### Per-turn prompt latency and decode per build

lcbench follow-up turns of the same sessions. The `lc-test.json` rows of the per-turn CSV hold only the last window (`-b 3072`), because each window overwrote the file; the other windows' medians come from their SUMMARY and lcsum lines, archived in [raw/2026-09-26-cold-read-windows.csv](raw/2026-09-26-cold-read-windows.csv). The "prompt median" of both tools is the upper median over all follow-up turns, and every third turn adds 358-416 tokens: with 6 turns it is the slowest of the four ~60-token turns, with 10 turns the second slowest of seven short turns ([`tools/lcbench.py`](../../../tools/lcbench.py), [`tools/lcsum.py`](../../../tools/lcsum.py)). So 6-turn and 10-turn medians are not strictly comparable, and the typical ~60-token turn is faster: in `lc-draftqsa` the six ~60-token turns have a plain median of 0.77 s against the 0.92 s below, in the `-b 3072` session the three short turns took 0.58-0.80 s ([raw/2026-09-26-lcbench-sessions-per-turn.csv](raw/2026-09-26-lcbench-sessions-per-turn.csv)).

| Tree or build | Session | Turns | Prompt median, ~60 new tokens | Decode median (min-max) tok/s | Acceptance | ms per step |
|---|---|---|---|---|---|---|
| dev, wip `9f41d810e` | `lc-mtpwin-0-allrows` | 10 | 1.24 s | 42.3 (37.8-48.4) | 69.7% | 73.4 |
| dev, wip `9f41d810e` | `lc-mtpwin-0-rowsonly` | 10 | 1.27 s | 41.4 (36.0-46.4) | 65.2% | 72.7 |
| dev, wip `9f41d810e` | `lc-mtpwin-2048` | 10 | 0.98 s | 42.3 (39.3-44.0) | 61.7% | 68.4 |
| dev, wip `9f41d810e` | `lc-mtpwin-8192` | 10 | 0.99 s | 43.0 (39.2-48.2) | 65.6% | 69.2 |
| dev, wip `0f23f62c8` | `lc-draftqsa` | 10 | 0.92 s | 41.1 (39.6-44.7) | 63.1% | 70.5 |
| dev, wip `0f23f62c8` | pairprobe, draft QSA / dense | 12 paired | not summarised | 43.4 / 41.9 overall | 67.6% / 65.7% | 70.3 / 71.5 |
| production `993baf141` | slot save/restore test | 2 | 0.96 s | 39.1 (37.7-39.1) | not summarised | not summarised |
| dev, wip `2c88bdf19` | lctest (broadcast) | 6 | 0.91 s | 43.9 (39.9-54.2) | 66.0% | 67.1 |
| dev at `517eb833d` | lctest | 6 | 0.90 s | 48.3 (40.8-52.7) | 72.2% | 68.4 |
| dev at `85b26781b` | lctest | 6 | 0.77 s | 48.7 (36.3-51.5) | 70.7% | 68.1 |
| dev, wip `a890bf8b0` | lctest | 10 | 0.76 s | 43.7 (37.6-50.2) | 63.8% | 68.0 |
| `2b84213a4`, `-b 3072` | lctest | 6 | 1.53 s, not comparable | 62.4 (47.1-62.6), not comparable | 93.1%, inflated | 64.4 |

The prompt turn fell in one step, 0.90 to 0.77 s, with the in-place grouped GEMM (`85b26781b`), and by another 0.01 s with `eh_proj`. Decode tok/s follows acceptance: the 48.3 and 48.7 medians came with 70.7-72.2% acceptance, the 43.7 median with 63.8%, at 68.0-68.4 ms per step throughout. The step time dropped from 70.3-70.5 ms on the draft-QSA tree to 67.1-68.4 ms on every build from the broadcast expansion on; whether that 2-3 ms is the expansion's decode-side cost was not tested with an A/B in one session (UNVERIFIED).

| Concurrency | Aggregate tok/s | Per-stream tok/s | TTFT p50 | TPOT p50 | Acceptance |
|---|---|---|---|---|---|
| 1 (final tree `a890bf8b0`, 10 turns) | 43.7 | 43.7 | 0.76 s for ~60 new tokens on a cached prefix | 22.9 ms | 63.8% |
| 2 and up | not applicable: one slot | | | | |

| Context length | TTFT (cold) | Single-stream decode tok/s |
|---|---|---|
| 134,862 (cold read), `993baf141` | 455.35 s | 39.1 |
| 134,862, `2c88bdf19` | 382.49 s | 43.9 |
| 134,862, `517eb833d` / `85b26781b` | 312.58 / 312.71 s | 48.3 / 48.7 |
| 134,862, `a890bf8b0` | 283.66 s | 43.7 |

## Observations

- **Run-to-run spread of the baseline.** Six clean runs of the same expansion code, on three trees, spanned 436.8-462.0 s (-4% to +2% around the median, a 5.5% span). The two fastest were the draft-QSA tree's (436.80, 438.65 s); whether the draft's sparse attention shortens the read is not separable from noise at N=2. Counting every archived pre-broadcast read, including the excluded 21:01 read (403.26 s) and the 2026-09-25 sessions on earlier builds, the span is 403.26-472.24 s over 14 sessions with differing settings; that is why the broadcast step (N=1) is given as about 5-17% (estimate). Every later build has N=1, so differences under about 3% between them are not established. The 517eb833d and 85b26781b reads agree within 0.13 s: the in-place I/O of `85b26781b` helped the prompt turns (0.90 to 0.77 s), not the cold read.
- **The second 135K read of an instance was faster than the first.** The KL probe's context read followed each lctest window in the same server instance and reached 122,880 tokens 4-15% sooner than that window's cold read: 317.26 against 337.12 s (`2c88bdf19`), 242.56 against 274.07 (`517eb833d`), 233.31 against 275.10 (`85b26781b`), 238.12 against 247.51 (`a890bf8b0`). Every number in the progression is the first read after a fresh start. A plausible cause is first-use just-in-time (JIT) compilation of SYCL kernels in a new process, which [findings/no-persistent-sycl-cache-on-bmg.md](../../../findings/no-persistent-sycl-cache-on-bmg.md) puts at about 30 s per start; the probe's prompt also differs (a raw `/completion` prefix without the chat template). UNVERIFIED; a repeated read in one instance would settle it.
- **Profiling runs.** On `85b26781b` the 01:26 run, with the op profiler on for 62 s in the middle of the read, finished in 295.68 s, 18 s faster than the 01:17 run whose profiler window fell on the read's last 16 s (314.11 s). The flag-file profiler is not the dominant cost; the gap (6%) is about the size of the baseline's 5.5% spread. Both are excluded anyway. The 23:27 profile run on `993baf141` (447.75 s) is the only pre-broadcast run with a server end line (447.01 s).
- **Commit message baseline.** Patch 0011's commit message gives "135K cold read: 447-461 s -> 382 s" and the production notes call it six runs. The clean runs archived here give 436.8-462.0 s; the 447-461 range matches the dense-draft and production runs without the two draft-QSA reads. Which six sessions the message counted is not recorded.
- **Correctness.** Each build's KL probe stayed within the same-build noise floor: top-1 agreement 116-118/128 and median KL 0.0019-0.0024 against the pooled-cache probe (`kl-ident`, `kl-mmid`, `kl-mmid18`), and 115/128, median 0.0025 for `kl-eh` against its predecessor `kl-mmid18`, against 115-116/128 and 0.0021-0.0026 for two runs of one build (`kl-poolC`/`kl-poolA`, `kl-spC`/`kl-spA`) ([raw/2026-09-26-klprobe-comparisons.csv](raw/2026-09-26-klprobe-comparisons.csv)). KL is the Kullback-Leibler divergence over the top-20 log-probabilities per position. The probe sends one-token requests, so no MTP draft step or multi-token verify batch runs in it: it shows the target model's next-token distribution is unchanged, not that the draft is.
- **Memory.** GPU memory after the lctest windows was 28,934 / 28,672 / 28,938 / 29,442 MiB (`517eb833d`), 28,748 / 28,486 / 28,752 / 29,256 MiB (`85b26781b`), 28,752 / 28,490 / 28,756 / 29,278 MiB (`a890bf8b0`) and 28,747 / 28,485 / 28,751 / 29,401 MiB (`-b 3072`) (source: session log, not archived).
