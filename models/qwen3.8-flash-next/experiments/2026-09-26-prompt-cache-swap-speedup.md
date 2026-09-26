# Faster host-RAM prompt-cache swaps: reuse the host buffers, stage host-to-device copies through pinned memory

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept (production build `20260926-5c258f538` since 2026-09-26 22:10 UTC) |
| **Baseline** | production build `20260926-2b84213a4`; the cost on real traffic is in [real agent traffic](../benchmarks/2026-09-26-real-agent-traffic.md) |
| **Result** | [raw/2026-09-26-prompt-cache-swap-timings.csv](../benchmarks/raw/2026-09-26-prompt-cache-swap-timings.csv) (server-side phase timings), [raw/2026-09-26-prompt-cache-swap-probe.csv](../benchmarks/raw/2026-09-26-prompt-cache-swap-probe.csv) (client-side probe), [raw/2026-09-26-slot-restore-and-start.csv](../benchmarks/raw/2026-09-26-slot-restore-and-start.csv) (restores, roundtrips, start times) |
| **Related** | [slot save/restore and `--cache-ram`](2026-09-25-slot-save-restore-and-ram-cache.md), patches 0019, 0021 and 0022 in [configs/patches/](../configs/patches/README.md), [finding: greedy decoding is not reproducible on this stack](../../../findings/llama-cpp-sycl-greedy-not-reproducible.md) |

## Result

Moving a 135K-token conversation between the GPUs and llama-server's host-RAM prompt cache (`--cache-ram`) got about four times faster:

| Direction | Before | After |
|---|---|---|
| Swap-out (GPU to host RAM) | about 2.5 s | 0.6-0.75 s |
| Swap-in (host RAM to GPU) | about 2.95 s | 0.60-0.68 s |

That is the work behind every switch between the agent's main conversation and a subagent. Measured on real traffic, those switches were 5.7% of the server's busy time.

Two changes did it:
- The server reuses the host buffer of the state it just loaded back into the slot for the next save.
- The SYCL backend copies host to device through two reusable pinned buffers per GPU instead of a fresh `malloc` per tensor.

The PCIe copies were never the problem: device to host already ran at about 9 GB/s (3,979 MiB in 448 ms). Most of the time went on page faults and zero-filling freshly allocated multi-GiB buffers.

Two side effects:
- A slot restore from disk went from 3.2 s to 1.0 s.
- A service restart went from about 75 s to about 33 s until the server answers, because model weights are uploaded through the same copy.

## Hypothesis

The journal showed 5.07 s median between llama-server picking the slot and starting a task whenever the agent switched conversations. The PCIe link would move 4 GB in well under a second, so something else dominated. Reading the code gave two suspects:
- The server allocates a new `std::vector<uint8_t>` of the state's size for every save. That zero-fills and page-faults every page of about 4 GiB.
- The SYCL `set_tensor` copies every tensor through a fresh `malloc` plus `memcpy`, a workaround for mmap'd sources on PVC. That page-faults the whole size again before the device copy starts.

## Change

Patch 0019 first added timing log lines (host allocation, device-to-host copy of the target and draft states, and total load time) to confirm where the time went. Then:

- **0021, server** (`tools/server/server-task.{h,cpp}`): `server_prompt_cache` keeps the buffers of the state it just loaded back into a slot, one for the target and one for the draft, instead of freeing them. The next save takes a spare buffer if its capacity fits the new state and is at most twice its size. Fresh buffers reserve 1/8 more capacity than needed. Those pages are never touched, so they are not resident. The first version matched spares by size, so a conversation a few tokens longer than last time never fit (the "fix v1" rows).
- **0022, SYCL** (`ggml_backend_sycl_buffer_set_tensor`): copy in 32 MiB chunks through two pinned host buffers per device (`sycl::malloc_host`, 256 MiB for four GPUs in all). The host `memcpy` of one chunk overlaps the device copy of the other. If pinned memory cannot be allocated, it falls back to the old path.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Before: `2b84213a4` (production) and `556c342cb` (0018-0020 on top, for the timing logs). After: `5c258f538` = `2b84213a4` + 0018-0023 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0; oneAPI 2026.1 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, production flags ([llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh)), `--cache-ram 24576`; host RAM 128 GB DDR4 behind a Ryzen 5800X, PCIe Gen4 x16 uplink to a PEX88096 switch |

## Procedure

[`tools/switchprobe.py`](../../../tools/switchprobe.py) on a server whose slot holds the saved 135,226-token agent conversation. It alternates:
- an unrelated 2,010-token prompt ("away": the conversation is copied to host RAM);
- the conversation plus a new user marker ("back": copied back to the GPUs).

Each request has `n_predict` 1. The swap time is the client wall time minus the server's prompt and decode timings. The build with patch 0019 also logs each phase. Before and after, the slot was restored from the same file.

```bash
SWITCH_SLOT=/path/to/slots/slot0.bin SWITCH_ROUNDS=3 python3 tools/switchprobe.py
```

The byte-level check restores the slot file into the server, saves it straight back under a new name and compares the files with `cmp`. That path goes through the changed `set_tensor` (restore) and the unchanged `get_tensor` (save).

## Results

Server-side phases at 135K ([swap timings CSV](../benchmarks/raw/2026-09-26-prompt-cache-swap-timings.csv)):

| Phase | Before | Fix v1 (reuse by size) | Fix v2 (reuse by capacity, deployed) |
|---|---|---|---|
| Swap-out: host allocation of the 3,979-3,988 MiB state | 2,104.7 / 2,165.5 ms | 1,920.1 / 1,948.5 / 2,184.4 ms (never reused) | 1,917.8 ms first time, then 52.8 / 156.0 / 364.0 ms |
| Swap-out: device-to-host copy of the target state | 447.0-448.2 ms | 447.5-454.4 ms | 449.7-452.6 ms |
| Swap-out: draft state (about 300 MiB) | 42.9-43.4 ms | 42.5-42.8 ms | 42.6-42.8 ms |
| Swap-in: load of the 135K state, total | 2,664.2 / 2,718.5 ms | 435.9 / 658.4 / 708.2 ms | 419.3 / 426.3 / 431.3 / 435.6 ms |

Client-side probe ([probe CSV](../benchmarks/raw/2026-09-26-prompt-cache-swap-probe.csv); time outside prompt evaluation and decode):

| | Unmodified production `2b84213a4` (3 runs) | Deployed `5c258f538` (rounds 1-2) |
|---|---|---|
| Away (135K conversation out, small prompt in) | 2.45-2.60 s | 0.64-0.75 s |
| Back (small prompt out, 135K conversation in) | 2.95-2.97 s | 0.60 s |

Round 0 of every run includes the first, fresh allocation; the "before" rows came from a build with profiling queues on and match unmodified production within 0.3 s.

Other effects ([restore and start CSV](../benchmarks/raw/2026-09-26-slot-restore-and-start.csv)):

| | Before | After |
|---|---|---|
| Slot restore from disk, 135,226 tokens | 3,156 ms | 980.6-1,052.6 ms (8 runs) |
| Service start to ready, from the journal | 78-79 s including a 3.2 s slot restore (3 starts) | 32-35 s, no restore (6 starts) |
| Restore then save roundtrip | byte-identical | byte-identical |

Expected on the real traffic of [the 2026-09-26 journal](../benchmarks/2026-09-26-real-agent-traffic.md): 42 switches at a median 5.07 s were 200 s. At about 1.0-1.3 s per switch (one save with a reused buffer plus one load) they would be 42-55 s, freeing roughly 4% of the server's busy time (estimate; not yet re-measured on real traffic). The switches in that journal moved a 105K-119K main conversation and a 5K-60K subagent each time, so each one did a save and a load.

## Correctness

The staged copy is byte-exact. A 3,863,800,112-byte target state and a 316,428,932-byte draft state went through restore and save and came out equal to the file, on the unmodified build and on the fix.

Greedy output after a swap out and back was compared with the same request without a swap:
- Fix v1: all 32 tokens identical.
- Fix v2: 7 of 32 identical.

That comparison cannot settle the question on this stack. On unmodified production, restoring the same slot file twice and running the same greedy request gave 32/32, then 19/32, then 1/32 tokens shared with the first run ([finding](../../../findings/llama-cpp-sycl-greedy-not-reproducible.md)). The swap changes nothing about what is copied: buffer reuse only changes which host memory holds the bytes, and the staged copy was shown byte-exact.

Decode after swaps was checked on the fix build: acceptance and step times were in the normal range (about 67 ms per step at 135K). The deployed server answered a known-answer question correctly.

## Decision

Kept. Deployed with production build `20260926-5c258f538` on 2026-09-26 at 22:10 UTC. The saved conversation was carried over by re-signing the slot, because the KV layout is unchanged.

## Follow-ups

- [ ] Re-measure the switch gap on real traffic with `tools/switchcost.py` after a day of use.
- [ ] The swap-out still copies device to host into pageable memory (about 0.45 s for 4 GB). A pinned bounce buffer there too would cut it further.
- [ ] Upstream: the per-call `malloc` in `ggml_backend_sycl_buffer_set_tensor` affects every SYCL user's model load and slot restore.
