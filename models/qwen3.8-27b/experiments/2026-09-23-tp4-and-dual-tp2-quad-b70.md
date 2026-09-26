# TP4 and two TP2 lanes on a four-B70 host

| | |
|---|---|
| **Date** | 2026-09-23 |
| **Model / checkpoint** | DavidAU Qwen3.8-27B TURBO Fable Cold Fusion 735-882 Heretic merge, GPTQ INT4 group 128, BF16 MTP head (details in the [benchmark](../benchmarks/2026-09-23-tp4-quad-b70.md)) |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | inconclusive (TP4 speed measured, correctness not checked, not deployed); PP2 x TP2 blocked |
| **Baseline** | TP2 on the same host, same benchmark: [2026-09-23-tp4-quad-b70](../benchmarks/2026-09-23-tp4-quad-b70.md) |
| **Result** | [2026-09-23-tp4-quad-b70](../benchmarks/2026-09-23-tp4-quad-b70.md) |
| **Related** | [findings/xe-vram-overcommit-spills-into-host-ram.md](../../../findings/xe-vram-overcommit-spills-into-host-ram.md), [findings/pcie-switch-uplink-can-train-at-gen1.md](../../../findings/pcie-switch-uplink-can-train-at-gen1.md), [intel/compute-runtime#953](https://github.com/intel/compute-runtime/issues/953), [intel/compute-runtime#980](https://github.com/intel/compute-runtime/issues/980), [vllm-project/vllm#46994](https://github.com/vllm-project/vllm/pull/46994) |

**Result: TP4 (tensor parallel over four cards) ran at 129.1 tok/s single-stream, +50% over TP2 at 86.0 tok/s, once every multi-GPU launch used vLLM's sleep-mode allocator.** Earlier TP4 attempts that night exhausted host RAM through the GPU driver and once hung the host. Two TP2 lanes side by side ran at 84.7 and 87.5 tok/s single-stream. PP2 x TP2 (pipeline parallel 2 over tensor parallel 2) does not start on vLLM 0.28.0 with MTP.

## Hypothesis

With four cards on one PCIe switch, TP4 should raise single-stream decode over TP2, because each card reads a quarter of the weights per token and the per-layer allreduce stays on peer-to-peer (P2P) links inside the switch. Two TP2 lanes should give the most aggregate throughput for mixed use. The expected size of the effect was not written down before the run; this entry was reconstructed afterwards from the logs.

## Change

Against the production vLLM service, which ran TP2 on GPUs 0 and 1:

```diff
- ZE_AFFINITY_MASK=0,1
+ # ZE_AFFINITY_MASK unset: all four cards visible
- --tensor-parallel-size 2
+ --tensor-parallel-size 4
- --api-key ${VLLM_API_KEY}
+ # no API key, test port 18082, private compile cache
```

Everything else (image, seven startup patches, oneCCL and Level Zero environment, MTP 3, FP8 KV, `--enable-sleep-mode`, XPU graphs) was copied from the production service. Full flags are in the [benchmark](../benchmarks/2026-09-23-tp4-quad-b70.md#environment).

The two-lane test ran two TP2 containers at once, one on GPUs 0 and 1 (port 18082) and one on GPUs 2 and 3 (port 18083). The PP2 x TP2 attempt used `--tensor-parallel-size 2 --pipeline-parallel-size 2`.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | vLLM 0.28.0, `vllm/vllm-openai-xpu:v0.28.0@sha256:4756b66a077627133cee653b551f6f5eaa1b9a981b5eea13edd33fcd3b0d3ca3` |
| compute-runtime / IGC | 26.27.39122.11 / IGC 2.38.2 inside the container |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic, in-tree `xe`, GuC 70.58.0 |
| oneCCL | TODO (not recorded) |
| PyTorch XPU | 2.13.0+xpu |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | TP4 / PP1; TP2 / PP1 (alone, and two instances); TP2 / PP2 (failed) |
| PCIe uplink | Gen1 x16 during all runs (retrained to Gen4 on 2026-09-24) |

## Procedure

1. Stop every other GPU workload on the host, and pause the supervisor that restarts the production vLLM container. It restarted the container once in the middle of an earlier test.
2. Start a watchdog that logs host memory and per-card VRAM every 2 s and kills the test container if `MemAvailable` falls below 10 GiB:

   ```bash
   #!/bin/bash
   # Usage: watchdog.sh <container> <logfile> [min_avail_gb]
   C=$1; LOG=$2; FLOOR_KB=$(( ${3:-10} * 1024 * 1024 ))
   echo "time avail_gb anon_gb shmem_gb slab_gb free_gb swapused_gb unaccounted_gb vram_mib(0,1,2,3)" > $LOG
   while docker inspect -f '{{.State.Running}}' $C 2>/dev/null | grep -q true; do
     eval $(awk '{gsub(":","",$1); printf "%s=%s\n", $1, $2}' /proc/meminfo | grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|AnonPages|Shmem|Slab|KernelStack|PageTables|VmallocUsed|SwapTotal|SwapFree|SwapCached)=' | sed 's/(//;s/)//')
     UNACC=$(( MemTotal - MemFree - Buffers - Cached - SwapCached - AnonPages - Slab - KernelStack - PageTables - VmallocUsed ))
     V=$(for i in 0 1 2 3; do xpu-smi stats -d $i 2>/dev/null | awk -F'current: ' '/GPU Memory Used/ {print $2+0}'; done | paste -sd, -)
     printf "%s %.1f %.1f %.1f %.1f %.1f %.1f %.1f %s\n" $(date -u +%H:%M:%S) \
       $(echo "$MemAvailable/1048576" | bc -l) $(echo "$AnonPages/1048576" | bc -l) $(echo "$Shmem/1048576" | bc -l) \
       $(echo "$Slab/1048576" | bc -l) $(echo "$MemFree/1048576" | bc -l) $(echo "($SwapTotal-$SwapFree)/1048576" | bc -l) \
       $(echo "$UNACC/1048576" | bc -l) "$V" >> $LOG
     if [ "$MemAvailable" -lt "$FLOOR_KB" ]; then
       echo "$(date -u +%H:%M:%S) WATCHDOG: MemAvailable ${MemAvailable}kB < floor, killing $C" >> $LOG
       docker kill $C >/dev/null 2>&1; break
     fi
     sleep 2
   done
   ```

3. Start the TP4 container with the production service's image, environment, mounts, patches and flags, changed as above.
4. Wait for the server, then run `bench.py` (in the [benchmark](../benchmarks/2026-09-23-tp4-quad-b70.md#workload)): one warm-up request, then 1, 4 and 16 concurrent requests of the same prompt, 512 tokens, greedy, `ignore_eos`.
5. Stop it. Repeat for TP2 alone on GPUs 0 and 1.
6. Start both TP2 lanes and run `bench.py` against both at the same time.
7. Try PP2 x TP2.

## Results

| Metric | Baseline (TP2) | TP4 | Delta |
|---|---|---|---|
| Single-stream tok/s | 86.0 | 129.1 | +50% |
| Aggregate tok/s at c4 | 220.6 | 303.8 | +38% |
| Aggregate tok/s at c16 | 756.9 | 969.0 | +28% |
| Aggregate tok/s at c8 | not run | not run | |
| TTFT at any context | not measured | not measured | |
| Spec-decode mean acceptance length | 2.73 to 3.01 | 2.89 to 3.04 | about the same |
| KV cache capacity | 694,169 tokens | 2,110,332 tokens | 3.0x |
| Peak driver-held host RAM | 12.3 GiB | 20.8 GiB | |

| Layout | c1 | c4 aggregate | c16 aggregate |
|---|---|---|---|
| 2x TP2, lane A (GPUs 0,1) | 84.7 | 252.7 | 772.2 |
| 2x TP2, lane B (GPUs 2,3) | 87.5 | 240.0 | 664.0 |
| Both lanes combined | | about 493 | about 1,436 |

The combined figures add two runs that overlapped but were not synchronized. Source for all speeds: [raw/2026-09-23-tp4-quad-b70.csv](../benchmarks/raw/2026-09-23-tp4-quad-b70.csv).

**PP2 x TP2 failed at config time:** `NotImplementedError: Pipeline parallelism is not supported for this model. Supported models implement the SupportsPP interface.` The main Qwen3.8 model implements `SupportsPP`; the MTP draft model class (`Qwen3_5MTP`) does not in vLLM 0.28.0. [vllm-project/vllm#46994](https://github.com/vllm-project/vllm/pull/46994) adds it and is in v0.30.0 (upstream-documented; untested on XPU).

**A first TP4 start with the production service's `ZE_AFFINITY_MASK=0,1` failed at once:** `AssertionError: local_world_size (4) must be less than or equal to the number of visible devices (2)`. TP4 needs all four cards visible.

### The TP4 out-of-memory episode

Before the successful run, three multi-GPU attempts on the same night ran out of host memory. The numbers below come from the kernel journal, the earlyoom log and the launch scripts at the time; those logs are not kept in this repository.

| Attempt | Layout | Sleep-mode allocator | Peak driver-held host RAM | Outcome |
|---|---|---|---|---|
| 1 | probably TP4 (inferred from the KV step size; flags lost) | no (inferred) | about 104 GiB | page-allocation failures; the OOM (out-of-memory) killer killed the user `systemd`, then a worker |
| 2 | unknown | unknown | about 104 GiB | the OOM killer took `containerd-shim` and `containerd`, then kernel panic "System is deadlocked on memory"; with `kernel.panic=0` the host hung for about 8 minutes until it was power-cycled |
| 3 | TP2 x DP2 (data parallel 2) | no | about 92 GiB | OOM killer took `systemd`, then a `Worker_TP` process |
| 4 (this experiment) | TP4 | yes | 20.8 GiB | served and benchmarked |
| 5 (this experiment) | TP2 | yes | 12.3 GiB | served and benchmarked |
| 6 (this experiment) | 2x TP2 | yes | 20.7 GiB | both lanes served at once |

Here "driver-held" is memory that is neither page cache, anonymous memory nor slab: the watchdog's `unaccounted` column for attempts 4 to 6, and `MemTotal - (MemAvailable + AnonPages)` (earlyoom's formula) for attempts 1 to 3. The two measures differ by about 2 to 3 GiB.

**Root cause: host RAM held by the `xe` driver, not vLLM's own memory.** Two mechanisms, ranked as in our analysis:

1. **dma-buf export backs VRAM buffers with unmovable host pages. Kernel path verified here; userspace trigger likely.** Every failure logged the same allocation path from comm `VLLM::Worker_TP` with `GFP_HIGHUSER|__GFP_ZERO|__GFP_RETRY_MAYFAIL`: `drm_prime_handle_to_fd_ioctl -> xe_gem_prime_export -> ttm_bo_setup_export -> ttm_bo_populate -> xe_ttm_tt_populate -> ttm_pool_alloc_page`. TTM (the kernel's Translation Table Manager) pool pages are not charged to the container's memory cgroup and cannot be swapped. The export-then-populate step is in `xe_dma_buf.c` from Linux 6.18, not in 6.17 ([v6.17](https://github.com/torvalds/linux/blob/v6.17/drivers/gpu/drm/xe/xe_dma_buf.c), [v6.18](https://github.com/torvalds/linux/blob/v6.18/drivers/gpu/drm/xe/xe_dma_buf.c)). The likely trigger is the container's compute-runtime 26.27.39122.11, which is in the range of [intel/compute-runtime#953](https://github.com/intel/compute-runtime/issues/953) (fixed in [26.35.39758.10](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10)) and [#980](https://github.com/intel/compute-runtime/issues/980) (open). That link is UNVERIFIED here.
2. **VRAM overcommit spills silently into host RAM. Mechanism verified here; its share in these failures is inferred.** Allocating past the card's VRAM does not fail on `xe`; the overflow lands in unswappable host memory. vLLM's KV sizing also over-commits on XPU (18.7 GiB allocated per card at TP4 against 17.13 GiB that its own log says fills the card). Details and the oversubscription test: [findings/xe-vram-overcommit-spills-into-host-ram.md](../../../findings/xe-vram-overcommit-spills-into-host-ram.md).

**Why the sleep-mode allocator avoids it (likely, not proven).** With `--enable-sleep-mode`, vLLM on XPU allocates weights and KV through its `xpumem` allocator, which maps SYCL physical memory into reserved virtual addresses ([vLLM xpumem.py](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/device_allocator/xpumem.py)). Every run with it stayed at or under about 21 GiB; every failed run without it reached 92 to 104 GiB. No run without sleep mode was repeated under the watchdog, so this is a strong correlation, not a controlled test.

**Why the whole host went down.** earlyoom could not see driver memory: it measures `MemAvailable / (MemAvailable + AnonPages)` ([earlyoom meminfo.c](https://github.com/rfjakob/earlyoom/blob/v1.9.0/meminfo.c)) and only acts when both free RAM and free swap are below 5%. The host still had 58 to 75 GB of free swap during the failures, which driver pages cannot use. The kernel OOM killer then killed `systemd` and `containerd` before the workers, ranks died one at a time (`memory.oom.group=0`), and `kernel.panic=0` left the panicked host hung instead of rebooting.

## Correctness

Not checked. No greedy output was compared against a non-speculative reference, single-stream or under concurrency. The config combines MTP 3, multi-GPU XPU graphs (experimental in vLLM 0.28.0) and prefix caching; combinations of these have community reports of corruption (see [findings/prefix-cache-mtp-corruption-hybrid.md](../../../findings/prefix-cache-mtp-corruption-hybrid.md)).

## Decision

Inconclusive, not deployed.

- TP4 is the fastest single-stream layout measured for this model on this host, but correctness is unchecked and the benchmark is synthetic.
- The host's single-user mode went to [Qwen3.8-Flash-Next](../../qwen3.8-flash-next/README.md) on llama.cpp instead, and its mixed mode keeps the production TP2 service on GPUs 0 and 1.
- **Kept as a rule for every multi-GPU vLLM launch on `xe`:** pass `--enable-sleep-mode` (allocator only; never call `/sleep` on a large model), make all cards visible for TP4, size the KV cache explicitly with `--kv-cache-memory-bytes` below vLLM's "fully utilize" value, start lanes one at a time, and run a host-memory watchdog.

## Follow-ups

- [ ] Greedy correctness check for TP4 with MTP 3, XPU graphs and prefix caching, at c1, c4 and c16.
- [ ] Re-run TP4 and TP2 with the Gen4 uplink and the standard methodology (c2, c8, c32, context sweep, prose and code workloads, 3 repetitions).
- [ ] Controlled test of the allocator link: the same TP4 launch without `--enable-sleep-mode`, under the watchdog with a high floor.
- [ ] Try a derived image with compute-runtime 26.35.39758.x to test the #953 / #980 trigger.
- [ ] Set `--kv-cache-memory-bytes` to about 16 GiB per card at TP4 (vLLM suggested 15.79 GiB to fit 0.90 utilization) and re-measure.
- [ ] Retest PP2 x TP2 on vLLM 0.30.0 or later.
