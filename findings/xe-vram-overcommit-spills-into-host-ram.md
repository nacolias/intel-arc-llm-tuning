# xe spills VRAM overcommit into host RAM that cannot be swapped

| | |
|---|---|
| **Date** | 2026-09-23 |
| **Confidence** | verified-here (the spill, and the dma-buf export path in the failed runs); likely (the runtime trigger and why the sleep-mode allocator avoids it) |
| **Applies to** | Arc Pro B70; Linux 7.0.0-34-generic `xe` (the export path exists from 6.18); vLLM XPU 0.28.0 (`vllm/vllm-openai-xpu:v0.28.0`, compute-runtime 26.27.39122.11); llama.cpp SYCL on host compute-runtime 26.35.39758.11; host [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md) with 121 GiB of usable RAM |
| **Area** | driver / host |

## Finding

On `xe`, allocating more than a card's VRAM does not fail. The overflow silently lands in host RAM that the driver holds, which cannot be swapped and is not charged to the process or its container. Separately, when a VRAM buffer is exported through dma-buf, `xe` backs it with host pages as well.

A vLLM TP=4 launch without the sleep-mode allocator drove this driver-held RAM to 92–104 GiB. The global OOM killer then took the wrong processes, and one run ended in a kernel panic, "System is deadlocked on memory". It looked like a GPU out-of-memory error; it was host RAM exhaustion.

## Evidence

**Driver-held RAM** below means `MemTotal − MemAvailable − AnonPages` from `/proc/meminfo` (earlyoom's formula). It is about 4 GiB at idle on this host. The watchdog logs use a slightly different "unaccounted" figure that runs 2–3 GiB lower.

### 1. Allocating past VRAM succeeds and spills

A test script allocated and filled 1 GiB chunks on one B70 (31.89 GiB total), `torch` 2.13.0+xpu in the vLLM v0.28.0 image:

| Allocated | `torch.xpu.mem_get_info` free | Host MemAvailable | Unaccounted host RAM |
|---|---|---|---|
| start | | 116.9 GiB | 2.2 GiB |
| 20 GiB | 11.6 GiB | 116.8 GiB | 2.2 GiB |
| 31 GiB | 0.6 GiB | 116.8 GiB | 2.2 GiB |
| 32 GiB | 0.8 GiB | 115.7 GiB | 3.4 GiB |
| 36 GiB | 0.8 GiB | 111.6 GiB | 7.4 GiB |
| 40 GiB | 0.7 GiB | 107.6 GiB | 11.5 GiB |
| 44 GiB | 0.7 GiB | 103.6 GiB | 15.4 GiB |

Every allocation succeeded, and the data read back intact. From about 32 GiB on, each extra GiB of "VRAM" became about 1 GiB of host RAM (a second reading of the same test at 42 GiB gave +11.2 GiB; the metrics differ). The free-memory query kept reporting 0.7–0.8 GiB free throughout. A separate host-memory watchdog logged the same jump, from 2.1 to 16.4 GiB unaccounted.

### 2. The TP=4 failures were the dma-buf export path

Kernel journal and pstore, 2026-09-23:

| Attempt | Layout | Sleep-mode allocator | Peak driver-held RAM | Outcome |
|---|---|---|---|---|
| 1a | probably TP=4 (flags lost) | no (inferred) | about 104 GiB | page-allocation failures; OOM killer killed user systemd, then a worker |
| 1b | unknown | unknown | about 104 GiB | OOM killer took containerd; kernel panic "System is deadlocked on memory"; hung about 8 minutes until power-cycled |
| 2 | TP=2 × DP=2 (tensor × data parallel) | no | about 92 GiB | OOM killer killed systemd, then a worker |
| 3 | TP=4 | yes | 20.8 GiB peak, about 12 steady (watchdog); lowest MemAvailable 86.4 GiB | served; 129.1 tok/s single-stream |
| 4 | TP=2 | yes | 12.3 GiB peak, 7.0 steady (watchdog) | served |
| 5 | 2 × TP=2 on GPUs 0,1 and 2,3 | yes | 20.7 GiB peak with both starting at once, 11.7 steady; lowest MemAvailable 83.5 GiB | both lanes served |

- Every failure shows the same kernel call chain from a `VLLM::Worker_TP` thread, with `GFP_HIGHUSER|__GFP_ZERO|__GFP_RETRY_MAYFAIL`: `drm_prime_handle_to_fd_ioctl → xe_gem_prime_export → ttm_bo_setup_export → ttm_bo_populate → xe_ttm_tt_populate → ttm_pool_alloc_page` (TTM is the kernel's GPU memory manager).
- The export-then-populate step exists in `xe` from v6.18, not v6.17 ([v6.17](https://github.com/torvalds/linux/blob/v6.17/drivers/gpu/drm/xe/xe_dma_buf.c), [v6.18](https://github.com/torvalds/linux/blob/v6.18/drivers/gpu/drm/xe/xe_dma_buf.c)).
- In attempt 2, driver-held RAM rose 35 GiB in 4 s as four workers loaded 9.26 GiB of weights each, and another 32 GiB after the KV cache was allocated. That is faster than the 3.6 GB/s host link of that boot could copy, so it was new zeroed memory, not evicted VRAM.
- The memory was tied to the worker processes. It was freed when they died.
- Attempts 3–5 came from [the Qwen3.8-27B quad-card experiment](../models/qwen3.8-27b/experiments/2026-09-23-tp4-and-dual-tp2-quad-b70.md).

### 3. What probably triggers it

- **Likely:** the container's compute-runtime 26.27.39122.11 is in the range of Intel's "defer backing" regression ([compute-runtime #953](https://github.com/intel/compute-runtime/issues/953), fixed in [26.35.39758.10](https://github.com/intel/compute-runtime/releases/tag/26.35.39758.10)) and the multi-device host mirror ([#980](https://github.com/intel/compute-runtime/issues/980), open). The host already runs 26.35.39758.11, but the container does not use it.
- **Likely:** vLLM's sleep-mode allocator (`--enable-sleep-mode`, which uses `xpumem` to map SYCL physical memory into reserved virtual addresses; [xpumem.py](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/device_allocator/xpumem.py)) avoids the mirror. All sleep-mode layouts stayed at or under about 21 GiB. This is a correlation across four layouts, not a controlled test: no run without sleep mode was repeated.
- **Upstream:** a pending patch on this path cut vLLM's host RAM on six B70s from 115.8 GB to 15.3 GB. It is unmerged; reviewers asked for a rework ([intel-xe thread](https://ratatoskr.run/intel-xe/2026/08/17436404/t)).
- **Unverified:** one user on Ubuntu 7.0.0-31 with the full 26.35 stack still hit this OOM in vLLM TP=2 (in the #953 thread). Intel called it a separate issue.

### 4. llama.cpp hits the spill too

Qwen3.8-Flash-Next on four cards at 262K context with `-ub 2048` and an MTP draft head, 2026-09-24: the memory fitter left the draft head out of its sizing. GPU3 peaked at 32,636 MiB, driver-held RAM climbed to 57.8 GiB, and decode fell to 0.6 tok/s. The run was stopped. See [the Flash-Next baseline experiment](../models/qwen3.8-flash-next/experiments/2026-09-24-baseline-llamacpp-mtp-262k.md).

Healthy llama.cpp levels on the same model: about 16 GiB of driver-held RAM at `-ub 1024`, 22 GiB at `-ub 1536`, and 27 GiB with pipeline parallelism enabled.

### 5. Why it took the whole box down

- **earlyoom cannot see it.** It measures `MemAvailable / (MemAvailable + AnonPages)`, which approaches 100% as driver memory grows ([earlyoom meminfo.c](https://github.com/rfjakob/earlyoom/blob/v1.9.0/meminfo.c)). It also acts only when both free RAM and free swap are under its threshold. Swap had 58–75 GB free during the failures, and driver pages cannot be swapped.
- **The OOM killer picked the wrong processes first:** user systemd (`oom_score_adj` 100), then containerd, before the workers.
- **Ranks died one at a time** (`memory.oom.group=0`). Buffers shared between ranks are freed only when every holder has exited.
- **The panic did not reboot the box** (`kernel.panic=0`).
- **Free CMA memory was unusable.** Ubuntu's 7.0 kernel reserves 17.3 GiB of Kexec HandOver (KHO) scratch as CMA (contiguous memory allocator) memory, which only movable allocations can use. The panic report showed `free_cma` of 4,535,980 pages, exactly 17.30 GiB, while the driver starved. `crashkernel=` reserves another 4.25 GiB that kdump does not use here.
- **fdinfo GTT sums mislead.** Summed `drm-total-gtt` or `gtt_mm` double-counts shared buffers: 30–59 GB during the healthy TP=4 run, while real driver-held RAM was about 15 GiB.

## Impact

- Without the sleep-mode allocator, vLLM TP=4 on four B70s is not usable on a 128 GB host, and a failed start can hang the machine.
- With it, TP=4 serves at 129.1 tok/s single-stream with at least 86 GiB of RAM still available ([benchmark](../models/qwen3.8-27b/benchmarks/2026-09-23-tp4-quad-b70.md)).
- Any VRAM overcommit, from a second process on a card or an engine that sizes its buffers wrong, turns into a slow, silent host-RAM leak instead of an error.

## What to do

- **One GPU tenant per card.** Stop other GPU services, and any timers or guards that restart them, before loading a model that fills VRAM.
- **vLLM multi-GPU:** pass `--enable-sleep-mode`, but do not actually put a large model to sleep; switch layouts by stopping and starting servers. Size the KV cache with `--kv-cache-memory-bytes` below the value vLLM logs as filling VRAM (17.13 GiB at TP=4 for Qwen3.8-27B int4, so about 16 GiB). With `--gpu-memory-utilization 0.90` it allocated 18.7 GiB, about 1.5 GiB over per card, and the XPU path does not count graph memory.
- **llama.cpp multi-GPU:** set `-ts` and `-ub` explicitly and keep VRAM headroom on the fullest card. Do not trust the fitter when a draft model is loaded.
- **Watch driver-held RAM** during every multi-GPU load, and kill the server before the box runs out:

  ```bash
  # gpuwatch.sh "<stop command>" [driver_held_limit_gib] [avail_floor_gib]
  STOP=$1; L=${2:-60}; F=${3:-10}
  while sleep 2; do
    read h a < <(awk '/^MemTotal/{t=$2}/^MemAvailable/{v=$2}/^AnonPages/{p=$2}
                      END{printf "%d %d\n",(t-v-p)/1048576,v/1048576}' /proc/meminfo)
    if [ "$h" -gt "$L" ] || [ "$a" -lt "$F" ]; then
      logger -t gpuwatch "driver-held ${h} GiB, available ${a} GiB: stopping"; sh -c "$STOP"; exit 1
    fi
  done
  ```

  A watchdog with a 10 GiB MemAvailable floor was the safeguard used in the 2026-09-23 runs. This exact script was not run.
- **Contain the next event** (proposed, untested here): `oom_score_adj: 1000` on GPU containers, `OOMScoreAdjust=-900` for `sshd`, `memory.oom.group=1` on each GPU container's cgroup ([oom_kill.c](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/plain/mm/oom_kill.c?h=v7.0)), and `kernel.panic=10`.
- **Give unmovable allocations more room** (proposed, not applied on this host): add `kho=off` and remove `crashkernel=` if kdump is not used. That returns about 21.5 GiB.
- **After a load, check fdinfo per process:** `drm-total-gtt` for the server should be a few GB, not the model size.

## Still open

- **Control test:** one TP=2 lane without `--enable-sleep-mode`, everything else stopped, with the watchdog on. It would show whether the allocator is really what prevents the growth.
- **Does compute-runtime 26.35 remove the mirror on kernel 7.0?** Intel says #953 is fixed there; one report says a kernel-7.0 path remains.
- **The TTM limit did not cap it.** `ttm` `pages_limit` was 15,920,472 pages (about 60.7 GiB), yet driver-held RAM reached 92–104 GiB. Not investigated.
- **Long sessions.** The sleep-mode TP=4 run was watched for about 5 minutes only.
- Re-check after any kernel, `xe` or compute-runtime upgrade.
