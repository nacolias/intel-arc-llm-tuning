# Optimizing Qwen 3.8-27B on Dual Intel Arc Pro B70: The Complete Technical Blueprint

**Target System:** `dual-b70-5800x`  
**CPU / RAM:** AMD Ryzen 7 5800X (8C/16T), 128 GB DDR4  
**GPUs:** 2× Intel Arc Pro B70 (Battlemage / Xe2 architecture, 32,656 MiB physical VRAM each, ~30.3 GiB usable)  
**Software Stack:** Docker, `vllm/vllm-openai-xpu:v0.28.0`, PyTorch 2.13.0+xpu, oneAPI / Level-Zero 1.32, oneCCL  
**Production Endpoint:** `https://<your-host>/v1` (served IDs: `Qwen3.8-27B-Uncensored`, `Qwen3.8-27B-int4-AutoRound`, `Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16`)

---

## 1. Executive Summary & Benchmark Evolution

Through a series of architectural breakthroughs across Level-Zero device affinity, PCIe P2P interconnect discovery, oneCCL SYCL collective routing, and MTP kernel patching, inference performance on dual Arc Pro B70 was elevated from an un-graphable, sluggish baseline to **champion-tier performance beating dual-card expectations**:

| Milestone / Architecture Stage | 1× Stream Decode | 8× Concurrent Decode | 16× Concurrent Decode | 150K Prompt Cold TTFT |
|---|---|---|---|---|
| **Baseline (Single GPU, FP8, Eager mode)** | ~25–45 tok/s | ~110 tok/s | ~180 tok/s | OOM / N/A |
| **Initial TP2 (No graphs, oneCCL default)** | ~28–35 tok/s | ~107 tok/s | ~193 tok/s | ~240 s |
| **TP2 + Affinity Patch + XPU Graphs (P2P=0)** | ~55–75 tok/s | ~176 tok/s | ~356 tok/s | ~213 s |
| **Production Champion (TP2 + Graphs + P2P=1 + MTP2)** | **77.8–83.7 tok/s** | **291.6 tok/s** | **420.9–441.8 tok/s** | **~183 s** |

- **Single stream:** ~78–83.7 tok/s steady state (~72–76 tok/s conservative baseline).
- **Aggregate throughput:** 405–442 tok/s at 16 concurrent requests.
- **Context support:** 262,144 tokens (`--max-model-len 262144`), FP8 KV cache, prefix caching enabled.

---

## 2. Model & Precision Architecture

### Checkpoint Selection: GPTQ-INT4 sym-G128 with BF16 MTP
- **Why not FP8?** Evaluated FP8 models natively. FP8 dense weights saturated memory bandwidth, resulting in only ~25 words/s and making draft speculative models ineffective.
- **Why INT4?** GPTQ-INT4 with symmetric group size 128 (`Qwen3.8-27B-Uncensored-GPTQ-Int4-sym-G128-MTP-BF16`) reduced memory footprint sufficiently to fit both model weights and massive KV caches onto the two cards while dramatically cutting memory bandwidth pressure during decode.
- **MTP2 (Multi-Token Prediction) vs External Drafters:**
  - Evaluated DFlash and DFlash2 external drafters (e.g. `z-lab/incoai` DFlash2). Due to quantization distribution shifts against the uncensored fine-tune, external drafters scored 0% acceptance.
  - Native MTP (Multi-Token Prediction) with a BF16 draft head and 2 speculative tokens (`SPEC={"method":"mtp","num_speculative_tokens":2}`) achieved high acceptance and boosted decode speeds to >80 tok/s.

---

## 3. Breakthrough 1: TP2 XPU Graphs via Per-Worker Level-Zero Affinity

### The Problem
Intel Arc GPUs using Level-Zero (L0) and oneCCL failed or crashed during XPU graph capture (`VLLM_XPU_ENABLE_XPU_GRAPH=1`) when running multi-GPU Tensor Parallelism (`--tensor-parallel-size 2`). When vLLM spawned worker processes, both workers initialized Level-Zero across all visible system devices, causing race conditions in memory pool tracking and SYCL command queue capture.

### The Solution (`patch_worker_affinity.py`)
Applying the "sergiiob" isolation pattern: each spawned worker process intercepts its multiprocessing entry point and explicitly binds its own Level-Zero affinity mask (`ZE_AFFINITY_MASK`) to its assigned rank **before Level-Zero or PyTorch initializes**:
- Rank 0 sets `ZE_AFFINITY_MASK=0`
- Rank 1 sets `ZE_AFFINITY_MASK=1`

Container prerequisites:
- `--privileged` / SYS_PTRACE capability to allow process inspection and shared memory access.
- `ipc: host` to enable fast inter-process communication.

**Impact:** XPU execution graphs captured cleanly (PIECEWISE 14/14, FULL 8/8), instantly doubling decode throughput compared to eager execution.

---

## 4. Breakthrough 2: Hardware P2P Discovery (Debunking the "No P2P" Myth)

### The Myth
Community consensus in Arc / Xe2 discussions asserted that Intel Battlemage cards do not support PCIe Peer-to-Peer (P2P) transfers. Intel's default environment variables and community scripts explicitly set `CCL_TOPO_P2P_ACCESS=0`.

### The Verification (`p2p_check.cpp`)
We built a direct Level-Zero hardware probe using `icpx` and L0 headers:
```cpp
zeDeviceCanAccessPeer(dev0, dev1, &canAccess);
```
**Probe Output:**
```
dev0 -> dev1: getP2P r=0 flags=0x1 canAccess r=0 val=1
dev1 -> dev0: getP2P r=0 flags=0x1 canAccess r=0 val=1
```
`zeDeviceCanAccessPeer = 1` in both directions. Because both B70 cards sit on the same PCIe root complex, hardware P2P was physically present and operational.

### Enabling P2P in Production (`CCL_TOPO_P2P_ACCESS=1`)
Flipping `CCL_TOPO_P2P_ACCESS=1` in the production environment produced immediate measurable gains:
- **40 MB Allreduce Latency:** Dropped from 26.8 ms (1.57 GB/s) to **16.4 ms (2.56 GB/s)** (1.6× faster).
- **150K Cold Prompt Prefill (TTFT):** Dropped from 213 s to **~183 s**.
- **Aggregate Decode (16 concurrent):** Rose from ~356 tok/s to **420.9–441.8 tok/s**.

---

## 5. Breakthrough 3: oneCCL SYCL Graph-Capturable Collectives

### The Constraint
oneCCL provides multiple execution algorithms for inter-GPU communication:
1. `topo` (topological scheduler): Uses worker threads to schedule transfers. **Fatal flaw:** cannot be recorded inside SYCL execution graphs (throws graph record errors).
2. `sycl` (SYCL kernel collectives): Kernel-based collectives executed directly on the GPU queue.

### The Configuration
To ensure all communication during graph replay remains inside graph-capturable paths, oneCCL was configured to route all collectives via the SYCL "simple" kernel path:
```bash
CCL_ENABLE_SYCL_KERNELS=1
CCL_ENABLE_TOPO_ALGO=1
CCL_WORKER_COUNT=1
CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_ALLTOALL_TMP_BUF=1
CCL_ATL_TRANSPORT=ofi
CCL_ZE_IPC_EXCHANGE=pidfd
FI_PROVIDER=shm
CCL_ATL_SHM=1
```
*Note:* The threshold `4294967296` (4 GiB) forces all message sizes up to 4 GB to use the graph-capturable simple kernel path rather than splitting into complex asynchronous sub-schedulers.

---

## 6. The In-Container Patch Chain

All patches live in `/opt/docker/vllm/patches` and are applied dynamically at container startup before launching `vllm serve`:

1. **`patch_mtp_nightly.py`**:
   Adapts vLLM's MTP draft model handling to support BF16 draft heads when paired with quantized base weights on Intel XPU (`B70_MTP_BF16_DRAFT=1`).
2. **`patch_mtp_boundary.py`**:
   Corrects an off-by-one boundary check in speculative verification where speculative tokens at context boundaries could trigger index exceptions.
3. **`patch_gdn_mixed_split_v5.py`**:
   Fixes the GDN (Gated Delta Net) / Mamba hybrid attention-linear kernel split for Qwen hybrid layers on Battlemage hardware.
4. **`patch_worker_affinity.py`**:
   Sets per-rank `ZE_AFFINITY_MASK` inside each spawned multiprocessing worker prior to Level-Zero initialization, unlocking TP2 XPU graph capture.
5. **`patch_mtp_ptr_wrap.py`**:
   Wraps 64-bit integer pointers (`data_ptr()`) in `mamba_utils.py`. When operating without `expandable_segments`, the XPU allocator can assign virtual addresses $\ge 2^{63}$, which previously threw `ValueError: Overflow when unpacking long long`.
6. **`patch_sleep_graphs_v5.py`**:
   The model-swapping fix. Releases and purges existing SYCL execution graphs before entering sleep mode, and safely recaptures them into a clean memory pool on wake-up.
7. **`patch_mem_report2.py`**:
   Adds diagnostic RPC endpoints (`Worker.b70_mem_segments` and `b70_empty_cache`) via `/collective_rpc` to monitor device memory and fragmentation.

---

## 7. Production Docker Compose Configuration

File location: `/opt/docker/vllm/docker-compose.yml`

```yaml
services:
  vllm-a:
    image: vllm/vllm-openai-xpu:v0.28.0
    container_name: vllm-a
    restart: unless-stopped
    privileged: true
    ipc: host
    expose:
      - "8081"
    environment:
      - VIRTUAL_HOST=<your-host>
      - VIRTUAL_PORT=8081
      - ZES_ENABLE_SYSMAN=1
      - VLLM_TARGET_DEVICE=xpu
      - ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
      - VLLM_XPU_ENABLE_XPU_GRAPH=1
      - VLLM_SERVER_DEV_MODE=1
      - B70_MTP_BF16_DRAFT=1
      - CCL_ENABLE_SYCL_KERNELS=1
      - CCL_ENABLE_TOPO_ALGO=1
      - CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
      - CCL_ATL_TRANSPORT=ofi
      - CCL_ZE_IPC_EXCHANGE=pidfd
      - FI_PROVIDER=shm
      - CCL_ATL_SHM=1
      - CCL_WORKER_COUNT=1
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
      - TRITON_INTEL_DEVICE_ARCH=bmg
      - CCL_TOPO_P2P_ACCESS=1
      - CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
      - CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
      - CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
      - CCL_SYCL_ALLTOALL_TMP_BUF=1
      - SPEC={"method":"mtp","num_speculative_tokens":2}
    entrypoint: ["/bin/bash", "-lc"]
    command:
      - "set -e; \
         python /patches/patch_mtp_nightly.py; \
         python /patches/patch_mtp_boundary.py; \
         python /patches/patch_gdn_mixed_split_v5.py; \
         python /patches/patch_worker_affinity.py; \
         python /patches/patch_mtp_ptr_wrap.py; \
         python /patches/patch_sleep_graphs_v5.py; \
         python /patches/patch_mem_report2.py; \
         exec vllm serve /models/Qwen3.8-27B-Uncensored-GPTQ-Int4-sym-G128-MTP-BF16 \
           --host 0.0.0.0 \
           --port 8081 \
           --quantization gptq \
           --dtype float16 \
           --tensor-parallel-size 2 \
           --kv-cache-dtype fp8 \
           --max-model-len 262144 \
           --gpu-memory-utilization 0.90 \
           --max-num-seqs 32 \
           --max-num-batched-tokens 16384 \
           --block-size 32 \
           --enable-prefix-caching \
           --language-model-only \
           --served-model-name Qwen3.8-27B-Uncensored Qwen3.8-27B-int4-AutoRound Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16 \
           --enable-auto-tool-choice \
           --tool-call-parser qwen3_coder \
           --reasoning-parser qwen3 \
           --speculative-config \"$$SPEC\" \
           --api-key ${VLLM_API_KEY} \
           --enable-log-requests \
           --generation-config vllm \
           --enable-sleep-mode"
    volumes:
      - "/opt/ai/models:/models:ro"
      - "/opt/docker/vllm/patches:/patches:ro"
      - "/opt/docker/vllm/compile_cache_a:/root/.cache/vllm"
    devices:
      - "/dev/dri:/dev/dri"
    networks:
      - llm_bridge
```

---

## 8. Verified Performance Verification Commands

### Single Stream Benchmark
```bash
KEY=$(grep -E '^VLLM_API_KEY' /opt/docker/vllm/.env | cut -d= -f2)
python3 /opt/docker/vllm/bench_vllm.py http://localhost:8081/v1/chat/completions $KEY 1 200
```
*Expected:* 78.0–83.7 tok/s.

### 16 Concurrent Stream Benchmark (Throughput Stress Test)
```bash
python3 /opt/docker/vllm/bench_vllm.py http://localhost:8081/v1/chat/completions $KEY 16 200
```
*Expected:* 405.0–441.8 tok/s aggregate throughput.
