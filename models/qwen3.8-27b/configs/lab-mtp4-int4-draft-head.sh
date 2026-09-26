#!/usr/bin/env bash
# Dual-B70 TP2 reference launch (steveseguin/b70-optimization-lab, public domain), adapted as an A/B box.
# Reproduces the 112.9 tok/s single-user configuration: TP2, XPU graphs, MTP depth 4, draft-only INT4 LM head.
# The image is vLLM 0.27.2rc1.dev77 with rebuilt vllm-xpu-kernels; use it to isolate "host vs stack" before porting.
# Their model was an AutoRound INT4 checkpoint relabelled to plain gptq; a GPTQ sym G128 checkpoint with BF16 MTP
# tensors (yours) takes the same XPUwNa16 path. Set MODEL_DIR and PORT. See models/qwen3.8-27b/research/2026-09-12-performance-gains-report.md section 3.2.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the model directory}
PORT=${PORT:-18134}
IMAGE=${IMAGE:-ghcr.io/steveseguin/vllm-openai-xpu-qwen38-int4@sha256:521eb277c0733f8c2ce47aea1bb98ed576c6f1ad63bf5baf22d38fc07abf54ad}
CACHE_DIR=${CACHE_DIR:-$(mktemp -d)}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
KV_DTYPE=${KV_DTYPE:-auto}          # lab used auto (FP16 KV); try fp8 as the second arm
SPEC_TOKENS=${SPEC_TOKENS:-4}
docker run --rm --name qwen38-lab-ab --ulimit core=0 \
  --device /dev/dri:/dev/dri --group-add "$(stat -c '%g' /dev/dri/render* | sort -u | head -1)" \
  --cap-add SYS_PTRACE --security-opt label=disable --ipc=host --shm-size=8g \
  --publish 127.0.0.1:${PORT}:8000 --volume "${MODEL_DIR}:/model:ro" --volume "${CACHE_DIR}:/root/.cache/vllm" \
  --env ZE_AFFINITY_MASK=0,1 --env ONEAPI_DEVICE_SELECTOR=level_zero:0,1 \
  --env VLLM_TARGET_DEVICE=xpu --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env VLLM_XPU_ENABLE_XPU_GRAPH=1 --env VLLM_XPU_ALLREDUCE_HOST_WAIT=1 \
  --env VLLM_XPU_FP8_BLOCK_W8A16=1 --env VLLM_XPU_W4A16_DETERMINISM_PAD=0 \
  --env VLLM_XPU_GEMMA_RMSNORM_TRITON=0 --env VLLM_XPU_RMSNORM_TRITON=0 \
  --env VLLM_XPU_GDN_SPLIT_MIXED=1 --env VLLM_XPU_GDN_SPEC_GROUP=16 --env VLLM_XPU_GDN_PREFILL_GROUP=1 \
  --env VLLM_XPU_FP16_LINEAR_ROWCHUNK=32 --env VLLM_BATCH_INVARIANT=0 --env VLLM_XPU_GDN_SPEC_PERSISTENT_SCRATCH=1 \
  --env VLLM_XPU_QWEN_GEMMA_RMSNORM_BATCH_INVARIANT=0 --env VLLM_XPU_QWEN_GEMMA_RMSNORM_PACKED_SERIAL_EXACT=1 \
  --env VLLM_XPU_DRAFT_LM_HEAD_INT4=1 --env VLLM_XPU_DRAFT_LM_HEAD_INT4_GROUP_SIZE=128 --env VLLM_XPU_DRAFT_LM_HEAD_INT4_SCALE_DTYPE=bf16 \
  --env VLLM_XPU_DRAFT_LM_HEAD_INT4_CHUNK_ROWS=2048 --env VLLM_XPU_GDN_NATIVE_FALLBACK=1 \
  --env TORCHINDUCTOR_DETERMINISTIC=1 --env VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0 --env VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0 --env PYTHONHASHSEED=0 \
  --env PYTORCH_ALLOC_CONF=expandable_segments:True \
  --env CCL_ATL_TRANSPORT=ofi --env FI_PROVIDER=tcp --env FI_TCP_IFACE=lo --env CCL_ZE_IPC_EXCHANGE=pidfd \
  --env CCL_SEND=direct --env CCL_RECV=direct --env CCL_TOPO_P2P_ACCESS=1 \
  --env CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296 --env CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296 \
  --env CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296 \
  "${IMAGE}" \
  --model /model --served-model-name qwen38-lab-ab --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --dtype float16 --quantization gptq --kv-cache-dtype "${KV_DTYPE}" \
  --gpu-memory-utilization 0.92 --max-model-len "${MAX_MODEL_LEN}" --block-size 64 \
  --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --no-enable-prefix-caching --language-model-only \
  --speculative-config "{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":${SPEC_TOKENS}}" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,8,10,15,16,20,25,30,32,40,50,60,64,80,100,120,160,200,240,320],"max_cudagraph_capture_size":320,"splitting_ops":[],"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false,"deterministic":true,"benchmark_epilogue_fusion":false,"split_reductions":false,"triton.autotune_pointwise":false}}'
