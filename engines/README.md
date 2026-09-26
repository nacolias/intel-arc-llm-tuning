# Inference engines

XPU-specific notes per engine: which images and versions work, which features are supported on Arc, required patches, and known gaps. Model-specific configs stay in `models/<model>/configs/`.

| Engine | Folder | Used for | Status on Arc Pro B70 |
|---|---|---|---|
| vLLM (XPU backend) | [vllm-xpu/](vllm-xpu/) | production serving, tensor parallel, MTP | primary |
| Intel llm-scaler (vLLM fork) | [llm-scaler/](llm-scaler/) | Intel-patched vLLM builds | reference for patches |
| llama.cpp (SYCL backend) | [llama-cpp-sycl/](llama-cpp-sycl/) | GGUF models, single-card experiments | not yet tested |
| OpenVINO / OpenVINO Model Server | [openvino/](openvino/) | Intel's own inference stack | not yet tested |

For each engine, record:

- Image or build, with tag and digest.
- Supported quantization formats and the XPU kernel each one lands on.
- Speculative decoding support (MTP, EAGLE, DFlash, n-gram).
- Graph capture and compile modes that work.
- Multi-GPU support and collective backend.
- Patches we apply, and the upstream PR or issue each one tracks.
