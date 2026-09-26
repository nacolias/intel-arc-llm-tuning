# OpenVINO and OpenVINO Model Server

Not yet tested on Arc Pro B70.

## To record

- OpenVINO and OVMS versions, and the GPU plugin version.
- Model conversion path and weight compression settings (INT4 and INT8).
- Continuous batching and speculative decoding support.
- Multi-GPU support.
- Comparison against vLLM XPU on the same model and workload.

## Notes

- **2026-09-26: no Qwen3.8-Flash-Next support.** Intel's Qwen3.8 listing on Hugging Face has only a Qwen3.8-27B OpenVINO IR (intermediate representation), `Intel/qwen3.8-27B-int4-ov`, created 2026-09-24 ([listing](https://huggingface.co/api/models?author=Intel&search=Qwen3.8), fetched 2026-09-26). The [community survey](../../models/qwen3.8-flash-next/research/2026-09-26-flash-next-on-vllm-intel-community.md) found no OpenVINO route for Flash-Next. Other routes for that model are compared in [engines/README.md](../README.md).
