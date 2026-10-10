# Strata (SYCL port)

Not yet run here. Status checked on 2026-10-10 from the upstream repository, its issue tracker and community reports. Every number on this page is a community or upstream measurement and is **UNVERIFIED** on our hardware. The planned first run is [this experiment](../../models/qwen3.8-flash-next/experiments/2026-10-10-strata-sycl-quad-b70.md); the reading notes are in [this research note](../../models/qwen3.8-flash-next/research/2026-10-10-strata-sycl-port.md).

## What it is

[Strata](https://github.com/Niko1221/Strata) (MIT) is an inference engine written for one model, Qwen3.8-Flash-Next, and for one 12 to 24 GB consumer GPU. It is a C++ engine with its own CUDA kernels and parts of ggml, behind a Python server with OpenAI, Anthropic and Responses APIs, tool calls, a web app and an MCP (Model Context Protocol) server. Releases came almost daily in early October 2026: 0.1.36 on 2026-10-02, 0.1.41 on 2026-10-08.

What the engine does differently from llama.cpp:

- **Expert tiers.** The 24,576 routed experts (512 per layer, 48 layers) sit in a VRAM cache filled from a per-model expert profile (`data/expert-profile.bin`), with the rest in pinned host RAM or read from the GGUF. On NVIDIA the CPU computes the experts the card lacks; the SYCL port instead reads a pinned host mirror over PCIe with the GPU (`--stream-experts`, `STRATA_MIRROR_MIB`).
- **Verify windows.** Decode runs as speculative windows: the model's own MTP (multi-token prediction) layer, packed from the base checkpoint, drafts up to 3 tokens per window and prompt lookup can extend them. Each window is a captured graph of about 2,400 to 2,600 nodes, replayed per round. This matters on Arc: llama.cpp's SYCL graphs are refused with more than one GPU ([finding](../../findings/llama-cpp-sycl-graphs-off-with-multiple-gpus.md)); Strata captures its windows per card.
- **Native QSA (Qwen sparse attention).** Block selection is implemented (top-k of pooled blocks per query), so attention reads about 2,000 cells per query at any context, like the reference model.
- **PLE (per-layer embedding) n-gram table** read by row from the SSD (`--ple-io direct`, the default) or locked in RAM (`--ple-io ram`).
- **Concurrency.** Up to 8 batch slots (`--batch N`), and with a layer split one pipeline group per GPU (`--batch-groups auto`, default from 0.1.41). Conversation checkpoints play the role of a prefix cache.

The PC-centric design is not what we need on four 32 GB cards, where everything fits in VRAM. The interest is the engine's speed with every expert resident, its prompt path (dequantize to FP16 plus oneMKL GEMM on XMX) and its concurrency.

## Intel support

| Item | Status (2026-10-10) |
|---|---|
| Port | `sycl/` in the main repository: the CUDA engine migrated with SYCLomatic (`dpct`) plus hand fixes, by the author of the fork `maxfridbe/Strata_B70` ([PR #423](https://github.com/Niko1221/Strata/pull/423)). No upstream file is changed; the port mirrors the files it alters and is re-migrated after each upstream merge |
| Official status | "Intel Arc officially supported, Linux only" since v0.1.40.2 (2026-10-07). The maintainers test one Arc Pro B70 (`xe`) and one Arc A750 (`i915`). Everything with two or more cards is community-tested |
| Main branch | `fb58e0d` = v0.1.41 (2026-10-08); no newer commit on 2026-10-10 |
| Build | Linux only, from source. Docker route: `./setup.sh --backend sycl` builds in the `strata-sycl-dev` image, which starts from a community llama.cpp SYCL image (`ghcr.io/snailium/llama.cpp-sycl-intel-b70/llama-sycl-b70:stable`) plus `dpct`. Native route: oneAPI DPC++ 2026.1 with oneMKL, `cmake -S sycl -B build-sycl-aot -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_SYCL_AOT=bmg-g31` (needs `ocloc`; without AOT the first start JIT-compiles for about 47 s) |
| Load-bearing compiler flags | `-fp-model=precise`, `-cl-fp32-correctly-rounded-divide-sqrt` (byte-exact quantizers), `-fsycl-default-sub-group-size=32` (kernels written for 32-wide warps) |
| Model files | the same GGUFs llama.cpp uses, plus a "native pack" from `tools/iq_pack.py` (dense tensors, tokenizer, index; about 30 s), the MTP draft layer from the base checkpoint (`tools/mtp_fetch.py`, `mtp_pack.py --experts q2_0`, `mtp_rt.py`; 4.9 GB download, 809 MiB of VRAM) and the bundled expert profile |
| Images (vision) | not wired on Intel |
| Windows | no native path (a community fork exists) |

### Which quantizations the port runs

Expert formats in the SYCL kernels (`sycl/src/kernels/cuda/iq_kernels.dp.cpp` at `fb58e0d`):

| Format | Decode kernels | Prompt path | Intel reports |
|---|---|---|---|
| IQ1_M (Coder), IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0 | the port's fused "decode-once" kernels | dequantize to FP16, oneMKL GEMM | Coder, IQ2_XS, IQ3_XXS, IQ3_S on B70, B60, B65; UD-IQ4_XS on 2x B70 ([#1484](https://github.com/Niko1221/Strata/issues/1484)) and Swift IQ4_XS on three cards |
| Q4_K, Q5_K, Q5_1, Q8_0 (the UD-Q4_K_XL mix) | per-entry kernels only (slower path; upstream's grouped kernels are opt-in `STRATA_EXPERT_V2K=1`) | the same FP16 dequant handles these types | **none**. Setup marks UD-Q4_K_XL "NVIDIA only" (its fused prompt kernels are HIP gfx11 and CUDA); one community port reports degenerate output from a Q4-class attempt on 2x B70 and dropped it |

Setup knows these GGUFs by fingerprint: ISTA-DASLab GSQ-RCO Q2_0, IQ2_XS, IQ3_XXS, IQ3_S; Swift 1.5 Q2_0, IQ2_XS, IQ3_XXS; Coder IQ1_M; unsloth UD-IQ4_XS and UD-Q4_K_XL. Other files, including the huihui-ai abliterated UD-Q4_K_XL we serve, need the manual pack route (`docs/ORCA.md` shows it for the OrcaRouter file). The abliterated file has the same tensor layout as unsloth's UD-Q4_K_XL ([quant note](../../models/qwen3.8-flash-next/research/2026-09-24-quants-and-uncensored-variants.md)), so `iq_pack.py --compat-bf16` should accept it (UNVERIFIED).

## Multi-GPU

Strata splits layers, not tensors: contiguous layer ranges per card, one hand-off per verify window through pinned host memory (a few hundred KB), no peer-to-peer access needed. `--peer-device` (a second card as an expert tier) is stubbed on SYCL. `--pipeline-windows` is for exactly two GPUs.

| Flag or variable | Meaning |
|---|---|
| `--layer-split 12,24,36` or `auto` | the first layer of each later card. `auto` scores every placement for up to four cards (since 0.1.40) by how much of the expert profile the caches hold |
| `ONEAPI_DEVICE_SELECTOR=level_zero:0,1,2,3` | which Level Zero devices the engine sees; the Docker image pins `level_zero:0`, the wrapper passes the host's value through |
| `--split-device` | the later stages' devices; default the next visible ones |
| `--trim-stage-weights` (engine flag, 0.1.41) or `STRATA_STAGE_TRIM=1` (0.1.39 to 0.1.40 ports) | each card loads only its own layers' dense weights; needs an explicit split |
| `--batch N --batch-groups auto` | several conversations pipelined through the cards |

What each card holds: its layers' dense weights, its session state and KV for its layers, its verify window and prompt buffers (about 1.5 GiB at 2,048-token chunks, more at 4,096), and an expert cache for its layers; the last card also the output head and the draft layer (about 0.8 GiB).

### Multi-card status on Arc

| Build | Hardware | Result | Source |
|---|---|---|---|
| 0.1.39-sycl `a5682fe` plus two local fixes | 2x B70, IQ3_XXS all resident, 32K, INT8 KV, `--spec 4 --mtp` | works: 56 to 61 tok/s short, 82 tok/s after a 720-token prompt, 51 after 3.2K; prompt 290 to 540 tok/s | [#1440](https://github.com/Niko1221/Strata/issues/1440) |
| 0.1.39-sycl `6f32ec0` plus two fixes | 2x B60 (24 GB), Coder IQ1_M and IQ2_XS, 8K | works: 56.6 to 61.3 tok/s, prompt 425 to 445 tok/s at 2K | [bench report in the repo](https://github.com/Niko1221/Strata/tree/main/bench/results/2026-10-04-community-2x-arc-pro-b60) |
| [PR #1111](https://github.com/Niko1221/Strata/pull/1111) head `3f37281` (branch `intel-arc-0.1.40` of `maxfridbe/Strata_B70`, 0.1.40-sycl) | 2x B70, IQ3_XXS all resident | works: 96.4 tok/s short, 2,477-token prompt in 3.94 s then 67.1 tok/s, 170 of 170 requests in a 12-minute soak; a second host: 64 short, 95 / 303 at 720 tokens, 67 / 756 at 3.2K with `--ple-io ram` (30 to 40% less decode with the default `--ple-io direct` on a slow NVMe) | [#1054](https://github.com/Niko1221/Strata/issues/1054) |
| PR #1111 branch `3f37281` | 2x B70, UD-IQ4_XS, 86 to 88% resident on a 30 GiB host | runs with `STRATA_EXPERT_SPLIT=1`: prompt 712 tok/s at 7.2 to 7.6K, decode 16 to 18 tok/s with `--spec 2` and no MTP; the default expert kernels gave corrupted output after 20 to 40 tokens | [#1484](https://github.com/Niko1221/Strata/issues/1484) |
| main v0.1.40.2 `e8ca9af` | 2x B70 | fails three ways: the host mirror takes later stages' layers, card 0's VRAM is charged to card 1, and the first request crashes | [#1440](https://github.com/Niko1221/Strata/issues/1440) |
| main v0.1.41 `fb58e0d` | 2x B70 and 2x B60 | starts with 100% of experts resident; the first request crashes in `sycl/src/core/verify.cpp:1706` ("Cannot submit to a queue with a dependency from a graph that is associated with a different context"). `STRATA_SH_STREAM=0` avoids the crash at about 6 tok/s on B70 and 1.7 on B60. Open on 2026-10-10 | [#1440](https://github.com/Niko1221/Strata/issues/1440), [#1054](https://github.com/Niko1221/Strata/issues/1054) |
| fork [Strata-Void](https://github.com/Jumbomuffin777/Strata-Void) (0.1.39-sycl base) | B70 + B65 + B60 (88 GB), Swift 1.5 IQ4_XS all resident, `--layer-split 18,35`, up to 128K | 40 to 45 tok/s single stream, about 113 tok/s over 6 slots, prompt 659 tok/s at 16K and 600 to 630 to 122K | fork README |
| fork "Strata-for-Intel-B70s" | 2x B70, IQ3_S, 262K with KV streaming | 18.5 to 21.7 tok/s decode, 317 to 345 tok/s prompt from 32K to 262K; base version unknown | fork README |

No report with four Intel cards exists (searched 2026-10-10). The three-card fork run is the closest. For a multi-B70 run today the working code is the PR #1111 branch, not main.

## Single-card results worth knowing

| Card and model | Decode | Prompt | Source |
|---|---|---|---|
| B70, Coder IQ1_M (all 12,288 experts resident), 0.1.39-sycl benchy on a Gen5 host | 77.3 tok/s at 20 tokens, 71.9 at 8K, 67.6 at 40K, 63.9 at 128K, 56.0 at 256K | 875 at 2,185, 1,285 at 8K, 1,392 at 40K, 1,327 at 128K, 1,271 at 256K | PR #1111 `docs/INTEL_PERFORMANCE.md` |
| B70, IQ2_XS (6,164 experts in a RAM mirror), same | 70.0 to 78.5 at 20 to 8K, 65.4 at 128K | 745 at 2,185, 1,050 at 8K, 1,117 at 40K | same |
| B70, IQ3_S (11.5k of 24.6k resident), maintainers, v0.1.40.2 | 30.5 prose, 40.8 code | 980 to 1,002 at 4K with `--prefill 4096` (618 with 2,048-token chunks) | release notes, `docs/INTEL.md` |
| B70, IQ3_S, PR #1111 build, same reporter's "current" 0.1.39 build in brackets | 55.0 (53.5) at 2,185 tokens, 54.0 (52.5) at 8K, 48.0 (50.6) at 40K, 48.75 (46.0) at 128K, 43.1 (38.5) at 256K | 433 (287), 682 (588), 744 (658), 736 (612), 721 (555) | PR #1111 comments |
| B65 (same die as the B70, 200 W), IQ2_XS full, v0.1.40.2, one card | 45.0 at 512 in, 43.5 at 7K in; 54.5 on a 20-token fixture; 262K profile 43.5 / 42.4 | 352 at 512-token chunks, 830 at 7K with 4,096-token chunks | bench report `2026-10-07-community-arc-b65-v01402` in the repo |
| B70 vs llama.cpp SYCL, same card, same GGUF (Coder), port author | 78 vs 23 to 25 tok/s | 790 vs about 150 tok/s | `docs/INTEL.md` (stock llama.cpp, not our patched build) |

For our production numbers with llama.cpp on four cards see the [model card](../../models/qwen3.8-flash-next/README.md#headline-numbers): 48.6 to 50.5 tok/s short, 60.6 to 67.2 at 10K, 40.6 at 219K, prompt 568 to 645 tok/s at 10 to 39K, with a different and heavier quant (UD-Q4_K_XL) and vision.

## Engine flags that matter

| Flag | Notes |
|---|---|
| `--pack DIR --native <shard 1>` | the native pack and the GGUF it was cut from; the engine finds the other shards by name |
| `--ple-gguf <shard>` | the shard holding `per_layer_token_embd.weight`; found by name from 0.1.38 on, pass it anyway |
| `--ple-io ram` | lock the n-gram table in RAM (28.8 GB for the unsloth files); the default `direct` reads rows from the SSD and cost 30 to 40% of decode on a DRAM-less NVMe in #1054 |
| `--expert-profile data/expert-profile.bin --expert-cache auto` | the original model's profile; `auto` fills VRAM down to the reserve |
| `--stream-experts` | the port's mode on `xe`: experts go from the GGUF into VRAM, no host arena |
| `--prefill 4096` | prompt chunk; setup writes 4,096 for 24 GB+ Arc cards. `--no-prefill-borrow` keeps own prompt buffers instead of lending cache slots (the port borrows above 32K by default; a borrow costs about 1 s of refill per prompt) |
| `--spec 4 --spec-min-p 0.5 --mtp <mtp/rt>` | setup's speculative defaults. `--spec 6` was +6.5% on code and -1.5% on prose on the B70 |
| `--max-context N --kv int8` | KV formats `fp16`, `int8`, `q4_0`, `k8v4`; about 20 KiB per token at int8 (3.6 GB at 262K over all cards). `--kv-resident 32768` streams the KV from pinned RAM, needed only when VRAM is short |
| `--vram-reserve-mib 1024` to `2048` | leave headroom: an allocation past VRAM on `xe` spills into host RAM ([our finding](../../findings/xe-vram-overcommit-spills-into-host-ram.md), and the port's own warning) |
| `--batch N`, `--batch-groups auto`, `--conversation-cache-mib N` | slots and parked conversations; off for the first measurements |

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `STRATA_VERIFY_DEVICE_PLAN` | 1 (wrapper) | the GPU plans each layer itself |
| `STRATA_VERIFY_NO_HOST` | 1 (wrapper, `xe`) | the host waits for the whole window graph instead of per-layer rings. Required on `xe` (host-to-GPU flag stores are not visible there), valid only when every missing expert is mirrored. `0` on `i915` A-series cards |
| `STRATA_MIRROR_MIB` | free RAM minus 4 GiB | size of the pinned host mirror for experts outside VRAM; `0` turns it off (set it on an all-resident split, or an unused mirror is allocated) |
| `STRATA_EXPERT_SPLIT` | 0 | `1` runs upstream's grouped expert kernels instead of the port's (11 to 16% slower on the Coder; the only correct ones for UD-IQ4_XS on 2x B70 in #1484) |
| `STRATA_SH_STREAM` | 1 on main, 0 in PR #1111 | forks the shared expert onto a side queue during window capture; suspected cause of the main-branch split crash |
| `STRATA_SPIN_MAX`, `-DSTRATA_SYCL_SPIN_MAX` | 20,000 reads on `xe` cards (per device at run time since 0.1.41; a fixed 20,000 in the PR #1111 build) | the GPU's bounded spin on host flags. 20,000 is right with `--mtp` (45 vs 26 tok/s in [#1473](https://github.com/Niko1221/Strata/issues/1473)); without `--mtp` the suffix drafter needs 2,000,000 or it drafts nothing |
| `STRATA_ARENA_ALIAS_CHECK` | 1 | checks allocations of 32 MiB or more for aliased pages, which `xe` returned on the B70 (wrong experts in the cache); fixed by reallocation in 0.1.40.2 |
| `STRATA_STAGE_TRIM` | 0 | `1`: each card loads only its layers' dense weights (ports up to 0.1.40; `--trim-stage-weights` on main) |
| `STRATA_STAGER_THREADS` | 12 (wrapper) | threads reading expert blobs from the GGUF |
| `STRATA_SELECT_GEMM`, `STRATA_ATTN_PERCELL`, `STRATA_LEND_MIRROR` | on (PR #1111 branch only) | QSA block scores as oneMKL GEMM tiles, per-cell prompt attention, lendable experts mirrored in RAM: the branch's prompt-path speedups |
| `STRATA_PROMPT_ATTN_XMX`, `STRATA_EXPERT_XMX` | off | XMX (matrix engine) kernels that lost to the plain ones on the B70; leave off |
| `SYCL_CACHE_PERSISTENT` | 0 | the persistent JIT cache crashed on Xe2 |
| `ZES_ENABLE_SYSMAN` | 1 | free-memory queries |
| `ONEAPI_DEVICE_SELECTOR` | `level_zero:0` in the image | set `level_zero:0,1,2,3` for four cards |

`sycl/serve/strata-sycl.sh` drops any variable whose value is exactly `0`, so a `0` must be written as `00` through that wrapper (#1440). The engine reads `STRATA_VERIFY_NO_HOST=0` as off.

## Known issues on Arc (open on 2026-10-10 unless noted)

| Issue | What | Status |
|---|---|---|
| [#1440](https://github.com/Niko1221/Strata/issues/1440) | 2x B70 layer split on main: three failures | two fixed in 0.1.41; the first-request graph crash open |
| [#1054](https://github.com/Niko1221/Strata/issues/1054) | 2x B70 `--layer-split auto` deadlock after the host-mirror fill | gone in 0.1.41, replaced by the crash above; working on the PR #1111 branch |
| [#1484](https://github.com/Niko1221/Strata/issues/1484) | 2x B70 + UD-IQ4_XS: the port's default expert kernels corrupt output after 20 to 40 tokens; `STRATA_EXPERT_SPLIT=1` restores it | open; cause suspected in synchronization, not the kernels' arithmetic |
| [#1794](https://github.com/Niko1221/Strata/issues/1794) | `sycl/setup_intel.py` reads the version as `0` on 0.1.41, so setup refuses UD-IQ4_XS and UD-Q4_K_XL on Intel | open; hand-written config avoids setup |
| [#1473](https://github.com/Niko1221/Strata/issues/1473), [#1397](https://github.com/Niko1221/Strata/issues/1397) | the spin bound: one constant cannot serve `--mtp` and suffix-only drafting | per-device bound in 0.1.41; the two-bound split not done |
| [#1745](https://github.com/Niko1221/Strata/issues/1745) | the PCIe probe times a pageable copy and reads the link about 4x low | fix PR #1746 open; irrelevant with nothing mirrored |
| [#867](https://github.com/Niko1221/Strata/issues/867) | 2x B60: per-layer ring waits never saw a slow layer ("never rang") | fixed in 0.1.40 |
| v0.1.40.2 notes | `xe` returned large allocations with aliased pages (wrong answers); the B70 hung when given pageable host memory without `--stream-experts` | fixed: alias check, `--pcie-frac 0` without the flag |
| #1145, #498 | `setup.py` refuses multi-GPU for the unsloth files unless RAM holds the GGUF plus 24 GB | hand-written config |
| `docs/INTEL_ARC.md` | `source setvars.sh` fails under `set -u` (`OCL_ICD_FILENAMES`) | source it before `set -u` |

## What we would lose against our llama.cpp setup

- Vision: not wired on Intel. Our production config serves the BF16 projector.
- The abliterated UD-Q4_K_XL checkpoint: no Intel report of any Q4_K_XL file on Strata; manual pack.
- A tested four-card path: none exists; main's split is broken and the working branch is a six-day-old fork.
- Our llama.cpp patches do not carry over; Strata's QSA and MoE paths are its own.

## Notes

- The maintainers' own SYCL port measurement baseline is one B70 in a PCIe Gen3 x8 slot; most speed claims after 2026-10-04 come from the port author's Gen5 x8 host with a B70 and a B65.
- `docs/INTEL.md` in the Strata repository is the port's reference (800+ lines of measurements and fixes); `docs/INTEL_ARC.md` is the short user page; the PR #1111 branch adds `docs/INTEL_PERFORMANCE.md` and `sycl/TODO.md`.
