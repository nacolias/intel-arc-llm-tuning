# Strata: an engine built for Qwen3.8-Flash-Next, with a SYCL port for Arc

| | |
|---|---|
| **Date read** | 2026-10-10 |
| **Source** | [Niko1221/Strata](https://github.com/Niko1221/Strata) at `fb58e0d` (v0.1.41): `docs/INTEL.md`, `docs/INTEL_ARC.md`, `docs/MULTI_GPU.md`, `docs/MODELS.md`, `docs/UNSLOTH_Q4.md`, `sycl/`; release notes v0.1.39 to v0.1.41; issues [#867](https://github.com/Niko1221/Strata/issues/867), [#1054](https://github.com/Niko1221/Strata/issues/1054), [#1397](https://github.com/Niko1221/Strata/issues/1397), [#1440](https://github.com/Niko1221/Strata/issues/1440), [#1473](https://github.com/Niko1221/Strata/issues/1473), [#1484](https://github.com/Niko1221/Strata/issues/1484), [#1545](https://github.com/Niko1221/Strata/issues/1545), [#1745](https://github.com/Niko1221/Strata/issues/1745), [#1794](https://github.com/Niko1221/Strata/issues/1794); [PR #1111](https://github.com/Niko1221/Strata/pull/1111) at `3f37281` with its `docs/INTEL_PERFORMANCE.md` and `sycl/TODO.md`; the forks [Strata-Void](https://github.com/Jumbomuffin777/Strata-Void) and [Strata-for-Intel-B70s](https://github.com/chmihimabb-max/Strata-for-Intel-B70s) |
| **Author / org** | Niko1221 (engine and server); the SYCL port by the author of `maxfridbe/Strata_B70`; community reporters on the issues above |
| **Type** | upstream source, release notes, issues and PRs, community benchmark reports |
| **Applies to** | Qwen3.8-Flash-Next on Arc Pro B70, B65 and B60; Strata 0.1.39-sycl to 0.1.41; oneAPI 2026.1; the `xe` driver |

## Summary

Strata is an MIT-licensed engine written for exactly this model, and since v0.1.40.2 (2026-10-07) its SYCL port is "officially supported" on Intel Arc under Linux. With every expert in VRAM it decodes about three times faster than stock llama.cpp SYCL on one B70 (port author's numbers), and a two-B70 layer split reached 96 tok/s on short prompts and 67 after a 2.5K prompt on the PR #1111 branch. Nobody has run it on four Intel cards, the main branch's multi-card split crashes on the first request, there is no vision on Intel, and our abliterated UD-Q4_K_XL file is untested on any Intel card. Every number here is **UNVERIFIED** on our hardware; the engine page is [engines/strata/](../../../engines/strata/README.md) and the planned run is [this experiment](../experiments/2026-10-10-strata-sycl-quad-b70.md).

## What it is, in one paragraph

A C++ engine with its own kernels (plus parts of ggml) and a Python server with OpenAI, Anthropic and Responses APIs. Decode runs as speculative verify windows: the model's own MTP (multi-token prediction) layer drafts up to 3 tokens, prompt lookup can add more, and each window is a captured graph (about 2,400 to 2,600 nodes) replayed per round. Experts live in a VRAM cache placed from a profile, with the remainder in pinned RAM or the GGUF. QSA (Qwen sparse attention) block selection is implemented natively. The PLE (per-layer embedding) n-gram table is read by row from the SSD or locked in RAM. The Intel version is the same engine migrated with SYCLomatic into `sycl/`, run through Docker by default or natively with oneAPI.

## Claims to verify

| Claim | Reported number | Their setup | Status |
|---|---|---|---|
| The SYCL engine decodes about 3x faster than llama.cpp SYCL on one B70 with the same GGUF | 78 vs 23 to 25 tok/s decode; 790 vs about 150 tok/s prompt | Coder IQ1_M, all 12,288 experts resident, B70 at PCIe Gen3 x8, stock llama.cpp (not our patched build) | UNVERIFIED |
| A two-B70 layer split with every expert resident runs at full speed on the PR #1111 branch | 96.4 tok/s short; a 2,477-token prompt in 3.94 s then 67.1 tok/s; 170 of 170 requests in a 12-minute soak | `3f37281`, IQ3_XXS, 32K context, INT8 KV, `--spec 4 --mtp`, EPYC 4464P, each card Gen5 x8 on its own root port | UNVERIFIED |
| `--ple-io ram` is worth 30 to 40% of decode over the default SSD row reads | 64 vs 38 to 59 tok/s short; 95 vs 71 after 720 tokens; 67 vs 43 after 3.2K | same branch, 2x B70 at Gen4 x16, EPYC 7F52, DRAM-less NVMe | UNVERIFIED (our NVMe is a Gen3 drive; our RAM is 121.5 GiB) |
| main v0.1.41 starts a split but crashes on the first request; `STRATA_SH_STREAM=0` avoids it at about 6 tok/s | `verify.cpp:1706`, graph from another context | 2x B70 and 2x B60 | upstream-documented, open |
| UD-IQ4_XS on 2x B70 corrupts output with the port's default expert kernels after 20 to 40 tokens; `STRATA_EXPERT_SPLIT=1` is clean | 0 of 24 probes corrupt with the flag | `3f37281`, 86 to 88% of experts resident, 30 GiB host RAM | UNVERIFIED; the reporter withdrew the kernel-tiling hypothesis and suspects synchronization |
| Three Arc Pro cards hold Swift 1.5 IQ4_XS fully resident | 40 to 45 tok/s single stream; about 113 tok/s over 6 slots; prompt 659 tok/s at 16K, 600 to 630 to 122K; a 120K prompt in about 200 s | Strata-Void fork (0.1.39-sycl base), B70 + B65 + B60, 88 GB, `--layer-split 18,35`, Ryzen 9 9950X | UNVERIFIED |
| One B70 holds decode at long context | IQ3_S: 48.75 tok/s and 736 tok/s prompt at 128K; 43.1 and 721 at 256K. Coder: 63.9 and 1,327 at 128K; 56.0 and 1,271 at 256K | PR #1111 build; IQ3_S partly in a RAM mirror | UNVERIFIED |
| 4,096-token prompt chunks read a 4K prompt at about 1,000 tok/s on one B70 | 980 to 1,002 tok/s (618 with 2,048-token chunks) | maintainers, v0.1.40.2, IQ3_S with 11.5k of 24.6k experts resident | UNVERIFIED |
| The GPU spin bound of 20,000 reads is right with `--mtp` and wrong without | 45.4 vs 26.2 tok/s with `--mtp`; 18.9 vs 69.7 without | B70 on Windows OpenCL (#1473); 0.1.41 chooses it per device | upstream-documented |
| KV cache is about 20 KiB per token at int8 | 3.6 GB at 262K over all cards | `docs/INTEL.md` | upstream-documented |
| UD-Q4_K_XL is NVIDIA-only in Strata | 71.7 GiB of experts; fused prompt kernels for Q4_K and Q5_K exist for CUDA and HIP gfx11 only; the SYCL port has per-entry decode kernels and a generic FP16 dequant for these types | `docs/UNSLOTH_Q4.md`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` | upstream-documented; running it on Intel is UNVERIFIED, one fork reports degenerate output from a Q4-class attempt |
| Decode is bound by the per-round overhead, not by expert bytes | a round costs about 53 ms whatever its size (2026-09-29, improved since); kernels stream 130 to 160 GB/s of a 608 GB/s card; launch gaps about 5 us per node | B70, port author | upstream-documented |

## How it compares with our production numbers

Our numbers are the production llama.cpp build ([benchmark](../benchmarks/2026-09-25-production-build.md)): four B70s, huihui-ai abliterated UD-Q4_K_XL, 262K context, MTP draft n-max 3, single slot, 230 W, GPU clock floor. The Strata rows are community reports on other hosts with lighter quantizations, so this is a sketch of the possible, not a comparison.

| Workload | Ours, llama.cpp | Strata reports (UNVERIFIED) |
|---|---|---|
| Short prompt decode | 48.6 to 50.5 tok/s | 96.4 on 2x B70 (IQ3_XXS, PR #1111); 77.3 on 1x B70 (Coder); 40 to 45 on three cards (Swift IQ4_XS, fork) |
| Decode after a 2 to 10K prompt | 60.6 to 67.2 at 10K | 67.1 after 2.5K (2x B70, IQ3_XXS); 74.6 after 2,185 and 71.9 after 8K (1x B70, Coder); 55.0 after 2,185 (1x B70, IQ3_S) |
| Decode at about 40K | 53.6 to 55.2 at 39K | 67.6 (Coder) and 48.0 (IQ3_S) at 40K, 1x B70 |
| Decode at 128 to 256K | about 39 at 135K; 40.6 at 219K | 63.9 (Coder) and 48.75 (IQ3_S) at 128K; 56.0 (Coder) and 43.1 (IQ3_S) at 256K, 1x B70 |
| Prompt processing, 10 to 40K | 568 to 645 tok/s | 1,392 (Coder) and 744 (IQ3_S) at 40K, 1x B70; 600 to 630 to 122K on three cards |
| Prompt processing, about 220K | 268 tok/s | 721 at 256K (IQ3_S, 1x B70); 317 at 262K (IQ3_S, 2x B70, older fork) |
| Concurrency | one slot | up to 8 slots; about 113 tok/s aggregate over 6 (three cards, fork) |
| Vision | yes (BF16 projector) | not on Intel |
| Context | 262,144 | 262,144 served on one B65 and one B70 |

Two structural reasons Strata could be faster here: its decode windows are captured graphs per card, where llama.cpp refuses SYCL graphs with more than one GPU ([finding](../../../findings/llama-cpp-sycl-graphs-off-with-multiple-gpus.md)), and its prompt path dequantizes experts to FP16 once per chunk and runs oneMKL GEMMs on XMX instead of llama.cpp's per-expert GEMM loop with 144 host syncs per micro-batch ([finding](../../../findings/llama-cpp-sycl-moe-prefill-host-sync.md)). Two reasons it could be slower: four pipeline stages cost three hand-offs per window instead of one, and the only four-way data point is none.

## What would differ on our host

- **Topology.** All four cards sit behind one PEX88096 switch on a shared 16 GT/s x16 uplink: 28.4 GB/s host to GPU in total, 28.5 GB/s for any P2P pair ([host](../../../hardware/hosts/quad-b70-5800x-pex88096.md)). The community two-B70 results had each card on its own root port. Strata's split moves a few hundred KB per window through pinned RAM, and with every expert resident there is no mirror traffic, so the shared uplink should matter little for decode. The PCIe probe bug ([#1745](https://github.com/Niko1221/Strata/issues/1745)) is moot when nothing is mirrored.
- **Host.** A Zen 3 desktop against the reporters' EPYC and Zen 5 hosts. Our llama.cpp decode is launch-bound at about 4,000 launches per token ([research](2026-09-24-decode-bottleneck-and-speed-plan.md)); Strata replays one graph of about 2,500 nodes per round and gets 2 to 4 tokens out of it, so the host should count for less.
- **VRAM.** With four 32 GB cards every candidate quantization fits in VRAM with room to spare (table below), which no community report has had. KV streaming (`--kv-resident`) and the host mirror are unnecessary; set `STRATA_MIRROR_MIB=0`.
- **RAM.** `--ple-io ram` locks 28.8 GB of the n-gram table; 121.5 GiB MemTotal covers it. Our `xe` overcommit finding still applies: watch `MemTotal − MemAvailable − AnonPages` during the load ([finding](../../../findings/xe-vram-overcommit-spills-into-host-ram.md)).
- **Clocks and power.** The GPU clock floor sped up llama.cpp's layer split ([finding](../../../findings/gpu-clock-floor-speeds-layer-split.md)); Strata's split is also sequential across cards, so measure with the floor on. ASPM L1 stays on.
- **Setup.** Strata's installer refuses our files on Intel ([#1794](https://github.com/Niko1221/Strata/issues/1794)) and refuses multi-GPU for the unsloth files without 135 GB of RAM, so the config is written by hand and the engine binary is pointed at directly, as the Strata-Void fork does. The Docker wrapper and its community base image are not needed: the host already has oneAPI 2026.1 and compute-runtime 26.35.

### VRAM budget on four B70s (estimate)

Per card, with 30.3 GiB usable ([hardware](../../../hardware/README.md)). Component sizes are Strata's documented ones; the startup log decides.

| Component | Per card, GiB | Source |
|---|---|---|
| Reserve (`--vram-reserve-mib 2048`) | 2.0 | setup's value above 32K |
| Dense weights of the stage's own layers (`STRATA_STAGE_TRIM=1`) | 1 to 2 | whole dense set 1.4 GB for UD-Q4_K_XL, about 3.4 GB for the Coder |
| Prompt buffers at `--prefill 4096` | 2 to 3 | 1.5 GiB at 2,048-token chunks per `docs/MULTI_GPU.md` |
| KV at 262K, int8, 12 QSA layers split four ways | about 0.9 | 20 KiB per token |
| Draft layer and output head (last card only) | 0.8 | `docs/MULTI_GPU.md` |
| Left for the expert cache | about 22 to 24 | |

Four caches of 22 to 24 GiB hold 88 to 96 GiB of experts: unsloth UD-IQ4_XS needs 55.4 GiB (59.5 GB), UD-Q4_K_XL 71.7 GiB, ISTA-DASLab IQ3_S about 43 GiB. All three fit; UD-IQ4_XS leaves about 35 GiB spare across the cards. Our llama.cpp peak was 28,957 MiB on the busiest card at 219K with UD-Q4_K_XL.

## Which checkpoint

| File | Size, experts | Strata on Intel | Note |
|---|---|---|---|
| unsloth UD-IQ4_XS | 94 GB, 59.5 GB of experts (IQ3_S gate/up, IQ4_NL down) | runs on 2x B70 with `STRATA_EXPERT_SPLIT=1` ([#1484](https://github.com/Niko1221/Strata/issues/1484)); the same formats as the three-card fork run | **first candidate**: the only 4-bit-class file with Intel reports; censored (stock) |
| huihui-ai abliterated UD-Q4_K_XL (production) | 111 GB, 71.7 GiB of experts (Q4_K, Q5_K, Q5_1, Q8_0) | no Intel report; per-entry decode kernels and FP16 prompt dequant exist for these types; setup calls the unsloth original NVIDIA-only | second candidate: same layout as unsloth's file, so `iq_pack.py --compat-bf16` should accept it (UNVERIFIED); fingerprint unknown to setup |
| ISTA-DASLab GSQ-RCO IQ3_S | 84 GB, about 46 GB of experts | the maintainers' B70 test model; 262K served on one B65 | fallback if the 4-bit files misbehave; quality below Q4 |
| ISTA-DASLab IQ3_XXS | 76 GB, 43 GB of experts | the 2x B70 configuration of #1054 and #1440 (96 and 67 tok/s) | diagnostic: reproduces the community result on two cards if the four-card run fails |

The uncensored IQ4-class files from the [quant note](2026-09-24-quants-and-uncensored-variants.md) (heretic-2, OrcaRouter, the apetersson LoRA) would all go through the manual pack route; none has a Strata report.

## Which Strata to build

| Tree | For | Against |
|---|---|---|
| PR #1111 head `3f37281` (branch `intel-arc-0.1.40` of `maxfridbe/Strata_B70`, engine 0.1.40-sycl, 2026-10-06) | the only tree with a full-speed multi-B70 result (96 and 67 tok/s); #1054 and #866 fixed; `STRATA_SH_STREAM` off; the lend mirror, QSA scores as GEMM tiles and per-cell prompt attention | a fork six days old; lacks 0.1.41's B70 dequant speedup (+6% prompt), per-device spin bound and `--batch-groups`; the port author said the merge would follow "in 3 days" on 2026-10-08 |
| main `fb58e0d` (v0.1.41) | current; two of #1440's three split bugs fixed; batch groups per GPU | the first request on a split crashes; the `STRATA_SH_STREAM=0` workaround runs at about 6 tok/s |
| Strata-Void fork | the only three-card all-resident Intel run; `--batch` groups and `--slot-context` ported to the SYCL engine; faster IQ4_XS/IQ4_NL prompt dequant | a 0.1.39-sycl base plus one person's changes; task-parallel features we do not need |

Build the PR #1111 tree first and keep main as a 10-minute check of whether the crash also appears with four cards, to report upstream.

## Risks

1. **No four-way precedent.** The branch author tested one card and a B70 + B65 pair; the four-card placement code (`auto` for four cards, three hand-offs per window) is untested on SYCL.
2. **Silent wrong output.** Three separate mechanisms have produced it on B70s: aliased pages from `xe` (fixed by a start-up check), the UD-IQ4_XS kernel issue of #1484, and the host handshake without `STRATA_VERIFY_NO_HOST=1`. The greedy checks in [benchmarks/README.md](../../../benchmarks/README.md#correctness-check) are not optional.
3. **The production checkpoint may not run.** UD-Q4_K_XL has never been reported on Intel; the per-entry kernels for its formats are the slow path.
4. **No vision**, so Strata cannot replace the production endpoint for image requests, and two engines cannot share the cards ([finding](../../../findings/xe-vram-overcommit-spills-into-host-ram.md)).
5. **Churn.** Six releases in seven days, a port that is re-migrated after each merge, and the working multi-card code living in a fork. Pin commits in every record.
6. **Memory.** `--ple-io ram` locks 28.8 GB; pinned prompt buffers and the hand-off buffers add to it. Run the RAM watchdog.
7. **Slots.** Eight at most; batch slots decode without MTP drafts in the fork's measurements.

## Relevance

For our host the question is narrow: with every expert resident on four cards, is Strata's window-graph decode and FP16-GEMM prompt path faster than our patched llama.cpp, and is it correct? The community data says yes on one and two cards against stock llama.cpp, by 1.5x to 3x on decode and 1.5x to 2x on prompts, at the cost of vision and with an unproven checkpoint. One test day settles it. Production stays on llama.cpp until the experiment reports.

## Actions

- [ ] Run [the planned experiment](../experiments/2026-10-10-strata-sycl-quad-b70.md): PR #1111 tree, UD-IQ4_XS, `--layer-split 12,24,36`, every expert resident, our fnbench and lcbench points, greedy checks.
- [ ] If the four-card split works, post the config and numbers in [#1440](https://github.com/Niko1221/Strata/issues/1440) or as a `bench/results` report, with private information stripped.
- [ ] If UD-IQ4_XS needs `STRATA_EXPERT_SPLIT=1` here too, add the data point to [#1484](https://github.com/Niko1221/Strata/issues/1484).
- [ ] Pack the abliterated UD-Q4_K_XL; if it loads, compare greedy tokens against llama.cpp on the same file.
- [ ] Re-check after PR #1111 merges or v0.1.42 ships: is [#1440](https://github.com/Niko1221/Strata/issues/1440)'s first-request crash fixed on main?
- [ ] Decide what a text-only endpoint is worth against the vision we would lose.
