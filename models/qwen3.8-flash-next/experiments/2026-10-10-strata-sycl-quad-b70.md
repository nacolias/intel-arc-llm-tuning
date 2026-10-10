# Strata SYCL port on four B70s: layer split with every expert in VRAM

| | |
|---|---|
| **Date** | 2026-10-10 (planned, not yet run) |
| **Model / checkpoint** | Qwen3.8-Flash-Next: unsloth `UD-IQ4_XS` first; then the huihui-ai abliterated `UD-Q4_K_XL` (production file) if its pack loads |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | planned |
| **Baseline** | [production build, llama.cpp](../benchmarks/2026-09-25-production-build.md) and the [135K agent session](../benchmarks/2026-09-25-agent-session-135k.md) |
| **Result** | not yet run |
| **Related** | [research note](../research/2026-10-10-strata-sycl-port.md), [engine page](../../../engines/strata/README.md), findings on [VRAM overcommit](../../../findings/xe-vram-overcommit-spills-into-host-ram.md) and the [clock floor](../../../findings/gpu-clock-floor-speeds-layer-split.md) |

## Hypothesis

Strata's SYCL engine, split over the four cards with every expert resident in VRAM, decodes faster than our patched llama.cpp build at short and medium context and reads prompts faster, because each card replays its verify window as one captured graph and the prompt path runs FP16 GEMMs instead of a host-synced per-expert loop. Community points (all UNVERIFIED, other hosts, lighter quantizations): 96 tok/s short and 67 after 2.5K on two B70s, 40 to 45 tok/s on three mixed Arc Pro cards with IQ4_XS, 744 to 1,392 tok/s prompt at 40K on one B70. Against our 48.6 to 50.5 tok/s short and 568 to 645 tok/s prompt, the expected range is +10% to +90% on short-prompt decode and +20% to +100% on prompt processing, with the four-way pipeline (three hand-offs per window) as the unknown that could cancel it. At 135K and beyond the expectation is parity to +50% (community: 48.75 tok/s at 128K on one B70 with IQ3_S against our 39 to 40).

Correctness is part of the hypothesis: greedy output must be identical with speculative decoding on and off, and free of the collapse reported for UD-IQ4_XS on two B70s.

## Change

An engine swap, not a flag change. Four arms, in this order:

| Arm | Tree | Cards | Checkpoint | Purpose |
|---|---|---|---|---|
| 1 | Strata [PR #1111](https://github.com/Niko1221/Strata/pull/1111) head `3f37281` (engine 0.1.40-sycl) | 4, `--layer-split 12,24,36` | unsloth UD-IQ4_XS | the target configuration |
| 2 | Strata main `fb58e0d` (v0.1.41), same config | 4 | same | 10-minute check: does [#1440](https://github.com/Niko1221/Strata/issues/1440)'s first-request crash appear with four cards? Report either way |
| 3 | the better of 1 and 2 | 4 | huihui-ai abliterated UD-Q4_K_XL | can the production checkpoint run on Strata on Intel? |
| fallback | PR #1111 | 2, `--layer-split 24`, cards `level_zero:0,1` | ISTA-DASLab IQ3_XXS | reproduces the community two-card result if arm 1 fails, to separate host problems from four-stage problems |

Against the production launch ([configs/llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh)) the service differences are: no vision, int8 KV instead of f16, Strata's MTP windows (`--spec 4`, up to 3 MTP drafts plus prompt lookup) instead of n-max 3, and 4,096-token prompt chunks instead of `-ub 1536`.

## Environment

Fill in from the running system before the first measurement.

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | Strata commit (arm 1: `3f37281`; arm 2: `fb58e0d`), native build, `-DSTRATA_SYCL_AOT=bmg-g31`, `STRATA_SYCL_PARITY` on or off; the engine's own version line |
| Compiler | oneAPI DPC++ `icpx` 2026.1.1, oneMKL 2026.1, `ocloc` version |
| compute-runtime / IGC | 26.35.39758.11 / 2.41.9 on 2026-09-25; re-read |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / `xe` / 70.58.0; re-read |
| oneCCL | n/a |
| PyTorch XPU | n/a |
| Power cap per card | 230 W (`power1_cap`) |
| GPU clock | `min_freq = rp0` while serving ([tools/gpu-clock-floor.sh](../../../tools/gpu-clock-floor.sh)), as in production |
| Tensor / pipeline parallel | Strata layer split, 4 stages of 12 layers; one hand-off per verify window through pinned host memory |
| Pack and draft | `tools/iq_pack.py` output directory, `mtp/rt` built from the base checkpoint's MTP layer with Q2_0 experts, `data/expert-profile.bin` of the Strata commit |
| Model files | SHA-256 of every shard |

## Procedure

Paths are placeholders. Keep the Strata checkout, packs, draft and models on the Gen4 NVMe if space allows; the Gen3 drive reads about 3.4 GB/s.

### 0. Pre-flight

1. Stop the production llama-server and any timer or guard that restarts it. One GPU tenant per card ([finding](../../../findings/xe-vram-overcommit-spills-into-host-ram.md)).
2. Check the switch uplink trained at 16 GT/s x16 ([finding](../../../findings/pcie-switch-uplink-can-train-at-gen1.md)).
3. Apply the clock floor: `sudo tools/gpu-clock-floor.sh on`.
4. Start the RAM watchdog in a second shell and stop the engine if `MemAvailable` falls under 20 GiB:

```bash
while sleep 5; do
  awk '/MemTotal|MemAvailable|AnonPages/ {a[$1]=$2}
       END {printf "driver-held %.1f GiB  MemAvailable %.1f GiB\n",
            (a["MemTotal:"]-a["MemAvailable:"]-a["AnonPages:"])/1048576, a["MemAvailable:"]/1048576}' /proc/meminfo
done
```

5. `dmesg -w | grep -i -E 'xe|reset'` in a third shell; any GT reset ends the arm.

### 1. Source

```bash
git clone https://github.com/Niko1221/Strata /path/to/Strata
cd /path/to/Strata
git fetch origin pull/1111/head:pr-1111
git checkout pr-1111            # 3f37281, 2026-10-06; arm 2 uses: git checkout fb58e0d
```

### 2. Build natively

The host already has oneAPI 2026.1 and compute-runtime 26.35, so the Docker image is not needed. Source oneAPI before any `set -u` (its `setvars.sh` trips on `OCL_ICD_FILENAMES`).

```bash
source /opt/intel/oneapi/2026.1/oneapi-vars.sh
# ocloc is needed for the AOT build; the package name on Ubuntu is intel-ocloc (check what the host has)
cmake -S sycl -B build-sycl-aot -G Ninja -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
      -DSTRATA_SYCL_AOT=bmg-g31
cmake --build build-sycl-aot --target strata -j 16
```

Optional, once, on one card: build the kernel tests too (drop nothing; `STRATA_SYCL_PARITY` is on by default) and run `ONEAPI_DEVICE_SELECTOR=level_zero:0 ctest --test-dir build-sycl-aot`. Expect the model-fixture tests (`iq_parity`, `ple_parity`) and the grouped-S2 bitwise test to fail, as in the B65 report (26 pass, 4 fail). Record the counts.

Without `ocloc`, build without `-DSTRATA_SYCL_AOT` (JIT): the first start compiles for about 47 s, and the spin bound stays at 20,000 in this tree, which is the right value with `--mtp`.

### 3. Python environment

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export STRATA_GGUF_PY=/path/to/llama.cpp/gguf-py     # our llama.cpp checkout
```

### 4. Model files, pack, draft layer

Download `UD-IQ4_XS` from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (about 94 GB) and record the shard hashes. Find the shard that holds the n-gram table:

```bash
.venv/bin/python - <<'PY'
import glob, sys
sys.path.insert(0, "/path/to/llama.cpp/gguf-py")
from gguf import GGUFReader
for f in sorted(glob.glob("/path/to/models/UD-IQ4_XS/*.gguf")):
    if any(t.name == "per_layer_token_embd.weight" for t in GGUFReader(f).tensors):
        print("PLE shard:", f)
PY
```

Pack the dense tensors (about 30 s) and build the MTP draft layer (4.9 GB download from the base checkpoint):

```bash
.venv/bin/python tools/iq_pack.py --gguf /path/to/models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-0000N.gguf \
    --out packs/ud-iq4_xs --compat-bf16
.venv/bin/python tools/mtp_fetch.py fetch --out mtp
.venv/bin/python tools/mtp_pack.py --src mtp --experts q2_0 --out mtp/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf mtp/mtp-q2_0.gguf --out mtp/rt
cp data/draft_vocab.bin mtp/rt/draft_vocab.bin
```

If the packer refuses the file, record the message; `docs/UNSLOTH_Q4.md` and `docs/ORCA.md` in the Strata tree describe the manual route. For arm 3, run the same `iq_pack.py --compat-bf16` on shard 1 of the abliterated UD-Q4_K_XL into `packs/ud-q4_k_xl-abliterated`.

### 5. Config

Save as `strata-flash-next-quad.json` in the Strata checkout. The engine is pointed at directly (no Docker wrapper); the `env` block reaches the engine through the server. Card order follows `ONEAPI_DEVICE_SELECTOR`; the last card carries the draft layer and the head.

```json
{
  "exe": "build-sycl-aot/strata",
  "backend": "sycl",
  "model_name": "Qwen3.8-Flash-Next",
  "tokenizer": "<pack dir>/tokenizer",
  "args": [
    "--pack", "packs/ud-iq4_xs",
    "--native", "/path/to/models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-0000N.gguf",
    "--ple-gguf", "/path/to/models/UD-IQ4_XS/<the shard that holds per_layer_token_embd.weight>",
    "--ple-io", "ram",
    "--mtp", "mtp/rt",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "auto",
    "--stream-experts",
    "--prefill", "4096",
    "--no-prefill-borrow",
    "--spec", "4", "--spec-min-p", "0.5",
    "--max-context", "262144",
    "--kv", "int8",
    "--vram-reserve-mib", "2048",
    "--layer-split", "12,24,36",
    "--conversation-cache-mib", "0"
  ],
  "env": {
    "ONEAPI_DEVICE_SELECTOR": "level_zero:0,1,2,3",
    "STRATA_VERIFY_DEVICE_PLAN": "1",
    "STRATA_VERIFY_NO_HOST": "1",
    "STRATA_MIRROR_MIB": "0",
    "STRATA_STAGE_TRIM": "1",
    "STRATA_EXPERT_SPLIT": "1",
    "SYCL_CACHE_PERSISTENT": "0",
    "ZES_ENABLE_SYSMAN": "1"
  }
}
```

Why these values:

- `<pack dir>` is the pack written in step 4 (`packs/ud-iq4_xs`); the server reads the tokenizer from it.
- `--layer-split 12,24,36`: 12 layers and 3 QSA layers per card; `auto` is the second try. On main (arm 2) replace `STRATA_STAGE_TRIM` with the flag `--trim-stage-weights`.
- `STRATA_VERIFY_NO_HOST=1` is valid only with every expert resident, which the startup log must confirm (`100% of the experts resident` for every stage). `STRATA_MIRROR_MIB=0` stops an unused pinned mirror from being allocated.
- `STRATA_EXPERT_SPLIT=1` is the kernel set that gave correct UD-IQ4_XS output in [#1484](https://github.com/Niko1221/Strata/issues/1484); A/B it against the default (`STRATA_EXPERT_SPLIT` unset) once the greedy checks pass, since the port's own kernels were 11 to 16% faster on the Coder.
- `--no-prefill-borrow`: own prompt buffers; VRAM is not short, and a borrow costs about 1 s of refill per prompt.
- `--ple-io ram`: 28.8 GB locked in RAM; the default SSD row reads cost 30 to 40% of decode on one reporter's NVMe.
- `--kv int8`: 3.6 GB at 262K over four cards; `--kv fp16` is the A/B if int8 changes output quality.
- `--conversation-cache-mib 0` and no `--batch` for the first measurements; concurrency comes after single-stream passes.

### 6. Start and check

```bash
source /opt/intel/oneapi/2026.1/oneapi-vars.sh
export SYCL_CACHE_PERSISTENT=0 ZES_ENABLE_SYSMAN=1
STRATA_API_KEY="$(cat "$API_KEY_FILE")" .venv/bin/python -m serve.server --engine strata \
    --config strata-flash-next-quad.json --host 127.0.0.1 --port 8085
```

Before measuring, read from the engine's startup lines and record: the layer ranges per card, `cache slots` and free VRAM per card, `100% of the experts resident` on every stage, the PLE mode, the load time, and peak host RAM held by the driver from the watchdog. If a stage reports experts "neither in VRAM nor mirrored", stop: lower the reserve or move the split, do not run with `NO_HOST` and missing experts.

### 7. Correctness

1. Three probes through `/v1/chat/completions` at temperature 0 with thinking off: "What is 2+2?", a Fibonacci function (execute it), and the 155-minute trip question from [tools/fnbench.py](../../../tools/fnbench.py).
2. Eight varied prompts, 256 tokens each, greedy: no `!!!!` runs, token soup, NaN or repetition loops. The UD-IQ4_XS collapse in #1484 appeared after 20 to 40 tokens.
3. The same eight prompts with speculative decoding off (no `--mtp`, `--suffix-draft 0`) must give identical tokens.
4. Send one 10K prompt twice and compare the outputs.
5. For arm 3 only: the fnbench short prompt against the llama.cpp production server on the same GGUF, greedy. The engines round differently, so expect the same answer and a possible late near-tie flip; a divergence in the first tokens is a defect.

### 8. Measurements

Same points as the production benchmark, read from the server's own `timings` object (Strata reports `prompt_per_second` and `predicted_per_second` in llama.cpp's form) and from its `/metrics` for draft acceptance. Warm up with three requests first.

- `BASE=http://127.0.0.1:8085 VLLM_API_KEY=... python3 tools/fnbench.py` for the short, 9,749- and 39,119-token points. fnbench sends llama.cpp's `cache_prompt` and `ignore_eos` fields; if Strata rejects them, run an adapted copy from the scratch directory and say so in the benchmark file.
- `LONG_TOKENS=200000 python3 tools/fnbench.py` for the 219K point, only after a 64K prompt has completed cleanly.
- [tools/lcbench.py](../../../tools/lcbench.py) for the 135K agent session: median decode, acceptance, follow-up turn latency.
- Peak VRAM per card and peak driver-held host RAM during the 219K run.
- Then the A/Bs, one at a time, interleaved with the arm-1 config: `STRATA_EXPERT_SPLIT` unset; `--layer-split auto`; `--ple-io direct`; `--spec 6`; `--kv fp16`; the clock floor off.
- Last, concurrency: `--batch 4` (and on main `--batch-groups auto`) at c4 with the standard greedy checks; `--batch 8` if c4 holds.

Save raw output under [benchmarks/raw/](../benchmarks/raw/) as `2026-MM-DD-strata-*.csv` and write the benchmark file from [templates/benchmark.md](../../../templates/benchmark.md).

### Abort criteria

- `MemAvailable` under 20 GiB, or driver-held host RAM still growing after the expert fill has finished.
- Any `xe` GT reset or engine reset in `dmesg`.
- The server reporting "still starting" past 10 minutes after the fill lines stop.
- Corrupt output in step 7: stop measuring, keep the log, try `STRATA_EXPERT_SPLIT` the other way once, then move to the fallback arm.

## Results

Not yet run. Fill in per arm.

| Metric | Baseline (llama.cpp production) | Arm 1: Strata PR #1111, UD-IQ4_XS | Arm 3: Strata, abliterated UD-Q4_K_XL | Delta |
|---|---|---|---|---|
| Single-stream decode, short | 48.6 to 50.5 tok/s | | | |
| Single-stream decode, 9,749-token prompt | 60.6 to 67.2 tok/s | | | |
| Single-stream decode, 39,119-token prompt | 53.6 to 55.2 tok/s | | | |
| Single-stream decode, 135K agent session (median) | about 39 tok/s | | | |
| Single-stream decode, 219K | 40.6 tok/s | | | |
| Prompt processing, 9,749 / 39,119 | 589.8 / 577.2 tok/s | | | |
| Prompt processing, 219K cold | 268.3 tok/s (816.9 s) | | | |
| Follow-up turn, about 60 tokens on a cached 135K prefix | 1.04 to 1.32 s | | | |
| Aggregate at c4 / c8 | n/a (single slot) | | | |
| Spec-decode acceptance | 62.5 to 82.7% at 135K | | | |
| Peak VRAM, busiest card, 219K | 28,957 MiB | | | |
| Peak driver-held host RAM | 16.8 to 22 GiB (ubatch runs) | | | |
| Load time | about 2.5 min (111 GB) | | | |

## Correctness

To be filled from step 7: identical tokens with and without speculative decoding (yes/no), the eight-prompt scan, the repeated-prompt check, and for arm 3 the comparison with llama.cpp on the same file.

## Decision

Pending. Rules set in advance:

- `kept` only if greedy output is identical with speculative decoding on and off, the eight-prompt scan is clean, and decode or prompt speed beats production at the fnbench points without losing the 262K context. Even then production keeps llama.cpp until the vision question is settled; Strata would be a second, text-only configuration.
- `inconclusive` if arm 1 runs but only the four-card split misbehaves and the two-card fallback matches the community numbers: report upstream and re-test after the next release.
- `blocked` if the pack or the engine refuses our files on Intel.

## Follow-ups

- [ ] Report the four-card result in [#1440](https://github.com/Niko1221/Strata/issues/1440) or [#1054](https://github.com/Niko1221/Strata/issues/1054), with hostnames, paths and keys stripped.
- [ ] If arm 1 is correct and fast, write a `configs/strata-*.json` with placeholders and a benchmark file; update the model card only if a headline number changes.
- [ ] If arm 3 works, measure the refusal smoke probe and the teacher-forced KL check against llama.cpp on the same file.
- [ ] Re-run arm 2 when main carries a fix for the split crash.
