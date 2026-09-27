#!/usr/bin/env bash
# Qwen3.8-Flash-Next on four Arc Pro B70 with llama.cpp SYCL: 262,144-token context, MTP draft head,
# layer split, sparse flash attention. This is the production launch of host quad-b70-5800x-pex88096,
# with host-specific paths, names and the API key source replaced by placeholders.
#
# Fill in (environment variables; defaults in brackets):
#   BIN           llama.cpp build dir holding llama-server, built with configs/patches/  [/path/to/llama.cpp/build/bin]
#   MODEL         first shard of the trunk GGUF                                          [/path/to/models/...-00001-of-00004.gguf]
#   MTP           MTP draft GGUF, or "off"                                               [/path/to/models/.../mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf]
#   MMPROJ        vision projector GGUF, or empty to serve text only                     [/path/to/models/.../mmproj-model-bf16.gguf]
#   API_KEY_FILE  file that holds the API key (mode 0600); required. Never pass the key itself on the command line.
#   HOST, PORT    bind address and port                                                  [127.0.0.1, 8080]
#   SLOT_DIR      directory for saving the prompt-cache slot across restarts (0700; see
#                 llama-server-slot-cache.sh); empty disables --slot-save-path              [empty]
#   CACHE_RAM     host-RAM prompt cache in MiB for displaced conversations (--cache-ram);
#                 empty keeps llama-server's default                                      [empty]
#   SIGNATURE     file to write the setup signature to, for llama-server-slot-cache.sh; empty skips it [empty]
#   ONEAPI_VARS   oneAPI environment script                                              [/opt/intel/oneapi/2026.1/oneapi-vars.sh]
#   CTX, UB, BATCH, TS, EXTRA_ARGS: see below.
#
# Measured with these defaults on build 20260925-f47a6a5f3, GPU clock floor 2800 MHz (tools/gpu-clock-floor.sh),
# 230 W power cap: decode 48.6 tok/s short, 60.6 at ~10k, 53.6 at ~39k, 40.6 at 219k; prompt ~580 tok/s at
# 10k-39k and 268 at 219k; busiest card 28,957 MiB peak. See benchmarks/2026-09-25-production-build.md.
set -euo pipefail

BIN=${BIN:-/path/to/llama.cpp/build/bin}
MODEL=${MODEL:-/path/to/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
MTP=${MTP:-/path/to/models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
MMPROJ=${MMPROJ-/path/to/models/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/mmproj-model-bf16.gguf}
ALIAS=${ALIAS:-Qwen3.8-Flash-Next}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
: "${API_KEY_FILE:?set API_KEY_FILE to a 0600 file that holds the API key}"
ONEAPI_VARS=${ONEAPI_VARS:-/opt/intel/oneapi/2026.1/oneapi-vars.sh}
SLOT_DIR=${SLOT_DIR:-}
CACHE_RAM=${CACHE_RAM:-}
SIGNATURE=${SIGNATURE:-}

CTX=${CTX:-262144}
# -ub 1536 against 1024 (measured, MTP on): prompt +4.5% at 39k and +6% at 219k, 28.2 against 27.8 GiB on the
# fullest card, 22 against 16 GiB of driver-held host RAM. -ub 2048 roughly doubles the QSA indexer's compute
# buffer (about 7 GiB per card at 262k) and does not fit with MTP.
UB=${UB:-1536}
BATCH=${BATCH:-2048}

# The memory fitter cannot size the MTP draft, so with MTP the split is explicit (-fit off). Whole numbers summing
# to 49 give exact layer counts (48 layers plus the output layer). The last card gets fewer layers because it also
# holds the draft (-devd SYCL3); the first gives up one for the -ub 1536 compute buffers and the vision projector.
if [ "$MTP" = off ]; then
  TS=${TS:-12,12,12,13}
  SPEC=()
else
  TS=${TS:-12,13,13,11}
  SPEC=(-fit off --spec-type draft-mtp --spec-draft-model "$MTP" --spec-draft-n-max 3 -devd SYCL3)
fi

VISION=()
[ -n "$MMPROJ" ] && VISION=(--mmproj "$MMPROJ")

# Extra llama-server flags, split on whitespace.
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"

# Prompt cache across restarts (see configs/README.md, "Prompt cache across restarts"): the unit's ExecStop saves
# slot 0 to SLOT_DIR and ExecStartPost restores it, but only into an identical setup. The signature written here
# is what llama-server-slot-cache.sh compares. --cache-ram keeps a displaced conversation in host RAM instead of
# re-reading it when a second client takes the single slot (about 4 GB of state for a 135K conversation with the draft;
# see configs/README.md, "Prompt cache across restarts").
SLOTS=()
if [ -n "$SLOT_DIR" ]; then
  mkdir -p "$SLOT_DIR" && chmod 700 "$SLOT_DIR"
  SLOTS+=(--slot-save-path "$SLOT_DIR")
fi
[ -n "$CACHE_RAM" ] && SLOTS+=(--cache-ram "$CACHE_RAM")
if [ -n "$SIGNATURE" ]; then
  # kv= is what this script passes; EXTRA_ARGS can override it (a later -ctk/-ctv wins), so it is part of the signature too
  printf '%s\n' "model=$MODEL" "mtp=$MTP" "bin=$BIN" "ctx=$CTX" "kv=f16/f16" "np=1" \
      "mtp_qsa=${LLAMA_MTP_QSA-}" "mtp_window=${LLAMA_MTP_WINDOW-}" "extra=${EXTRA_ARGS:-}" > "$SIGNATURE"
fi

set +eu
# shellcheck disable=SC1090
source "$ONEAPI_VARS" >/dev/null 2>&1
set -eu

export ZES_ENABLE_SYSMAN=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
# Sparse flash attention for the 12 QSA layers: attend the ~2,051 cells the indexer selects instead of scanning
# the whole cache. Needs upstream #28796; configs/patches/0006 extends it to MTP verify batches.
export GGML_SYCL_SPARSE_FA=${GGML_SYCL_SPARSE_FA:-1}
export LD_LIBRARY_PATH=$BIN:${LD_LIBRARY_PATH:-}
unset ZE_AFFINITY_MASK ONEAPI_DEVICE_SELECTOR

# -ot per_layer_token_embd=CPU pins the n-gram (PLE) table to CPU memory. llama.cpp already places this lazily read
# tensor in CPU memory, so the flag is redundant; it also disables multi-GPU pipeline parallelism, which gained
# nothing here (experiments/2026-09-25-pipeline-parallel-prefill.md). It is kept to match the measured config.
exec "$BIN/llama-server" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  -dev SYCL0,SYCL1,SYCL2,SYCL3 -sm layer -ngl 99 -ts "$TS" \
  -ot 'per_layer_token_embd=CPU' \
  -c "$CTX" -np 1 -fa on -ctk f16 -ctv f16 -b "$BATCH" -ub "$UB" -t 8 --jinja \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  --host "$HOST" --port "$PORT" \
  --api-key-file "$API_KEY_FILE" --no-webui \
  "${SLOTS[@]}" "${SPEC[@]}" "${VISION[@]}" "${EXTRA[@]}"
