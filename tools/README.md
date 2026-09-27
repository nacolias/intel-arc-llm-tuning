# Tools

Probes and scripts that work across models. Model-specific launch scripts live in `models/<model>/configs/`.

| Tool | What it measures |
|---|---|
| [`xccl_allreduce_census.py`](xccl_allreduce_census.py) | two-rank XCCL allreduce exactness (correctly rounded fp16 sum, row-count invariance, repeat equality) and latency at 2, 32, 64 and 900 rows of `[rows, 5120]` fp16 |
| [`host_submission_latency_probe.py`](host_submission_latency_probe.py) | per-launch host cost on one card (async launch and launch plus sync, in microseconds); tells you the host latency class |
| [`fnbench.py`](fnbench.py) | llama-server quick bench: a sanity question with a known answer, median decode tok/s over 5 short runs, prefill and decode at about 10k and 39k prompt tokens, and optionally one very long prompt; MTP draft acceptance when reported |
| [`lcbench.py`](lcbench.py) | llama-server agent-like session: one cold read of a fixed ~135K-token C++ context, then 6 short turns on the cached prefix; per-turn prompt latency, decode tok/s and MTP acceptance, optional JSON of the outputs for parity checks |
| [`klprobe.py`](klprobe.py) | teacher-forced next-token check after the same long context: top-20 log-probabilities at 128 positions; `--compare` prints top-1 agreement and KL divergence between two runs |
| [`accprobe.py`](accprobe.py) | greedy MTP (multi-token prediction) draft acceptance, step time and decode tok/s over 12 turns on top of the final prompt of an `lcbench.py` run; two builds that differ only in the draft head produce nearly the same text, so their acceptance compares like for like |
| [`pairprobe.py`](pairprobe.py) | paired A/B of a runtime flag file on one server: every turn is decoded twice, with the feature off and on, on the same context; resolves acceptance changes of about 2 points that two separate sessions cannot (run-to-run noise is about 3 points on this workload); prints the number of turns with identical greedy outputs |
| [`lcsum.py`](lcsum.py) | one line per `lcbench.py` or `accprobe.py` JSON: turns, MTP acceptance, milliseconds per MTP step, decode tok/s overall and median prompt latency, the cold read left out |
| [`gpuprof.py`](gpuprof.py) | while llama-server decodes one request: per-GPU engine busy % from `xe` fdinfo, actual GT clock, llama-server CPU use and busiest threads; shows whether a layer split is host-bound |
| [`gpu-clock-floor.sh`](gpu-clock-floor.sh) | pins (or restores) the minimum GT0 clock of every Intel GPU to its hardware maximum; for llama.cpp layer split, see [findings/gpu-clock-floor-speeds-layer-split.md](../findings/gpu-clock-floor-speeds-layer-split.md) |
| [`bmg-aspm-l1.sh`](bmg-aspm-l1.sh) | enables PCIe ASPM L1 on the Arc Pro B70 card links only (switch uplink and other links left off) and checks the result; `status` prints per-card board power; see [findings/bmg-aspm-l1-idle-power.md](../findings/bmg-aspm-l1-idle-power.md) |
| [`bmg-aspm-l1.service`](bmg-aspm-l1.service) | systemd unit that runs `bmg-aspm-l1.sh apply` at boot and `revert` on stop |

Run the census inside the engine container with both cards visible, once with the default protocol and once with `CCL_SYCL_ALLREDUCE_LL=twoshots`:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 xccl_allreduce_census.py /tmp/ar.json
```

Run it again after any oneCCL, compute-runtime or graph-mode change.

Both probes are copied verbatim from `steveseguin/b70-optimization-lab`, which is public domain (Unlicense).

## llama-server benchmarks and probes

`fnbench.py`, `lcbench.py`, `klprobe.py`, `accprobe.py`, `pairprobe.py`, `lcsum.py` and `gpuprof.py` are our own (MIT) and need only the Python standard library. They talk to llama-server's native endpoints. Set `BASE` (default `http://127.0.0.1:8080`) and, if the server was started with `--api-key`, `VLLM_API_KEY`. All requests are greedy (temperature 0) with `ignore_eos`.

```bash
export BASE=http://127.0.0.1:8080
export VLLM_API_KEY="$(cat /path/to/api-key-file)"          # only if the server needs a key
python3 fnbench.py                                           # sanity, short, ~10k, ~39k
LONG_TOKENS=200000 python3 fnbench.py                        # adds a ~219K-token prompt
LC_CORPUS=lc-corpus.txt LC_OUT=a.json python3 lcbench.py     # 135K-token agent session
KL_OUT=a.json python3 klprobe.py                             # run on build A, then on build B
python3 klprobe.py --compare a.json b.json
sudo -E python3 gpuprof.py 512                               # decode profile, 512 tokens
```

Paired A/B of a runtime switch, and the acceptance probe. Both start from the final prompt of an `lcbench.py` run, so save it first; `pairprobe.py` creates the flag file in the server's own `/tmp`, so run it on the server host. The flag files each patch honours are listed in the [patches README](../models/qwen3.8-flash-next/configs/patches/README.md#runtime-switches).

```bash
LC_SAVE_PROMPT=lc-turnprof-prompt.json LC_TURNS=1 python3 lcbench.py       # cold read, then save the context prompt
PAIR_FLAG=/tmp/llama-mtp-qsa-off PAIR_TURNS=12 python3 pairprobe.py        # each turn decoded with draft QSA off, then on
ACC_OUT=acc-A.json python3 accprobe.py                                     # 12 greedy turns on build A; repeat as acc-B.json on build B
python3 lcsum.py a.json b.json acc-A.json acc-B.json                       # acceptance, ms/step, decode tok/s, prompt median per file
```

The draft QSA A/B in [the experiment](../models/qwen3.8-flash-next/experiments/2026-09-25-mtp-draft-qsa.md) and the `eh_proj` acceptance check in [that experiment](../models/qwen3.8-flash-next/experiments/2026-09-26-mtp-eh-proj-2d-product.md) were run this way; the per-turn numbers of every session are in [the raw CSV](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-lcbench-sessions-per-turn.csv).

`lcbench.py`, `accprobe.py`, `pairprobe.py` and `klprobe.py` need a corpus file. Its docstring shows how to rebuild ours byte for byte from the public llama.cpp tree at commit `6fcaa16` (PR #28243). Use the same corpus and `LC_TOKENS` for every build you compare.

## Host scripts

Both need root and change sysfs state. Try them by hand before installing them as services.

```bash
sudo ./gpu-clock-floor.sh pin        # min GT clock = hardware max while serving
sudo ./gpu-clock-floor.sh restore    # back to the driver default
sudo ./bmg-aspm-l1.sh apply          # L1 on the B70 links, policy powersave
sudo ./bmg-aspm-l1.sh status         # policy, links with L1 on, board power per card
sudo ./bmg-aspm-l1.sh revert         # policy default
sudo install -m 755 bmg-aspm-l1.sh /usr/local/sbin/ && sudo install -m 644 bmg-aspm-l1.service /etc/systemd/system/ \
  && sudo systemctl enable --now bmg-aspm-l1.service
```

`gpu-clock-floor.sh` fits `ExecStartPre=+` / `ExecStopPost=+` lines in the llama-server unit; see its header.
