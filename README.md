# intel-arc-llm-tuning

Research, experiments, benchmarks and findings from running large language models on the **Intel Arc Pro** platform (Battlemage / Xe2, starting with the Arc Pro B70 32 GB).

The goal is a reproducible record: what was tried, on which exact stack, what it measured, and what we concluded. A number without its versions, flags and workload is not useful here, so every entry records them.

## Repository layout

```
.
├── README.md                 you are here
├── AGENTS.md                 pointer to .agents/ for AI agents
├── .agents/                  instructions, privacy rules and secret scanner for contributors
├── hardware/                 GPUs and host machines (specs, topology, BIOS, power)
│   └── hosts/                one file per physical box
├── platform/                 kernel driver, firmware, compute-runtime, Level Zero, oneCCL, PyTorch XPU
├── engines/                  inference engines and their XPU-specific notes
│   ├── vllm-xpu/
│   ├── llm-scaler/
│   ├── llama-cpp-sycl/
│   └── openvino/
├── models/                   one folder per model; the core of the repo
│   ├── README.md             model index and status board
│   ├── _template/            copy this to start a new model
│   ├── qwen3.8-27b/
│   │   ├── README.md         model card: checkpoints, best config, headline numbers
│   │   ├── research/         reading notes, upstream issues, external reports
│   │   ├── experiments/      change attempts, one file per attempt
│   │   ├── benchmarks/       measured results (raw/ holds JSON/CSV output)
│   │   └── configs/          compose files, launch scripts, patches/
│   └── qwen3.8-flash-next/   same layout
├── benchmarks/               shared methodology, workloads, cross-model summary
│   └── workloads/            standard prompt sets
├── findings/                 cross-cutting, model-independent learnings
├── templates/                entry templates (experiment, benchmark, finding, research note, host)
└── tools/                    probes and scripts usable across models
```

**Where does a note go?**

| You have... | Put it in |
|---|---|
| something you read (a paper, a PR, a forum post, a vendor doc) | `models/<model>/research/`, or the relevant `engines/` or `platform/` README if it is model-independent |
| a change you tried (flag, patch, driver, image, quant) | `models/<model>/experiments/YYYY-MM-DD-<slug>.md` |
| a measured result | `models/<model>/benchmarks/YYYY-MM-DD-<slug>.md`, raw output in `benchmarks/raw/` |
| a launch config worth keeping | `models/<model>/configs/` |
| a lesson that applies beyond one model | `findings/<slug>.md` |
| a driver, firmware or runtime version change | the version matrix in `platform/README.md` |

## Workflow

1. **Research.** Capture the source and the claims it makes in a research note. Mark each claim `UNVERIFIED` until you measure it.
2. **Experiment.** Write the hypothesis and the exact change before running it. Link the baseline benchmark you are comparing against.
3. **Benchmark.** Follow [`benchmarks/README.md`](benchmarks/README.md): same workload, warm-up and concurrency levels as the baseline, with power cap and versions recorded.
4. **Decide.** Mark the experiment `kept`, `reverted` or `inconclusive`. Kept changes update the model card's best config and headline numbers.
5. **Generalize.** If the lesson holds beyond this model, write a finding and link it from the experiment.

## Conventions

- **Dates** are ISO `YYYY-MM-DD`. Entry file names are `YYYY-MM-DD-short-slug.md`.
- **Model folder names** are lowercase and match the upstream family and size, for example `qwen3.8-27b`. Quantized checkpoints are variants inside one model folder, not separate folders.
- **Experiment status:** `planned`, `running`, `kept`, `reverted`, `inconclusive`, `blocked`.
- **Finding confidence:** `verified-here` (measured on our hardware), `reproduced-community` (someone else measured it on the same GPU), `upstream-documented` (source code or issue tracker), `unverified`.
- **Throughput** is output tokens per second. Say whether it is single-stream or aggregate, and at what concurrency and context length.
- **Always record** image tag or digest, engine version, compute-runtime version, kernel version, power cap per card, and the full launch flags. The templates have a table for this.
- **Correctness before speed.** A change that alters greedy output is a regression until proven otherwise. See the correctness section of [`benchmarks/README.md`](benchmarks/README.md).

## Keep private information out

This repository is public. Never commit IP addresses, hostnames, domains, API keys, tokens, passwords, secrets, `.env` files or their values, usernames, email addresses or hardware serial numbers. Use placeholders such as `<your-host>` and `${VLLM_API_KEY}`, and name hosts by their hardware, for example `dual-b70-5800x`.

Turn on the pre-commit scanner once per clone, and read [`.agents/privacy.md`](.agents/privacy.md) for the full rules:

```bash
git config core.hooksPath .agents/hooks
python3 .agents/scan-private-info.py
```

AI agents contributing to this repository must follow [`.agents/README.md`](.agents/README.md).

## Models

See [`models/README.md`](models/README.md) for the full status board.

| Model | Best single-stream | Status |
|---|---|---|
| [Qwen3.8-27B](models/qwen3.8-27b/) | 78 to 84 tok/s (dual B70, TP2, MTP2) | active tuning |
| [Qwen3.8-Flash-Next](models/qwen3.8-flash-next/) | 48.6 to 50.5 tok/s short, about 39 tok/s at 135K context (quad B70, llama.cpp SYCL layer split, MTP) | production |

## Adding a new model

```bash
cp -r models/_template models/<model-name>
```

Then fill in the model card, add a row to `models/README.md`, and record a baseline benchmark before changing anything.
