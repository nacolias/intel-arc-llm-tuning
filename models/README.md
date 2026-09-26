# Models

One folder per model family and size. Quantized checkpoints, fine-tunes and draft models are variants inside that folder.

## Status board

| Model | Architecture | Best config | Single-stream | Aggregate (c16) | Status | Last updated |
|---|---|---|---|---|---|---|
| [Qwen3.8-27B](qwen3.8-27b/) | hybrid Gated DeltaNet + attention, 27B dense, native MTP | GPTQ INT4 G128, TP2, MTP2, XPU graphs, FP8 KV | 78 to 84 tok/s | 405 to 442 tok/s | active tuning | 2026-09-12 |
| [Qwen3.8-Flash-Next](qwen3.8-flash-next/) | MoE, about 125B plus a 51B per-layer n-gram table (PLE), about 6B active; 48 layers: 12 QSA sparse attention + 36 Gated DeltaNet; native MTP | `UD-Q4_K_XL` (abliterated), llama.cpp SYCL layer split over 4 cards (build `20260926-2b84213a4`, patches 0002-0017), 262K context, MTP draft 3 with draft QSA, sparse FA, pooled QSA key cache, grouped MoE matmul on the XMX (Xe Matrix Extensions) units for prompts, slot save/restore and `--cache-ram 24576`, GPU clock floor | 43.7 to 48.7 tok/s at 135K (greedy, code); cold 135K read about 284 s, ~60-token follow-up turns 0.76-0.77 s ([benchmark](qwen3.8-flash-next/benchmarks/2026-09-26-cold-read-135k-by-build.md)); 48.6 to 50.5 tok/s short on build `20260925-f47a6a5f3` ([benchmark](qwen3.8-flash-next/benchmarks/2026-09-25-production-build.md)) | n/a (single slot) | production | 2026-09-26 |

Status values: `candidate` (not tried yet), `baseline` (runs, untuned), `active tuning`, `production`, `parked`, `abandoned`.

## Starting a new model

```bash
cp -r models/_template models/<model-name>
```

1. Fill in the model card (`README.md`) with architecture and candidate checkpoints.
2. Get it running with the simplest working config and save that config in `configs/`.
3. Record a baseline benchmark before tuning anything.
4. Add a row to the status board above.

## Folder layout

| Path | Contents |
|---|---|
| `README.md` | model card: architecture, checkpoints, current best config, headline numbers, open questions |
| `research/` | reading notes, upstream issues, external reports, long-form analysis |
| `experiments/` | one file per change attempt, plus an index in `experiments/README.md` |
| `benchmarks/` | one file per measured run, plus an index; `raw/` holds JSON and CSV output |
| `configs/` | compose files and launch scripts; `patches/` holds engine patches applied at startup |
