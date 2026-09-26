# Models

One folder per model family and size. Quantized checkpoints, fine-tunes and draft models are variants inside that folder.

## Status board

| Model | Architecture | Best config | Single-stream | Aggregate (c16) | Status | Last updated |
|---|---|---|---|---|---|---|
| [Qwen3.8-27B](qwen3.8-27b/) | hybrid Gated DeltaNet + attention, 27B dense, native MTP | GPTQ INT4 G128, TP2, MTP2, XPU graphs, FP8 KV | 78 to 84 tok/s | 405 to 442 tok/s | active tuning | 2026-09-12 |

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
