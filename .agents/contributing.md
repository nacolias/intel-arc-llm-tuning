# Contributing

Read the root [`README.md`](../README.md) first. It has the folder layout, the workflow and the naming conventions. This file adds the rules agents most often get wrong.

## Where things go

| You are adding | Location | Template |
|---|---|---|
| a new model | `models/<model>/`, copied from `models/_template/` | model card in `_template/README.md` |
| notes on something you read | `models/<model>/research/YYYY-MM-DD-<slug>.md` | [`templates/research-note.md`](../templates/research-note.md) |
| a change attempt | `models/<model>/experiments/YYYY-MM-DD-<slug>.md` | [`templates/experiment.md`](../templates/experiment.md) |
| a measured result | `models/<model>/benchmarks/YYYY-MM-DD-<slug>.md`, raw output in `benchmarks/raw/` | [`templates/benchmark.md`](../templates/benchmark.md) |
| a launch config or patch | `models/<model>/configs/`, patches in `configs/patches/` | none |
| a lesson beyond one model | `findings/<slug>.md` | [`templates/finding.md`](../templates/finding.md) |
| a host | `hardware/hosts/<hardware-label>.md` | [`templates/host.md`](../templates/host.md) |
| a driver or runtime version combination | the version matrix in `platform/README.md` | none |
| an engine capability or patch status | `engines/<engine>/README.md` | none |

## Indexes to update

Every folder with entries has an index table in its `README.md`. When you add an entry, add a row.

- New experiment: `models/<model>/experiments/README.md`.
- New benchmark: `models/<model>/benchmarks/README.md`.
- New research note: `models/<model>/research/README.md`.
- New finding: `findings/README.md`.
- New host: `hardware/README.md`.
- New model: `models/README.md` and the models table in the root `README.md`.
- Headline number changed: the model card, `models/README.md`, and the cross-model summary in `benchmarks/README.md`.

## Evidence rules

- **Every number needs a source.** Link the benchmark file, or the external source for community numbers.
- **Mark what you did not measure.** Write `UNVERIFIED` next to any claim not measured on our hardware or not confirmed in source code.
- **Use the confidence labels** on findings: `verified-here`, `reproduced-community`, `upstream-documented`, `unverified`.
- **Say which workload.** Speculative decoding speed varies a lot between prose and code. A number without its workload class, context length and concurrency is incomplete.
- **Correctness comes with speed.** An experiment that changes kernels, graphs, collectives, speculative decoding or caching must report the greedy correctness check from [`benchmarks/README.md`](../benchmarks/README.md).
- **Do not guess versions.** Read them from the running system, the image or the source. Write `TODO` if unknown.

## Editing rules

- Add new dated files rather than rewriting old results. If an old entry was wrong, add a dated correction note to it and link the new entry.
- Keep upstream references precise: repository, PR or issue number, commit hash, tag.
- Do not commit model weights, large binaries or large logs. Raw benchmark output should be small JSON or CSV. Link to large artifacts instead.
- Copied third-party code or text must be license-compatible. Name the source and license.
- Mark statements about "latest" versions with the date you checked.

## Writing style

- Lead with the result. Put the conclusion in the first sentence of a section.
- Short sentences, one idea each.
- Tables for numbers and comparisons; bullets for lists; prose for reasoning.
- Expand uncommon acronyms the first time they appear in a file.
- Commands and config go in fenced code blocks.
- Dates are ISO `YYYY-MM-DD`. Throughput is output tokens per second.
