# Decode at 133K-136K: where a verify step's time goes, from GPU timestamps

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Config** | [llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh) production flags with the draft at n-max 3, no cut (3 drafted tokens every step), GPU clock floor 2800 MHz |
| **Experiment** | diagnostics for the decode lever; related: [draft settings under the production sampler](../experiments/2026-09-26-mtp-draft-settings-sampled-replay.md) |
| **Raw data** | [raw/2026-09-26-decode-gpu-busy-95k.csv](raw/2026-09-26-decode-gpu-busy-95k.csv), [raw/2026-09-26-decode-devprof-per-op-133k.csv](raw/2026-09-26-decode-devprof-per-op-133k.csv), [raw/2026-09-26-decode-devprof-coarse-135k.csv](raw/2026-09-26-decode-devprof-coarse-135k.csv), [raw/2026-09-26-decode-driver-runs.csv](raw/2026-09-26-decode-driver-runs.csv) |

## Result

At 135K a decode step is about 63 ms: a 4-token target verify plus 3 draft-head steps. Of that, the GPUs are busy for about 52 ms one card after another: 11.0-11.5 ms each on cards 0-2, and 18.2 ms on card 3, which also runs the draft head. The remaining **~10.6 ms per step is outside any GPU graph**: host work between the graphs.

The host submits each card's graph in about 7 ms, faster than the card executes it, so kernel launch is not what limits the step. At 95K the four cards' compute engines were busy 100-112% summed. With a layer split one card works at a time, so the chain of cards is never idle for long. On 2026-09-24, at short context, the sum was about 65%.

Two consequences:
- Faster decode now needs less GPU time per layer, or a shorter host gap between the cards' graphs, not faster kernel submission.
- Moving the target's sampling onto the GPU (`--backend-sampling`) did not shorten the gap, and it slowed short prompts from 0.27 s to 1.45 s.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container: `2be97a7d9` (passive sample), `556c342cb` (per-op profile, patch 0020), `8f95cb5d2` (coarse spans, patch 0023) and `5c258f538` (backend-sampling A/B); all are the production tree `2b84213a4` plus test hooks, diagnostics and the [prompt-cache swap fix](../experiments/2026-09-26-prompt-cache-swap-speedup.md), none of which touch the decode kernels |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0; oneAPI 2026.1 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| KV cache dtype / block size | f16 K and V, 262,144 cells |
| Speculative decoding | MTP draft on `SYCL3`, n-max 3, no p-min, draft QSA (Qwen sparse attention) on |
| Graph / compile mode | eager (SYCL graphs refused with more than one device); `GGML_SYCL_SPARSE_FA=1`; pooled QSA key cache on |

## Workload

| | |
|---|---|
| Workload set | the saved real agent conversation (135,226 tokens; private, not in this repository), restored into the server. Driven by [`tools/specreplay.py`](../../../tools/specreplay.py) (reply to one of the conversation's own turns) or by [`tools/devdrive.py`](../../../tools/devdrive.py) (a new assistant turn at the end, `ignore_eos`) |
| Input length | 94.8K-97.4K (passive sample), 132.6K-134.2K (per-op), 135.2K-136.8K (coarse, backend sampling) |
| Output length | 512-768 tokens per request |
| Sampling | server sampler: temperature 1.0, top-k 20, top-p 0.95, min-p 0.05 |
| Warm-up | one request per server before the measured ones; first requests after a restore are slower (pooled-key rebuild) and are marked |
| Repetitions | 3-4 requests per setting |

## Results

### Engine busy at about 95K, no profiler

Three 10 s windows of [`tools/gpupassive.py`](../../../tools/gpupassive.py) (xe fdinfo cycles, no GPU access of its own) during the draft-settings replay ([CSV](raw/2026-09-26-decode-gpu-busy-95k.csv)):

| Window | Card 0 | Card 1 | Card 2 | Card 3 | Summed | llama-server CPU (one core = 100%) |
|---|---|---|---|---|---|---|
| 0 | 22% | 22% | 22% | 43% | 110% | 94% |
| 1 | 23% | 22% | 22% | 45% | 112% | 97% |
| 2 | 22% | 22% | 22% | 35% | 100% | 95% |

One thread does almost all the CPU work. Card 3's extra share is the draft head plus the LM head.

### One span per graph at 135K (coarse mode)

With a barrier before each graph's first op and after its last (`GGML_SYCL_OP_DEVPROF=20`, file content `coarse`), decode ran 62.4-63.0 ms per step. That is the same as with the profiler off (61.3-67.2 ms per step on the same kind of requests, [driver CSV](raw/2026-09-26-decode-driver-runs.csv)). Over the 311 verify steps in the printed block ([coarse CSV](raw/2026-09-26-decode-devprof-coarse-135k.csv)):

| Card | GPU span per step | Host submit per step | Graphs |
|---|---|---|---|
| 0 | 11.39 ms | 6.93 ms | 311 (target) |
| 1 | 11.03 ms | 6.96 ms | 311 |
| 2 | 11.48 ms | 6.94 ms | 310 |
| 3 | 18.19 ms | 7.86 ms | 1,541: 311 target, 311 small (under 200 nodes, 0.54 ms each) and about 3 draft-head graphs per step |
| Sum | 52.1 ms | | step time 62.7 ms, so about 10.6 ms per step is outside the GPU graphs |

Cards 0-2 spend 0.85-0.95 ms per layer (12, 13 and 13 layers of the `-ts 12,13,13,11` split). Card 3's 11 target layers plus the target LM head (about 1.2 ms) come to about 10.9 ms (estimate). That leaves roughly 7.3 ms for the small graph and the draft head's three steps, about 2.3 ms per draft step (estimate).

### Per-op profile at 133K (host-paced; counts only)

The per-op mode submits a barrier before every op. It doubled the step time: 24.7 tok/s against about 46 unprofiled on the same turns in the [draft-settings replay](../experiments/2026-09-26-mtp-draft-settings-sampled-replay.md). Host submission (about 18 ms per card graph) then exceeded the device time (about 15.7 ms), so most op times measure the host's pace, not the GPU. What it does show ([per-op CSV](raw/2026-09-26-decode-devprof-per-op-133k.csv); the printed blocks cover 612 of the 754 verify steps, since card 0 ran 612 graphs, and the counts below are per covered step):

- At least 1,894 ops per verify step across the four cards, target plus three draft-head steps. That is the count in the listed rows; each block lists only its top 60 op keys. That includes 153 MoE matmuls per step (gate, up and down in 48 layers, plus 3 per draft step) and about 584 other matmuls: hyper-connection mixing, linear-attention projections, gates.
- The LM head (`result_output`, 2,433 calls, 4 per step: the target plus 3 draft steps) averaged 1.2 ms per call, the largest single op, on card 3.
- Sparse flash attention: 15 calls per step (12 QSA layers plus 3 draft steps), about 0.32 ms each.
- MoE matmuls averaged 97-155 µs per call.

### Target sampling on the GPU (`--backend-sampling`, experimental)

Same server build, same restored conversation, 4 requests of 512 tokens each ([driver CSV](raw/2026-09-26-decode-driver-runs.csv), parts E1 and E2):

| | ms per verify step | Acceptance | 6-token prompt |
|---|---|---|---|
| Host sampling (E1) | 66.8-67.6 | 49.3% | 0.26-0.27 s |
| `--backend-sampling` (E2) | 63.1-67.3 | 76.2% | 1.45-1.85 s |
| Host sampling, repetitive text of the same kind (coarse run) | 62.4-63.0 | 78.7% | 0.15-0.27 s |

Step times track the text: repetitive text routes the 4 verify tokens to fewer distinct experts, and every step drafts 3 tokens here. Against host sampling on text of the same kind, backend sampling saved nothing measurable, and it made short prompts 5 times slower. Not adopted.

## Observations

- **The ~10.6 ms gap per step** is the next thing to measure: host work between the cards' graphs (inter-card copies and waits, output reads, draft bookkeeping, recurrent-state snapshots for rollback). A host timeline of one step (`LLAMA_HOST_PROF` on the current build) would split it.
- **The draft's LM head** costs about 1.2 ms per draft step at full vocabulary (248K) to pick a top-10. A reduced-vocabulary draft head would cut most of that, with some loss of acceptance on rare tokens (UNVERIFIED; not built).
- **The first request after a slot restore** runs slower (67-81 ms per step in its first 512 tokens), while the pooled QSA key rows are rebuilt.
