# Workloads

Standard prompt sets used by every benchmark. Keep prompts fixed once published so results stay comparable; add a new version instead of editing one.

| Set | Class | Input length | Output length | Purpose |
|---|---|---|---|---|
| `prose-v1` | prose | TODO | 200 | general chat decode speed |
| `code-v1` | Python code | TODO | 200 | high speculative acceptance case |
| `docs-v1` | structured documents | TODO | 200 | middle case |
| `long-context-v1` | mixed | 2K, 32K, 128K | 200 | TTFT and long-context decode |
| `correctness-v1` | mixed | short | 256, greedy | token-exact regression check |

Store each set as a JSONL file of OpenAI-style message lists, named after the set, for example `prose-v1.jsonl`.
