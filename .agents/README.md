# Instructions for agents

This folder is for AI agents (and humans) adding to this repository. Read all of it before your first change.

| File | Read it for |
|---|---|
| [`privacy.md`](privacy.md) | what must never be committed, and how to redact it. **Mandatory.** |
| [`contributing.md`](contributing.md) | where each kind of entry goes, required fields, writing rules |
| [`checklist.md`](checklist.md) | the steps to run before every commit |
| [`scan-private-info.py`](scan-private-info.py) | the scanner that checks for private information |
| [`hooks/pre-commit`](hooks/pre-commit) | git hook that runs the scanner on every commit |

## The rules that matter most

1. **This repository is public.** Never commit IP addresses, hostnames, domain names, API keys, tokens, passwords, secrets, `.env` files or their contents, private keys, usernames, email addresses, MAC addresses or serial numbers. If you are unsure whether something is private, treat it as private. Details in [`privacy.md`](privacy.md).
2. **Run the scanner before every commit** and fix everything it reports:
   ```bash
   python3 .agents/scan-private-info.py
   ```
3. **Never invent measurements.** Every number needs a source: a benchmark file in this repo, or a linked external source. Anything not measured on our hardware is marked `UNVERIFIED`.
4. **Record the environment.** A result without engine version, image digest, runtime versions, power cap and launch flags is not useful. Use the tables in [`templates/`](../templates/).
5. **Do not overwrite history.** Add new experiment and benchmark files; do not edit old results to match new ones. Correct a mistake with a dated note.
6. **Keep the indexes current.** When you add an entry, add its row to the folder's `README.md` index, and update the model card and status board if headline numbers change.

## Getting started

```bash
git config core.hooksPath .agents/hooks
```

This turns on the pre-commit scanner for your clone. Then read [`contributing.md`](contributing.md) and the root [`README.md`](../README.md).
