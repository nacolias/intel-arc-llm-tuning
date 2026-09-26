# Before every commit

1. **Run the scanner** and fix every finding:
   ```bash
   python3 .agents/scan-private-info.py
   ```
2. **Read your own diff** (`git diff --staged`) for private information the scanner cannot recognize: machine nicknames, people's names, locations, private model names, pasted logs.
3. **No secret files staged.** `git status` shows no `.env`, key, certificate or credential file.
4. **Every new number has a source**, and unmeasured claims say `UNVERIFIED`.
5. **Environment recorded** for every benchmark and experiment: engine version, image digest, runtime versions, power cap, launch flags.
6. **Indexes updated** for every entry you added (see [`contributing.md`](contributing.md#indexes-to-update)).
7. **Relative links resolve.** Every relative Markdown link you added points to a file that exists.
8. **Commit message** says what was added and why, for example "Add MTP4 experiment for Qwen3.8-27B".
