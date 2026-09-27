# Idle autosave of the prompt-cache slot

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | adopted |
| **Baseline** | [slot save/restore](2026-09-25-slot-save-restore-and-ram-cache.md): the slot is saved only when the service stops cleanly |
| **Result** | this file (one manual run and the timer's first runs); save timings in [raw/2026-09-26-slot-cache-timings.csv](../benchmarks/raw/2026-09-26-slot-cache-timings.csv) for the earlier saves |
| **Related** | [real agent traffic](../benchmarks/2026-09-26-real-agent-traffic.md), [finding: a restart drops the prefix cache](../../../findings/llama-server-restart-drops-prefix-cache.md) |

## Result

A timer now saves the prompt-cache slot every time the agent has been idle for a whole 5-minute period and the slot changed since the last save. Before, the only save ran on a clean stop. A crash, an out-of-memory kill or a power cut would have restored the last clean-stop save. On the day this was added, that save held 41,364 tokens and was 18 hours old, while the live conversation was at 135,226 tokens. A restore would have left at least 94K tokens to re-read, if the old save was a prefix of the same conversation, and all 135K if not: roughly 3.5 to 4.7 minutes at the current cold-read rate (estimate from the [cold-read segment times](../benchmarks/2026-09-26-cold-read-135k-by-build.md)). The first autosave wrote the 135,226-token slot (3.6 GiB plus a 302 MiB draft file) in 1.8 s.

## Hypothesis

Clean stops are covered: the unit saves on `ExecStop`. Unclean ones are not. The host has a memory watchdog that kills the server before host RAM runs out, and a kill skips `ExecStop`. The agent works in bursts with idle gaps, so a save in each idle gap costs nothing the user would notice. The save is queued by llama-server and runs only when the slot is free. It takes about 1.5-1.8 s at 135K and would delay a request that arrived during it by at most that. A restore then needs only the turns after the last idle gap.

## Change

`slot-cache.sh autosave`, run by a systemd timer every 5 minutes (the sanitized script is [configs/llama-server-slot-cache.sh](../configs/llama-server-slot-cache.sh)):

1. Skip if a hold is set (tests and benchmarks), or if the server's unit is not active.
2. Read `/slots`: skip if the slot is processing. Otherwise take the last task ID, keyed with the unit's `InvocationID` so a restart starts afresh.
3. If that task ID was not seen on the previous tick, the agent was active during the period: remember it and stop. The next tick saves if nothing new arrived in between, so a save follows 5-10 minutes of idleness.
4. Skip if this task ID was already saved.
5. Skip, but mark as saved, if the slot holds under half the tokens of a save made in the past 12 hours. That is a side request or subagent that displaced the main conversation into the host-RAM prompt cache, which a slot save cannot reach. Saving it would replace the useful save with a small one.
6. Save through the same code path as the stop-time save (temporary file, then rename, signature copied), under a `flock` so the timer and the stop-time save cannot write at once.

Timer and service units, as installed (unit names generic here):

```ini
# llama-server-autosave.service
[Unit]
Description=save llama-server's prompt cache once the agent is idle (slot-cache.sh autosave)
After=llama-server.service

[Service]
Type=oneshot
User=<service user>
Environment=UNIT=llama-server.service SLOT_DIR=/path/to/slots
Environment=API_KEY_FILE=<0600 file holding the API key>
ExecStart=/path/to/slot-cache.sh autosave

# llama-server-autosave.timer
[Unit]
Description=check every 5 min whether llama-server's prompt cache needs saving

[Timer]
OnBootSec=10min
OnUnitActiveSec=5min
AccuracySec=30s

[Install]
WantedBy=timers.target
```

## Environment

Production service, build `20260926-2b84213a4` (see [real agent traffic](../benchmarks/2026-09-26-real-agent-traffic.md) for the full table). The slot directory is on a local NVMe ext4 file system.

## Procedure

Ran `slot-cache.sh autosave` three times by hand on the idle production server, then enabled the timer and ran its service once.

## Results

| Run | Expected | Observed |
|---|---|---|
| 1st manual | task not seen before: record it, no save | recorded the task, no save |
| 2nd manual | same task, not saved yet, not smaller than half the last save (41,364): save | `saved 135226 tokens (3.6G, draft 302M) in 1.8s`; wall time 3.3 s including the size checks |
| 3rd manual | same task, already saved: nothing | nothing |
| timer's first run | same task, already saved: nothing | nothing; next run scheduled 5 min later |

Disk cost (estimate): about 4 GiB per save at 135K. With one save per idle gap, a day of 30 gaps writes about 120 GiB, well within the endurance of a 1 TB NVMe drive rated in the hundreds of TB written (the drive's rating was not checked).

## Correctness

The save and restore paths are unchanged from [slot save/restore](2026-09-25-slot-save-restore-and-ram-cache.md), which checked restored-state decoding. A restore is still gated on the setup signature.

## Decision

Adopted on 2026-09-26.

## Follow-ups

- [ ] Kill the server with SIGKILL after an autosave and check that the restart restores the autosaved state (not yet done).
- [ ] The half-size guard is a heuristic. If the agent compacts its conversation to under half, autosaves stop for up to 12 hours; the stop-time save still runs.
