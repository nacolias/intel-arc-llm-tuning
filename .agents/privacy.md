# Privacy and secrets

This repository is public. Anything committed can be read, copied and indexed by anyone, and stays in git history even after it is deleted. Removing it later requires rewriting history and rotating whatever leaked.

## Never commit

| Category | Examples |
|---|---|
| Network addresses | IPv4 and IPv6 addresses (public or private ranges), subnets, MAC addresses, port-forwarding details |
| Hostnames and domains | a machine's real hostname, internal DNS names, personal or home domains, reverse-proxy vhosts, Tailscale or VPN names |
| Credentials | API keys, bearer tokens, Hugging Face tokens, GitHub tokens, cloud keys, passwords, session cookies |
| Secret files | `.env` and `.env.*` files or any values copied out of them, private keys (`*.pem`, `*.key`, `id_rsa`), certificates with private keys, `known_hosts`, kubeconfig, Docker `config.json` |
| Personal information | real names beyond the LICENSE, email addresses, usernames, home directory paths such as `/home/<name>`, physical locations |
| Hardware identifiers | GPU, board or disk serial numbers, UUIDs from `xpu-smi` or `lspci -vv`, BIOS asset tags |
| Private endpoints | URLs of your own servers, dashboards, model endpoints, internal registries |
| Private models or data | gated model weights, private prompts, user conversations, request logs containing user content |

If you are not sure, leave it out or replace it with a placeholder.

## Use placeholders

| Instead of | Write |
|---|---|
| a real hostname or domain | `<your-host>` |
| an IP address | `<host-ip>` or `127.0.0.1` for local-only examples |
| an API key | `${VLLM_API_KEY}` (a variable reference, never the value) |
| a token | `${HF_TOKEN}` |
| a home directory | `/path/to/models` |
| a real machine name | a hardware label such as `dual-b70-5800x` |

Public, non-identifying values are fine: `0.0.0.0` and `127.0.0.1` in launch flags, public URLs (GitHub, Hugging Face, vendor docs), public image names and digests, public model IDs, and version numbers.

## Hosts are named by their hardware

Host files in `hardware/hosts/` are named after the hardware, for example `dual-b70-5800x.md`. Never use the machine's real hostname, even without a domain.

## Before you paste command output

Tool output often carries private data. Strip it before committing:

- `ip`, `ifconfig`, `hostname`, `uname -a`: hostnames and addresses.
- `lspci -vv`, `xpu-smi discovery`, `dmidecode`, `sycl-ls`: serial numbers and UUIDs.
- `docker inspect`, `docker compose config`: environment variables, including secrets from `.env`.
- vLLM and proxy logs: client IPs, API keys in headers, user prompts.
- Shell prompts: `user@host:~$` reveals both.

Keep only the lines that matter for the finding.

## When copying files from a real system

Compose files, launch scripts and patches from a working server usually reference `.env` values, hostnames and local paths. Before committing:

1. Replace every hostname, domain and address with a placeholder.
2. Replace secret values with `${VARIABLE}` references and never commit the `.env` file.
3. Add a comment at the top saying which placeholders to fill in.
4. Run the scanner.

## If something private gets committed

- **Not pushed yet:** amend or rewrite the local commit to remove it, then re-run the scanner.
- **Already pushed:** stop and tell the repository owner. Assume the value is compromised. Rotate any credential immediately; deleting the file in a new commit does not remove it from history.

## The scanner

```bash
python3 .agents/scan-private-info.py          # scan every tracked and new file
python3 .agents/scan-private-info.py --staged # scan only staged files (used by the pre-commit hook)
```

It flags IP addresses, MAC addresses, email addresses, internal domains, URLs to hosts that are not on the public allowlist, home directory paths, common token formats, literal secrets assigned to key-like names, private key blocks, and forbidden file names such as `.env`.

The root `.gitignore` also blocks common secret file names and model weights. Never force-add an ignored file with `git add -f`.

The scanner is a safety net, not a guarantee. It cannot recognize a bare hostname such as a machine nickname. You are still responsible for reading your diff.

**False positives.** If a flagged value is genuinely public, such as a four-part version number that looks like an IP address or a public domain not yet on the allowlist, add the exact string to [`scan-allowlist.txt`](scan-allowlist.txt) with a comment saying why. Never allowlist a real private value.
