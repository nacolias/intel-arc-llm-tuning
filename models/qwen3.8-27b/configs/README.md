# Configs

| File | Description | Status |
|---|---|---|
| [production-mtp2-compose.yml](production-mtp2-compose.yml) | vLLM XPU 0.28.0, TP2, MTP2, XPU graphs, FP8 KV, 262K context | production |
| [lab-mtp4-int4-draft-head.sh](lab-mtp4-int4-draft-head.sh) | community lab reference launch: MTP4 plus INT4 draft LM head, 112.9 tok/s on their dual-B70 host | A/B candidate |

Patches applied at startup are listed in [patches/README.md](patches/README.md).
