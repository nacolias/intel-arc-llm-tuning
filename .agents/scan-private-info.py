#!/usr/bin/env python3
"""Scan the repository for private information before it is committed.

Usage:
  python3 .agents/scan-private-info.py            # tracked and untracked (not ignored) files
  python3 .agents/scan-private-info.py --staged   # staged content only (pre-commit hook)

Exit status is 1 when anything is found. False positives go in .agents/scan-allowlist.txt.
"""
import ipaddress
import os
import re
import subprocess
import sys

ROOT = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
ALLOWLIST_PATH = os.path.join(ROOT, ".agents", "scan-allowlist.txt")
SKIP_FILES = {".agents/scan-allowlist.txt", ".agents/scan-private-info.py"}

FORBIDDEN_NAMES = [
    (re.compile(r"^\.env(\..+)?$"), "environment file"),
    (re.compile(r"\.(pem|key|p12|pfx|jks|keystore)$", re.I), "key or certificate file"),
    (re.compile(r"^id_(rsa|dsa|ecdsa|ed25519)(\.pub)?$"), "SSH key"),
    (re.compile(r"^(known_hosts|authorized_keys|\.netrc|\.pgpass|kubeconfig)$"), "credential file"),
    (re.compile(r"^(credentials|secrets?)(\..+)?$", re.I), "credential file"),
]
ALLOWED_NAMES = {".env.example", ".env.sample", ".env.template"}

PUBLIC_TLDS = r"com|net|org|io|dev|app|me|xyz|cloud|ai|co|us|uk|de|info|biz|tech|site|online|home|lan|gg"

PATTERNS = [
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Anthropic/OpenAI style key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}")),
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{20,})")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
    ("MAC address", re.compile(r"(?<![\w:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![\w:-])")),
    ("email address", re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")),
    ("home directory path", re.compile(r"(?:/home/[A-Za-z_][\w.-]*|/Users/[\w.-]+|[A-Za-z]:\\Users\\[\w.-]+)")),
    ("internal domain", re.compile(r"(?i)\b[\w-]+(?:\.[\w-]+)*\.(?:lan|local|internal|localdomain|corp|home\.arpa|ts\.net)\b")),
]

IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
IPV6 = re.compile(r"(?<![\w:.])[0-9A-Fa-f:]{6,}(?![\w:])")
DOMAIN = re.compile(r"(?i)(?<![\w@.-])((?:[a-z0-9][a-z0-9-]*\.)+(?:" + PUBLIC_TLDS + r"))(?![\w-])")
SECRET_ASSIGN = re.compile(
    r"(?i)\b[\w-]*(?:api[_-]?key|secret|token|passw(?:or)?d|pwd)[\w-]*[\"']?\s*[:=]\s*[\"']?([^\s\"'`,;)}\]]+)"
)
SECRET_FLAG = re.compile(r"(?i)--(?:api-key|token|password|secret)[= ]+([^\s\\\"']+)")
SAFE_IPS = {"0.0.0.0", "127.0.0.1", "255.255.255.255"}


def load_allowlist():
    entries = set()
    if os.path.exists(ALLOWLIST_PATH):
        for line in open(ALLOWLIST_PATH, encoding="utf-8"):
            line = line.split("#", 1)[0].strip()
            if line:
                entries.add(line.lower())
    return entries


def allowed(value, allowlist):
    v = value.lower().rstrip(".")
    return any(v == a or v.endswith("." + a) for a in allowlist)


def placeholder(value):
    return value[:1] in "$<{%" or value.lower() in {"true", "false", "none", "null", "required", "optional"}


def scan_line(line, allowlist):
    hits = []
    for name, rx in PATTERNS:
        for m in rx.finditer(line):
            if name == "email address" and m.group(0).lower().startswith(("noreply@", "no-reply@")):
                continue
            if not allowed(m.group(0), allowlist):
                hits.append((name, m.group(0)))
    for m in IPV4.finditer(line):
        ip = m.group(0)
        if all(int(o) <= 255 for o in ip.split(".")) and ip not in SAFE_IPS and not ip.startswith(("127.", "0.")) and not allowed(ip, allowlist):
            hits.append(("IPv4 address", ip))
    for m in IPV6.finditer(line):
        if m.group(0).count(":") < 2:
            continue
        try:
            addr = ipaddress.IPv6Address(m.group(0))
        except ValueError:
            continue
        if not addr.is_loopback and not addr.is_unspecified and not allowed(m.group(0), allowlist):
            hits.append(("IPv6 address", m.group(0)))
    for m in DOMAIN.finditer(line):
        if not allowed(m.group(1), allowlist):
            hits.append(("domain or hostname", m.group(1)))
    for rx in (SECRET_ASSIGN, SECRET_FLAG):
        for m in rx.finditer(line):
            val = m.group(1)
            if placeholder(val) or val.isdigit() or len(val) < 8:
                continue
            hits.append(("literal secret value", m.group(0)))
    return hits


def list_files(staged):
    if staged:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                             capture_output=True, text=True, cwd=ROOT, check=True).stdout
    else:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                             capture_output=True, text=True, cwd=ROOT, check=True).stdout
    return [f for f in out.splitlines() if f]


def read_file(path, staged):
    if staged:
        data = subprocess.run(["git", "show", ":" + path], capture_output=True, cwd=ROOT).stdout
    else:
        full = os.path.join(ROOT, path)
        if not os.path.isfile(full):
            return None
        with open(full, "rb") as fh:
            data = fh.read()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def main():
    staged = "--staged" in sys.argv[1:]
    allowlist = load_allowlist()
    findings = []
    for path in list_files(staged):
        base = os.path.basename(path)
        if base not in ALLOWED_NAMES:
            for rx, label in FORBIDDEN_NAMES:
                if rx.search(base):
                    findings.append((path, 0, "forbidden file (" + label + ")", base))
        if path in SKIP_FILES:
            continue
        text = read_file(path, staged)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            seen = set()
            for kind, value in scan_line(line, allowlist):
                if value not in seen:
                    seen.add(value)
                    findings.append((path, lineno, kind, value))

    if not findings:
        print("scan-private-info: no private information found")
        return 0
    print("scan-private-info: possible private information found\n")
    for path, lineno, kind, value in findings:
        print(f"  {path}:{lineno}: {kind}: {value}")
    print("\nRemove or replace these with placeholders (see .agents/privacy.md).")
    print("If a value is genuinely public, add it to .agents/scan-allowlist.txt with a comment.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
