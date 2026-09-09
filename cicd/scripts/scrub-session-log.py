#!/usr/bin/env python3
"""Search a Claude Code session log (.jsonl) for credentials, and trim them.

The file-issue workflow copies session logs into .sessions/ and commits them. This
repo is a PUBLIC fork, so a log carrying a live secret publishes it irreversibly.
Run `scan` before committing a log; run `scrub` to write a redacted copy.

Two sources of truth:
  - the built-in regexes below, which recognise secrets by SHAPE and are safe to commit;
  - a local denylist of LITERAL strings, which is not, and so lives outside the repo.

A password like "hunter2" has no shape a regex can catch. Put those literals in the
denylist file (one per line, blank lines and #-comments ignored):

  ~/.claude/scrub-denylist.txt        (override with $SCRUB_DENYLIST_FILE)

Usage:
  scrub-session-log.py scan  <file.jsonl> [...]      exit 1 if anything is found
  scrub-session-log.py scrub <in.jsonl> <out.jsonl>  write a redacted copy
"""

import json
import os
import re
import sys

# Shape-based. Safe to commit — none of these encode a secret.
PATTERNS = [
    ("private-key", r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
    ("github-token", r"gh[pousr]_[A-Za-z0-9]{20,}"),
    ("github-pat", r"github_pat_[A-Za-z0-9_]{30,}"),
    ("anthropic-key", r"sk-ant-[A-Za-z0-9_-]{20,}"),
    ("openai-key", r"sk-(?:proj-)?[A-Za-z0-9]{32,}"),
    ("aws-access-key", r"AKIA[0-9A-Z]{16}"),
    ("slack-token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ("google-api-key", r"AIza[0-9A-Za-z_-]{35}"),
    ("jwt", r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    # scheme://user:pass@host
    ("url-inline-cred", r"[a-z][a-z0-9+.-]*://[^/\s:@\"\\]+:[^/\s@\"\\]+@"),
    # curl -u user:pass
    ("basic-auth-flag", r"-u\s+[^\s:\"\\]{1,64}:[^\s\"\\]{4,}"),
    ("auth-header", r"[Aa]uthorization[\"'\s:]+(?:Bearer|Basic|token)\s+[A-Za-z0-9_.=+/-]{8,}"),
    # key = value. ${{ secrets.X }} does not match: '$' is not in the value class.
    ("assigned-secret",
     r"(?i)\b(?:api[_-]?key|apikey|access[_-]?token|auth[_-]?token|secret[_-]?key|password|passwd)\b"
     r"\s*[:=]\s*\\?[\"']?[A-Za-z0-9_.=+/-]{6,}"),
]
COMPILED = [(kind, re.compile(rx)) for kind, rx in PATTERNS]

# Not a credential, but this project's memory forbids publishing the K80 host's address.
PRIVATE_IP = re.compile(r"\b(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
                        r"|\b192\.168\.\d{1,3}\.\d{1,3}\b"
                        r"|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b")


def load_denylist():
    path = os.environ.get("SCRUB_DENYLIST_FILE",
                          os.path.expanduser("~/.claude/scrub-denylist.txt"))
    if not os.path.exists(path):
        return [], path
    literals = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            s = raw.strip()
            if s and not s.startswith("#"):
                literals.append(s)
    # longest first, so a substring never redacts ahead of the full secret
    return sorted(set(literals), key=len, reverse=True), path


def mask(s):
    """Show enough to identify a hit, never enough to use it."""
    s = s.strip()
    return (s[:3] + "*" * min(len(s) - 3, 12)) if len(s) > 6 else "*" * len(s)


def find(line, denylist, flag_ips):
    hits = []
    for kind, rx in COMPILED:
        for m in rx.finditer(line):
            hits.append((kind, m.group(0)))
    for lit in denylist:
        if lit in line:
            hits.append(("denylist", lit))
    if flag_ips:
        for m in PRIVATE_IP.finditer(line):
            hits.append(("private-ip", m.group(0)))
    return hits


def redact(line, denylist, flag_ips):
    # Placeholders contain no quotes or backslashes, so JSON stays valid.
    for lit in denylist:
        line = line.replace(lit, "[REDACTED-denylist]")
    for kind, rx in COMPILED:
        line = rx.sub("[REDACTED-%s]" % kind, line)
    if flag_ips:
        line = PRIVATE_IP.sub("[REDACTED-private-ip]", line)
    return line


def main():
    argv = [a for a in sys.argv[1:] if a != "--no-ips"]
    flag_ips = "--no-ips" not in sys.argv[1:]
    if len(argv) < 2 or argv[0] not in ("scan", "scrub"):
        print(__doc__.strip())
        return 2

    mode, args = argv[0], argv[1:]
    denylist, denylist_path = load_denylist()
    if not denylist:
        print("WARNING: no denylist at %s — shape-less secrets (plain passwords) "
              "will NOT be caught.\n" % denylist_path, file=sys.stderr)

    if mode == "scan":
        total = 0
        for path in args:
            per_file = {}
            with open(path, encoding="utf-8", errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    for kind, value in find(line, denylist, flag_ips):
                        per_file.setdefault(kind, []).append((n, value))
            if per_file:
                print("%s" % path)
                for kind in sorted(per_file):
                    rows = per_file[kind]
                    lines = ",".join(str(n) for n, _ in rows[:5])
                    more = " +%d" % (len(rows) - 5) if len(rows) > 5 else ""
                    print("  %-18s %3d hit(s)  line %s%s   e.g. %s"
                          % (kind, len(rows), lines, more, mask(rows[0][1])))
                    total += len(rows)
            else:
                print("%s\n  clean" % path)
        print("\n%d finding(s)." % total)
        return 1 if total else 0

    if len(args) != 2:
        print("scrub takes <in.jsonl> <out.jsonl>", file=sys.stderr)
        return 2
    src, dst = args
    changed = 0
    with open(src, encoding="utf-8", errors="replace") as fin, \
            open(dst, "w", encoding="utf-8") as fout:
        for n, line in enumerate(fin, 1):
            out = redact(line, denylist, flag_ips)
            if out != line:
                changed += 1
            fout.write(out)
            # a redacted log nothing can parse is not a log
            if out.strip():
                try:
                    json.loads(out)
                except json.JSONDecodeError as e:
                    print("ERROR: line %d is not valid JSON after redaction: %s"
                          % (n, e), file=sys.stderr)
                    return 3
    leftover = sum(len(find(l, denylist, flag_ips))
                   for l in open(dst, encoding="utf-8", errors="replace"))
    print("%s -> %s: %d line(s) redacted, %d finding(s) remaining"
          % (src, dst, changed, leftover))
    return 1 if leftover else 0


if __name__ == "__main__":
    sys.exit(main())
