#!/usr/bin/env python3
"""Block sensitive data from being committed to this public repo.

Checks tracked files (or, with --staged, what is about to be committed) plus the
git author/committer identity for:

  * hardware serial numbers (DGX_SERIAL_NUMBER, board/chassis serials)
  * real email addresses (anything that is not a GitHub noreply address)
  * personal names
  * absolute home directories that leak a username
  * credential-shaped strings (tokens, API keys, private keys)

Run directly, via `make check-sensitive`, or install as a pre-commit hook:

    python3 scripts/check_no_sensitive.py --install-hook

Exit 0 = clean, 1 = findings. Use `--staged` in hooks so it only inspects what
is being committed.

If a match is a false positive, add a narrow pattern to ALLOW below rather than
weakening a rule.
"""
import argparse
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Exact substrings that may make the matching value exempt. These never exempt
# the rest of a line: a placeholder beside a real credential must still fail.
ALLOW = (
    "users.noreply.github.com",   # GitHub's own anonymised address form
    "noreply@anthropic.com",      # tool co-author trailer
    "noreply@github.com",         # GitHub merge committer
    "example.com",
    "your-email",
    "REDACTED",
    "<redacted>",
    "placeholder",
)

# Paths that are themselves the scanner / its docs, where the patterns appear
# as literals by necessity.
SKIP_PATHS = (
    "scripts/check_no_sensitive.py",
    ".githooks/pre-commit",
)

RULES = [
    # --- hardware identifiers ---
    ("hardware serial",
     re.compile(r"(DGX_SERIAL_NUMBER|BOARD_SERIAL|CHASSIS_SERIAL|SERIAL_NUMBER)"
                r"\s*[=:]\s*[\"']?([A-Za-z0-9-]{6,})", re.I),
     "Redact the value, e.g. DGX_SERIAL_NUMBER=\"<redacted>\""),

    # --- identity ---
    ("real email address",
     re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
     "Use a GitHub noreply address instead of a personal one"),

    ("personal name",
     re.compile(r"\b(benjiconner|benjamin\s+conner|benjamin)\b", re.I),
     "Replace with the GitHub handle or a generic role name"),

    ("username assignment",
     re.compile(r"\b(?:ssh_user|user(?:name)?|login(?:_user)?)\s*[=:]\s*"
                r"[\"']?([A-Za-z][A-Za-z0-9._-]{2,})", re.I),
     "Replace the account name with a documented placeholder"),

    # --- management network / hardware identity ---
    ("management-network address",
     re.compile(r"\b192\.168\.1\.[0-9]{1,3}\b"),
     "Use the documented placeholder range 192.168.10.0/24 instead"),

    ("MAC address",
     re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b"),
     "Redact, e.g. <mac-node0-p0>"),

    ("IPv6 link-local address",
     re.compile(r"fe80::[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4})+"),
     "Redact to <link-local>; the interface identifier fingerprints the host"),

    ("real DGX hostname",
     re.compile(r"\b(gx10-[0-9a-f]{4}|spark-[0-9a-f]{4})\b"),
     "Use a generic node label such as node0/node1/node2"),

    # --- paths that leak a username ---
    ("home directory with username",
     re.compile(r"(/home/(?!sparkmain|spark1|spark2|spark-sep)[a-z][a-z0-9_-]{2,}"
                r"|[Cc]:[\\/]Users[\\/][A-Za-z][A-Za-z0-9_-]+"
                r"|/Users/(?!shared)[a-z][a-z0-9_-]{2,})"),
     "Use a relative path, $HOME, or a node hostname"),

    # --- credentials ---
    ("credential-shaped string",
     re.compile(r"(gh[pousr]_[A-Za-z0-9]{16,}"
                r"|xox[baprs]-[A-Za-z0-9-]{10,}"
                r"|-----BEGIN (?:RSA |OPENSSH |EC |PGP )?PRIVATE KEY-----"
                r"|(?:api[_-]?key|secret|passwd|password|token)\s*[=:]\s*"
                r"[\"'][^\"'\s]{8,}[\"'])", re.I),
     "Remove the secret and rotate it"),
]

# Node hostnames are intentionally allowed in /home/ paths above; they are
# generic and carry no personal identity.


def tracked_files():
    out = subprocess.run(["git", "-C", REPO, "ls-files"],
                         capture_output=True, text=True).stdout
    return [p for p in out.splitlines() if p]


def staged_files():
    out = subprocess.run(
        ["git", "-C", REPO, "diff", "--cached", "--name-only",
         "--diff-filter=ACMR"],
        capture_output=True, text=True).stdout
    return [p for p in out.splitlines() if p]


def is_binary(path):
    try:
        with open(path, "rb") as fh:
            return b"\0" in fh.read(4096)
    except OSError:
        return True


def staged_content(path):
    """Return the exact blob that Git would commit, not the working-tree copy."""
    result = subprocess.run(
        ["git", "-C", REPO, "show", ":%s" % path],
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def allowed_match(label, value):
    """Apply narrow, rule-specific exceptions to one regex match."""
    if label == "real email address":
        return any(allowed.lower() in value.lower() for allowed in ALLOW)
    if label == "credential-shaped string":
        return any(allowed.lower() in value.lower() for allowed in
                   ("REDACTED", "<redacted>", "placeholder"))
    if label == "username assignment":
        username = re.split(r"[=:]", value, maxsplit=1)[-1].strip(" \t\"'").lower()
        return username in {"youruser", "username", "sparkmain", "spark1", "spark2"}
    return False


def scan_files(paths, staged=False):
    findings = []
    for rel in paths:
        if rel in SKIP_PATHS:
            continue
        if staged:
            data = staged_content(rel)
            if data is None or b"\0" in data[:4096]:
                continue
            lines = data.decode("utf-8", errors="replace").splitlines()
        else:
            full = os.path.join(REPO, rel)
            if not os.path.isfile(full) or is_binary(full):
                continue
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
        for n, line in enumerate(lines, 1):
            for label, pattern, fix in RULES:
                for hit in pattern.finditer(line):
                    if allowed_match(label, hit.group(0)):
                        continue
                    findings.append((rel, n, label, hit.group(0)[:70], fix))
    return findings


def scan_identity():
    """Author/committer identity is permanent once pushed - check it too."""
    findings = []
    for key in ("user.email", "user.name"):
        val = subprocess.run(["git", "-C", REPO, "config", "--get", key],
                             capture_output=True, text=True).stdout.strip()
        if not val:
            continue
        for label, pattern, fix in RULES:
            if label not in ("real email address", "personal name"):
                continue
            hit = pattern.search(val)
            if hit and not allowed_match(label, hit.group(0)):
                findings.append(("git config " + key, 0, label, val, fix))
    return findings


HOOK = """#!/bin/sh
# Installed by scripts/check_no_sensitive.py --install-hook
# Resolve an interpreter portably: `python3` is absent on Windows/Git-Bash,
# where the launcher is `py`.
ROOT="$(git rev-parse --show-toplevel)"
for PY in python3 py python; do
  if command -v "$PY" >/dev/null 2>&1 && "$PY" -c "" >/dev/null 2>&1; then
    exec "$PY" "$ROOT/scripts/check_no_sensitive.py" --staged
  fi
done
echo "pre-commit: no working Python interpreter found; blocking commit" >&2
echo "            install Python or run the scanner from another environment" >&2
exit 1
"""


def install_hook():
    hooks_dir = os.path.join(REPO, ".githooks")
    os.makedirs(hooks_dir, exist_ok=True)
    path = os.path.join(hooks_dir, "pre-commit")
    with open(path, "w", newline="\n") as fh:
        fh.write(HOOK)
    os.chmod(path, 0o755)
    subprocess.run(["git", "-C", REPO, "config", "core.hooksPath", ".githooks"],
                   check=True)
    print("Installed .githooks/pre-commit and set core.hooksPath=.githooks")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staged", action="store_true",
                    help="scan only staged files (for pre-commit hooks)")
    ap.add_argument("--install-hook", action="store_true",
                    help="install as a pre-commit hook and exit")
    args = ap.parse_args()

    if args.install_hook:
        return install_hook()

    paths = staged_files() if args.staged else tracked_files()
    findings = scan_files(paths, staged=args.staged) + scan_identity()

    if not findings:
        scope = "staged files" if args.staged else "%d tracked files" % len(paths)
        print("OK - no sensitive data found in %s" % scope)
        return 0

    print("BLOCKED - %d potential leak(s):\n" % len(findings))
    for rel, n, label, snippet, fix in findings:
        where = rel if n == 0 else "%s:%d" % (rel, n)
        print("  %s" % where)
        print("      %s -> %s" % (label, snippet))
        print("      fix: %s\n" % fix)
    print("If a match is a false positive, add a narrow entry to ALLOW in")
    print("scripts/check_no_sensitive.py - do not weaken a rule.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
