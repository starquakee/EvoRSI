"""Reproducible secret and host-path audit for tracked repository files (US-001).

Two rule classes:
- secrets (hard fail, all tracked text files): sandbox keys, cloud/provider
  tokens, private key blocks, and generic secret-looking assignments.
- host absolute paths (hard fail only in research/**/*.py): new consolidated
  code must resolve repository paths relative to the repo root. The same
  patterns in other tracked files (docs, PRD, manifests) are reported as
  warnings, since documentation legitimately describes machine layout.

The audit reads tracked files via `git ls-files`; secret-named files are never
tracked (root .gitignore), and this tool never prints matched secret values —
only file, line number, and pattern name.

Usage:
    <venv-python> research/tools/audit_secrets.py            # text report
    <venv-python> research/tools/audit_secrets.py --json OUT # machine report
Exit code 1 on any hard violation.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

SECRET_PATTERNS: dict[str, str] = {
    "openmle_sandbox_key": r"mlsandbox-[0-9A-Za-z-]{16,}",
    "openai_style_key": r"sk-[A-Za-z0-9_-]{20,}",
    "aws_access_key_id": r"AKIA[0-9A-Z]{16}",
    "github_or_gitlab_token": r"(?:ghp_|github_pat_|glpat-|xox[bap]-)[A-Za-z0-9_-]{16,}",
    "private_key_block": r"BEGIN [A-Z ]*PRIVATE KEY",
    "generic_secret_assignment": (
        r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*"
        r"['\"][^'\"\s]{12,}['\"]"
    ),
}

HOST_PATH_PATTERNS: dict[str, str] = {
    "linux_home": r"(?<![\w/])/home/",
    "wsl_windows_drive": r"(?<![\w/])/mnt/[a-z]/",
    "windows_user_dir": r"[A-Za-z]:\\Users",
    "macos_users": r"(?<![\w/])/Users/",
}

# Placeholders that are documentation, not credentials.
PLACEHOLDER_RE = re.compile(
    r"(?i)^(<[^>]*>|\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*|x+|\*+|"
    r"your[_-].*|replace[-_].*|change[-_]?me.*|example.*|dummy.*)$"
)

_MAX_FILE_BYTES = 5 * 1024 * 1024


def _is_text(data: bytes) -> bool:
    return b"\x00" not in data[:8192]


# Self-referential files: the audit module defines host-path literals and the
# generic-assignment regex, and its test plants samples of them. They are still
# scanned with the high-confidence secret patterns (specific key formats,
# private key blocks) — only the self-matching rule classes are skipped.
SELF_REFERENTIAL_PATHS = frozenset({
    "research/tools/audit_secrets.py",
    "research/tests/test_audit_secrets.py",
})

_HIGH_CONFIDENCE_SECRETS = (
    "openmle_sandbox_key", "openai_style_key", "aws_access_key_id",
    "github_or_gitlab_token", "private_key_block",
)


def scan_text_for_secrets(
    text: str, *, patterns: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    selected = patterns if patterns is not None else SECRET_PATTERNS.keys()
    for name in selected:
        for match in re.finditer(SECRET_PATTERNS[name], text):
            if name == "generic_secret_assignment":
                value = match.group(0).rsplit("=", 1)[-1].rsplit(":", 1)[-1]
                value = value.strip().strip("'\"")
                if PLACEHOLDER_RE.match(value):
                    continue
            findings.append({
                "pattern": name,
                "line": text.count("\n", 0, match.start()) + 1,
            })
    return findings


def scan_text_for_host_paths(text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for name, pattern in HOST_PATH_PATTERNS.items():
        for match in re.finditer(pattern, text):
            findings.append({
                "pattern": name,
                "line": text.count("\n", 0, match.start()) + 1,
            })
    return findings


def tracked_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=repo_root,
        check=True, capture_output=True, text=True,
    )
    return [repo_root / line for line in out.stdout.splitlines() if line]


def audit(paths: Iterable[Path], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    secrets: list[dict[str, Any]] = []
    path_violations: list[dict[str, Any]] = []
    path_warnings: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    scanned = 0
    for path in sorted(paths):
        rel = path.relative_to(repo_root).as_posix()
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                skipped.append({"path": rel, "reason": "too_large"})
                continue
            data = path.read_bytes()
        except OSError as exc:
            skipped.append({"path": rel, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if not _is_text(data):
            skipped.append({"path": rel, "reason": "binary"})
            continue
        scanned += 1
        text = data.decode("utf-8", errors="replace")
        if rel in SELF_REFERENTIAL_PATHS:
            for finding in scan_text_for_secrets(text, patterns=_HIGH_CONFIDENCE_SECRETS):
                secrets.append({"file": rel, **finding})
            continue
        for finding in scan_text_for_secrets(text):
            secrets.append({"file": rel, **finding})
        for finding in scan_text_for_host_paths(text):
            entry = {"file": rel, **finding}
            if rel.startswith("research/") and rel.endswith(".py"):
                path_violations.append(entry)
            else:
                path_warnings.append(entry)
    return {
        "scanned_files": scanned,
        "secret_findings": secrets,
        "host_path_violations": path_violations,
        "host_path_warnings": path_warnings,
        "skipped": skipped,
        "ok": not secrets and not path_violations,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write machine report")
    args = parser.parse_args(argv)

    report = audit(tracked_files())
    print(f"scanned {report['scanned_files']} tracked text files")
    print(f"secret findings: {len(report['secret_findings'])}")
    for finding in report["secret_findings"]:
        print(f"  SECRET {finding['file']}:{finding['line']} {finding['pattern']}")
    print(f"host-path violations (research/**/*.py): {len(report['host_path_violations'])}")
    for finding in report["host_path_violations"]:
        print(f"  PATH {finding['file']}:{finding['line']} {finding['pattern']}")
    print(f"host-path warnings (docs/other): {len(report['host_path_warnings'])}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("AUDIT " + ("OK" if report["ok"] else "FAILED"))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
