"""Fail the build if anything private is about to become public.

Scans the working tree and, with --check-commits, recent commit messages.
Forbidden terms live in a gitignored `.secretscan-deny` so the deny-list is
never itself a public artifact.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

TEXT_SUFFIXES = {
    ".py", ".md", ".json", ".yaml", ".yml", ".txt", ".toml", ".cfg", ".ini",
    ".jsonl", ".sh", ".ps1", ".gitignore", ".gitattributes", "",
}

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("token", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("token", re.compile(r"\bghp_[A-Za-z0-9]{30,}\b")),
    ("abs_path", re.compile(r"[A-Za-z]:\\+Users\\+", re.IGNORECASE)),
    ("abs_path", re.compile(r"[A-Za-z]:\\+Projects\\+", re.IGNORECASE)),
    ("abs_path", re.compile(r"(?<![\w./])/home/[A-Za-z0-9._-]+")),
    ("abs_path", re.compile(r"(?<![\w./])/Users/[A-Za-z0-9._-]+")),
]

# Lines allowed to contain path patterns because they document them.
ALLOWLIST_SUBSTRINGS = (
    "ALLOWLIST_SUBSTRINGS",
    "check_no_secrets",
    "secretscan-deny",
)

# Files that legitimately contain pattern examples rather than real
# secrets: the scanner's own tests, and the design docs that quote
# path patterns as documentation.
EXEMPT_PATHS = (
    "tests/test_check_no_secrets.py",
    "docs/",
)


class Finding(NamedTuple):
    line_no: int
    kind: str
    snippet: str


def load_deny_terms(root: Path) -> list[str]:
    path = root / ".secretscan-deny"
    if not path.exists():
        return []
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line.lower())
    return terms


def scan_text(text: str, deny_terms: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if any(s in line for s in ALLOWLIST_SUBSTRINGS):
            continue
        for kind, pattern in PATTERNS:
            m = pattern.search(line)
            if m:
                findings.append(Finding(i, kind, m.group(0)))
                break
        else:
            low = line.lower()
            for term in deny_terms:
                if term in low:
                    findings.append(Finding(i, "deny_term", term))
                    break
    return findings


def _parse_gitignore(root: Path) -> set[str]:
    """Parse .gitignore patterns (simple implementation for directories)."""
    gitignore_path = root / ".gitignore"
    if not gitignore_path.exists():
        return set()

    patterns = set()
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.add(line.rstrip("/"))
    return patterns


def _should_skip(path: Path, root: Path, gitignore_patterns: set[str]) -> bool:
    """Check if a path matches gitignore patterns.

    A pattern containing `*` or `?` is matched with fnmatch against each
    path component; everything else is matched literally, as before.
    """
    rel = path.relative_to(root)
    parts = rel.parts
    for pattern in gitignore_patterns:
        if "*" in pattern or "?" in pattern:
            if any(fnmatch.fnmatch(part, pattern) for part in parts):
                return True
        elif pattern in parts:
            return True
    return False


def _tracked_files(root: Path) -> list[Path]:
    """Files git would keep — honours .gitignore without reimplementing it."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
        return [root / line for line in out.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        gitignore_patterns = _parse_gitignore(root)
        return [p for p in root.rglob("*") if p.is_file() and not _should_skip(p, root, gitignore_patterns)]


def scan_repo(root: Path, deny_terms: list[str]) -> list[tuple[Path, Finding]]:
    results: list[tuple[Path, Finding]] = []
    for path in _tracked_files(root):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        if any(rel == e or rel.startswith(e) for e in EXEMPT_PATHS):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for finding in scan_text(text, deny_terms):
            results.append((path, finding))
    return results


def scan_commits(root: Path, deny_terms: list[str], limit: int = 50) -> list[Finding]:
    try:
        out = subprocess.run(
            ["git", "log", f"-{limit}", "--pretty=%H%n%B"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return scan_text(out, deny_terms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--check-commits", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not (root / ".secretscan-deny").exists():
        print(
            "warning: no .secretscan-deny found — deny-term scanning is disabled",
            file=sys.stderr,
        )
    deny = load_deny_terms(root)
    failures = 0

    for path, finding in scan_repo(root, deny):
        rel = path.relative_to(root)
        print(f"{rel}:{finding.line_no}: {finding.kind}: {finding.snippet}")
        failures += 1

    if args.check_commits:
        for finding in scan_commits(root, deny):
            print(f"commit message: {finding.kind}: {finding.snippet}")
            failures += 1

    if failures:
        print(f"\n{failures} finding(s). Nothing may be committed or deployed.")
        return 1
    print("check_no_secrets: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
