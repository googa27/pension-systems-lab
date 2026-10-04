"""Fail-closed publication privacy guard for the public repository surface.

The guard intentionally scans an explicit file list. In a standalone repository it
can obtain that list from Git-tracked/staged files; when the project is only a
nested directory inside a larger parent Git checkout, it refuses Git mode so the
parent repository cannot accidentally define the publication boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".ipynb",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".typed",
    ".yaml",
    ".yml",
}
FORBIDDEN_SUFFIXES = {
    ".db",
    ".duckdb",
    ".key",
    ".p12",
    ".parquet",
    ".pdf",
    ".sqlite",
    ".sqlite3",
    ".xls",
    ".xlsm",
    ".xlsx",
}
FORBIDDEN_PATH_PARTS = {
    ".aider.chat.history.md",
    ".aws",
    ".codex",
    ".env",
    ".gcp",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "agent_history",
    "credentials",
    "secrets",
}
ALLOWED_BINARY_SUFFIXES = {".png"}
TEXT_NAMES = {"LICENSE", ".gitignore", ".gitattributes"}
PRIVATE_TEXT_PATTERNS = [
    re.compile(r"/home/[A-Za-z0-9_.-]+"),
    re.compile("Desktop/" + "Projects"),
    re.compile(r"/tmp/pension-prior-" + r"discovery\.md"),
    re.compile(r"BEGIN (?:RSA |EC |OPENSSH |)PRIVATE KEY"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|password|passwd|secret|token)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
    ),
    re.compile(r"(?i)github\.com/[A-Za-z0-9_.-]*(?:private|internal|personal)[A-Za-z0-9_.-]*/"),
]


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str
    detail: str


def _run_git(root: Path, args: list[str]) -> list[str]:
    output = subprocess.check_output(["git", "-C", str(root), *args], text=True)
    return [line for line in output.splitlines() if line]


def _nearest_git_dir(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def git_publication_paths(root: Path = ROOT) -> list[Path]:
    """Return tracked plus staged paths only when *root* is its own Git root."""
    own_git = root / ".git"
    nearest = _nearest_git_dir(root)
    if not own_git.exists():
        if nearest is not None:
            raise RuntimeError(
                f"refusing Git publication scan: {root} is nested under parent Git root {nearest}"
            )
        raise RuntimeError(f"refusing Git publication scan: {root} is not a Git repository")
    top = Path(_run_git(root, ["rev-parse", "--show-toplevel"])[0]).resolve()
    if top != root.resolve():
        raise RuntimeError(f"refusing Git publication scan: Git root is {top}, not {root}")
    tracked = set(_run_git(root, ["ls-files"]))
    staged = set(_run_git(root, ["diff", "--cached", "--name-only"]))
    return [root / name for name in sorted(tracked | staged)]


def all_public_paths(root: Path = ROOT) -> list[Path]:
    paths: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = set(path.relative_to(root).parts)
        if relative_parts & {
            ".venv",
            ".ruff_cache",
            ".mypy_cache",
            ".pytest_cache",
            ".hypothesis",
            "__pycache__",
            ".git",
            "dist",
            "build",
        }:
            continue
        paths.append(path)
    return sorted(paths)


def explicit_paths(names: Iterable[str], root: Path = ROOT) -> list[Path]:
    paths = []
    for name in names:
        candidate = (root / name).resolve()
        if root.resolve() not in (candidate, *candidate.parents):
            raise RuntimeError(f"path escapes repository root: {name}")
        paths.append(candidate)
    return paths


def _is_binary_allowed(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_BINARY_SUFFIXES


def scan_paths(paths: Iterable[Path], root: Path = ROOT) -> list[Finding]:
    findings: list[Finding] = []
    root = root.resolve()
    for path in paths:
        if path.is_symlink():
            findings.append(Finding(path.name, "symlink", "symlinks are not public artifacts"))
            continue
        if not path.exists() or not path.is_file():
            findings.append(Finding(str(path), "missing", "listed publication file does not exist"))
            continue
        relative = path.resolve().relative_to(root).as_posix()
        parts = set(Path(relative).parts)
        lowered = {part.lower() for part in parts}
        bad_parts = sorted(lowered & FORBIDDEN_PATH_PARTS)
        if bad_parts:
            findings.append(Finding(relative, "forbidden-path", ", ".join(bad_parts)))
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            findings.append(Finding(relative, "forbidden-suffix", suffix))
        if suffix not in TEXT_EXTENSIONS and path.name not in TEXT_NAMES:
            if not _is_binary_allowed(path):
                findings.append(Finding(relative, "unknown-binary", suffix or "no suffix"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(Finding(relative, "decode", "text file is not UTF-8"))
            continue
        for pattern in PRIVATE_TEXT_PATTERNS:
            if pattern.search(text):
                findings.append(Finding(relative, "private-text", pattern.pattern))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("git", "all", "paths"), default="git")
    parser.add_argument(
        "--path", action="append", default=[], help="Repository-relative path for --mode paths"
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable result")
    args = parser.parse_args(argv)
    try:
        if args.mode == "git":
            paths = git_publication_paths(ROOT)
        elif args.mode == "all":
            paths = all_public_paths(ROOT)
        else:
            paths = explicit_paths(args.path, ROOT)
        findings = scan_paths(paths, ROOT)
    except RuntimeError as exc:
        payload = {"status": "refused", "reason": str(exc), "findings": []}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(payload["reason"], file=sys.stderr)
        return 2
    payload = {
        "status": "ok" if not findings else "failed",
        "scanned": len(paths),
        "findings": [finding.__dict__ for finding in findings],
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
