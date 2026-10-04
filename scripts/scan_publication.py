"""Scan the exact Git index export, never unrelated ignored local caches.

Refuses unstaged modifications or untracked candidates. Runs the publication guard
and redacted Gitleaks against a temporary standalone copy of the publication set.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gitleaks", default=shutil.which("gitleaks"))
    args = parser.parse_args()
    if not args.gitleaks:
        parser.error("gitleaks is required; no silent skipped privacy gate")
    top = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True
    ).strip()
    if Path(top).resolve() != ROOT:
        parser.error("refusing a parent Git repository")
    if subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode:
        parser.error("unstaged changes: stage the final intended publication before scanning")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT
    )
    if untracked:
        parser.error("untracked candidates: review/stage or explicitly ignore before scanning")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    if not tracked:
        parser.error("empty publication set")
    with tempfile.TemporaryDirectory(prefix="public-index-") as directory:
        export = Path(directory)
        subprocess.run(
            ["git", "checkout-index", "--all", f"--prefix={export}/"], cwd=ROOT, check=True
        )
        guard = subprocess.run(
            [
                sys.executable,
                str(export / "scripts/publication_guard.py"),
                "--mode",
                "all",
                "--json",
            ],
            cwd=export,
        )
        if guard.returncode:
            return guard.returncode
        command = [args.gitleaks, "dir", "--no-banner", "--redact", "--exit-code", "1"]
        config = export / ".gitleaks.toml"
        if config.exists():
            command += ["--config", str(config)]
        command.append(str(export))
        result = subprocess.run(command, cwd=export)
        print(f"publication_scope=exact_git_index scanner_exit_code={result.returncode}")
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
