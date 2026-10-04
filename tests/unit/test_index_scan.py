"""Regression: publication scanner cannot attest stale or partial staging."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_index_scan_refuses_unstaged_and_untracked_candidates(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts/scan_publication.py", scripts / "scan_publication.py")
    shutil.copy2(ROOT / "scripts/publication_guard.py", scripts / "publication_guard.py")
    readme = tmp_path / "README.md"
    readme.write_text("Synthetic scanner regression fixture.\n")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    command = [sys.executable, str(scripts / "scan_publication.py"), "--gitleaks", "/bin/true"]
    # This fixture's scanner is an explicit mock; production runs require real Gitleaks.
    readme.write_text("Unstaged synthetic change.\n")
    stale = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert stale.returncode != 0
    assert "unstaged changes" in stale.stderr
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    (tmp_path / "new.md").write_text("Untracked candidate.\n")
    partial = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert partial.returncode != 0
    assert "untracked candidates" in partial.stderr
