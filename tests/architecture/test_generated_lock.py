from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_generated_lock_size_exception_is_scoped_bounded_and_documented() -> None:
    architecture = json.loads((ROOT / "docs/ARCHITECTURE.yaml").read_text())
    exceptions = architecture["publication_file_exceptions"]
    assert set(exceptions) == {"uv.lock"}
    for relative, record in exceptions.items():
        assert (ROOT / relative).stat().st_size <= record["max_kib"] * 1024
        for field in ("owner", "reason", "risk", "refactoring_trigger", "verification"):
            assert record[field]
    config = (ROOT / ".pre-commit-config.yaml").read_text()
    assert "exclude: ^uv\\.lock$" in config
    assert "https://github.com/gitleaks/gitleaks" in config
