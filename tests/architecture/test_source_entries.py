"""Constitution fitness gate: runtime modules plus immediate package directories."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_source_entries_are_bounded_and_legacy_exceptions_do_not_grow() -> None:
    contract = json.loads((ROOT / "docs/ARCHITECTURE.yaml").read_text())
    source = ROOT / "src"
    exceptions = contract["source_entry_exceptions"]
    directories = [
        source,
        *(p for p in source.rglob("*") if p.is_dir() and p.name != "__pycache__"),
    ]
    for directory in directories:
        entries = [
            p
            for p in directory.iterdir()
            if (p.is_file() and p.suffix == ".py" and p.name != "__init__.py")
            or (p.is_dir() and (p / "__init__.py").is_file())
        ]
        relative = directory.relative_to(ROOT).as_posix()
        exception = exceptions.get(relative)
        ceiling = contract["source_entry_limit"] if exception is None else exception["max_entries"]
        assert len(entries) <= ceiling, (relative, len(entries), ceiling)
        if exception is not None:
            for field in ("owner", "reason", "risk", "refactoring_trigger", "evidence"):
                assert exception[field], (relative, field)
    assert set(exceptions).issubset({p.relative_to(ROOT).as_posix() for p in directories})
