from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD_PATH = ROOT / "scripts" / "publication_guard.py"


def _guard_module():
    spec = importlib.util.spec_from_file_location("publication_guard_under_test", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_scan_accepts_clean_markdown(tmp_path: Path) -> None:
    guard = _guard_module()
    clean = tmp_path / "README.md"
    clean.write_text("# Clean\n\nSynthetic public fixture.\n")
    assert guard.scan_paths([clean], tmp_path) == []


def test_scan_rejects_credentials_and_raw_exports(tmp_path: Path) -> None:
    guard = _guard_module()
    secret = tmp_path / "config.md"
    secret.write_text("token = " + "x" * 16 + "\n")
    workbook = tmp_path / "raw.xlsx"
    workbook.write_bytes(b"placeholder")
    findings = guard.scan_paths([secret, workbook], tmp_path)
    rules = {finding.rule for finding in findings}
    assert "private-text" in rules
    assert "forbidden-suffix" in rules


def test_scan_rejects_private_paths_without_literal_fixture_leak(tmp_path: Path) -> None:
    guard = _guard_module()
    note = tmp_path / "note.md"
    note.write_text("bad path: " + "/home/" + "example/private.txt")
    findings = guard.scan_paths([note], tmp_path)
    assert any(finding.rule == "private-text" for finding in findings)


def test_git_mode_refuses_nested_parent_git(tmp_path: Path) -> None:
    guard = _guard_module()
    parent = tmp_path / "parent"
    child = parent / "child"
    (parent / ".git").mkdir(parents=True)
    child.mkdir()
    try:
        guard.git_publication_paths(child)
    except RuntimeError as exc:
        assert "nested under parent Git root" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected nested parent Git refusal")
