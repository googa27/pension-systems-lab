from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src" / "chile_demographic_pde"
REQUIRED_FILES = [
    "README.md",
    "AGENTS.md",
    "LICENSE",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".github/workflows/ci.yml",
    "scripts/publication_guard.py",
    "scripts/render_readme_assets.py",
    "docs/ECOSYSTEM.md",
    "docs/RESEARCH.md",
    "docs/REGULATION.md",
    "docs/ARCHITECTURE.yaml",
    "docs/ROADMAP.md",
]


def _architecture() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((ROOT / "docs" / "ARCHITECTURE.yaml").read_text()))


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
        length = struct.unpack(">I", stream.read(4))[0]
        chunk = stream.read(4)
        assert length == 13 and chunk == b"IHDR"
        width, height = struct.unpack(">II", stream.read(8))
    return width, height


def test_publication_surface_files_exist() -> None:
    for relative in REQUIRED_FILES:
        assert (ROOT / relative).is_file(), relative
    assert (ROOT / "README.md").read_text().count("https://github.com/googa27/afp-operations") >= 1
    assert "not published on PyPI" in (ROOT / "README.md").read_text()


def test_hero_asset_is_dark_high_resolution_png() -> None:
    document = _architecture()
    asset = ROOT / document["publication_contract"]["hero_asset"]  # type: ignore[index]
    width, height = _png_size(asset)
    assert width >= 2400
    assert height >= 1200
    assert asset.stat().st_size > 100_000


def test_source_directory_file_counts_have_exact_documented_exceptions() -> None:
    document = _architecture()
    limit = int(document["directory_file_count_limit"])
    exceptions = document["directory_file_count_exceptions"]
    for directory in sorted({path.parent for path in SRC_ROOT.rglob("*.py")}):
        relative = directory.relative_to(ROOT).as_posix()
        count = len(list(directory.glob("*.py")))
        allowed = exceptions.get(relative, {}).get("max_files", limit)
        assert count <= allowed, f"{relative} has {count} files; allowed {allowed}"
        if count > limit:
            record = exceptions[relative]
            assert record["owner"]
            assert record["risk"]
            assert record["trigger"]
            assert record["evidence"]


def test_source_modules_obey_500_line_no_growth_contract() -> None:
    document = _architecture()
    limit = int(document["max_source_lines_without_exception"])
    exceptions = document["legacy_size_exceptions"]
    for path in SRC_ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        lines = len(path.read_text().splitlines())
        allowed = exceptions.get(relative, {}).get("max_lines", limit)
        assert lines <= allowed, f"{relative} has {lines} lines; allowed {allowed}"
        if lines > limit:
            assert relative in exceptions


def test_public_docs_do_not_contain_private_discovery_paths() -> None:
    forbidden = [
        "pension-prior-discovery",
        "/home/" + "example",
        "Desktop/" + "Projects",
        ".hermes/profiles",
    ]
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = path.read_text()
        for token in forbidden:
            assert token not in text, f"{path} leaks {token}"
