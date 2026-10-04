from __future__ import annotations

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src" / "chile_demographic_pde"
PACKAGE = "chile_demographic_pde"


def _architecture() -> dict[str, Any]:
    return json.loads((ROOT / "docs" / "ARCHITECTURE.yaml").read_text())


def _module_name(path: Path) -> str:
    relative = path.relative_to(SRC_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE, *parts)) if parts else PACKAGE


def _source_modules() -> dict[str, Path]:
    return {_module_name(path): path for path in SRC_ROOT.rglob("*.py")}


def _internal_imports(path: Path, modules: dict[str, Path]) -> set[str]:
    tree = ast.parse(path.read_text())
    imports: set[str] = set()

    def add_name(name: str | None) -> None:
        if name is None or not name.startswith(PACKAGE):
            return
        parts = name.split(".")
        while parts:
            candidate = ".".join(parts)
            if candidate in modules:
                imports.add(candidate)
                return
            parts.pop()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_name(alias.name)
        elif isinstance(node, ast.ImportFrom):
            add_name(node.module)
    return imports


def _coarse_fanout(imports: set[str]) -> set[str]:
    fanout = set()
    for name in imports:
        parts = name.split(".")
        fanout.add(".".join(parts[:3]) if len(parts) >= 3 else name)
    return fanout


def test_architecture_yaml_has_audited_exception_metadata() -> None:
    document = _architecture()
    for section, limit_key in (
        ("legacy_size_exceptions", "max_lines"),
        ("legacy_fanout_exceptions", "max_internal_fanout"),
    ):
        assert section in document
        for relative_path, record in document[section].items():
            assert (ROOT / relative_path).is_file(), relative_path
            assert isinstance(record[limit_key], int)
            assert record["owner"]
            assert record["risk"]
            assert record["trigger"]
            assert record["evidence"]


def test_source_file_size_no_growth_except_documented_legacy_modules() -> None:
    document = _architecture()
    default_limit = int(document["max_source_lines_without_exception"])
    exceptions = document["legacy_size_exceptions"]
    for path in SRC_ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT).as_posix()
        lines = len(path.read_text().splitlines())
        allowed = exceptions.get(relative, {}).get("max_lines", default_limit)
        assert lines <= allowed, f"{relative} has {lines} lines; allowed {allowed}"
        if lines > default_limit:
            assert relative in exceptions, f"{relative} needs an audited no-growth exception"


def test_internal_fanout_no_growth_except_documented_legacy_modules() -> None:
    document = _architecture()
    default_limit = int(document["max_internal_fanout_without_exception"])
    exceptions = document["legacy_fanout_exceptions"]
    modules = _source_modules()
    for module, path in modules.items():
        relative = path.relative_to(ROOT).as_posix()
        fanout = _coarse_fanout(_internal_imports(path, modules) - {module})
        allowed = exceptions.get(relative, {}).get("max_internal_fanout", default_limit)
        assert len(fanout) <= allowed, f"{relative} has fanout {len(fanout)}: {sorted(fanout)}"
        if len(fanout) > default_limit:
            assert relative in exceptions, f"{relative} needs an audited fanout exception"


def test_no_internal_import_cycles() -> None:
    modules = _source_modules()
    graph = {
        module: _internal_imports(path, modules) - {module} for module, path in modules.items()
    }
    index = 0
    stack: list[str] = []
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    cycles: list[list[str]] = []

    def strongconnect(module: str) -> None:
        nonlocal index
        indexes[module] = index
        lowlinks[module] = index
        index += 1
        stack.append(module)
        on_stack.add(module)
        for dependency in graph[module]:
            if dependency not in graph:
                continue
            if dependency not in indexes:
                strongconnect(dependency)
                lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
            elif dependency in on_stack:
                lowlinks[module] = min(lowlinks[module], indexes[dependency])
        if lowlinks[module] == indexes[module]:
            component: list[str] = []
            while True:
                dependency = stack.pop()
                on_stack.remove(dependency)
                component.append(dependency)
                if dependency == module:
                    break
            if len(component) > 1:
                cycles.append(sorted(component))

    for module in graph:
        if module not in indexes:
            strongconnect(module)
    assert cycles == []


def test_layer_import_boundaries_and_no_sibling_hacks() -> None:
    document = _architecture()
    modules = _source_modules()
    violations: dict[str, list[str]] = defaultdict(list)
    boundary = document["import_boundaries"]
    allowed_exceptions = {
        (module, imported)
        for module, record in boundary["allowed_boundary_exceptions"].items()
        for imported in record["imports"]
    }
    for module, path in modules.items():
        source = path.read_text()
        assert "sys.path" not in source
        assert "../afp" not in source
        for owner_prefix, forbidden_prefixes in boundary["forbidden_prefixes"].items():
            if not module.startswith(owner_prefix):
                continue
            for imported in _internal_imports(path, modules):
                if (module, imported) in allowed_exceptions:
                    continue
                if any(imported.startswith(prefix) for prefix in forbidden_prefixes):
                    violations[module].append(imported)
    assert dict(violations) == {}
