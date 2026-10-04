from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
PENSION_LAB = ROOT / ".venv" / "bin" / "pension-lab"
PENSION_LAB_COMMAND = PENSION_LAB.as_posix() if PENSION_LAB.exists() else "pension-lab"


def _run_pension_lab(command: str) -> dict[str, Any]:
    completed = subprocess.run(
        [PENSION_LAB_COMMAND, command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_console_script_capabilities_and_demo() -> None:
    capabilities = _run_pension_lab("capabilities")
    assert capabilities["cli"] == "pension-lab"
    assert "demo" in capabilities["commands"]

    demo = _run_pension_lab("demo")
    assert demo["scenario"] == "small_two_sex_fem_smoke"
    assert demo["nonnegative"] is True


def test_console_script_pension_demo_is_public_optional_integration() -> None:
    payload = _run_pension_lab("pension-demo")
    assert payload["dependency"] == "afp_operations"
    assert payload["status"] in {
        "ok",
        "optional_dependency_not_installed",
        "missing_public_demo",
    }
    assert "sibling" not in json.dumps(payload).lower()


def test_installed_afp_operations_demo_through_public_cli_contract() -> None:
    if importlib.util.find_spec("afp_operations") is None:
        pytest.skip("optional public afp_operations package is not installed")

    payload = _run_pension_lab("pension-demo")

    assert payload["dependency"] == "afp_operations"
    assert payload["status"] == "ok"
    assert payload["entrypoint"] in {
        "afp_operations.public_demo",
        "afp_operations.demo",
        "afp_operations.run_demo",
    }
    assert "result" in payload
    assert "sibling" not in json.dumps(payload).lower()
