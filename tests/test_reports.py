from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = PROJECT_ROOT / "scripts" / "build_results.py"


def _summary() -> dict[str, Any]:
    return {
        "data": {
            "monthly_periods": 16,
            "censo_population": 18_480_432,
        },
        "calibration": {
            "iterations": 3,
            "converged": True,
            "birth_deviance": 12.3456,
            "death_deviance": 23.4567,
            "birth_rolling_mae": 321.89,
            "death_rolling_mae": 210.12,
        },
        "bayesian_births": {
            "method": "SVI smoke fixture",
            "final_loss": 123.4567,
        },
        "scenarios": {
            "baseline": {
                "population_after_10y": 18_700_000,
                "cumulative_births": 1_450_000,
                "cumulative_deaths": 1_200_000,
            },
            "low_fertility": {
                "population_after_10y": 18_500_000,
                "cumulative_births": 1_250_000,
                "cumulative_deaths": 1_200_000,
            },
        },
    }


def _number(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    assert match is not None
    return float(match.group(1).replace(",", ""))


def test_render_results_uses_model_summary_values() -> None:
    from chile_demographic_pde.reporting import render_results

    summary = _summary()
    report = render_results(summary)

    assert "Model results and interpretation" in report
    assert (
        _number(r"block-coordinate iterations: \*\*(\d+)\*\*", report)
        == (summary["calibration"]["iterations"])
    )
    assert (
        _number(r"births \*\*([0-9.]+)\*\*, deaths", report)
        == summary["calibration"]["birth_deviance"]
    )
    assert (
        _number(r"deaths \*\*([0-9.]+)\*\*\.", report) == summary["calibration"]["death_deviance"]
    )
    assert (
        _number(r"rolling MAE: births \*\*([0-9.]+)\*\*", report)
        == summary["calibration"]["birth_rolling_mae"]
    )
    assert _number(r"baseline\*\*: population after 10 years \*\*([0-9,]+)\*\*", report) == (
        round(summary["scenarios"]["baseline"]["population_after_10y"])
    )
    assert "neither official INE projections" in report


def test_write_results_uses_supplied_summary_and_output_paths(tmp_path: Path) -> None:
    from chile_demographic_pde.reporting import render_results, write_results

    summary_path = tmp_path / "nested" / "model_summary.json"
    output_path = tmp_path / "published" / "RESULTS.md"
    summary_path.parent.mkdir()
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")

    written = write_results(summary_path, output_path)

    assert written == output_path
    assert output_path.read_text(encoding="utf-8") == render_results(_summary())


def test_build_results_script_is_a_thin_package_wrapper() -> None:
    source = BUILDER.read_text(encoding="utf-8")

    assert "from chile_demographic_pde.reporting import write_results" in source
    assert "def render_results" not in source
