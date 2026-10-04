"""Maintained human-readable reporting from machine-readable artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _whole_number(value: float | int) -> str:
    return f"{round(float(value)):,.0f}"


def render_results(summary: Mapping[str, Any]) -> str:
    """Return the maintained Markdown report for one model-summary mapping."""

    data = summary["data"]
    calibration = summary["calibration"]
    bayesian = summary["bayesian_births"]
    scenarios = summary["scenarios"]
    baseline_population = float(scenarios["baseline"]["population_after_10y"])
    scenario_lines = []
    for name, values in scenarios.items():
        population = float(values["population_after_10y"])
        difference = population - baseline_population
        scenario_lines.append(
            f"- **{name}**: population after 10 years "
            f"**{_whole_number(population)}** "
            f"({difference:+,.0f} relative to baseline); cumulative births "
            f"**{_whole_number(values['cumulative_births'])}**; cumulative deaths "
            f"**{_whole_number(values['cumulative_deaths'])}**."
        )
    scenarios_markdown = "\n".join(scenario_lines)
    return f"""# Model results and interpretation

## Reproducible fitted results

- Official monthly periods: **{int(data["monthly_periods"])}**; Censo-enumerated population: **{_whole_number(data["censo_population"])}**.
- PDE/statistical block-coordinate iterations: **{int(calibration["iterations"])}**; converged: **{bool(calibration["converged"])}**.
- In-sample Poisson deviance: births **{float(calibration["birth_deviance"]):.4f}**, deaths **{float(calibration["death_deviance"]):.4f}**.
- Leakage-free one-step rolling MAE: births **{float(calibration["birth_rolling_mae"]):.2f}**, deaths **{float(calibration["death_rolling_mae"]):.2f}** registrations per month.
- Bayesian births fit: **{bayesian["method"]}**; final optimization loss **{float(bayesian["final_loss"]):.4f}**.

The small in-sample deviances are not evidence of strong forecasting
performance: there are only {int(data["monthly_periods"])} monthly observations
and the smooth latent time factors are flexible. The rolling-origin errors are
the more honest predictive benchmark.

## Conditional structural interventions

{scenarios_markdown}

These are model interventions with future rate multipliers held fixed and
age-specific migration set to zero. They are neither official INE projections
nor causal estimates.

## Scientific interpretation

1. The age-bin likelihood must integrate rate times exposure; treating grouped counts as point evaluations is wrong.
2. The FEM state equation makes cohort ageing and the birth inflow boundary part of the fit rather than an afterthought.
3. Only one detailed age profile is bundled, so dynamic age-period interactions are not identified; the model uses a fixed age shape times a monthly structural multiplier.
4. Censo-enumerated grouped counts are not resident-population estimates. The optional descriptive interpolation preserves every official total exactly, while its within-bin shape remains grid-dependent.
5. The open 85+ exposure bin cannot identify extreme-age hazard curvature. The nondecreasing mortality-tail constraint and the model-domain truncation at age 105 are explicit assumptions.
6. More age-resolved historical files, exact person-year exposures, and age-specific migration would materially improve identification and forecasts.
"""


def write_results(summary_path: Path, output_path: Path) -> Path:
    """Render and write RESULTS from a model-summary JSON artifact."""

    summary: Mapping[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_results(summary), encoding="utf-8")
    return output_path
