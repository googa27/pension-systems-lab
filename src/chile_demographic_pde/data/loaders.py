"""Load the curated official Chilean demographic data bundled with the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True, eq=False)
class ProjectData:
    """All input tables required by the reference model."""

    __hash__ = None  # type: ignore[assignment]

    births_age: pd.DataFrame
    deaths_age: pd.DataFrame
    population_age: pd.DataFrame
    monthly: pd.DataFrame


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame


def load_project_data(directory: Path) -> ProjectData:
    """Load validated, analysis-ready CSV files.

    The CSVs are deterministic public-source extracts from INE and Censo 2024
    source material.  This public copy intentionally bundles only curated CSVs;
    raw Excel/workbook redistribution is unresolved and therefore out of scope.
    Keeping the curated tables in the repository makes smoke checks reproducible
    without network access or spreadsheet engines.
    """

    births = _read_csv(
        directory / "ine_april_2026_births_by_maternal_age.csv",
        {"age_group", "births_total", "period"},
    )
    deaths = _read_csv(
        directory / "ine_april_2026_deaths_by_age_sex.csv",
        {"age_group", "deaths_total", "deaths_male", "deaths_female", "period"},
    )
    population = _read_csv(
        directory / "censo2024_chile_population_by_age_sex.csv",
        {"age_group", "population_total", "population_male", "population_female"},
    )
    monthly = _read_csv(
        directory / "ine_monthly_births_deaths_2025_2026.csv",
        {"period", "births_total", "deaths_total", "source_url"},
    )
    monthly = monthly.sort_values("period", kind="stable").reset_index(drop=True)
    return ProjectData(
        births_age=births,
        deaths_age=deaths,
        population_age=population,
        monthly=monthly,
    )
