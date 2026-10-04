"""Age-bin harmonization utilities for INE and Censo labels."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from chile_demographic_pde.data.observation import AgeBin

FloatArray = NDArray[np.float64]


def _normalize(label: str) -> str:
    text = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.lower().strip().split())


def parse_age_bin(label: str) -> AgeBin:
    """Parse the Spanish age labels used by the curated INE/Censo tables."""

    text = _normalize(label)
    if text.startswith("menores de 1"):
        return AgeBin(0.0, 1.0, label)
    if text.startswith("menores de 15"):
        return AgeBin(0.0, 15.0, label)
    if "85 o mas" in text:
        return AgeBin(85.0, None, label)
    if "50 y mas" in text:
        return AgeBin(50.0, None, label)
    if "100 anos y mas" in text or text.startswith("100"):
        return AgeBin(100.0, None, label)
    match = re.search(r"(\d+)\s+a\s+(\d+)", text)
    if match is None:
        raise ValueError(f"Cannot parse age group: {label!r}")
    lower, inclusive_upper = (float(value) for value in match.groups())
    return AgeBin(lower, inclusive_upper + 1.0, label)


def censo_age_bins(population: pd.DataFrame) -> list[AgeBin]:
    return [parse_age_bin(label) for label in population["age_group"]]


def fertility_age_bins(births: pd.DataFrame) -> tuple[list[AgeBin], pd.DataFrame]:
    selected = births[births["age_group"].str.match(r"^(15|20|25|30|35|40|45) a")].copy()
    selected = selected.reset_index(drop=True)
    return [parse_age_bin(label) for label in selected["age_group"]], selected


def aggregate_deaths_to_population_bins(
    deaths: pd.DataFrame,
    population_bins: list[AgeBin],
    column: str,
    maximum_age: float = 105.0,
) -> FloatArray:
    result = np.zeros(len(population_bins), dtype=float)
    for _, row in deaths.iterrows():
        source = parse_age_bin(str(row["age_group"]))
        midpoint = 0.5 * (source.lower + source.resolved_upper(maximum_age))
        candidates = [
            index
            for index, target in enumerate(population_bins)
            if target.lower <= midpoint < target.resolved_upper(maximum_age)
        ]
        if len(candidates) != 1:
            raise ValueError(f"Could not map death bin {source} uniquely")
        result[candidates[0]] += float(row[column])
    return result
