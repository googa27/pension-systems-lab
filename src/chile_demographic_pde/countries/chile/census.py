"""Normalization of published Censo 2024 grouped population stocks."""

from __future__ import annotations

import re
from datetime import date, timedelta
from itertools import pairwise
from typing import Final, cast

import numpy as np
import pandas as pd

from chile_demographic_pde.core.errors import (
    DataContractError,
    ObservationOperatorError,
)
from chile_demographic_pde.countries.chile._text import normalize_spanish_label
from chile_demographic_pde.countries.chile.curated import (
    CuratedDomainRelease,
    CuratedTableContract,
    _attest_curated_release,
    _finalize_demographic_batch,
    canonical_curated_table_bytes,
)
from chile_demographic_pde.data.age_groups import parse_age_bin
from chile_demographic_pde.data.domain_schemas import DemographicObservationBatch
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.observation import AgeBin

_REFERENCE_START: Final = date(2024, 4, 16)
_REQUIRED_COLUMNS: Final = frozenset({"age_group", "population_male", "population_female"})
_VALUE_TO_SEX: Final = {
    "population_male": "male",
    "population_female": "female",
    "population_total": "all",
}
_CENSO_AGE_LABEL: Final = re.compile(r"^(?:\d+\s+a\s+\d+|85\s+o\s+mas)$")

__all__ = [
    "CuratedDomainRelease",
    "CuratedTableContract",
    "canonical_curated_table_bytes",
    "normalize_censo_grouped_curated",
]


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Censo frame is missing required columns: {missing!r}.")
    if frame.empty:
        raise ValueError("Censo frame must contain at least one published age group.")


def _enumerated_counts(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if any(
        frame[column].map(lambda value: isinstance(value, (bool, np.bool_))).any()
        for column in columns
    ):
        raise ValueError(
            "Censo population columns cannot contain boolean values in place of enumerated counts."
        )
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if (
        numeric.isna().any(axis=None)
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError(
            "Censo population columns must contain finite nonnegative integer enumerated counts."
        )
    return cast(pd.DataFrame, numeric.astype(float))


def _parse_groups(labels: pd.Series) -> list[AgeBin]:
    bins: list[AgeBin] = []
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Censo age_group labels must be nonblank strings.")
        normalized_label = normalize_spanish_label(label, field="Censo age_group")
        if _CENSO_AGE_LABEL.fullmatch(normalized_label) is None:
            raise ValueError(f"Invalid Censo age group {label!r}.")
        try:
            bins.append(parse_age_bin(label))
        except (ValueError, ObservationOperatorError) as error:
            raise ValueError(f"Invalid Censo age group {label!r}.") from error

    ordered = sorted(bins, key=lambda age_bin: age_bin.lower)
    for left, right in pairwise(ordered):
        if left.upper is None or right.lower < left.upper:
            raise ValueError(
                "Censo age groups overlap or duplicate one another; each published "
                "person must belong to at most one group."
            )
    return bins


def _validate_optional_census_year(frame: pd.DataFrame) -> None:
    if "census_year" not in frame:
        return
    year = pd.to_numeric(frame["census_year"], errors="coerce")
    values = year.to_numpy(dtype=float)
    if (
        year.isna().any()
        or not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or not year.eq(_REFERENCE_START.year).all()
    ):
        raise ValueError(
            "census_year must be the integral year 2024, consistent with the "
            "documented Censo reference date."
        )


def _primary_provisional(release: CuratedDomainRelease) -> bool:
    primary_key = release.manifest.primary_fact_source_key
    return next(
        source.provisional
        for source in release.provenance.sources
        if source.source_key == primary_key
    )


def normalize_censo_grouped_curated(
    frame: pd.DataFrame,
    *,
    release: CuratedDomainRelease,
    reference_date: date,
    identity_registry: IdentityRegistry,
) -> DemographicObservationBatch:
    """Normalize an attested Censo table without reconstructing single ages."""

    if type(reference_date) is not date or reference_date != _REFERENCE_START:
        raise DataContractError(
            "reference_date must equal the reviewed Censo 2024 reference date 2024-04-16."
        )
    _attest_curated_release(frame, release)

    _require_columns(frame)
    _validate_optional_census_year(frame)
    bins = _parse_groups(frame["age_group"])
    value_columns = ["population_male", "population_female"]
    if "population_total" in frame:
        value_columns.append("population_total")
    counts = _enumerated_counts(frame, value_columns)

    if "population_total" in counts and not counts["population_total"].equals(
        counts["population_male"] + counts["population_female"]
    ):
        raise ValueError(
            "Published Censo total must equal the sum of its male and female counts "
            "for every age group."
        )

    parsed = pd.DataFrame(
        {
            "age_lower": [age_bin.lower for age_bin in bins],
            "age_upper": [age_bin.upper for age_bin in bins],
            "age_open": [age_bin.upper is None for age_bin in bins],
        },
        index=frame.index,
    )
    wide = pd.concat([parsed, counts], axis=1)
    long = wide.melt(
        id_vars=["age_lower", "age_upper", "age_open"],
        value_vars=value_columns,
        var_name="_published_sex_column",
        value_name="value",
    )
    sex = long.pop("_published_sex_column").map(_VALUE_TO_SEX)
    aggregation = np.where(
        sex.eq("all"),
        "published Censo 2024 grouped enumerated stock by age, all sexes",
        "published Censo 2024 grouped enumerated stock by age and sex",
    )
    period_end = reference_date + timedelta(days=1)
    normalized = pd.DataFrame(
        {
            "domain": "demographic",
            "variable": "population",
            "value": long["value"],
            "semantic_kind": "population",
            "unit": "person",
            "population_basis": "census_enumerated",
            "observation_role": "observed_fact",
            "parity_scope": "not_applicable",
            "period_start": pd.Series(
                [reference_date] * len(long),
                index=long.index,
                dtype=object,
            ),
            "period_end": pd.Series(
                [period_end] * len(long),
                index=long.index,
                dtype=object,
            ),
            "source_key": release.manifest.primary_fact_source_key,
            "vintage": release.vintage,
            "released_at": release.manifest.primary_fact_asset.released_at,
            "release_missingness_reason": (
                None
                if release.manifest.primary_fact_asset.release_missingness_reason is None
                else release.manifest.primary_fact_asset.release_missingness_reason.value
            ),
            "available_at": release.manifest.available_at,
            "provisional": _primary_provisional(release),
            "transformation_id": None,
            "transformation_version": None,
            "aggregation_rule": aggregation,
            "missingness_reason": "not_applicable_observed",
            "source_status_code": None,
            "age_lower": long["age_lower"],
            "age_upper": pd.Series(
                (
                    None if pd.isna(value) else float(cast(float, value))
                    for value in long["age_upper"]
                ),
                index=long.index,
                dtype=object,
            ),
            "age_open": long["age_open"],
            "sex": sex,
            "region": "CL",
            "cause": "not_applicable",
            "date_basis": "census_reference_instant",
            "age_role": "enumerated_habitual_resident",
            "age_measure_unit": "grouped_completed_years",
            "age_quantity": None,
            "sex_role": "enumerated_habitual_resident",
            "commune": None,
            "geography_basis": "habitual_residence",
            "geography_vintage": "censo2024",
            "cause_code_system": None,
            "cause_role": None,
            "external_cause": None,
            "registration_start": None,
            "registration_end": None,
            "birth_order": None,
            "nationality": None,
            "nationality_role": None,
            "country_of_birth": None,
            "education_level": None,
            "residence_five_years_ago": None,
            "migration_status": None,
        }
    )
    return _finalize_demographic_batch(
        normalized,
        release=release,
        identity_registry=identity_registry,
        source_identity={
            "publisher": "Instituto Nacional de Estadisticas de Chile",
            "dataset": "Censo 2024 grouped population",
            "contract_id": release.table_contract.contract_id,
        },
    )
