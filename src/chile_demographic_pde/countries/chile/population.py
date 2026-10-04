"""Normalization of INE Base-2024 resident-population stocks."""

from __future__ import annotations

import re
from datetime import date, timedelta
from itertools import pairwise
from numbers import Real
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
    _attest_curated_release,
    _finalize_demographic_batch,
)
from chile_demographic_pde.data.age_groups import parse_age_bin
from chile_demographic_pde.data.domain_schemas import DemographicObservationBatch
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.observation import AgeBin

_REQUIRED_COLUMNS: Final = frozenset({"year", "age", "sex", "population"})
_SEX_LABELS: Final = {
    "male": "male",
    "hombre": "male",
    "hombres": "male",
    "female": "female",
    "mujer": "female",
    "mujeres": "female",
    "all": "all",
    "ambos sexos": "all",
    "total": "all",
}
_NUMERIC_LABEL: Final = re.compile(r"^(?P<age>\d+(?:\.0+)?)(?:\s+(?:ano|anos))?$")
_GROUPED_AGE_LABEL: Final = re.compile(
    r"^(?:"
    r"\d+\s+a\s+\d+"
    r"|menores\s+de\s+(?:1|15)(?:\s+(?:ano|anos))?"
    r"|85\s+o\s+mas"
    r"|50\s+y\s+mas"
    r"|100\s+anos\s+y\s+mas"
    r")$"
)


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"INE resident-population frame is missing required columns: {missing!r}.")
    if frame.empty:
        raise ValueError("INE resident-population frame must contain at least one official stock.")


def _integral_years(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float)
    if (
        numeric.isna().any()
        or not np.isfinite(array).all()
        or not np.equal(array, np.floor(array)).all()
    ):
        raise ValueError("INE population year must be a finite integer.")
    years = numeric.astype(int)
    if not years.between(1992, 2070).all():
        raise ValueError("INE Base-2024 population year must lie from 1992 through 2070.")
    return years


def _population_values(values: pd.Series) -> pd.Series:
    if values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError(
            "INE population cannot contain boolean values in place of official stocks."
        )
    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy(dtype=float)
    if numeric.isna().any() or not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError("INE population must contain finite nonnegative official stock values.")
    return numeric.astype(float)


def _parse_single_age(value: object) -> AgeBin:
    if isinstance(value, bool):
        raise ValueError("INE population age must be a nonnegative integral age.")
    if isinstance(value, Real):
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0.0 or numeric != np.floor(numeric):
            raise ValueError("INE population age must be a nonnegative integral age.")
        return AgeBin(numeric, numeric + 1.0, str(value))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("INE population age must be a nonblank age label.")

    label = normalize_spanish_label(value, field="INE population age")
    numeric_label = _NUMERIC_LABEL.fullmatch(label)
    if numeric_label:
        numeric = float(numeric_label.group("age"))
        return AgeBin(numeric, numeric + 1.0, value)
    if _GROUPED_AGE_LABEL.fullmatch(label) is None:
        raise ValueError(f"Invalid INE population age label {value!r}.")
    try:
        age_bin = parse_age_bin(value)
    except (ValueError, ObservationOperatorError) as error:
        raise ValueError(f"Invalid INE population age label {value!r}.") from error
    if age_bin.upper is not None and age_bin.upper - age_bin.lower != 1.0:
        raise ValueError(
            "INE Base-2024 population requires single ages or an explicitly "
            "open-ended terminal age."
        )
    return age_bin


def _normalize_sex(values: pd.Series) -> pd.Series:
    def normalize(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("INE population sex labels must be nonblank strings.")
        canonical = _SEX_LABELS.get(normalize_spanish_label(value, field="INE population sex"))
        if canonical is None:
            raise ValueError(f"Unknown INE population sex label {value!r}.")
        return canonical

    return values.map(normalize)


def _normalize_region(frame: pd.DataFrame) -> pd.Series:
    if "region" not in frame:
        return pd.Series("CL", index=frame.index, dtype=str)
    region = frame["region"]
    valid = region.map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not valid.all():
        raise ValueError(
            "Published region values must be nonblank strings; missing regions are "
            "not replaced by national observations."
        )
    return region.astype(str).str.strip()


def _validate_age_partitions(normalized: pd.DataFrame) -> None:
    group_columns = ["period_start", "sex", "region"]
    for group_key, group in normalized.groupby(group_columns, sort=False, dropna=False):
        ordered = group.sort_values("age_lower", kind="stable")
        rows = list(ordered.itertuples(index=False))
        for index, row in enumerate(rows):
            if bool(row.age_open) and index != len(rows) - 1:
                raise ValueError(
                    "An open INE population age bin must be terminal within each "
                    f"year-sex-region group; invalid group {group_key!r}."
                )
        for left, right in pairwise(rows):
            if bool(left.age_open) or (
                left.age_upper is not None
                and not pd.isna(left.age_upper)
                and float(cast(Real, right.age_lower)) < float(cast(Real, left.age_upper))
            ):
                raise ValueError(
                    f"INE population age bins overlap within year-sex-region group {group_key!r}."
                )


def _primary_provisional(release: CuratedDomainRelease) -> bool:
    primary_key = release.manifest.primary_fact_source_key
    return next(
        source.provisional
        for source in release.provenance.sources
        if source.source_key == primary_key
    )


def normalize_ine_population_curated(
    frame: pd.DataFrame,
    *,
    release: CuratedDomainRelease,
    historical_end_year: int,
    identity_registry: IdentityRegistry,
) -> DemographicObservationBatch:
    """Normalize an attested INE Base-2024 resident-population table."""

    if type(historical_end_year) is not int or not 1992 <= historical_end_year <= 2070:
        raise DataContractError("historical_end_year must be an integer from 1992 through 2070.")
    _attest_curated_release(frame, release)

    _require_columns(frame)
    years = _integral_years(frame["year"])
    population = _population_values(frame["population"])
    age_bins = frame["age"].map(_parse_single_age)
    sex = _normalize_sex(frame["sex"])
    region = _normalize_region(frame)

    period_start = pd.Series(
        (date(int(year), 6, 30) for year in years),
        index=frame.index,
        dtype=object,
    )
    period_end = period_start.map(lambda value: value + timedelta(days=1))
    observation_role = np.where(
        years.le(historical_end_year),
        "observed_fact",
        "external_projection",
    )
    aggregation_rule = np.where(
        years.le(historical_end_year),
        "published INE Base-2024 historical estimate at 30 June by single age and sex",
        "published INE Base-2024 projection at 30 June by single age and sex",
    )
    normalized = pd.DataFrame(
        {
            "domain": "demographic",
            "variable": "population",
            "value": population,
            "semantic_kind": "population",
            "unit": "person",
            "population_basis": "resident_estimate",
            "observation_role": observation_role,
            "parity_scope": "not_applicable",
            "period_start": period_start,
            "period_end": period_end,
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
            "aggregation_rule": aggregation_rule,
            "missingness_reason": "not_applicable_observed",
            "source_status_code": None,
            "age_lower": age_bins.map(lambda age_bin: age_bin.lower),
            "age_upper": pd.Series(
                (age_bin.upper for age_bin in age_bins),
                index=frame.index,
                dtype=object,
            ),
            "age_open": age_bins.map(lambda age_bin: age_bin.upper is None),
            "sex": sex,
            "region": region,
            "cause": "not_applicable",
            "date_basis": "stock_reference_instant",
            "age_role": "resident",
            "age_measure_unit": "completed_years",
            "age_quantity": pd.Series(
                (age_bin.lower if age_bin.upper is not None else None for age_bin in age_bins),
                index=frame.index,
                dtype=object,
            ),
            "sex_role": "resident",
            "commune": None,
            "geography_basis": "habitual_residence",
            "geography_vintage": "ine-base2024",
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
        },
        index=frame.index,
    )
    duplicate_key = [
        "age_lower",
        "age_upper",
        "age_open",
        "period_start",
        "period_end",
        "sex",
        "region",
    ]
    if normalized.duplicated(subset=duplicate_key, keep=False).any():
        raise ValueError("INE population contains duplicate normalized year-age-sex-region keys.")
    _validate_age_partitions(normalized)
    return _finalize_demographic_batch(
        normalized,
        release=release,
        identity_registry=identity_registry,
        source_identity={
            "publisher": "Instituto Nacional de Estadisticas de Chile",
            "dataset": "Base-2024 resident population",
            "contract_id": release.table_contract.contract_id,
        },
    )
