from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.countries.chile.census import (
    CuratedDomainRelease,
    CuratedTableContract,
    canonical_curated_table_bytes,
    normalize_censo_grouped_curated,
)
from chile_demographic_pde.countries.chile.population import (
    normalize_ine_population_curated,
)
from chile_demographic_pde.data.domain_schemas import (
    DemographicObservationSchema,
    validate_demographic_observations,
)
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CENSO_REFERENCE_DATE = date(2024, 4, 16)


def _censo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age_group": ["0 a 4", "85 o más"],
            "population_male": [100, 10],
            "population_female": [105, 15],
        }
    )


def _resident_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2024, 2024],
            "age": [0, 1],
            "sex": ["Hombres", "Mujeres"],
            "population": [102.0, 107.0],
        }
    )


def _logical_types(frame: pd.DataFrame) -> tuple[str, ...]:
    integer_columns = {
        "census_year",
        "population_female",
        "population_male",
        "population_total",
        "year",
    }
    float_columns = {"population"}
    string_columns = {"age_group", "region", "sex"}
    logical_types: list[str] = []
    for column, dtype in frame.dtypes.items():
        if column in integer_columns:
            logical_types.append("integer")
        elif column in float_columns:
            logical_types.append("float64")
        elif column in string_columns:
            logical_types.append("string")
        elif column == "age" and pd.api.types.is_integer_dtype(dtype):
            logical_types.append("integer")
        elif column == "age" and pd.api.types.is_float_dtype(dtype):
            logical_types.append("float64")
        elif column == "age":
            logical_types.append("string")
        elif pd.api.types.is_datetime64tz_dtype(dtype):
            logical_types.append("timestamp_utc")
        else:
            raise AssertionError(f"Test fixture has no logical type for {column!r}.")
    return tuple(logical_types)


def _release(
    frame: pd.DataFrame,
    *,
    source_key: str,
    released_at: date,
    vintage: str,
    provisional: bool,
    observation_start: date,
    observation_end: date,
) -> tuple[CuratedDomainRelease, IdentityRegistry]:
    registry = IdentityRegistry()
    contract = CuratedTableContract(
        contract_id=f"{source_key}-curated-v1",
        columns=tuple(frame.columns),
        logical_types=_logical_types(frame),
    )
    digest = hashlib.sha256(canonical_curated_table_bytes(frame, contract)).hexdigest()
    available_at = datetime(2026, 7, 18, 12, tzinfo=UTC)
    manifest = ReleaseManifest.create(
        schema_version="population-adapter-test-v1",
        assets=(
            RequiredReleaseAsset(
                source_key=source_key,
                sha256=digest,
                available_at=available_at,
                released_at=released_at,
                release_missingness_reason=None,
            ),
        ),
        primary_fact_source_key=source_key,
        identity_registry=registry,
    )
    provenance = VariableProvenance(
        sources=(
            SourceRef(
                source_key=source_key,
                name=source_key,
                url="https://example.invalid/curated",
                release_date=released_at,
                release_missingness_reason=None,
                retrieved_at=available_at,
                sha256=digest,
                vintage=vintage,
                provisional=provisional,
            ),
        ),
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=observation_start,
        observation_end=observation_end,
        dimensions=("period", "age", "sex", "geography"),
        aggregation_rules=("published population stock",),
        missingness_reason="not_applicable_observed",
    )
    return (
        CuratedDomainRelease(
            manifest=manifest,
            provenance=provenance,
            vintage=vintage,
            curated_source_key=source_key,
            table_contract=contract,
        ),
        registry,
    )


def _normalize_censo(frame: pd.DataFrame, **metadata: object) -> pd.DataFrame:
    vintage = str(metadata.pop("vintage", "2024-final"))
    provisional = bool(metadata.pop("provisional", False))
    released_at = metadata.pop("released_at", date(2025, 3, 27))
    source_key = str(metadata.pop("source_key", "censo2024_grouped"))
    assert not metadata
    if isinstance(released_at, str):
        released_at = date.fromisoformat(released_at)
    assert type(released_at) is date
    release, registry = _release(
        frame,
        source_key=source_key,
        released_at=released_at,
        vintage=vintage,
        provisional=provisional,
        observation_start=CENSO_REFERENCE_DATE,
        observation_end=date(2024, 4, 17),
    )
    return normalize_censo_grouped_curated(
        frame,
        release=release,
        reference_date=CENSO_REFERENCE_DATE,
        identity_registry=registry,
    ).observations


def _normalize_resident(frame: pd.DataFrame, **metadata: object) -> pd.DataFrame:
    vintage = str(metadata.pop("vintage", "base2024"))
    provisional = bool(metadata.pop("provisional", False))
    released_at = metadata.pop("released_at", date(2026, 1, 28))
    source_key = str(metadata.pop("source_key", "ine_population_base2024"))
    historical_end_year = int(metadata.pop("historical_end_year", 2024))
    assert not metadata
    if isinstance(released_at, str):
        released_at = date.fromisoformat(released_at)
    assert type(released_at) is date
    release, registry = _release(
        frame,
        source_key=source_key,
        released_at=released_at,
        vintage=vintage,
        provisional=provisional,
        observation_start=date(1992, 6, 30),
        observation_end=date(2070, 7, 1),
    )
    return normalize_ine_population_curated(
        frame,
        release=release,
        historical_end_year=historical_end_year,
        identity_registry=registry,
    ).observations


def test_censo_and_resident_estimates_keep_distinct_stock_bases() -> None:
    grouped = _normalize_censo(_censo_frame())
    estimates = _normalize_resident(_resident_frame())

    assert set(grouped["population_basis"]) == {"census_enumerated"}
    assert set(estimates["population_basis"]) == {"resident_estimate"}
    assert set(grouped["semantic_kind"]) == {"population"}
    assert set(estimates["semantic_kind"]) == {"population"}


def test_censo_uses_documented_reference_instant_and_preserves_open_age() -> None:
    grouped = _normalize_censo(_censo_frame())

    assert grouped["period_start"].unique().tolist() == [date(2024, 4, 16)]
    assert grouped["period_end"].unique().tolist() == [date(2024, 4, 17)]
    open_rows = grouped.loc[grouped["age_lower"].eq(85.0)]
    assert open_rows["age_open"].all()
    assert open_rows["age_upper"].isna().all()
    closed_rows = grouped.loc[grouped["age_lower"].eq(0.0)]
    assert not closed_rows["age_open"].any()
    assert closed_rows["age_upper"].eq(5.0).all()


def test_resident_estimates_use_30_june_reference_instant() -> None:
    estimates = _normalize_resident(_resident_frame())

    assert estimates["period_start"].unique().tolist() == [date(2024, 6, 30)]
    assert estimates["period_end"].unique().tolist() == [date(2024, 7, 1)]


def test_censo_declares_enumerated_roles_and_reference_bases() -> None:
    grouped = _normalize_censo(_censo_frame())

    assert set(grouped["date_basis"]) == {"census_reference_instant"}
    assert set(grouped["age_role"]) == {"enumerated_habitual_resident"}
    assert set(grouped["age_measure_unit"]) == {"grouped_completed_years"}
    assert grouped["age_quantity"].isna().all()
    assert set(grouped["sex_role"]) == {"enumerated_habitual_resident"}
    assert set(grouped["geography_basis"]) == {"habitual_residence"}
    assert set(grouped["geography_vintage"]) == {"censo2024"}


def test_ine_population_declares_resident_roles_reference_bases_and_age() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024, 2024],
            "age": ["0", "100 años y más"],
            "sex": ["Hombres", "Mujeres"],
            "population": [102.0, 8.0],
        }
    )

    estimates = _normalize_resident(frame)

    assert set(estimates["date_basis"]) == {"stock_reference_instant"}
    assert set(estimates["age_role"]) == {"resident"}
    assert set(estimates["age_measure_unit"]) == {"completed_years"}
    assert estimates["age_quantity"].tolist()[0] == 0.0
    assert pd.isna(estimates["age_quantity"].tolist()[1])
    assert set(estimates["sex_role"]) == {"resident"}
    assert set(estimates["geography_basis"]) == {"habitual_residence"}
    assert set(estimates["geography_vintage"]) == {"ine-base2024"}


def test_adapters_return_strict_schema_frames_without_mutating_inputs() -> None:
    censo = _censo_frame()
    resident = _resident_frame()
    censo_original = censo.copy(deep=True)
    resident_original = resident.copy(deep=True)

    grouped = _normalize_censo(censo)
    estimates = _normalize_resident(resident)

    pd.testing.assert_frame_equal(censo, censo_original)
    pd.testing.assert_frame_equal(resident, resident_original)
    strict_columns = tuple(DemographicObservationSchema.to_schema().columns)
    assert tuple(grouped.columns) == strict_columns
    assert tuple(estimates.columns) == strict_columns
    pd.testing.assert_frame_equal(validate_demographic_observations(grouped), grouped)
    pd.testing.assert_frame_equal(validate_demographic_observations(estimates), estimates)
    assert grouped["transformation_id"].isna().all()
    assert estimates["transformation_id"].isna().all()
    unrelated = [
        "commune",
        "cause_code_system",
        "cause_role",
        "external_cause",
        "registration_start",
        "registration_end",
        "birth_order",
        "nationality",
        "nationality_role",
        "country_of_birth",
        "education_level",
        "residence_five_years_ago",
        "migration_status",
    ]
    assert grouped.loc[:, unrelated].isna().all(axis=None)
    assert estimates.loc[:, unrelated].isna().all(axis=None)
    foreign = {
        "currency",
        "facility",
        "station",
        "quote_type",
        "pollutant",
    }
    assert foreign.isdisjoint(grouped.columns)
    assert foreign.isdisjoint(estimates.columns)
    assert set(grouped["missingness_reason"]) == {"not_applicable_observed"}
    assert set(estimates["missingness_reason"]) == {"not_applicable_observed"}


def test_censo_expands_published_total_and_preserves_exact_group_totals() -> None:
    frame = _censo_frame().assign(population_total=[205, 25])

    grouped = _normalize_censo(frame)

    assert set(grouped["sex"]) == {"male", "female", "all"}
    totals = grouped.pivot(index="age_lower", columns="sex", values="value")
    np.testing.assert_array_equal(totals["all"], totals["male"] + totals["female"])


def test_official_censo_fixture_preserves_18_480_432_grand_total() -> None:
    fixture = pd.read_csv(PROJECT_ROOT / "data/processed/censo2024_chile_population_by_age_sex.csv")

    grouped = _normalize_censo(fixture)

    published_total = grouped.loc[grouped["sex"].eq("all"), "value"].sum()
    assert published_total == 18_480_432
    assert grouped.loc[grouped["sex"].eq("male"), "value"].sum() == 8_967_033
    assert grouped.loc[grouped["sex"].eq("female"), "value"].sum() == 9_513_399


def test_censo_accepts_unicode_case_and_spacing_age_variants() -> None:
    frame = _censo_frame()
    frame["age_group"] = ["  0   A   4 ", "85 O M\u00c1S"]

    grouped = _normalize_censo(frame)

    assert set(grouped["age_lower"]) == {0.0, 85.0}
    assert grouped.loc[grouped["age_lower"].eq(85), "age_open"].all()


def test_censo_rejects_unsupported_unicode_in_age_labels() -> None:
    frame = _censo_frame()
    frame.loc[0, "age_group"] = "0 a 4😀"

    with pytest.raises(ValueError, match="age"):
        _normalize_censo(frame)


@pytest.mark.parametrize(
    "age_groups",
    [
        ["not an age", "85 o más"],
        ["garbage 0 a 4", "85 o más"],
        ["0 a 4", "85 o más garbage"],
        ["0 a 4", "0 a 4"],
        ["0 a 9", "5 a 14"],
        ["85 o más", "90 a 94"],
    ],
)
def test_censo_rejects_malformed_duplicate_or_overlapping_age_groups(
    age_groups: list[str],
) -> None:
    frame = _censo_frame()
    frame["age_group"] = age_groups

    with pytest.raises(ValueError, match="age"):
        _normalize_censo(frame)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("population_male", True),
        ("population_male", "not-numeric"),
        ("population_male", np.inf),
        ("population_female", -1),
        ("population_female", 1.5),
    ],
)
def test_censo_rejects_invalid_enumerated_counts(column: str, bad_value: object) -> None:
    frame = _censo_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = bad_value

    with pytest.raises((ValueError, DataContractError), match="population"):
        _normalize_censo(frame)


def test_censo_rejects_inconsistent_published_total() -> None:
    frame = _censo_frame().assign(population_total=[204, 25])

    with pytest.raises(ValueError, match="total"):
        _normalize_censo(frame)


def test_censo_rejects_raw_census_year_inconsistent_with_reference_date() -> None:
    frame = _censo_frame().assign(census_year=[2024, 2023])

    with pytest.raises(ValueError, match="census_year"):
        _normalize_censo(frame)


@pytest.mark.parametrize(
    "missing_column",
    ["age_group", "population_male", "population_female"],
)
def test_censo_rejects_missing_required_raw_columns(missing_column: str) -> None:
    with pytest.raises(ValueError, match="required"):
        _normalize_censo(_censo_frame().drop(columns=missing_column))


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("male", "male"),
        ("HOMBRE", "male"),
        ("Hombres", "male"),
        ("female", "female"),
        ("MUJER", "female"),
        ("Mujeres", "female"),
        ("all", "all"),
        ("Ambos sexos", "all"),
        ("Total", "all"),
    ],
)
def test_resident_adapter_normalizes_documented_sex_labels(
    raw_label: str,
    expected: str,
) -> None:
    frame = _resident_frame().iloc[[0]].reset_index(drop=True)
    frame["sex"] = raw_label

    estimates = _normalize_resident(frame)

    assert estimates.loc[estimates.index[0], "sex"] == expected


def test_resident_adapter_rejects_unknown_sex_labels() -> None:
    frame = _resident_frame()
    frame.loc[0, "sex"] = "unknown"

    with pytest.raises(ValueError, match="sex"):
        _normalize_resident(frame)


def test_resident_adapter_rejects_unsupported_unicode_in_sex_labels() -> None:
    frame = _resident_frame()
    frame.loc[0, "sex"] = "male😀"

    with pytest.raises(ValueError, match="sex"):
        _normalize_resident(frame)


def test_resident_adapter_preserves_published_region_and_does_not_fill_missing() -> None:
    frame = _resident_frame().assign(region=[" CL-13 ", "CL-08"])

    estimates = _normalize_resident(frame)

    assert estimates["region"].tolist() == ["CL-13", "CL-08"]
    frame.loc[0, "region"] = None
    with pytest.raises(ValueError, match="region"):
        _normalize_resident(frame)


def test_resident_adapter_marks_national_series_only_when_region_column_absent() -> None:
    estimates = _normalize_resident(_resident_frame())

    assert set(estimates["region"]) == {"CL"}


def test_resident_adapter_preserves_explicit_open_terminal_age() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024, 2024],
            "age": ["99", "100 años y más"],
            "sex": ["male", "female"],
            "population": [7.0, 8.0],
        }
    )

    estimates = _normalize_resident(frame)

    closed = estimates.loc[estimates["age_lower"].eq(99.0)].iloc[0]
    opened = estimates.loc[estimates["age_lower"].eq(100.0)].iloc[0]
    assert closed["age_upper"] == 100.0
    assert not bool(closed["age_open"])
    assert pd.isna(opened["age_upper"])
    assert bool(opened["age_open"])


def test_resident_adapter_does_not_guess_numeric_terminal_age_is_open() -> None:
    frame = _resident_frame().iloc[[0]].reset_index(drop=True)
    frame["age"] = 100

    estimates = _normalize_resident(frame)

    assert estimates.iloc[0]["age_upper"] == 101.0
    assert not bool(estimates.iloc[0]["age_open"])


def test_resident_adapter_does_not_guess_ordinary_100_year_label_is_open() -> None:
    frame = _resident_frame().iloc[[0]].reset_index(drop=True)
    frame["age"] = "100 años"

    estimates = _normalize_resident(frame)

    assert estimates.iloc[0]["age_upper"] == 101.0
    assert not bool(estimates.iloc[0]["age_open"])


def test_resident_adapter_rejects_an_open_age_bin_overlapped_in_same_group() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024, 2024],
            "age": ["50 y más", "60"],
            "sex": ["male", "male"],
            "population": [100.0, 20.0],
            "region": ["CL", "CL"],
        }
    )

    with pytest.raises(ValueError, match=r"open|overlap"):
        _normalize_resident(frame)


def test_resident_age_bin_validation_is_independent_by_year_sex_and_region() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024, 2024, 2024, 2025],
            "age": ["50 y más", "60", "60", "60"],
            "sex": ["male", "female", "male", "male"],
            "population": [100.0, 20.0, 30.0, 40.0],
            "region": ["CL-13", "CL-13", "CL-08", "CL-13"],
        }
    )

    estimates = _normalize_resident(frame)

    assert len(estimates) == 4
    assert estimates["age_open"].sum() == 1


def test_resident_estimates_and_projections_are_distinguished_without_provisional_abuse() -> None:
    frame = pd.DataFrame(
        {
            "year": [2024, 2025],
            "age": [40, 40],
            "sex": ["male", "male"],
            "population": [100.0, 101.0],
        }
    )

    estimates = _normalize_resident(frame)

    years = estimates["period_start"].map(lambda value: value.year)
    historical = estimates.loc[years.eq(2024)].iloc[0]
    projected = estimates.loc[years.eq(2025)].iloc[0]
    assert "historical estimate" in historical["aggregation_rule"]
    assert "projection" in projected["aggregation_rule"]
    assert historical["observation_role"] == "observed_fact"
    assert projected["observation_role"] == "external_projection"
    assert not estimates["provisional"].any()


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("year", 2024.5, "year"),
        ("year", "not-a-year", "year"),
        ("age", -1, "age"),
        ("age", 2.5, "age"),
        ("age", "garbage 0 a 0", "age"),
        ("population", "not-numeric", "population"),
        ("population", True, "population"),
        ("population", np.inf, "population"),
        ("population", -1, "population"),
    ],
)
def test_resident_adapter_rejects_invalid_values(
    column: str,
    bad_value: object,
    message: str,
) -> None:
    frame = _resident_frame()
    values = frame[column].astype(object)
    values.iloc[0] = bad_value
    frame[column] = values

    with pytest.raises((ValueError, DataContractError), match=message):
        _normalize_resident(frame)


@pytest.mark.parametrize("missing_column", ["year", "age", "sex", "population"])
def test_resident_adapter_rejects_missing_required_raw_columns(
    missing_column: str,
) -> None:
    with pytest.raises(ValueError, match="required"):
        _normalize_resident(_resident_frame().drop(columns=missing_column))


def test_resident_adapter_rejects_duplicate_normalized_keys() -> None:
    frame = _resident_frame().iloc[[0, 0]].reset_index(drop=True)

    with pytest.raises(ValueError, match="duplicate"):
        _normalize_resident(frame)


def test_adapter_metadata_is_explicit_and_raw_normalization_is_not_a_transformation() -> None:
    estimates = _normalize_resident(
        _resident_frame(),
        vintage="base2024-reviewed",
        provisional=True,
    )

    assert set(estimates["source_key"]) == {"ine_population_base2024"}
    assert set(estimates["vintage"]) == {"base2024-reviewed"}
    assert estimates["provisional"].all()
    assert estimates["transformation_id"].isna().all()
