from __future__ import annotations

import copy
from datetime import UTC, date, datetime

import numpy as np
import pytest
import xarray as xr

from chile_demographic_pde.core.cube import DemographicCube
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.data.exposure import (
    annual_central_exposure,
    central_death_rate,
)
from chile_demographic_pde.data.identity import ReleaseIdFactory


def _population(
    values: list[list[float]] | None = None,
    *,
    years: list[int] | None = None,
) -> xr.DataArray:
    return xr.DataArray(
        values or [[100.0, 102.0, 104.0], [80.0, 82.0, 84.0]],
        dims=("age", "year"),
        coords={"age": [40, 41], "year": years or [2023, 2024, 2025]},
        attrs={
            "semantic_kind": "population",
            "unit": "person",
            "population_basis": "resident_estimate",
            "source_key": "ine_population_base2024",
            "source_sha256": "a" * 64,
            "transformation_ids": ("normalized-official-stock-v1",),
            "nested_provenance": {"vintage": "base2024"},
        },
    )


def _deaths(values: list[list[float]] | None = None) -> xr.DataArray:
    return xr.DataArray(
        values or [[10.0, 20.0], [8.0, 16.0]],
        dims=("age", "year"),
        coords={"age": [40, 41], "year": [2023, 2024]},
        attrs={
            "semantic_kind": "count",
            "unit": "event",
            "population_basis": "resident_estimate",
            "source_key": "deis_deaths",
        },
    )


def _exposure(values: list[list[float]] | None = None) -> xr.DataArray:
    return xr.DataArray(
        values or [[100.0, 200.0], [80.0, 160.0]],
        dims=("age", "year"),
        coords={"age": [40, 41], "year": [2023, 2024]},
        attrs={
            "semantic_kind": "exposure",
            "unit": "person * year",
            "population_basis": "person_time",
            "source_key": "ine_population_base2024",
        },
    )


def _source(
    *,
    name: str,
    sha256: str,
    release_date: date,
) -> SourceRef:
    return SourceRef(
        source_key=name.lower().replace(" ", "_").replace("-", "_"),
        name=name,
        url="https://www.ine.gob.cl/",
        release_date=release_date,
        release_missingness_reason=None,
        retrieved_at=datetime(2026, 7, 19, tzinfo=UTC),
        sha256=sha256,
        vintage="review-fixture",
        provisional=False,
    )


def _provenance(
    *,
    source: SourceRef,
    observation_start: date,
    observation_end: date,
    aggregation_rule: str,
    missingness_reason: str,
    transformation_ids: tuple[str, ...],
) -> VariableProvenance:
    return VariableProvenance(
        sources=(source,),
        release_id="cldemopde:release:v1:" + source.sha256,
        available_at=source.retrieved_at,
        observation_start=observation_start,
        observation_end=observation_end,
        dimensions=("age", "year"),
        aggregation_rules=(aggregation_rule,),
        missingness_reason=missingness_reason,
        transformation_ids=transformation_ids,
        notes=(f"{source.name} provenance note",),
    )


def test_adjacent_midyear_stocks_become_person_time_exposure() -> None:
    exposure = annual_central_exposure(_population(), years=[2023, 2024])

    np.testing.assert_allclose(
        exposure.values,
        np.array([[101.0, 103.0], [81.0, 83.0]]),
    )
    assert exposure.dims == ("age", "year")
    assert exposure.coords["year"].values.tolist() == [2023, 2024]
    assert exposure.attrs["semantic_kind"] == "exposure"
    assert exposure.attrs["unit"] == "person * year"
    assert exposure.attrs["population_basis"] == "person_time"
    assert exposure.attrs["day_count_convention"] == "unit-calendar-year"


def test_exposure_default_uses_only_bounded_adjacent_intervals() -> None:
    exposure = annual_central_exposure(_population())

    assert exposure.coords["year"].values.tolist() == [2023, 2024]
    assert 2025 not in exposure.coords["year"]


def test_exposure_preserves_missing_endpoint_without_imputation() -> None:
    population = _population([[100.0, np.nan, 104.0], [80.0, 82.0, np.nan]])

    exposure = annual_central_exposure(population)

    assert np.isnan(exposure.sel(age=40, year=2023))
    assert np.isnan(exposure.sel(age=40, year=2024))
    assert exposure.sel(age=41, year=2023) == 81.0
    assert np.isnan(exposure.sel(age=41, year=2024))


def test_exposure_does_not_extrapolate_terminal_year() -> None:
    with pytest.raises(ValueError, match="following official stock"):
        annual_central_exposure(_population(), years=[2025])


def test_exposure_rejects_a_series_with_only_one_terminal_stock() -> None:
    population = _population(
        [[100.0], [80.0]],
        years=[2024],
    )

    with pytest.raises(ValueError, match="two"):
        annual_central_exposure(population)


def test_exposure_rejects_missing_intermediate_official_year() -> None:
    population = _population(
        [[100.0, 104.0], [80.0, 84.0]],
        years=[2023, 2025],
    )

    with pytest.raises(ValueError, match="consecutive"):
        annual_central_exposure(population)


@pytest.mark.parametrize("years", [[2024, 2023], [2023, 2023], [2022]])
def test_exposure_rejects_invalid_requested_years(years: list[int]) -> None:
    with pytest.raises(ValueError, match="years"):
        annual_central_exposure(_population(), years=years)


@pytest.mark.parametrize("stock_years", [[2024, 2023, 2025], [2023, 2023, 2024]])
def test_exposure_rejects_nonmonotone_or_duplicate_stock_years(
    stock_years: list[int],
) -> None:
    with pytest.raises(ValueError, match="year"):
        annual_central_exposure(_population(years=stock_years))


def test_exposure_rejects_nonmonotone_numeric_stock_coordinates() -> None:
    population = _population().assign_coords(age=[41, 40])

    with pytest.raises(ValueError, match="ordered"):
        annual_central_exposure(population)


def test_exposure_requires_every_dimension_to_have_an_explicit_unique_coordinate() -> None:
    population = _population().drop_indexes("age").drop_vars("age")

    with pytest.raises(ValueError, match="labeled"):
        annual_central_exposure(population)


def test_exposure_rejects_year_dependent_auxiliary_coordinates() -> None:
    population = _population().assign_coords(
        stock_date=(
            "year",
            np.array(["2023-06-30", "2024-06-30", "2025-06-30"], dtype="datetime64[D]"),
        )
    )

    with pytest.raises(ValueError, match="auxiliary"):
        annual_central_exposure(population)


@pytest.mark.parametrize("bad_value", [-1.0, np.inf])
def test_exposure_rejects_invalid_present_population_stocks(bad_value: float) -> None:
    population = _population()
    population.loc[{"age": 40, "year": 2023}] = bad_value

    with pytest.raises(ValueError, match="population"):
        annual_central_exposure(population)


def test_exposure_rejects_boolean_population_values() -> None:
    population = _population().astype(bool)

    with pytest.raises(ValueError, match="population"):
        annual_central_exposure(population)


@pytest.mark.parametrize(
    ("attribute", "bad_value"),
    [
        ("semantic_kind", "count"),
        ("unit", "event"),
        ("population_basis", "census_enumerated"),
    ],
)
def test_exposure_rejects_incompatible_declared_stock_semantics(
    attribute: str,
    bad_value: str,
) -> None:
    population = _population()
    population.attrs[attribute] = bad_value

    with pytest.raises(ValueError, match=attribute):
        annual_central_exposure(population)


def test_exposure_preserves_provenance_appends_transformation_and_does_not_mutate() -> None:
    population = _population()
    original = population.copy(deep=True)
    original_attrs = copy.deepcopy(population.attrs)

    exposure = annual_central_exposure(population, years=[2023])

    assert population.identical(original)
    assert population.attrs == original_attrs
    assert exposure.attrs["source_key"] == "ine_population_base2024"
    assert exposure.attrs["source_sha256"] == "a" * 64
    assert exposure.attrs["nested_provenance"] == {"vintage": "base2024"}
    assert exposure.attrs["transformation_id"] == ("resident-stock-trapezoidal-exposure-v1")
    assert exposure.attrs["transformation_ids"] == (
        "normalized-official-stock-v1",
        "resident-stock-trapezoidal-exposure-v1",
    )


def test_exposure_updates_canonical_provenance_and_passes_cube_validation() -> None:
    population = _population()
    source = _source(
        name="INE Base-2024 population",
        sha256="a" * 64,
        release_date=date(2026, 1, 28),
    )
    source_provenance = _provenance(
        source=source,
        observation_start=date(2023, 6, 30),
        observation_end=date(2025, 6, 30),
        aggregation_rule="published resident stock at 30 June",
        missingness_reason="source_missing_if_official_stock_absent",
        transformation_ids=("ine-base2024-normalization-v1",),
    )
    population.attrs["provenance"] = source_provenance.model_dump(mode="json")

    exposure = annual_central_exposure(population)
    provenance = VariableProvenance.model_validate(exposure.attrs["provenance"])
    cube = DemographicCube(xr.Dataset({"exposure": exposure}))

    assert cube.dataset["exposure"].identical(exposure)
    assert provenance.sources == (source,)
    assert provenance.release_id == source_provenance.release_id
    assert provenance.available_at == source_provenance.available_at
    assert provenance.observation_start == date(2023, 6, 30)
    assert provenance.observation_end == date(2025, 6, 30)
    assert provenance.dimensions == exposure.dims
    assert provenance.aggregation_rules[0] == ("published resident stock at 30 June")
    assert "trapezoidal" in provenance.aggregation_rules[-1]
    assert provenance.transformation_ids == (
        "ine-base2024-normalization-v1",
        "resident-stock-trapezoidal-exposure-v1",
    )
    assert "source_missing_if_official_stock_absent" in provenance.missingness_reason
    assert "endpoint" in provenance.missingness_reason


def test_central_death_rate_uses_exact_count_over_person_time_alignment() -> None:
    rate = central_death_rate(_deaths(), _exposure())

    np.testing.assert_allclose(rate.values, np.full((2, 2), 0.1))
    assert rate.dims == ("age", "year")
    assert rate.attrs["semantic_kind"] == "rate"
    assert rate.attrs["unit"] == "1 / year"
    assert rate.attrs["population_basis"] == "person_time"
    assert rate.attrs["transformation_id"] == "death-count-central-rate-v1"


def test_central_rate_merges_canonical_provenance_and_passes_cube_validation() -> None:
    population = _population()
    population_source = _source(
        name="INE Base-2024 population",
        sha256="a" * 64,
        release_date=date(2026, 1, 28),
    )
    population.attrs["provenance"] = _provenance(
        source=population_source,
        observation_start=date(2023, 6, 30),
        observation_end=date(2025, 6, 30),
        aggregation_rule="published resident stock at 30 June",
        missingness_reason="population_missing_if_official_stock_absent",
        transformation_ids=("ine-base2024-normalization-v1",),
    ).model_dump(mode="json")
    exposure = annual_central_exposure(population)

    deaths = _deaths()
    death_source = _source(
        name="DEIS deaths",
        sha256="b" * 64,
        release_date=date(2025, 6, 30),
    )
    deaths.attrs["provenance"] = _provenance(
        source=death_source,
        observation_start=date(2023, 1, 1),
        observation_end=date(2024, 12, 31),
        aggregation_rule="sum registered deaths by age and year",
        missingness_reason="deaths_missing_if_official_count_absent",
        transformation_ids=("deis-death-normalization-v1",),
    ).model_dump(mode="json")

    rate = central_death_rate(deaths, exposure)
    provenance = VariableProvenance.model_validate(rate.attrs["provenance"])
    cube = DemographicCube(xr.Dataset({"mortality_rate": rate}))

    assert cube.dataset["mortality_rate"].identical(rate)
    assert provenance.sources == (death_source, population_source)
    expected_release = ReleaseIdFactory.v1(
        (
            (death_source.source_key, death_source.sha256),
            (population_source.source_key, population_source.sha256),
        )
    )
    assert provenance.release_id == expected_release.value
    assert provenance.available_at == max(
        death_source.retrieved_at,
        population_source.retrieved_at,
    )
    assert provenance.observation_start == date(2023, 1, 1)
    assert provenance.observation_end == date(2025, 6, 30)
    assert provenance.dimensions == rate.dims
    assert provenance.aggregation_rules == (
        "sum registered deaths by age and year",
        "published resident stock at 30 June",
        "trapezoidal average of adjacent annual resident stocks with unit-calendar-year intervals",
        "central death rate from exact-aligned death counts and person-time exposure",
    )
    assert provenance.transformation_ids == (
        "deis-death-normalization-v1",
        "ine-base2024-normalization-v1",
        "resident-stock-trapezoidal-exposure-v1",
        "death-count-central-rate-v1",
    )
    assert "deaths_missing_if_official_count_absent" in provenance.missingness_reason
    assert "population_missing_if_official_stock_absent" in (provenance.missingness_reason)
    assert "either input" in provenance.missingness_reason
    assert provenance.notes == (
        "DEIS deaths provenance note",
        "INE Base-2024 population provenance note",
    )


def test_central_rate_rejects_conflicting_duplicate_provenance_source_keys() -> None:
    population = _population()
    population_source = _source(
        name="shared source",
        sha256="a" * 64,
        release_date=date(2026, 1, 28),
    )
    population.attrs["provenance"] = _provenance(
        source=population_source,
        observation_start=date(2023, 6, 30),
        observation_end=date(2025, 6, 30),
        aggregation_rule="published stock",
        missingness_reason="official stock absent",
        transformation_ids=("stock-v1",),
    ).model_dump(mode="json")
    exposure = annual_central_exposure(population)

    deaths = _deaths()
    deaths.attrs["provenance"] = _provenance(
        source=_source(
            name="shared source",
            sha256="b" * 64,
            release_date=date(2025, 6, 30),
        ),
        observation_start=date(2023, 1, 1),
        observation_end=date(2024, 12, 31),
        aggregation_rule="published deaths",
        missingness_reason="official count absent",
        transformation_ids=("deaths-v1",),
    ).model_dump(mode="json")

    with pytest.raises(ValueError, match="conflicting artifacts"):
        central_death_rate(deaths, exposure)


def test_central_death_rate_rejects_coordinate_mismatch_instead_of_intersecting() -> None:
    exposure = _exposure().assign_coords(age=[41, 42])

    with pytest.raises(ValueError, match="exact"):
        central_death_rate(_deaths(), exposure)


def test_central_death_rate_rejects_dimension_order_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        central_death_rate(_deaths(), _exposure().transpose("year", "age"))


def test_central_death_rate_rejects_auxiliary_coordinate_mismatch() -> None:
    deaths = _deaths().assign_coords(age_label=("age", ["40", "41"]))
    exposure = _exposure().assign_coords(age_label=("age", ["age 40", "age 41"]))

    with pytest.raises(ValueError, match="exact"):
        central_death_rate(deaths, exposure)


@pytest.mark.parametrize("bad_value", [0.0, -1.0, np.inf])
def test_central_death_rate_rejects_invalid_present_exposure(
    bad_value: float,
) -> None:
    exposure = _exposure()
    exposure.loc[{"age": 40, "year": 2023}] = bad_value

    with pytest.raises(ValueError, match="exposure"):
        central_death_rate(_deaths(), exposure)


@pytest.mark.parametrize("bad_value", [-1.0, 1.5, np.inf])
def test_central_death_rate_rejects_invalid_present_death_counts(
    bad_value: float,
) -> None:
    deaths = _deaths()
    deaths.loc[{"age": 40, "year": 2023}] = bad_value

    with pytest.raises(ValueError, match="death"):
        central_death_rate(deaths, _exposure())


def test_central_death_rate_rejects_boolean_death_counts() -> None:
    deaths = _deaths().astype(bool)

    with pytest.raises(ValueError, match="death"):
        central_death_rate(deaths, _exposure())


def test_central_death_rate_propagates_missingness_from_either_input() -> None:
    deaths = _deaths([[np.nan, 20.0], [8.0, 16.0]])
    exposure = _exposure([[100.0, np.nan], [80.0, 160.0]])

    rate = central_death_rate(deaths, exposure)

    assert np.isnan(rate.sel(age=40, year=2023))
    assert np.isnan(rate.sel(age=40, year=2024))
    assert rate.sel(age=41, year=2023) == 0.1
    assert rate.sel(age=41, year=2024) == 0.1


@pytest.mark.parametrize(
    ("array_name", "attribute", "bad_value"),
    [
        ("deaths", "semantic_kind", "rate"),
        ("deaths", "unit", "person"),
        ("exposure", "semantic_kind", "population"),
        ("exposure", "unit", "person"),
        ("exposure", "population_basis", "resident_estimate"),
    ],
)
def test_central_death_rate_rejects_incompatible_declared_semantics(
    array_name: str,
    attribute: str,
    bad_value: str,
) -> None:
    deaths = _deaths()
    exposure = _exposure()
    target = deaths if array_name == "deaths" else exposure
    target.attrs[attribute] = bad_value

    with pytest.raises(ValueError, match=attribute):
        central_death_rate(deaths, exposure)


def test_central_death_rate_preserves_inputs() -> None:
    deaths = _deaths()
    exposure = _exposure()
    original_deaths = deaths.copy(deep=True)
    original_exposure = exposure.copy(deep=True)

    central_death_rate(deaths, exposure)

    assert deaths.identical(original_deaths)
    assert exposure.identical(original_exposure)
