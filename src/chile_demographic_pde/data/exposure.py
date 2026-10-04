"""Explicit transformations from official stocks to exposures and central rates."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from itertools import pairwise
from numbers import Integral, Real
from typing import Final, cast

import numpy as np
import xarray as xr

from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.data.identity import ReleaseIdFactory

_EXPOSURE_TRANSFORMATION: Final = "resident-stock-trapezoidal-exposure-v1"
_RATE_TRANSFORMATION: Final = "death-count-central-rate-v1"
_EXPOSURE_AGGREGATION_RULE: Final = (
    "trapezoidal average of adjacent annual resident stocks with unit-calendar-year intervals"
)
_RATE_AGGREGATION_RULE: Final = (
    "central death rate from exact-aligned death counts and person-time exposure"
)


def _string_dimensions(array: xr.DataArray) -> tuple[str, ...]:
    if any(not isinstance(dimension, str) for dimension in array.dims):
        raise ValueError("Demographic array dimension names must be strings.")
    return cast(tuple[str, ...], array.dims)


def _require_data_array(array: xr.DataArray, *, name: str) -> None:
    if not isinstance(array, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray.")
    if not array.dims:
        raise ValueError(f"{name} must have at least one labeled dimension.")
    _string_dimensions(array)
    for dimension in array.dims:
        if dimension not in array.coords or dimension not in array.indexes:
            raise ValueError(
                f"{name} requires an explicit labeled coordinate for dimension {dimension!r}."
            )
        coordinate = array.coords[dimension]
        if coordinate.dims != (dimension,) or not array.get_index(dimension).is_unique:
            raise ValueError(
                f"{name} dimension {dimension!r} requires a unique one-dimensional "
                "labeled coordinate."
            )


def _numeric_values(array: xr.DataArray, *, name: str) -> np.ndarray:
    values = np.asarray(array.values)
    if np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} values must be numeric and cannot be boolean.")
    return values.astype(float, copy=False)


def _reject_time_dependent_auxiliary_coordinates(
    array: xr.DataArray,
    *,
    time_dimension: str,
) -> None:
    conflicting = sorted(
        str(name)
        for name, coordinate in array.coords.items()
        if name not in array.dims and time_dimension in coordinate.dims
    )
    if conflicting:
        raise ValueError(
            "Population stock arrays cannot carry year-dependent auxiliary "
            f"coordinates {conflicting!r}; encode stock provenance in attributes "
            "and interval identity in the year dimension coordinate."
        )


def _require_ordered_numeric_dimension_coordinates(array: xr.DataArray) -> None:
    for dimension in array.dims:
        values = np.asarray(array.coords[dimension].values)
        if np.issubdtype(values.dtype, np.number) and not np.issubdtype(values.dtype, np.bool_):
            numeric = values.astype(float, copy=False)
            if (
                not np.isfinite(numeric).all()
                or numeric.ndim != 1
                or (np.diff(numeric) <= 0.0).any()
            ):
                raise ValueError(
                    f"Numeric stock coordinate {dimension!r} must be finite, "
                    "strictly ordered, and unique."
                )


def _require_exact_coordinates(left: xr.DataArray, right: xr.DataArray) -> None:
    if set(left.coords) != set(right.coords):
        raise ValueError(
            "Deaths and exposure require exactly the same dimension and auxiliary coordinates."
        )
    for name in left.coords:
        if not left.coords[name].equals(right.coords[name]):
            raise ValueError(
                "Deaths and exposure require exact coordinate alignment, including "
                f"auxiliary coordinate {name!r}."
            )


def _validate_declared_semantics(
    array: xr.DataArray,
    *,
    name: str,
    expected: dict[str, str],
) -> None:
    for attribute, expected_value in expected.items():
        actual = array.attrs.get(attribute)
        if actual is not None and actual != expected_value:
            raise ValueError(
                f"{name} attribute {attribute!r} must be {expected_value!r}; got {actual!r}."
            )


def _transformation_history(
    attrs: dict[str, object],
    transformation_id: str,
) -> tuple[str, ...]:
    existing = attrs.get("transformation_ids", ())
    history: tuple[str, ...]
    if isinstance(existing, str):
        history = (existing,)
    elif isinstance(existing, (tuple, list)) and all(
        isinstance(item, str) and item for item in existing
    ):
        history = tuple(existing)
    elif existing in (None, ()):
        history = ()
    else:
        raise ValueError("transformation_ids provenance must be a sequence of strings.")
    if transformation_id in history:
        return history
    return (*history, transformation_id)


def _unique_strings(*groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def _unique_sources(
    *groups: Sequence[SourceRef],
) -> tuple[SourceRef, ...]:
    sources: list[SourceRef] = []
    seen: dict[str, SourceRef] = {}
    for group in groups:
        for source in group:
            existing = seen.get(source.source_key)
            if existing is not None and existing != source:
                raise ValueError(
                    "Derived provenance cannot combine conflicting artifacts "
                    f"under source key {source.source_key!r}."
                )
            if existing is None:
                seen[source.source_key] = source
                sources.append(source)
    return tuple(sources)


def _canonical_provenance(
    array: xr.DataArray,
    *,
    name: str,
) -> VariableProvenance | None:
    payload = array.attrs.get("provenance")
    if payload is None:
        return None
    try:
        provenance = VariableProvenance.model_validate(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} has invalid canonical provenance: {error}") from error
    if provenance.dimensions != array.dims:
        raise ValueError(
            f"{name} canonical provenance dimensions must exactly equal "
            f"{array.dims!r}; got {provenance.dimensions!r}."
        )
    return provenance


def _exposure_provenance(
    provenance: VariableProvenance,
    *,
    dimensions: tuple[str, ...],
) -> VariableProvenance:
    return VariableProvenance(
        sources=provenance.sources,
        release_id=provenance.release_id,
        available_at=provenance.available_at,
        observation_start=provenance.observation_start,
        observation_end=provenance.observation_end,
        dimensions=dimensions,
        aggregation_rules=_unique_strings(
            provenance.aggregation_rules, (_EXPOSURE_AGGREGATION_RULE,)
        ),
        missingness_reason=(
            f"source={provenance.missingness_reason}; derived exposure missing if "
            "either adjacent resident-stock endpoint is missing"
        ),
        transformation_ids=_unique_strings(
            provenance.transformation_ids, (_EXPOSURE_TRANSFORMATION,)
        ),
        notes=provenance.notes,
    )


def _rate_provenance(
    deaths: VariableProvenance,
    exposure: VariableProvenance,
    *,
    dimensions: tuple[str, ...],
) -> VariableProvenance:
    sources = _unique_sources(deaths.sources, exposure.sources)
    release_id = ReleaseIdFactory.v1(
        tuple(sorted((source.source_key, source.sha256) for source in sources))
    )
    return VariableProvenance(
        sources=sources,
        release_id=release_id.value,
        available_at=max(deaths.available_at, exposure.available_at),
        observation_start=min(deaths.observation_start, exposure.observation_start),
        observation_end=max(deaths.observation_end, exposure.observation_end),
        dimensions=dimensions,
        aggregation_rules=_unique_strings(
            deaths.aggregation_rules,
            exposure.aggregation_rules,
            (_RATE_AGGREGATION_RULE,),
        ),
        missingness_reason=(
            f"deaths={deaths.missingness_reason}; "
            f"exposure={exposure.missingness_reason}; derived rate missing if "
            "either input is missing"
        ),
        transformation_ids=_unique_strings(
            deaths.transformation_ids,
            exposure.transformation_ids,
            (_RATE_TRANSFORMATION,),
        ),
        notes=_unique_strings(deaths.notes, exposure.notes),
    )


def _integer_years(coordinate: xr.DataArray) -> list[int]:
    years: list[int] = []
    for value in coordinate.values.tolist():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Population year coordinates must be finite integers.")
        numeric = float(value)
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValueError("Population year coordinates must be finite integers.")
        years.append(int(numeric))
    if years != sorted(years) or len(years) != len(set(years)):
        raise ValueError("Population year coordinates must be strictly increasing and unique.")
    return years


def _requested_years(
    available: list[int],
    requested: Sequence[int] | None,
) -> list[int]:
    if requested is None:
        selected = available[:-1]
    else:
        selected = []
        for value in requested:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("Requested exposure years must be integers.")
            selected.append(int(value))
        if not selected:
            raise ValueError("Requested exposure years must not be empty.")
        if selected != sorted(selected) or len(selected) != len(set(selected)):
            raise ValueError("Requested exposure years must be strictly increasing and unique.")
        if any(year not in available for year in selected):
            raise ValueError("Requested exposure years must be official stock years.")
    if any(year + 1 not in available for year in selected):
        raise ValueError(
            "Every requested exposure year requires the following official stock year; "
            "terminal stocks are never extrapolated."
        )
    return selected


def annual_central_exposure(
    population: xr.DataArray,
    *,
    years: Sequence[int] | None = None,
) -> xr.DataArray:
    """Integrate adjacent annual resident stocks with the trapezoidal rule."""

    _require_data_array(population, name="population")
    if "year" not in population.dims:
        raise ValueError("population dimensions must include the labeled 'year' dimension.")
    _require_ordered_numeric_dimension_coordinates(population)
    _reject_time_dependent_auxiliary_coordinates(population, time_dimension="year")
    _validate_declared_semantics(
        population,
        name="population",
        expected={
            "semantic_kind": "population",
            "unit": "person",
            "population_basis": "resident_estimate",
        },
    )
    canonical_provenance = _canonical_provenance(population, name="population")
    values = _numeric_values(population, name="population")
    present = ~np.isnan(values)
    if not np.isfinite(values[present]).all() or (values[present] < 0.0).any():
        raise ValueError(
            "Present population stocks must be finite and nonnegative; missing "
            "official endpoints must be NaN."
        )

    available = _integer_years(population.coords["year"])
    if len(available) < 2:
        raise ValueError("Annual central exposure requires at least two official stock years.")
    if any(right != left + 1 for left, right in pairwise(available)):
        raise ValueError("Annual central exposure requires consecutive official stock years.")
    selected = _requested_years(available, years)

    left = population.sel(year=selected).copy(deep=True)
    right = population.sel(year=[year + 1 for year in selected]).copy(deep=True)
    right = right.assign_coords(year=selected)
    try:
        left, right = xr.align(left, right, join="exact", copy=True)
    except ValueError as error:
        raise ValueError(
            "Adjacent population stocks must have exact non-time coordinate alignment."
        ) from error
    exposure_values = 0.5 * (
        _numeric_values(left, name="population") + _numeric_values(right, name="population")
    )
    exposure = xr.DataArray(
        exposure_values,
        dims=left.dims,
        coords=left.coords,
        name="exposure",
    )
    attrs: dict[str, object] = deepcopy(population.attrs)
    attrs.update(
        {
            "semantic_kind": "exposure",
            "unit": "person * year",
            "population_basis": "person_time",
            "day_count_convention": "unit-calendar-year",
            "transformation_id": _EXPOSURE_TRANSFORMATION,
        }
    )
    attrs["transformation_ids"] = _transformation_history(attrs, _EXPOSURE_TRANSFORMATION)
    if canonical_provenance is not None:
        derived_provenance = _exposure_provenance(
            canonical_provenance, dimensions=_string_dimensions(exposure)
        )
        attrs["provenance"] = derived_provenance.model_dump(mode="json")
        attrs["transformation_ids"] = derived_provenance.transformation_ids
    exposure.attrs = attrs
    return exposure


def central_death_rate(
    deaths: xr.DataArray,
    exposure: xr.DataArray,
) -> xr.DataArray:
    """Compute central death rates only after exact labeled alignment checks."""

    _require_data_array(deaths, name="deaths")
    _require_data_array(exposure, name="exposure")
    if deaths.dims != exposure.dims:
        raise ValueError(
            "Deaths and exposure dimensions must have exactly the same names and order."
        )
    _require_exact_coordinates(deaths, exposure)
    _validate_declared_semantics(
        deaths,
        name="deaths",
        expected={"semantic_kind": "count", "unit": "event"},
    )
    death_provenance = _canonical_provenance(deaths, name="deaths")
    exposure_provenance = _canonical_provenance(exposure, name="exposure")
    if (death_provenance is None) != (exposure_provenance is None):
        raise ValueError(
            "Deaths and exposure must either both declare canonical provenance "
            "or both omit it; partial provenance cannot produce an authoritative rate."
        )
    _validate_declared_semantics(
        exposure,
        name="exposure",
        expected={
            "semantic_kind": "exposure",
            "unit": "person * year",
            "population_basis": "person_time",
        },
    )
    try:
        aligned_deaths, aligned_exposure = xr.align(deaths, exposure, join="exact", copy=True)
    except ValueError as error:
        raise ValueError(
            "Deaths and exposure require exact coordinate alignment; coordinates "
            "are never intersected or reordered."
        ) from error

    death_values = _numeric_values(aligned_deaths, name="death")
    death_present = ~np.isnan(death_values)
    if (
        not np.isfinite(death_values[death_present]).all()
        or (death_values[death_present] < 0.0).any()
        or not np.equal(death_values[death_present], np.floor(death_values[death_present])).all()
    ):
        raise ValueError(
            "Present death counts must be finite nonnegative integers; missing "
            "official counts must be NaN."
        )
    exposure_values = _numeric_values(aligned_exposure, name="exposure")
    exposure_present = ~np.isnan(exposure_values)
    if (
        not np.isfinite(exposure_values[exposure_present]).all()
        or (exposure_values[exposure_present] <= 0.0).any()
    ):
        raise ValueError(
            "Present exposure values must be finite and strictly positive; missing "
            "official exposures must be NaN."
        )

    rate_values = death_values / exposure_values
    rate = xr.DataArray(
        rate_values,
        dims=aligned_deaths.dims,
        coords=aligned_deaths.coords,
        name="mortality_rate",
    )
    attrs: dict[str, object] = deepcopy(exposure.attrs)
    source_keys = tuple(
        dict.fromkeys(
            source_key
            for source_key in (
                deaths.attrs.get("source_key"),
                exposure.attrs.get("source_key"),
            )
            if isinstance(source_key, str) and source_key
        )
    )
    if source_keys:
        attrs["source_keys"] = source_keys
    attrs.update(
        {
            "semantic_kind": "rate",
            "unit": "1 / year",
            "population_basis": "person_time",
            "transformation_id": _RATE_TRANSFORMATION,
        }
    )
    attrs["transformation_ids"] = _transformation_history(attrs, _RATE_TRANSFORMATION)
    if death_provenance is not None and exposure_provenance is not None:
        merged_provenance = _rate_provenance(
            death_provenance,
            exposure_provenance,
            dimensions=_string_dimensions(rate),
        )
        attrs["provenance"] = merged_provenance.model_dump(mode="json")
        attrs["transformation_ids"] = merged_provenance.transformation_ids
    rate.attrs = attrs
    return rate
