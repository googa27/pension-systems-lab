from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any

import xarray as xr

from chile_demographic_pde.core.errors import (
    PopulationBasisError,
    SurfaceAlignmentError,
    UnitMismatchError,
)
from chile_demographic_pde.core.semantics import (
    UNIT_REGISTRY,
    PopulationBasis,
    Unit,
    assert_unit_compatible,
)


def _assert_exact_alignment(left: xr.DataArray, right: xr.DataArray) -> None:
    if left.dims != right.dims:
        raise SurfaceAlignmentError(f"Surface dimensions differ: {left.dims!r} and {right.dims!r}.")
    if set(left.coords) != set(right.coords):
        raise SurfaceAlignmentError("Surface coordinates differ.")
    for dimension in left.dims:
        if not left.get_index(dimension).equals(right.get_index(dimension)):
            raise SurfaceAlignmentError(f"Surface coordinates differ on dimension {dimension!r}.")
    for coordinate_name in left.coords:
        left_coordinate = left.coords[coordinate_name]
        right_coordinate = right.coords[coordinate_name]
        if left_coordinate.dims != right_coordinate.dims or not left_coordinate.equals(
            right_coordinate
        ):
            raise SurfaceAlignmentError(
                f"Surface coordinates differ for coordinate {coordinate_name!r}."
            )


def _assert_same_basis(left: PopulationBasis, right: PopulationBasis) -> None:
    if left is not right:
        raise PopulationBasisError(f"Population bases differ: {left.value!r} and {right.value!r}.")


class _SurfaceValue:
    """Shared value equality for immutable surface wrappers."""

    __slots__ = ()
    __hash__ = None  # type: ignore[assignment]

    data: xr.DataArray
    unit: Unit
    population_basis: PopulationBasis

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.unit is other.unit
            and self.population_basis is other.population_basis
            and self.data.identical(other.data)
        )


@dataclass(frozen=True, slots=True, eq=False)
class ExpectedCountSurface(_SurfaceValue):
    """Expected event counts over a labeled demographic surface."""

    data: xr.DataArray
    unit: Unit = Unit.PERSON
    population_basis: PopulationBasis = PopulationBasis.RESIDENT_ESTIMATE

    def __post_init__(self) -> None:
        if self.unit is not Unit.PERSON:
            raise UnitMismatchError("ExpectedCountSurface requires unit 'person'.")


@dataclass(frozen=True, slots=True, eq=False)
class ExposureSurface(_SurfaceValue):
    """Person-time exposure over a labeled demographic surface."""

    data: xr.DataArray
    unit: Unit
    population_basis: PopulationBasis

    def __post_init__(self) -> None:
        if self.unit is not Unit.PERSON_YEAR:
            raise UnitMismatchError("ExposureSurface requires unit 'person * year'.")


@dataclass(frozen=True, slots=True, eq=False)
class RateSurface(_SurfaceValue):
    """Continuous reciprocal-time rate over a labeled demographic surface."""

    data: xr.DataArray
    unit: Unit
    population_basis: PopulationBasis

    def __post_init__(self) -> None:
        if self.unit not in {Unit.PER_YEAR, Unit.PER_MONTH}:
            raise UnitMismatchError("RateSurface requires a reciprocal-time unit.")

    def __call__(self, **coordinates: Any) -> xr.DataArray:
        unknown = set(coordinates).difference(self.data.dims)
        if unknown:
            raise KeyError(f"Unknown interpolation dimensions: {sorted(unknown)}")
        return self.data.interp(coordinates)

    def __add__(self, other: object) -> RateSurface:
        if not isinstance(other, RateSurface):
            return NotImplemented
        _assert_exact_alignment(self.data, other.data)
        _assert_same_basis(self.population_basis, other.population_basis)
        assert_unit_compatible(self.unit, other.unit)
        if self.unit is not other.unit:
            raise UnitMismatchError(
                "Rate addition requires identical units; convert explicitly first."
            )
        return RateSurface(
            self.data + other.data,
            unit=self.unit,
            population_basis=self.population_basis,
        )

    def __mul__(self, other: object) -> RateSurface | ExpectedCountSurface:
        if isinstance(other, Real):
            return RateSurface(
                self.data * float(other),
                unit=self.unit,
                population_basis=self.population_basis,
            )
        if not isinstance(other, ExposureSurface):
            return NotImplemented
        _assert_exact_alignment(self.data, other.data)
        _assert_same_basis(self.population_basis, other.population_basis)
        if self.unit is not Unit.PER_YEAR:
            raise UnitMismatchError(
                "Rate-exposure multiplication requires identical time units; "
                "convert explicitly first."
            )
        product = UNIT_REGISTRY.parse_units(self.unit.value) * UNIT_REGISTRY.parse_units(
            other.unit.value
        )
        target = UNIT_REGISTRY.parse_units(Unit.PERSON.value)
        if product.dimensionality != target.dimensionality:
            raise UnitMismatchError(
                f"{self.unit.value!r} times {other.unit.value!r} is not a count."
            )
        return ExpectedCountSurface(
            self.data * other.data,
            population_basis=self.population_basis,
        )

    def __rmul__(self, other: object) -> RateSurface:
        if not isinstance(other, Real):
            return NotImplemented
        return RateSurface(
            self.data * float(other),
            unit=self.unit,
            population_basis=self.population_basis,
        )
