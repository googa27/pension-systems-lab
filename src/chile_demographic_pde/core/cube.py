from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import xarray as xr

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.core.provenance import VariableProvenance
from chile_demographic_pde.core.semantics import (
    PopulationBasis,
    SemanticKind,
    Unit,
    assert_semantic_compatible,
)

REQUIRED_ATTRS = ("semantic_kind", "unit", "population_basis", "provenance")


def _validate_variable_metadata(name: Hashable, variable: xr.DataArray) -> None:
    missing = [key for key in REQUIRED_ATTRS if key not in variable.attrs]
    if missing:
        raise DataContractError(f"Variable {name!r} is missing attributes {missing!r}.")

    try:
        kind = SemanticKind(variable.attrs["semantic_kind"])
        unit = Unit(variable.attrs["unit"])
        population_basis = PopulationBasis(variable.attrs["population_basis"])
        assert_semantic_compatible(kind, unit, population_basis)
        provenance = VariableProvenance.model_validate(variable.attrs["provenance"])
        if provenance.dimensions != variable.dims:
            raise ValueError(
                "provenance dimensions must exactly equal the variable's ordered "
                f"dimensions: {provenance.dimensions!r} != {variable.dims!r}"
            )
    except (DataContractError, TypeError, ValueError) as error:
        raise DataContractError(f"Variable {name!r} has invalid metadata: {error}") from error


def _coordinates_on_dimensions(
    array: xr.Dataset | xr.DataArray,
    dimensions: set[Hashable],
) -> set[Hashable]:
    return {
        name
        for name, coordinate in array.coords.items()
        if dimensions.intersection(coordinate.dims)
    }


def _coordinates_identical(
    left: xr.DataArray,
    right: xr.DataArray,
) -> bool:
    left_coordinate = xr.DataArray(left.variable, name=left.name)
    right_coordinate = xr.DataArray(right.variable, name=right.name)
    return left_coordinate.identical(right_coordinate)


def _scalar_coordinate_names(
    array: xr.Dataset | xr.DataArray,
) -> set[Hashable]:
    return {name for name, coordinate in array.coords.items() if not coordinate.dims}


def _assert_exact_assignment_alignment(
    dataset: xr.Dataset,
    data: xr.DataArray,
) -> None:
    dataset_scalars = _scalar_coordinate_names(dataset)
    data_scalars = _scalar_coordinate_names(data)
    if dataset_scalars != data_scalars:
        cube_only = sorted(dataset_scalars.difference(data_scalars), key=repr)
        incoming_only = sorted(data_scalars.difference(dataset_scalars), key=repr)
        raise DataContractError(
            "Scalar coordinate names differ between the cube and incoming "
            f"variable: cube-only {cube_only!r}, incoming-only {incoming_only!r}."
        )
    for coordinate_name in sorted(dataset_scalars, key=repr):
        if not _coordinates_identical(
            dataset.coords[coordinate_name],
            data.coords[coordinate_name],
        ):
            raise DataContractError(
                f"Scalar coordinate {coordinate_name!r} differs between the cube "
                "and incoming variable."
            )

    shared_dimensions = set(dataset.dims).intersection(data.dims)
    shared_coordinates = set(dataset.coords).intersection(data.coords).difference(dataset_scalars)

    for coordinate_name in sorted(shared_coordinates, key=repr):
        if not _coordinates_identical(
            dataset.coords[coordinate_name],
            data.coords[coordinate_name],
        ):
            raise DataContractError(
                f"Shared coordinate {coordinate_name!r} differs between the cube "
                "and incoming variable."
            )

    for dimension in sorted(shared_dimensions, key=repr):
        if dataset.sizes[dimension] != data.sizes[dimension]:
            raise DataContractError(
                f"Dimension {dimension!r} has different sizes in the cube and incoming variable."
            )
        dataset_has_labels = dimension in dataset.coords
        data_has_labels = dimension in data.coords
        if dataset_has_labels != data_has_labels:
            raise DataContractError(
                f"Coordinate labels for shared dimension {dimension!r} must be "
                "present on both the cube and incoming variable."
            )
        if dataset_has_labels and not _coordinates_identical(
            dataset.coords[dimension],
            data.coords[dimension],
        ):
            raise DataContractError(f"Coordinate labels differ on shared dimension {dimension!r}.")

    dataset_coordinates = _coordinates_on_dimensions(dataset, shared_dimensions)
    data_coordinates = _coordinates_on_dimensions(data, shared_dimensions)
    if dataset_coordinates != data_coordinates:
        raise DataContractError(
            "Coordinates on shared dimensions differ between the cube and incoming variable."
        )
    for coordinate_name in sorted(dataset_coordinates, key=repr):
        if not _coordinates_identical(
            dataset.coords[coordinate_name],
            data.coords[coordinate_name],
        ):
            raise DataContractError(
                f"Shared coordinate {coordinate_name!r} differs between the cube "
                "and incoming variable."
            )

    for new_dimension in set(data.dims).difference(dataset.dims):
        if new_dimension in dataset.variables:
            raise DataContractError(
                f"New dimension {new_dimension!r} collides with an existing variable."
            )
    for colliding_coordinate in set(data.coords).intersection(dataset.data_vars):
        raise DataContractError(
            f"Incoming coordinate {colliding_coordinate!r} collides with an existing data variable."
        )


@dataclass(frozen=True, slots=True, eq=False)
class DemographicCube:
    """Canonical labeled demographic dataset with typed variable metadata."""

    __hash__ = None  # type: ignore[assignment]

    dataset: xr.Dataset

    def __post_init__(self) -> None:
        for name, variable in self.dataset.data_vars.items():
            _validate_variable_metadata(name, variable)

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.dataset.identical(other.dataset)

    def __getitem__(
        self,
        selection: dict[str, Any] | str,
    ) -> xr.Dataset | xr.DataArray:
        if isinstance(selection, str):
            return self.dataset[selection]
        return self.dataset.sel(selection)

    def with_variable(
        self,
        name: str,
        data: xr.DataArray,
        *,
        kind: SemanticKind | str,
        unit: Unit | str,
        population_basis: PopulationBasis | str,
        provenance: VariableProvenance | Mapping[str, Any],
    ) -> DemographicCube:
        if not isinstance(name, str) or not name.strip():
            raise DataContractError("Variable name must be a nonblank string.")
        if name in self.dataset.coords or name in data.coords:
            raise DataContractError(f"Variable name {name!r} collides with a coordinate.")

        try:
            normalized_kind = SemanticKind(kind)
            normalized_unit = Unit(unit)
            normalized_basis = PopulationBasis(population_basis)
            assert_semantic_compatible(
                normalized_kind,
                normalized_unit,
                normalized_basis,
            )
            normalized_provenance = VariableProvenance.model_validate(provenance)
            if normalized_provenance.dimensions != data.dims:
                raise ValueError(
                    "provenance dimensions must exactly equal the variable's ordered "
                    f"dimensions: {normalized_provenance.dimensions!r} != {data.dims!r}"
                )
        except (DataContractError, TypeError, ValueError) as error:
            raise DataContractError(f"Variable {name!r} has invalid metadata: {error}") from error

        _assert_exact_assignment_alignment(self.dataset, data)
        variable = data.copy(deep=True)
        variable.attrs.update(
            semantic_kind=normalized_kind.value,
            unit=normalized_unit.value,
            population_basis=normalized_basis.value,
            provenance=normalized_provenance.model_dump(mode="json"),
        )
        return DemographicCube(self.dataset.assign({name: variable}))
