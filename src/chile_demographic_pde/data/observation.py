from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from typing import cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from chile_demographic_pde.core.errors import ObservationOperatorError
from chile_demographic_pde.core.surfaces import RateSurface

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AgeBin:
    lower: float
    upper: float | None
    label: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or self.lower < 0.0:
            raise ObservationOperatorError("Age-bin lower endpoint must be finite and nonnegative.")
        if self.upper is not None and (not np.isfinite(self.upper) or self.upper <= self.lower):
            raise ObservationOperatorError("Age-bin upper endpoint must exceed its lower endpoint.")

    def __contains__(self, age: object) -> bool:
        if isinstance(age, bool) or not isinstance(age, Real):
            return False
        numeric_age = float(age)
        if not np.isfinite(numeric_age):
            return False
        return self.lower <= numeric_age and (self.upper is None or numeric_age < self.upper)

    def resolved_upper(self, open_bin_end: float | None) -> float:
        if self.upper is not None:
            return self.upper
        if open_bin_end is not None and not np.isfinite(open_bin_end):
            raise ObservationOperatorError("Open-bin domain end must be finite.")
        if open_bin_end is None or open_bin_end <= self.lower:
            raise ObservationOperatorError(
                f"Age bin {self.label!r} is open-ended; supply a valid open_bin_end."
            )
        return open_bin_end


@dataclass(frozen=True, slots=True, eq=False)
class ObservationOperator:
    matrix: xr.DataArray
    input_dimension: str
    output_dimension: str
    output_labels: tuple[str, ...]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, xr.DataArray):
            raise ObservationOperatorError("Operator matrix must be an xarray DataArray.")
        if self.input_dimension == self.output_dimension:
            raise ObservationOperatorError("Operator dimensions must be distinct.")
        expected_dimensions = (self.output_dimension, self.input_dimension)
        if self.matrix.ndim != 2 or self.matrix.dims != expected_dimensions:
            raise ObservationOperatorError(
                "Operator matrix dimensions must be exactly "
                f"{expected_dimensions!r}; got {self.matrix.dims!r}."
            )
        for dimension in expected_dimensions:
            if dimension not in self.matrix.coords or dimension not in self.matrix.indexes:
                raise ObservationOperatorError(
                    f"Operator dimension {dimension!r} requires a labeled coordinate index."
                )
            coordinate = self.matrix.coords[dimension]
            if coordinate.dims != (dimension,):
                raise ObservationOperatorError(
                    f"Operator coordinate {dimension!r} must index only its own dimension."
                )
            if not self.matrix.get_index(dimension).is_unique:
                raise ObservationOperatorError(
                    f"Operator coordinate index {dimension!r} must be unique."
                )
        values = np.asarray(self.matrix.values)
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise ObservationOperatorError("Operator matrix values must be finite numbers.")
        actual_labels = tuple(self.matrix.get_index(self.output_dimension).tolist())
        if actual_labels != self.output_labels:
            raise ObservationOperatorError(
                "Operator output coordinate labels must exactly match output_labels."
            )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        assert isinstance(other, ObservationOperator)
        return (
            self.input_dimension == other.input_dimension
            and self.output_dimension == other.output_dimension
            and self.output_labels == other.output_labels
            and self.matrix.identical(other.matrix)
        )

    def __matmul__(self, other: object) -> xr.DataArray | ObservationOperator:
        if isinstance(other, RateSurface):
            if self.input_dimension not in other.data.dims:
                raise ObservationOperatorError(
                    f"Surface lacks input dimension {self.input_dimension!r}."
                )
            shared_dimensions = set(self.matrix.dims).intersection(other.data.dims)
            if shared_dimensions != {self.input_dimension}:
                raise ObservationOperatorError(
                    "Operator and surface shared dimensions must be exactly "
                    f"the input dimension {self.input_dimension!r}; got "
                    f"{shared_dimensions!r}."
                )
            expected = self.matrix.get_index(self.input_dimension)
            actual = other.data.get_index(self.input_dimension)
            if not expected.equals(actual):
                raise ObservationOperatorError("Operator and surface coordinates do not match.")
            return cast(xr.DataArray, xr.dot(self.matrix, other.data, dim=self.input_dimension))
        if isinstance(other, ObservationOperator):
            if self.input_dimension != other.output_dimension:
                raise ObservationOperatorError(
                    "Operator composition requires matching intermediate dimensions."
                )
            shared_dimensions = set(self.matrix.dims).intersection(other.matrix.dims)
            if shared_dimensions != {self.input_dimension}:
                raise ObservationOperatorError(
                    "Operator shared dimensions must be exactly the contracted "
                    f"intermediate dimension {self.input_dimension!r}; got "
                    f"{shared_dimensions!r}."
                )
            expected = self.matrix.get_index(self.input_dimension)
            actual = other.matrix.get_index(other.output_dimension)
            if not expected.equals(actual):
                raise ObservationOperatorError("Operator composition coordinates do not match.")
            composed = cast(
                xr.DataArray,
                xr.dot(
                    self.matrix,
                    other.matrix,
                    dim=self.input_dimension,
                ),
            )
            return ObservationOperator(
                matrix=composed,
                input_dimension=other.input_dimension,
                output_dimension=self.output_dimension,
                output_labels=self.output_labels,
            )
        return NotImplemented


def p1_bin_integral_operator(
    nodes: ArrayLike,
    bins: list[AgeBin],
    *,
    open_bin_end: float | None = None,
) -> ObservationOperator:
    grid = np.asarray(nodes, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ObservationOperatorError("Age nodes must be a strictly increasing vector.")
    if not np.all(np.isfinite(grid)):
        raise ObservationOperatorError("Age nodes must contain only finite values.")
    if np.any(np.diff(grid) <= 0.0):
        raise ObservationOperatorError("Age nodes must be a strictly increasing vector.")
    matrix = np.zeros((len(bins), grid.size), dtype=float)
    for row, age_bin in enumerate(bins):
        upper_bound = age_bin.resolved_upper(open_bin_end)
        if age_bin.lower < grid[0] or upper_bound > grid[-1]:
            raise ObservationOperatorError("Every resolved bin must lie inside the age grid.")
        for left_index, (x0, x1) in enumerate(pairwise(grid)):
            lower = max(age_bin.lower, x0)
            upper = min(upper_bound, x1)
            if upper <= lower:
                continue
            width = x1 - x0
            z0 = lower - x0
            z1 = upper - x0
            right = (z1**2 - z0**2) / (2.0 * width)
            matrix[row, left_index] += upper - lower - right
            matrix[row, left_index + 1] += right
    labels = tuple(age_bin.label for age_bin in bins)
    data = xr.DataArray(
        matrix,
        dims=("age_bin", "age"),
        coords={"age_bin": list(labels), "age": grid},
    )
    return ObservationOperator(
        matrix=data,
        input_dimension="age",
        output_dimension="age_bin",
        output_labels=labels,
    )
