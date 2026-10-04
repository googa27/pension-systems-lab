"""Exact descriptive reconstruction of grouped population counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize

from chile_demographic_pde.core.errors import (
    ObservationOperatorError,
    ReconstructionError,
)
from chile_demographic_pde.data.observation import AgeBin, p1_bin_integral_operator

FloatArray = NDArray[np.float64]
OPTIMIZER_FEASIBILITY_TOLERANCE = 1e-10


@dataclass(frozen=True, slots=True)
class ReconstructionDiagnostics:
    """Numerical evidence for an exact grouped-count reconstruction."""

    converged: bool
    message: str
    objective: float
    max_absolute_bin_error: float
    max_relative_bin_error: float
    total_residual: float
    constraint_condition_number: float
    active_lower_bounds: int
    minimum_normalized_density: float
    minimum_density: float
    max_normalized_constraint_violation: float
    local_curvature_indicator: float


@dataclass(frozen=True, slots=True)
class GridSensitivityComparison:
    """Observed change when the exact reconstruction grid is refined.

    The refined density is interpolated back to the coarse nodes. The relative
    L-infinity difference divides the maximum absolute difference by the
    largest absolute density on either aligned curve.
    """

    coarse_node_count: int
    refined_node_count: int
    refinement_factor: int
    max_absolute_density_difference: float
    density_scale: float
    relative_linf_density_difference: float


@dataclass(frozen=True, slots=True, eq=False)
class ReconstructionResult:
    """Descriptive within-bin curve and its grouped-count diagnostics."""

    density: xr.DataArray
    reconstructed_bins: xr.DataArray
    residuals: xr.DataArray
    absolute_residuals: xr.DataArray
    relative_residuals: xr.DataArray
    diagnostics: ReconstructionDiagnostics

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        assert isinstance(other, ReconstructionResult)
        return (
            self.density.identical(other.density)
            and self.reconstructed_bins.identical(other.reconstructed_bins)
            and self.residuals.identical(other.residuals)
            and self.absolute_residuals.identical(other.absolute_residuals)
            and self.relative_residuals.identical(other.relative_residuals)
            and self.diagnostics == other.diagnostics
        )


def _validated_inputs(
    nodes: ArrayLike,
    bins: list[AgeBin],
    counts: ArrayLike,
    open_bin_end: object,
    tolerance: object,
) -> tuple[FloatArray, FloatArray, float | None, float]:
    try:
        grid = np.asarray(nodes, dtype=float)
    except (TypeError, ValueError) as error:
        raise ReconstructionError(
            "Age nodes must be a finite strictly increasing vector."
        ) from error
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ReconstructionError("Age nodes must be a finite strictly increasing vector.")
    if not bins:
        raise ReconstructionError("Grouped totals and bins must be nonempty.")
    try:
        totals = np.asarray(counts, dtype=float)
    except (TypeError, ValueError) as error:
        raise ReconstructionError(
            "Counts must be a finite nonnegative vector matching bins."
        ) from error
    if totals.size == 0:
        raise ReconstructionError("Grouped totals and bins must be nonempty.")
    if totals.ndim != 1 or totals.shape != (len(bins),):
        raise ReconstructionError("Counts must be a finite nonnegative vector matching bins.")
    if not np.all(np.isfinite(totals)):
        raise ReconstructionError("Counts must contain only finite values.")
    if np.any(totals < 0.0):
        raise ReconstructionError("Counts must be nonnegative.")
    validated_tolerance = _finite_scalar(tolerance, "Reconstruction tolerance")
    if validated_tolerance <= 0.0:
        cause = ValueError("Tolerance is not positive.")
        raise ReconstructionError(
            "Reconstruction tolerance must be finite and positive."
        ) from cause
    validated_open_bin_end = (
        None if open_bin_end is None else _finite_scalar(open_bin_end, "open_bin_end")
    )
    open_bins = [age_bin for age_bin in bins if age_bin.upper is None]
    if open_bins:
        if validated_open_bin_end is None:
            raise ReconstructionError(
                "Every open-ended bin requires an explicit finite open_bin_end."
            )
        if any(validated_open_bin_end <= age_bin.lower for age_bin in open_bins):
            cause = ValueError("Open-bin endpoint does not exceed its lower endpoint.")
            raise ReconstructionError(
                "open_bin_end must be finite and exceed every open-bin lower endpoint."
            ) from cause
    return grid, totals, validated_open_bin_end, validated_tolerance


def _finite_scalar(value: object, name: str) -> float:
    try:
        scalar = np.asarray(value)
        if scalar.ndim != 0:
            raise TypeError(f"{name} must be scalar.")
        normalized = float(scalar)
    except (TypeError, ValueError, OverflowError) as error:
        raise ReconstructionError(f"{name} must be a finite scalar.") from error
    if not np.isfinite(normalized):
        cause = ValueError(f"{name} is not finite.")
        raise ReconstructionError(f"{name} must be a finite scalar.") from cause
    return normalized


def _roughness_matrix(grid: FloatArray) -> FloatArray:
    if grid.size == 2:
        return np.eye(2, dtype=float)
    left_width = grid[1:-1] - grid[:-2]
    right_width = grid[2:] - grid[1:-1]
    curvature = np.zeros((grid.size - 2, grid.size), dtype=float)
    interior = np.arange(grid.size - 2)
    curvature[interior, interior] = 2.0 / (left_width * (left_width + right_width))
    curvature[interior, interior + 1] = -2.0 / (left_width * right_width)
    curvature[interior, interior + 2] = 2.0 / (right_width * (left_width + right_width))
    quadrature_weight = 0.5 * (left_width + right_width)
    roughness = curvature.T @ (quadrature_weight[:, None] * curvature)
    roughness_scale = max(float(np.trace(roughness)) / grid.size, 1.0)
    return roughness / roughness_scale + 1e-12 * np.eye(grid.size)


def _local_curvature_indicator(density: FloatArray) -> float:
    """Return a dimensionless local interpolation-curvature indicator."""

    if density.size < 3:
        return 0.0
    midpoint_adjustment = 0.125 * np.abs(np.diff(density, n=2))
    density_scale = max(float(np.max(np.abs(density))), np.finfo(float).tiny)
    return float(np.max(midpoint_adjustment, initial=0.0) / density_scale)


def reconstruct_grouped_density(
    nodes: ArrayLike,
    bins: list[AgeBin],
    counts: ArrayLike,
    *,
    open_bin_end: float | None = None,
    tolerance: float = 1e-7,
) -> ReconstructionResult:
    """Select a nonnegative minimum-curvature curve with exact grouped totals.

    The returned curve is a descriptive interpolation of Censo-enumerated
    grouped counts. It is neither observed single-age data nor an official
    resident-population estimate.
    """

    grid, totals, validated_open_bin_end, validated_tolerance = _validated_inputs(
        nodes,
        bins,
        counts,
        open_bin_end,
        tolerance,
    )
    try:
        operator = p1_bin_integral_operator(
            grid,
            bins,
            open_bin_end=validated_open_bin_end,
        )
    except ObservationOperatorError as error:
        raise ReconstructionError(
            f"Could not construct the grouped-count observation operator: {error}"
        ) from error
    matrix = np.asarray(operator.matrix.values, dtype=float)
    roughness = _roughness_matrix(grid)

    count_scale = float(np.max(totals, initial=0.0))
    if count_scale == 0.0:
        count_scale = 1.0
    normalized_totals = totals / count_scale
    try:
        initial_result = lsq_linear(
            matrix,
            normalized_totals,
            bounds=(0.0, np.inf),
            lsmr_tol="auto",
            max_iter=2_000,
        )
        if not initial_result.success:
            initial_failure = RuntimeError(str(initial_result.message))
            raise ReconstructionError(
                f"Exact grouped reconstruction initial solve failed: {initial_result.message}"
            ) from initial_failure
        optimized = minimize(
            fun=lambda values: 0.5 * float(values @ roughness @ values),
            x0=np.asarray(initial_result.x, dtype=float),
            jac=lambda values: roughness @ values,
            method="SLSQP",
            bounds=Bounds(0.0, np.inf),
            constraints=(
                LinearConstraint(
                    matrix,
                    normalized_totals,
                    normalized_totals,
                ),
            ),
            options={
                "ftol": 1e-13,
                "maxiter": 2_000,
            },
        )
    except ReconstructionError:
        raise
    except Exception as error:
        raise ReconstructionError(
            f"Exact grouped reconstruction optimizer failed: {error}"
        ) from error

    try:
        normalized_density = np.asarray(optimized.x, dtype=float)
        optimizer_objective = float(optimized.fun)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ReconstructionError(
            "Exact grouped reconstruction optimizer did not return a finite solution "
            "and finite objective."
        ) from error
    if normalized_density.shape != grid.shape or not np.all(np.isfinite(normalized_density)):
        solution_error = ValueError("Optimizer solution is nonfinite or has the wrong shape.")
        raise ReconstructionError(
            "Exact grouped reconstruction optimizer must return a finite solution "
            "matching the age grid."
        ) from solution_error
    if not np.isfinite(optimizer_objective):
        objective_error = ValueError("Optimizer objective is nonfinite.")
        raise ReconstructionError(
            "Exact grouped reconstruction optimizer must return a finite objective."
        ) from objective_error
    minimum_normalized = float(np.min(normalized_density))
    if minimum_normalized < -OPTIMIZER_FEASIBILITY_TOLERANCE:
        bound_error = ValueError(
            "Optimizer solution violates the normalized nonnegative bound by "
            f"{-minimum_normalized:.6g}."
        )
        raise ReconstructionError(
            "Exact grouped reconstruction optimizer returned a materially "
            "negative, nonnegative-bound-violating solution."
        ) from bound_error
    if minimum_normalized < 0.0:
        normalized_density = np.maximum(normalized_density, 0.0)
    density_values = count_scale * normalized_density
    if not np.all(np.isfinite(density_values)):
        density_error = ValueError("Physical density is nonfinite after rescaling.")
        raise ReconstructionError(
            "Exact grouped reconstruction produced a nonfinite physical density."
        ) from density_error
    physical_objective = 0.5 * float(density_values @ roughness @ density_values)
    if not np.isfinite(physical_objective):
        physical_objective_error = ValueError("Physical roughness objective is nonfinite.")
        raise ReconstructionError(
            "Exact grouped reconstruction produced a nonfinite objective."
        ) from physical_objective_error
    normalized_constraint_residual = matrix @ normalized_density - normalized_totals
    max_normalized_constraint_violation = float(
        np.max(np.abs(normalized_constraint_residual), initial=0.0)
    )
    reconstructed = matrix @ density_values
    residuals = reconstructed - totals
    absolute = np.abs(residuals)
    relative = absolute / np.maximum(totals, 1.0)
    maximum_absolute = float(np.max(absolute, initial=0.0))
    maximum_relative = float(np.max(relative, initial=0.0))
    converged = bool(
        optimized.success
        and maximum_absolute <= validated_tolerance
        and float(np.min(normalized_density)) >= 0.0
    )
    if not converged:
        convergence_error = RuntimeError(str(optimized.message))
        raise ReconstructionError(
            "Exact grouped reconstruction failed: "
            f"{optimized.message}; max bin error={maximum_absolute:.6g}."
        ) from convergence_error

    labels = list(operator.output_labels)
    bin_coordinates = {"age_bin": labels}
    density = xr.DataArray(
        density_values,
        dims=("age",),
        coords={"age": grid},
        attrs={
            "long_name": "Descriptive within-bin Censo count density",
            "units": "enumerated people per year of age",
            "population_basis": "censo_enumerated",
            "interpretation": (
                "Descriptive interpolation of grouped Censo-enumerated counts; "
                "not observed single-age data or a resident-population estimate."
            ),
        },
    )
    return ReconstructionResult(
        density=density,
        reconstructed_bins=xr.DataArray(
            reconstructed,
            dims=("age_bin",),
            coords=bin_coordinates,
            attrs={"units": "enumerated people"},
        ),
        residuals=xr.DataArray(
            residuals,
            dims=("age_bin",),
            coords=bin_coordinates,
            attrs={"units": "enumerated people"},
        ),
        absolute_residuals=xr.DataArray(
            absolute,
            dims=("age_bin",),
            coords=bin_coordinates,
            attrs={"units": "enumerated people"},
        ),
        relative_residuals=xr.DataArray(
            relative,
            dims=("age_bin",),
            coords=bin_coordinates,
            attrs={"units": "relative"},
        ),
        diagnostics=ReconstructionDiagnostics(
            converged=True,
            message=str(optimized.message),
            objective=physical_objective,
            max_absolute_bin_error=maximum_absolute,
            max_relative_bin_error=maximum_relative,
            total_residual=float(np.sum(residuals)),
            constraint_condition_number=float(np.linalg.cond(matrix)),
            active_lower_bounds=int(
                np.count_nonzero(
                    normalized_density
                    <= max(validated_tolerance / count_scale, OPTIMIZER_FEASIBILITY_TOLERANCE)
                )
            ),
            minimum_normalized_density=float(np.min(normalized_density)),
            minimum_density=float(np.min(density_values)),
            max_normalized_constraint_violation=max_normalized_constraint_violation,
            local_curvature_indicator=_local_curvature_indicator(density_values),
        ),
    )


def compare_reconstruction_grids(
    nodes: ArrayLike,
    bins: list[AgeBin],
    counts: ArrayLike,
    *,
    open_bin_end: float | None = None,
    tolerance: float = 1e-7,
    refinement_factor: int = 2,
    coarse_result: ReconstructionResult | None = None,
) -> GridSensitivityComparison:
    """Compare exact grouped reconstructions on coarse and refined age grids."""

    grid, totals, validated_open_bin_end, validated_tolerance = _validated_inputs(
        nodes,
        bins,
        counts,
        open_bin_end,
        tolerance,
    )
    if (
        isinstance(refinement_factor, bool)
        or not isinstance(refinement_factor, (int, np.integer))
        or refinement_factor < 2
    ):
        raise ReconstructionError("refinement_factor must be an integer of at least two.")
    factor = int(refinement_factor)
    coarse = (
        reconstruct_grouped_density(
            grid,
            bins,
            totals,
            open_bin_end=validated_open_bin_end,
            tolerance=validated_tolerance,
        )
        if coarse_result is None
        else coarse_result
    )
    try:
        coarse_operator = p1_bin_integral_operator(
            grid,
            bins,
            open_bin_end=validated_open_bin_end,
        )
    except ObservationOperatorError as error:
        raise ReconstructionError(
            f"Could not validate coarse_result observation operator: {error}"
        ) from error
    _validate_coarse_result(
        coarse,
        grid,
        totals,
        np.asarray(coarse_operator.matrix.values, dtype=float),
        coarse_operator.output_labels,
        validated_tolerance,
    )
    fractions = np.arange(factor, dtype=float) / factor
    refined_intervals = grid[:-1, None] + (grid[1:] - grid[:-1])[:, None] * fractions[None, :]
    refined_grid = np.concatenate([refined_intervals.ravel(), grid[-1:]])
    refined = reconstruct_grouped_density(
        refined_grid,
        bins,
        totals,
        open_bin_end=validated_open_bin_end,
        tolerance=validated_tolerance,
    )
    coarse_density = np.asarray(coarse.density.values, dtype=float)
    refined_on_coarse = np.interp(
        grid,
        np.asarray(refined.density.coords["age"].values, dtype=float),
        np.asarray(refined.density.values, dtype=float),
    )
    maximum_absolute = float(np.max(np.abs(refined_on_coarse - coarse_density), initial=0.0))
    density_scale = max(
        float(np.max(np.abs(coarse_density), initial=0.0)),
        float(np.max(np.abs(refined_on_coarse), initial=0.0)),
    )
    relative_linf = 0.0 if density_scale == 0.0 else maximum_absolute / density_scale
    if not np.all(np.isfinite([maximum_absolute, density_scale, relative_linf])):
        comparison_error = ValueError("Grid-comparison metrics are nonfinite.")
        raise ReconstructionError(
            "Grid comparison must produce finite density-difference metrics."
        ) from comparison_error
    return GridSensitivityComparison(
        coarse_node_count=int(grid.size),
        refined_node_count=int(refined_grid.size),
        refinement_factor=factor,
        max_absolute_density_difference=maximum_absolute,
        density_scale=density_scale,
        relative_linf_density_difference=relative_linf,
    )


def _invalid_coarse_result(message: str, cause: Exception | None = None) -> NoReturn:
    boundary_error = ValueError(message) if cause is None else cause
    raise ReconstructionError(f"Invalid coarse_result: {message}") from boundary_error


def _validate_coarse_result(
    coarse: object,
    grid: FloatArray,
    totals: FloatArray,
    matrix: FloatArray,
    labels: tuple[str, ...],
    tolerance: float,
) -> None:
    """Validate a caller-supplied exact reconstruction without trusting caches."""

    if type(coarse) is not ReconstructionResult:
        _invalid_coarse_result(
            "expected an exact ReconstructionResult instance.",
            TypeError(f"Got {type(coarse).__name__}."),
        )
    assert isinstance(coarse, ReconstructionResult)
    if type(coarse.density) is not xr.DataArray or coarse.density.dims != ("age",):
        _invalid_coarse_result("density dimensions must be exactly ('age',).")
    if "age" not in coarse.density.coords or coarse.density.coords["age"].dims != ("age",):
        _invalid_coarse_result("density requires a one-dimensional age coordinate.")
    try:
        coarse_ages = np.asarray(coarse.density.coords["age"].values, dtype=float)
        coarse_density = np.asarray(coarse.density.values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        _invalid_coarse_result("density and age coordinates must be numeric.", error)
    if (
        coarse_ages.shape != grid.shape
        or coarse_density.shape != grid.shape
        or not np.array_equal(coarse_ages, grid)
    ):
        _invalid_coarse_result("density shape and age coordinates must equal the supplied grid.")
    if not np.all(np.isfinite(coarse_density)):
        _invalid_coarse_result("density values must be finite.")
    if np.any(coarse_density < 0.0):
        _invalid_coarse_result("density values must be nonnegative.")
    if (
        type(coarse.diagnostics) is not ReconstructionDiagnostics
        or not coarse.diagnostics.converged
    ):
        _invalid_coarse_result("diagnostics must record a converged reconstruction.")
    if (
        type(coarse.reconstructed_bins) is not xr.DataArray
        or coarse.reconstructed_bins.dims != ("age_bin",)
        or "age_bin" not in coarse.reconstructed_bins.coords
        or coarse.reconstructed_bins.coords["age_bin"].dims != ("age_bin",)
    ):
        _invalid_coarse_result(
            "reconstructed_bins must be a labeled one-dimensional age_bin array."
        )
    actual_labels = tuple(coarse.reconstructed_bins.coords["age_bin"].values.tolist())
    if actual_labels != labels:
        _invalid_coarse_result("reconstructed_bins labels must match the supplied bins.")
    try:
        stored_bins = np.asarray(coarse.reconstructed_bins.values, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        _invalid_coarse_result("reconstructed_bins values must be numeric.", error)
    if stored_bins.shape != totals.shape or not np.all(np.isfinite(stored_bins)):
        _invalid_coarse_result("reconstructed_bins values must be finite and match totals.")
    recomputed_bins = matrix @ coarse_density
    if not np.all(np.isfinite(recomputed_bins)):
        _invalid_coarse_result("recomputed bin totals must be finite.")
    if not np.allclose(recomputed_bins, totals, rtol=0.0, atol=tolerance):
        _invalid_coarse_result(
            "density must recompute every supplied grouped total within tolerance."
        )
    if not np.allclose(stored_bins, recomputed_bins, rtol=0.0, atol=tolerance) or not np.allclose(
        stored_bins, totals, rtol=0.0, atol=tolerance
    ):
        _invalid_coarse_result(
            "reconstructed_bins values must match recomputed and supplied totals."
        )
