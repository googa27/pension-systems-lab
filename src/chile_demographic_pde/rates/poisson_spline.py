"""Penalized spline rate estimation from grouped Poisson event counts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint, minimize

from chile_demographic_pde.data.observation import AgeBin

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class _QuadratureCell:
    """Precomputed basis values and exposure weights for one observed bin."""

    __hash__ = None  # type: ignore[assignment]

    basis: FloatArray
    weights: FloatArray


def _knot_vector(lower: float, upper: float, n_basis: int, degree: int = 3) -> FloatArray:
    if n_basis <= degree:
        raise ValueError("n_basis must exceed spline degree")
    n_internal = n_basis - degree - 1
    internal = (
        np.linspace(lower, upper, n_internal + 2, dtype=float)[1:-1]
        if n_internal > 0
        else np.empty(0, dtype=float)
    )
    return np.concatenate([np.repeat(lower, degree + 1), internal, np.repeat(upper, degree + 1)])


def _design(x: ArrayLike, knots: FloatArray, degree: int) -> FloatArray:
    points = np.asarray(x, dtype=float)
    return np.asarray(BSpline.design_matrix(points, knots, degree, extrapolate=True).toarray())


def _derivative_design(x: ArrayLike, knots: FloatArray, degree: int) -> FloatArray:
    """Return the derivative of every B-spline basis function at ``x``."""

    points = np.asarray(x, dtype=float)
    n_basis = len(knots) - degree - 1
    vector_spline = BSpline(knots, np.eye(n_basis), degree, axis=0, extrapolate=True)
    return np.asarray(vector_spline.derivative()(points), dtype=float)


@dataclass(frozen=True, slots=True, eq=False)
class FittedBinnedPoissonSplineRate:
    """A positive continuous-age rate represented on the log scale by B-splines."""

    __hash__ = None  # type: ignore[assignment]

    coefficients: FloatArray
    knots: FloatArray
    degree: int
    objective: float
    converged: bool
    lower: float
    upper: float

    def rate(self, ages: ArrayLike) -> FloatArray:
        points = np.asarray(ages, dtype=float)
        basis = _design(points, self.knots, self.degree)
        values = np.exp(np.clip(basis @ self.coefficients, -30.0, 10.0))
        return np.where((points >= self.lower) & (points <= self.upper), values, 0.0)

    def expected_bin_counts(
        self,
        bins: list[AgeBin],
        exposure: ArrayLike,
        quadrature_order: int = 32,
        *,
        open_bin_end: float | None = None,
    ) -> FloatArray:
        exposures = np.asarray(exposure, dtype=float)
        if exposures.shape != (len(bins),):
            raise ValueError("Exposure and bins must have the same length.")
        return _expected_counts(
            self.coefficients,
            self.knots,
            self.degree,
            bins,
            exposures,
            quadrature_order,
            open_bin_end=open_bin_end,
        )


class BinnedPoissonSplineRate:
    """Estimate a smooth rate from binned counts and bin-level exposures.

    Within each published age bin, exposure is assumed uniform unless a finer
    exposure distribution is supplied externally. The likelihood integrates
    the exponentiated spline up to Gauss--Legendre quadrature error:

    ``Y_i ~ Poisson(E_i / |A_i| * integral_Ai exp(B(a) theta) da)``.

    ``monotone_from`` optionally imposes a nonnegative derivative of the
    log-rate over the specified tail. This is useful when a broad open-ended
    age bin would otherwise induce an unsupported old-age hazard decline.
    """

    def __init__(
        self,
        df: int = 8,
        penalty: float = 20.0,
        degree: int = 3,
        monotone_from: float | None = None,
    ) -> None:
        if df <= degree:
            raise ValueError("df must exceed degree")
        if penalty < 0.0:
            raise ValueError("penalty must be nonnegative")
        if monotone_from is not None and monotone_from < 0.0:
            raise ValueError("monotone_from must be nonnegative")
        self.df = df
        self.penalty = penalty
        self.degree = degree
        self.monotone_from = monotone_from

    def fit(
        self,
        bins: list[AgeBin],
        counts: ArrayLike,
        exposure: ArrayLike,
        *,
        exposure_nodes: ArrayLike | None = None,
        exposure_density: ArrayLike | None = None,
        exposure_period_years: float = 1.0,
        open_bin_end: float | None = None,
    ) -> FittedBinnedPoissonSplineRate:
        y = np.asarray(counts, dtype=float)
        e = np.asarray(exposure, dtype=float)
        if y.shape != (len(bins),) or e.shape != y.shape:
            raise ValueError("counts, exposure, and bins must have matching lengths")
        if np.any(y < 0.0) or np.any(e <= 0.0):
            raise ValueError("counts must be nonnegative and exposure strictly positive")
        if (exposure_nodes is None) != (exposure_density is None):
            raise ValueError("exposure_nodes and exposure_density must be supplied together")
        density_nodes = None if exposure_nodes is None else np.asarray(exposure_nodes, dtype=float)
        density_values = (
            None if exposure_density is None else np.asarray(exposure_density, dtype=float)
        )
        if density_nodes is not None:
            if (
                density_nodes.ndim != 1
                or density_values is None
                or density_values.shape != density_nodes.shape
            ):
                raise ValueError("Fine exposure nodes and density must be matching vectors")
            if np.any(np.diff(density_nodes) <= 0.0) or np.any(density_values < 0.0):
                raise ValueError("Fine exposure nodes must increase and density be nonnegative")
            if exposure_period_years <= 0.0:
                raise ValueError("exposure_period_years must be positive")

        lower = min(age_bin.lower for age_bin in bins)
        upper = max(age_bin.resolved_upper(open_bin_end) for age_bin in bins)
        if self.monotone_from is not None and not lower <= self.monotone_from < upper:
            raise ValueError("monotone_from must lie inside the fitted age domain")

        knots = _knot_vector(lower, upper, self.df, self.degree)
        difference = np.diff(np.eye(self.df), n=2, axis=0)
        penalty_matrix = self.penalty * difference.T @ difference
        cells = _prepare_quadrature(
            knots=knots,
            degree=self.degree,
            bins=bins,
            exposure=e,
            quadrature_order=32,
            exposure_nodes=density_nodes,
            exposure_density=density_values,
            exposure_period_years=exposure_period_years,
            open_bin_end=open_bin_end,
        )
        base_rate = max(float(y.sum() / e.sum()), 1e-12)
        initial = np.full(self.df, np.log(base_rate), dtype=float)

        def objective(theta: FloatArray) -> tuple[float, FloatArray]:
            expected, jacobian = _expected_counts_and_jacobian(theta, cells)
            nll = float(np.sum(expected - y * np.log(expected)))
            residual_weight = 1.0 - y / expected
            gradient = jacobian.T @ residual_weight
            penalty_value = 0.5 * float(theta @ penalty_matrix @ theta)
            penalty_gradient = penalty_matrix @ theta
            return nll + penalty_value, np.asarray(gradient + penalty_gradient, dtype=float)

        constraints: tuple[LinearConstraint, ...] = ()
        method = "L-BFGS-B"
        options: dict[str, float | int] = {
            "maxiter": 1_000,
            "ftol": 1e-11,
            "gtol": 1e-7,
        }
        if self.monotone_from is not None:
            # The derivative is linear in the spline coefficients. A dense grid
            # gives a transparent shape constraint without changing the rate
            # representation or the grouped-count likelihood.
            constraint_ages = np.linspace(self.monotone_from, upper, 16 * self.df + 1)
            derivative_basis = _derivative_design(
                constraint_ages,
                knots,
                self.degree,
            )
            # SLSQP permits small feasibility errors. A negligible positive
            # numerical margin keeps the evaluated curve nondecreasing after
            # optimization without materially changing the demographic fit.
            constraints = (LinearConstraint(derivative_basis, 1e-4, np.inf),)
            method = "SLSQP"
            options = {"maxiter": 2_000, "ftol": 1e-11}

        result = minimize(
            fun=lambda theta: objective(theta)[0],
            x0=initial,
            jac=lambda theta: objective(theta)[1],
            method=method,
            constraints=constraints,
            options=options,
        )
        return FittedBinnedPoissonSplineRate(
            coefficients=np.asarray(result.x, dtype=float),
            knots=knots,
            degree=self.degree,
            objective=float(result.fun),
            converged=bool(result.success),
            lower=lower,
            upper=upper,
        )


def _prepare_quadrature(
    *,
    knots: FloatArray,
    degree: int,
    bins: list[AgeBin],
    exposure: FloatArray,
    quadrature_order: int,
    exposure_nodes: FloatArray | None,
    exposure_density: FloatArray | None,
    exposure_period_years: float,
    open_bin_end: float | None,
) -> list[_QuadratureCell]:
    canonical_x, canonical_w = np.polynomial.legendre.leggauss(quadrature_order)
    cells: list[_QuadratureCell] = []
    for index, age_bin in enumerate(bins):
        upper = age_bin.resolved_upper(open_bin_end)
        width = upper - age_bin.lower
        half_width = 0.5 * width
        midpoint = 0.5 * (age_bin.lower + upper)
        points = midpoint + half_width * canonical_x
        integration_weights = half_width * canonical_w
        if exposure_nodes is None or exposure_density is None:
            exposure_weights = integration_weights * exposure[index] / width
        else:
            local_density = np.interp(points, exposure_nodes, exposure_density)
            exposure_weights = exposure_period_years * integration_weights * local_density
        cells.append(
            _QuadratureCell(
                basis=_design(points, knots, degree),
                weights=np.asarray(exposure_weights, dtype=float),
            )
        )
    return cells


def _expected_counts_and_jacobian(
    theta: FloatArray,
    cells: list[_QuadratureCell],
) -> tuple[FloatArray, FloatArray]:
    expected = np.empty(len(cells), dtype=float)
    jacobian = np.empty((len(cells), theta.size), dtype=float)
    for index, cell in enumerate(cells):
        linear_predictor = cell.basis @ theta
        clipped = np.clip(linear_predictor, -30.0, 10.0)
        rates = np.exp(clipped)
        active = ((linear_predictor > -30.0) & (linear_predictor < 10.0)).astype(float)
        weighted_rates = cell.weights * rates
        expected[index] = max(float(np.sum(weighted_rates)), 1e-12)
        jacobian[index] = (weighted_rates * active) @ cell.basis
    return expected, jacobian


def _expected_counts(
    theta: FloatArray,
    knots: FloatArray,
    degree: int,
    bins: list[AgeBin],
    exposure: FloatArray,
    quadrature_order: int,
    *,
    exposure_nodes: FloatArray | None = None,
    exposure_density: FloatArray | None = None,
    exposure_period_years: float = 1.0,
    open_bin_end: float | None = None,
) -> FloatArray:
    cells = _prepare_quadrature(
        knots=knots,
        degree=degree,
        bins=bins,
        exposure=exposure,
        quadrature_order=quadrature_order,
        exposure_nodes=exposure_nodes,
        exposure_density=exposure_density,
        exposure_period_years=exposure_period_years,
        open_bin_end=open_bin_end,
    )
    expected, _ = _expected_counts_and_jacobian(theta, cells)
    return expected
