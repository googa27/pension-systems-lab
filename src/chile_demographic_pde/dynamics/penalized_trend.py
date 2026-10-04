"""Smooth dynamic Poisson trends and extrapolative forecasts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import norm

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class TrendForecast:
    __hash__ = None  # type: ignore[assignment]

    mean: FloatArray
    lower: FloatArray
    upper: FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class FittedPoissonTrend:
    __hash__ = None  # type: ignore[assignment]

    observed: FloatArray
    log_fitted: FloatArray
    fitted: FloatArray
    penalty: float
    innovation_scale: float

    def forecast(self, horizon: int, confidence: float = 0.95) -> TrendForecast:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        slope = self.log_fitted[-1] - self.log_fitted[-2] if self.log_fitted.size > 1 else 0.0
        steps = np.arange(1, horizon + 1, dtype=float)
        log_mean = self.log_fitted[-1] + slope * steps
        z_score = float(norm.ppf(0.5 + confidence / 2.0))
        # Local-linear extrapolation uncertainty increases faster than a random walk.
        standard_error = np.maximum(self.innovation_scale, 0.03) * np.sqrt(
            steps * (1.0 + steps / 2.0)
        )
        return TrendForecast(
            mean=np.exp(log_mean),
            lower=np.exp(log_mean - z_score * standard_error),
            upper=np.exp(log_mean + z_score * standard_error),
        )


class PenalizedPoissonTrend:
    """Poisson trend with a second-difference penalty on latent log means."""

    def __init__(self, penalty: float = 30.0) -> None:
        if penalty < 0.0:
            raise ValueError("penalty must be nonnegative")
        self.penalty = penalty

    def fit(self, observations: ArrayLike, offset: ArrayLike | None = None) -> FittedPoissonTrend:
        y = np.asarray(observations, dtype=float)
        if y.ndim != 1 or y.size < 3 or np.any(y < 0.0):
            raise ValueError(
                "observations must be a nonnegative vector with at least three entries"
            )
        exposure = np.ones_like(y) if offset is None else np.asarray(offset, dtype=float)
        if exposure.shape != y.shape or np.any(exposure <= 0.0):
            raise ValueError("offset must be strictly positive and match observations")
        initial = np.log((y + 0.5) / exposure)
        d2 = np.diff(np.eye(y.size), n=2, axis=0)

        def objective(z: FloatArray) -> tuple[float, FloatArray]:
            mean = exposure * np.exp(np.clip(z, -30.0, 30.0))
            difference = d2 @ z
            value = float(
                np.sum(mean - y * (np.log(exposure) + z))
                + 0.5 * self.penalty * np.sum(difference**2)
            )
            gradient = mean - y + self.penalty * (d2.T @ difference)
            return value, gradient

        result = minimize(
            lambda z: objective(z)[0],
            initial,
            jac=lambda z: objective(z)[1],
            method="L-BFGS-B",
            options={"maxiter": 2_000, "ftol": 1e-12, "gtol": 1e-8},
        )
        log_rate = np.asarray(result.x, dtype=float)
        fitted = exposure * np.exp(log_rate)
        second_differences = np.diff(log_rate, n=2)
        innovation_scale = float(
            np.sqrt(np.mean(second_differences**2)) if second_differences.size else 0.05
        )
        return FittedPoissonTrend(
            observed=y,
            log_fitted=log_rate,
            fitted=fitted,
            penalty=self.penalty,
            innovation_scale=max(innovation_scale, 1e-3),
        )
