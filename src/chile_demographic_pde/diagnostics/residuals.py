"""Statistical diagnostics for count forecasts and fitted intensities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from statsmodels.stats.diagnostic import acorr_ljungbox

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class DiagnosticReport:
    __hash__ = None  # type: ignore[assignment]

    deviance: float
    pearson_dispersion: float
    mean_absolute_error: float
    root_mean_squared_error: float
    ljung_box_statistic: float
    ljung_box_pvalue: float
    pearson_residuals: FloatArray


def _validated(observed: ArrayLike, expected: ArrayLike) -> tuple[FloatArray, FloatArray]:
    y = np.asarray(observed, dtype=float)
    mu = np.asarray(expected, dtype=float)
    if y.shape != mu.shape or y.ndim != 1:
        raise ValueError("observed and expected must be matching vectors")
    if np.any(y < 0.0) or np.any(mu <= 0.0):
        raise ValueError("observed must be nonnegative and expected strictly positive")
    return y, mu


def poisson_deviance(observed: ArrayLike, expected: ArrayLike) -> float:
    """Twice the Poisson log-likelihood ratio against a saturated model."""

    y, mu = _validated(observed, expected)
    log_term = np.zeros_like(y)
    positive = y > 0.0
    log_term[positive] = y[positive] * np.log(y[positive] / mu[positive])
    return float(2.0 * np.sum(log_term - (y - mu)))


def diagnostic_report(
    observed: ArrayLike,
    expected: ArrayLike,
    *,
    lags: int = 4,
    estimated_parameters: int = 0,
) -> DiagnosticReport:
    y, mu = _validated(observed, expected)
    residuals = (y - mu) / np.sqrt(mu)
    degrees_of_freedom = max(y.size - estimated_parameters, 1)
    dispersion = float(np.sum(residuals**2) / degrees_of_freedom)
    effective_lag = min(max(lags, 1), max(y.size // 2 - 1, 1))
    ljung_box = acorr_ljungbox(residuals, lags=[effective_lag], return_df=True)
    statistic = float(ljung_box["lb_stat"].iloc[-1])
    pvalue = float(ljung_box["lb_pvalue"].iloc[-1])
    return DiagnosticReport(
        deviance=poisson_deviance(y, mu),
        pearson_dispersion=dispersion,
        mean_absolute_error=float(np.mean(np.abs(y - mu))),
        root_mean_squared_error=float(np.sqrt(np.mean((y - mu) ** 2))),
        ljung_box_statistic=statistic,
        ljung_box_pvalue=pvalue,
        pearson_residuals=residuals,
    )
