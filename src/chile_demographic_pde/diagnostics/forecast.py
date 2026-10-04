"""Honest time-ordered backtesting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from chile_demographic_pde.dynamics.penalized_trend import PenalizedPoissonTrend


def rolling_origin_backtest(
    observations: ArrayLike,
    *,
    minimum_training: int = 8,
    penalty: float = 30.0,
) -> pd.DataFrame:
    """One-step rolling-origin forecasts without future-data leakage."""

    y = np.asarray(observations, dtype=float)
    if y.ndim != 1 or minimum_training < 3 or minimum_training >= y.size:
        raise ValueError("Invalid observations or minimum_training")
    records: list[dict[str, float | int]] = []
    for origin in range(minimum_training, y.size):
        fitted = PenalizedPoissonTrend(penalty).fit(y[:origin])
        forecast = float(fitted.forecast(1).mean[0])
        actual = float(y[origin])
        records.append(
            {
                "origin": origin,
                "actual": actual,
                "forecast": forecast,
                "error": actual - forecast,
            }
        )
    return pd.DataFrame.from_records(records, columns=["origin", "actual", "forecast", "error"])
