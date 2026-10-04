import numpy as np

from chile_demographic_pde.dynamics.penalized_trend import PenalizedPoissonTrend


def test_penalized_trend_fits_and_forecasts_positive_counts() -> None:
    y = np.array([100, 105, 103, 108, 111, 109, 115, 118], dtype=float)
    fit = PenalizedPoissonTrend(penalty=20.0).fit(y)
    assert fit.fitted.shape == y.shape
    assert np.all(fit.fitted > 0.0)
    forecast = fit.forecast(4)
    assert forecast.mean.shape == (4,)
    assert np.all(forecast.lower > 0.0)
    assert np.all(forecast.upper > forecast.lower)
