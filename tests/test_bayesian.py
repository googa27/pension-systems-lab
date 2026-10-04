import numpy as np

from chile_demographic_pde.dynamics.bayesian import fit_bayesian_local_linear_trend


def test_bayesian_trend_returns_positive_posterior_and_forecast() -> None:
    y = np.array([100.0, 103.0, 101.0, 106.0, 108.0])
    result = fit_bayesian_local_linear_trend(
        y, num_samples=100, random_seed=7, inference_method="svi", svi_steps=300
    )
    assert result.posterior_mean.shape == y.shape
    assert np.all(result.posterior_mean > 0.0)
    assert result.inference_method == "svi"
    assert result.trend_damping == 0.8
    assert result.divergences == 0
    assert np.isfinite(result.final_loss)
    forecast = result.forecast(3, random_seed=8)
    assert forecast.mean.shape == (3,)
    assert forecast.median.shape == (3,)
    assert np.all(forecast.lower > 0.0)
