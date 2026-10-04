import numpy as np

from chile_demographic_pde.diagnostics.residuals import diagnostic_report, poisson_deviance


def test_poisson_diagnostics_are_finite() -> None:
    observed = np.array([100.0, 110.0, 105.0, 120.0, 115.0, 125.0])
    expected = np.array([102.0, 108.0, 107.0, 118.0, 117.0, 123.0])
    report = diagnostic_report(observed, expected, lags=2, estimated_parameters=2)
    assert poisson_deviance(observed, observed) == 0.0
    assert report.deviance >= 0.0
    assert np.isfinite(report.pearson_dispersion)
    assert 0.0 <= report.ljung_box_pvalue <= 1.0
