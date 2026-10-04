from pathlib import Path

import numpy as np

from chile_demographic_pde.calibration.pde_constrained import PDEConstrainedCalibrator
from chile_demographic_pde.config import ModelConfig
from chile_demographic_pde.data.loaders import load_project_data
from chile_demographic_pde.diagnostics.residuals import poisson_deviance


def test_coupled_calibration_reduces_monthly_deviance() -> None:
    data = load_project_data(Path("data/processed"))
    calibration = PDEConstrainedCalibrator(
        ModelConfig(age_nodes=81), max_iterations=3, damping=0.8
    ).fit(data)
    observed = data.monthly["births_total"].to_numpy(dtype=float)
    baseline = calibration.baseline_births
    fitted = calibration.simulation.predicted_births
    assert poisson_deviance(observed, fitted) < poisson_deviance(observed, baseline)
    assert calibration.iterations >= 1
    assert np.all(calibration.fertility_scale > 0.0)
