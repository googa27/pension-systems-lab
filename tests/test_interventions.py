from pathlib import Path

from chile_demographic_pde.calibration.pde_constrained import PDEConstrainedCalibrator
from chile_demographic_pde.config import ModelConfig
from chile_demographic_pde.data.loaders import load_project_data
from chile_demographic_pde.diagnostics.interventions import (
    fertility_boost,
    mortality_reduction,
    simulate_scenario,
)


def test_structural_interventions_move_the_expected_flows() -> None:
    data = load_project_data(Path("data/processed"))
    calibrated = PDEConstrainedCalibrator(ModelConfig(age_nodes=81), max_iterations=2).fit(data)
    baseline = simulate_scenario(calibrated, years=1.0)
    more_births = simulate_scenario(
        calibrated,
        years=1.0,
        fertility_multiplier=fertility_boost(25.0, 35.0, 1.10),
    )
    fewer_old_age_deaths = simulate_scenario(
        calibrated,
        years=1.0,
        mortality_multiplier=mortality_reduction(65.0, 105.0, 0.90),
    )
    assert (
        more_births.simulation.predicted_births.sum() > baseline.simulation.predicted_births.sum()
    )
    assert (
        fewer_old_age_deaths.simulation.predicted_deaths.sum()
        < baseline.simulation.predicted_deaths.sum()
    )
