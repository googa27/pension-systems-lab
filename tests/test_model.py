from pathlib import Path

import numpy as np

from chile_demographic_pde.config import ModelConfig
from chile_demographic_pde.data.loaders import load_project_data
from chile_demographic_pde.pipeline import fit_reference_model


def test_reference_pipeline_runs_and_preserves_nonnegative_population() -> None:
    data = load_project_data(Path("data/processed"))
    result = fit_reference_model(data, ModelConfig(age_nodes=81))
    assert result.states_male.shape[0] == len(data.monthly) + 1
    assert result.states_female.shape == result.states_male.shape
    assert np.all(result.states_male >= -1e-6)
    assert np.all(result.states_female >= -1e-6)
    assert result.predicted_births.shape == (len(data.monthly),)
    assert result.predicted_deaths.shape == (len(data.monthly),)
    assert result.censo_reconstruction_male.diagnostics.converged
    assert result.censo_reconstruction_female.diagnostics.converged
    assert result.censo_reconstruction_male.diagnostics.max_absolute_bin_error < 1e-7
    assert result.censo_reconstruction_female.diagnostics.max_absolute_bin_error < 1e-7
    assert result.censo_grid_comparison_male.coarse_node_count == 81
    assert result.censo_grid_comparison_male.refined_node_count == 161
    assert result.censo_grid_comparison_female.coarse_node_count == 81
    assert result.censo_grid_comparison_female.refined_node_count == 161
    assert 0.0 <= result.censo_grid_comparison_male.relative_linf_density_difference < 0.08
    assert 0.0 <= result.censo_grid_comparison_female.relative_linf_density_difference < 0.08
    np.testing.assert_array_equal(
        result.initial_male,
        result.censo_reconstruction_male.density.values,
    )
    np.testing.assert_array_equal(
        result.initial_female,
        result.censo_reconstruction_female.density.values,
    )
    tail_age = np.linspace(40.0, 105.0, 131)
    assert np.min(np.diff(np.log(result.mortality_male.rate(tail_age)))) > -1e-7
    assert np.min(np.diff(np.log(result.mortality_female.rate(tail_age)))) > -1e-7
