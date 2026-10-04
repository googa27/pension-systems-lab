from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from chile_demographic_pde.calibration.pde_constrained import (
    PDEConstrainedCalibration,
)
from chile_demographic_pde.data.loaders import ProjectData
from chile_demographic_pde.diagnostics.residuals import DiagnosticReport
from chile_demographic_pde.dynamics.penalized_trend import TrendForecast
from chile_demographic_pde.pde.model import SimulationResult
from chile_demographic_pde.pipeline import ReferenceModelResult
from chile_demographic_pde.rates.poisson_spline import (
    FittedBinnedPoissonSplineRate,
)

ContainerFactory = Callable[[], object]
_NESTED_RESULT = object()


def _dynamics_result() -> TrendForecast:
    return TrendForecast(
        mean=np.array([1.0, 2.0]),
        lower=np.array([0.5, 1.5]),
        upper=np.array([1.5, 2.5]),
    )


def _pde_result() -> SimulationResult:
    values = np.array([1.0, 2.0])
    return SimulationResult(values, values, values, values, values)


def _rate_result() -> FittedBinnedPoissonSplineRate:
    return FittedBinnedPoissonSplineRate(
        coefficients=np.array([1.0, 2.0]),
        knots=np.array([0.0, 0.0, 1.0, 1.0]),
        degree=1,
        objective=0.0,
        converged=True,
        lower=0.0,
        upper=1.0,
    )


def _loaded_data() -> ProjectData:
    frame = pd.DataFrame({"value": [1, 2]})
    return ProjectData(frame, frame, frame, frame)


def _diagnostic_result() -> DiagnosticReport:
    return DiagnosticReport(
        deviance=1.0,
        pearson_dispersion=1.0,
        mean_absolute_error=1.0,
        root_mean_squared_error=1.0,
        ljung_box_statistic=1.0,
        ljung_box_pvalue=0.5,
        pearson_residuals=np.array([1.0, 2.0]),
    )


def _pipeline_result() -> ReferenceModelResult:
    values = np.array([1.0, 2.0])
    return ReferenceModelResult(
        age=values,
        initial_male=values,
        initial_female=values,
        censo_reconstruction_male=_NESTED_RESULT,
        censo_reconstruction_female=_NESTED_RESULT,
        censo_grid_comparison_male=_NESTED_RESULT,
        censo_grid_comparison_female=_NESTED_RESULT,
        fertility=_NESTED_RESULT,
        mortality_male=_NESTED_RESULT,
        mortality_female=_NESTED_RESULT,
        birth_trend=_NESTED_RESULT,
        death_trend=_NESTED_RESULT,
        simulation=_NESTED_RESULT,
        fertility_scale=values,
        mortality_scale=values,
    )


def _calibration_result() -> PDEConstrainedCalibration:
    values = np.array([1.0, 2.0])
    return PDEConstrainedCalibration(
        reference=_NESTED_RESULT,
        simulation=_NESTED_RESULT,
        fertility_scale=values,
        mortality_scale=values,
        birth_scale_fit=_NESTED_RESULT,
        death_scale_fit=_NESTED_RESULT,
        baseline_births=values,
        baseline_deaths=values,
        iterations=1,
        converged=True,
        convergence_history=values,
    )


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(_dynamics_result, id="dynamics"),
        pytest.param(_pde_result, id="pde"),
        pytest.param(_rate_result, id="rates"),
        pytest.param(_loaded_data, id="loaders"),
        pytest.param(_diagnostic_result, id="diagnostics"),
        pytest.param(_pipeline_result, id="pipeline"),
        pytest.param(_calibration_result, id="calibration"),
    ],
)
def test_numerical_data_containers_use_identity_equality_and_are_unhashable(
    factory: ContainerFactory,
) -> None:
    left = factory()
    right = factory()

    assert left == left
    assert left != right
    assert type(left).__hash__ is None
    with pytest.raises(TypeError):
        hash(left)
