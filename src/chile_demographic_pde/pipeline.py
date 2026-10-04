"""Reference end-to-end calibration and PDE simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chile_demographic_pde.config import ModelConfig
from chile_demographic_pde.data.age_groups import (
    aggregate_deaths_to_population_bins,
    censo_age_bins,
    fertility_age_bins,
)
from chile_demographic_pde.data.loaders import ProjectData
from chile_demographic_pde.data.reconstruction import (
    GridSensitivityComparison,
    ReconstructionResult,
    compare_reconstruction_grids,
    reconstruct_grouped_density,
)
from chile_demographic_pde.dynamics.penalized_trend import FittedPoissonTrend, PenalizedPoissonTrend
from chile_demographic_pde.pde.fem import AgeTransportFEM
from chile_demographic_pde.pde.model import SimulationResult, TwoSexDemographicModel
from chile_demographic_pde.rates.poisson_spline import (
    BinnedPoissonSplineRate,
    FittedBinnedPoissonSplineRate,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class ReferenceModelResult:
    __hash__ = None  # type: ignore[assignment]

    age: FloatArray
    initial_male: FloatArray
    initial_female: FloatArray
    censo_reconstruction_male: ReconstructionResult
    censo_reconstruction_female: ReconstructionResult
    censo_grid_comparison_male: GridSensitivityComparison
    censo_grid_comparison_female: GridSensitivityComparison
    fertility: FittedBinnedPoissonSplineRate
    mortality_male: FittedBinnedPoissonSplineRate
    mortality_female: FittedBinnedPoissonSplineRate
    birth_trend: FittedPoissonTrend
    death_trend: FittedPoissonTrend
    simulation: SimulationResult
    fertility_scale: FloatArray
    mortality_scale: FloatArray

    @property
    def states_male(self) -> FloatArray:
        return self.simulation.states_male

    @property
    def states_female(self) -> FloatArray:
        return self.simulation.states_female

    @property
    def predicted_births(self) -> FloatArray:
        return self.simulation.predicted_births

    @property
    def predicted_deaths(self) -> FloatArray:
        return self.simulation.predicted_deaths


def fit_reference_model(
    data: ProjectData, config: ModelConfig | None = None
) -> ReferenceModelResult:
    """Fit age schedules, dynamic multipliers, and the coupled FEM PDE.

    The available official age profile is April 2026 while monthly totals span
    January 2025--April 2026.  The identifiable reference specification is
    therefore separable: each age schedule is fixed up to a smooth monthly
    multiplicative factor.  This limitation is explicit rather than hidden.
    """

    cfg = ModelConfig() if config is None else config
    solver = AgeTransportFEM(
        maximum_age=cfg.maximum_age,
        nodes=cfg.age_nodes,
        artificial_diffusion=cfg.artificial_diffusion,
    )
    population_bins = censo_age_bins(data.population_age)
    censo_reconstruction_male = reconstruct_grouped_density(
        solver.age,
        population_bins,
        data.population_age["population_male"].to_numpy(dtype=float),
        open_bin_end=cfg.maximum_age,
    )
    censo_reconstruction_female = reconstruct_grouped_density(
        solver.age,
        population_bins,
        data.population_age["population_female"].to_numpy(dtype=float),
        open_bin_end=cfg.maximum_age,
    )
    initial_male = np.asarray(censo_reconstruction_male.density.values, dtype=float)
    initial_female = np.asarray(censo_reconstruction_female.density.values, dtype=float)
    censo_grid_comparison_male = compare_reconstruction_grids(
        solver.age,
        population_bins,
        data.population_age["population_male"].to_numpy(dtype=float),
        open_bin_end=cfg.maximum_age,
        coarse_result=censo_reconstruction_male,
    )
    censo_grid_comparison_female = compare_reconstruction_grids(
        solver.age,
        population_bins,
        data.population_age["population_female"].to_numpy(dtype=float),
        open_bin_end=cfg.maximum_age,
        coarse_result=censo_reconstruction_female,
    )

    fertility_bins, births_selected = fertility_age_bins(data.births_age)
    female_exposure_by_group = data.population_age.set_index("age_group")["population_female"]
    fertility_exposure_month = np.asarray(
        [female_exposure_by_group.loc[str(label)] / 12.0 for label in births_selected["age_group"]],
        dtype=float,
    )
    fertility_fit = BinnedPoissonSplineRate(
        df=cfg.fertility_spline_df,
        penalty=cfg.spline_penalty,
    ).fit(
        fertility_bins,
        births_selected["births_total"].to_numpy(dtype=float),
        fertility_exposure_month,
        exposure_nodes=solver.age,
        exposure_density=initial_female,
        exposure_period_years=cfg.time_step_years,
        open_bin_end=cfg.maximum_age,
    )

    deaths_male = aggregate_deaths_to_population_bins(
        data.deaths_age, population_bins, "deaths_male", cfg.maximum_age
    )
    deaths_female = aggregate_deaths_to_population_bins(
        data.deaths_age, population_bins, "deaths_female", cfg.maximum_age
    )
    mortality_male_fit = BinnedPoissonSplineRate(
        df=cfg.mortality_spline_df,
        penalty=cfg.spline_penalty,
        monotone_from=cfg.mortality_monotone_from,
    ).fit(
        population_bins,
        deaths_male,
        data.population_age["population_male"].to_numpy(dtype=float) / 12.0,
        exposure_nodes=solver.age,
        exposure_density=initial_male,
        exposure_period_years=cfg.time_step_years,
        open_bin_end=cfg.maximum_age,
    )
    mortality_female_fit = BinnedPoissonSplineRate(
        df=cfg.mortality_spline_df,
        penalty=cfg.spline_penalty,
        monotone_from=cfg.mortality_monotone_from,
    ).fit(
        population_bins,
        deaths_female,
        data.population_age["population_female"].to_numpy(dtype=float) / 12.0,
        exposure_nodes=solver.age,
        exposure_density=initial_female,
        exposure_period_years=cfg.time_step_years,
        open_bin_end=cfg.maximum_age,
    )

    birth_trend = PenalizedPoissonTrend(cfg.dynamic_penalty).fit(
        data.monthly["births_total"].to_numpy(dtype=float)
    )
    death_trend = PenalizedPoissonTrend(cfg.dynamic_penalty).fit(
        data.monthly["deaths_total"].to_numpy(dtype=float)
    )
    fertility_scale = birth_trend.fitted / birth_trend.fitted[-1]
    mortality_scale = death_trend.fitted / death_trend.fitted[-1]

    def fertility(age: FloatArray, period: int) -> FloatArray:
        return np.asarray(fertility_fit.rate(age) * fertility_scale[period], dtype=float)

    def mortality_male(age: FloatArray, period: int) -> FloatArray:
        return np.asarray(mortality_male_fit.rate(age) * mortality_scale[period], dtype=float)

    def mortality_female(age: FloatArray, period: int) -> FloatArray:
        return np.asarray(mortality_female_fit.rate(age) * mortality_scale[period], dtype=float)

    simulation = TwoSexDemographicModel(solver, cfg.sex_ratio_male).simulate(
        initial_male=initial_male,
        initial_female=initial_female,
        fertility_rate=fertility,
        mortality_male=mortality_male,
        mortality_female=mortality_female,
        periods=len(data.monthly),
        dt=cfg.time_step_years,
    )
    return ReferenceModelResult(
        age=solver.age,
        initial_male=initial_male,
        initial_female=initial_female,
        censo_reconstruction_male=censo_reconstruction_male,
        censo_reconstruction_female=censo_reconstruction_female,
        censo_grid_comparison_male=censo_grid_comparison_male,
        censo_grid_comparison_female=censo_grid_comparison_female,
        fertility=fertility_fit,
        mortality_male=mortality_male_fit,
        mortality_female=mortality_female_fit,
        birth_trend=birth_trend,
        death_trend=death_trend,
        simulation=simulation,
        fertility_scale=fertility_scale,
        mortality_scale=mortality_scale,
    )
