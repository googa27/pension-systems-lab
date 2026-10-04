"""Block-coordinate calibration that repeatedly solves the demographic PDE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chile_demographic_pde.config import ModelConfig
from chile_demographic_pde.data.loaders import ProjectData
from chile_demographic_pde.dynamics.penalized_trend import FittedPoissonTrend, PenalizedPoissonTrend
from chile_demographic_pde.pde.fem import AgeTransportFEM
from chile_demographic_pde.pde.model import SimulationResult, TwoSexDemographicModel
from chile_demographic_pde.pipeline import ReferenceModelResult, fit_reference_model

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class PDEConstrainedCalibration:
    __hash__ = None  # type: ignore[assignment]

    reference: ReferenceModelResult
    simulation: SimulationResult
    fertility_scale: FloatArray
    mortality_scale: FloatArray
    birth_scale_fit: FittedPoissonTrend
    death_scale_fit: FittedPoissonTrend
    baseline_births: FloatArray
    baseline_deaths: FloatArray
    iterations: int
    converged: bool
    convergence_history: FloatArray


class PDEConstrainedCalibrator:
    """Alternate between structural Poisson fitting and FEM state propagation.

    Conditional on a population path, the smooth log multipliers are fitted by
    penalized Poisson likelihood with PDE-implied counts as offsets.  Conditional
    on those multipliers, the population path is recomputed by the FEM solver.
    The iteration is a transparent, stable alternative to differentiating a
    large black-box optimizer through every sparse linear solve.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        max_iterations: int = 8,
        tolerance: float = 1e-4,
        damping: float = 0.7,
    ) -> None:
        self.config = ModelConfig() if config is None else config
        if max_iterations < 1 or tolerance <= 0.0 or not 0.0 < damping <= 1.0:
            raise ValueError("Invalid PDE calibration settings")
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping

    def _simulate(
        self,
        reference: ReferenceModelResult,
        fertility_scale: FloatArray,
        mortality_scale: FloatArray,
    ) -> SimulationResult:
        solver = AgeTransportFEM(
            maximum_age=self.config.maximum_age,
            nodes=self.config.age_nodes,
            artificial_diffusion=self.config.artificial_diffusion,
        )

        def fertility(age: FloatArray, period: int) -> FloatArray:
            return np.asarray(reference.fertility.rate(age) * fertility_scale[period], dtype=float)

        def mortality_male(age: FloatArray, period: int) -> FloatArray:
            return np.asarray(
                reference.mortality_male.rate(age) * mortality_scale[period], dtype=float
            )

        def mortality_female(age: FloatArray, period: int) -> FloatArray:
            return np.asarray(
                reference.mortality_female.rate(age) * mortality_scale[period], dtype=float
            )

        return TwoSexDemographicModel(solver, self.config.sex_ratio_male).simulate(
            initial_male=reference.initial_male,
            initial_female=reference.initial_female,
            fertility_rate=fertility,
            mortality_male=mortality_male,
            mortality_female=mortality_female,
            periods=fertility_scale.size,
            dt=self.config.time_step_years,
        )

    def _unit_rate_offsets(
        self, reference: ReferenceModelResult, simulation: SimulationResult
    ) -> tuple[FloatArray, FloatArray]:
        age = reference.age
        periods = simulation.predicted_births.size
        births = np.empty(periods, dtype=float)
        deaths = np.empty(periods, dtype=float)
        fertility = reference.fertility.rate(age)
        mu_male = reference.mortality_male.rate(age)
        mu_female = reference.mortality_female.rate(age)
        for period in range(periods):
            births[period] = self.config.time_step_years * float(
                np.trapezoid(fertility * simulation.states_female[period], age)
            )
            deaths[period] = self.config.time_step_years * float(
                np.trapezoid(
                    mu_male * simulation.states_male[period]
                    + mu_female * simulation.states_female[period],
                    age,
                )
            )
        return np.maximum(births, 1e-9), np.maximum(deaths, 1e-9)

    def fit(self, data: ProjectData) -> PDEConstrainedCalibration:
        reference = fit_reference_model(data, self.config)
        periods = len(data.monthly)
        fertility_scale = np.ones(periods, dtype=float)
        mortality_scale = np.ones(periods, dtype=float)
        baseline_simulation = self._simulate(reference, fertility_scale, mortality_scale)
        baseline_births = baseline_simulation.predicted_births.copy()
        baseline_deaths = baseline_simulation.predicted_deaths.copy()
        simulation = baseline_simulation
        history: list[float] = []
        observed_births = data.monthly["births_total"].to_numpy(dtype=float)
        observed_deaths = data.monthly["deaths_total"].to_numpy(dtype=float)
        birth_fit: FittedPoissonTrend | None = None
        death_fit: FittedPoissonTrend | None = None
        converged = False

        iterations = 0
        for _iteration in range(1, self.max_iterations + 1):
            iterations = _iteration
            birth_offset, death_offset = self._unit_rate_offsets(reference, simulation)
            birth_fit = PenalizedPoissonTrend(self.config.dynamic_penalty).fit(
                observed_births, offset=birth_offset
            )
            death_fit = PenalizedPoissonTrend(self.config.dynamic_penalty).fit(
                observed_deaths, offset=death_offset
            )
            target_fertility = np.exp(birth_fit.log_fitted)
            target_mortality = np.exp(death_fit.log_fitted)
            updated_fertility = np.exp(
                (1.0 - self.damping) * np.log(fertility_scale)
                + self.damping * np.log(target_fertility)
            )
            updated_mortality = np.exp(
                (1.0 - self.damping) * np.log(mortality_scale)
                + self.damping * np.log(target_mortality)
            )
            change = float(
                max(
                    np.max(np.abs(np.log(updated_fertility / fertility_scale))),
                    np.max(np.abs(np.log(updated_mortality / mortality_scale))),
                )
            )
            history.append(change)
            fertility_scale = updated_fertility
            mortality_scale = updated_mortality
            simulation = self._simulate(reference, fertility_scale, mortality_scale)
            if change < self.tolerance:
                converged = True
                break

        if birth_fit is None or death_fit is None:
            raise RuntimeError("Calibration did not perform an iteration")
        return PDEConstrainedCalibration(
            reference=reference,
            simulation=simulation,
            fertility_scale=fertility_scale,
            mortality_scale=mortality_scale,
            birth_scale_fit=birth_fit,
            death_scale_fit=death_fit,
            baseline_births=baseline_births,
            baseline_deaths=baseline_deaths,
            iterations=iterations,
            converged=converged,
            convergence_history=np.asarray(history, dtype=float),
        )
