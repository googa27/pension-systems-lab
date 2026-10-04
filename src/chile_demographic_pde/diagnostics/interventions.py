"""Counterfactual interventions propagated through the structural PDE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from chile_demographic_pde.calibration.pde_constrained import PDEConstrainedCalibration
from chile_demographic_pde.pde.fem import AgeTransportFEM
from chile_demographic_pde.pde.model import SimulationResult, TwoSexDemographicModel

FloatArray = NDArray[np.float64]
AgeMultiplier = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True, slots=True, eq=False)
class ScenarioResult:
    __hash__ = None  # type: ignore[assignment]

    simulation: SimulationResult
    age: FloatArray
    population_total: FloatArray
    annual_births: FloatArray
    annual_deaths: FloatArray


def fertility_boost(lower: float, upper: float, multiplier: float) -> AgeMultiplier:
    if lower < 0.0 or upper <= lower or multiplier <= 0.0:
        raise ValueError("Invalid fertility intervention")

    def intervention(age: FloatArray) -> FloatArray:
        return np.where((age >= lower) & (age < upper), multiplier, 1.0)

    return intervention


def mortality_reduction(lower: float, upper: float, multiplier: float) -> AgeMultiplier:
    if lower < 0.0 or upper <= lower or not 0.0 < multiplier <= 1.0:
        raise ValueError("Invalid mortality intervention")

    def intervention(age: FloatArray) -> FloatArray:
        return np.where((age >= lower) & (age < upper), multiplier, 1.0)

    return intervention


def _ones(age: FloatArray) -> FloatArray:
    return np.ones_like(age)


def _annual_aggregate(values: FloatArray, periods_per_year: int) -> FloatArray:
    complete_years = values.size // periods_per_year
    if complete_years == 0:
        return np.asarray([values.sum()], dtype=float)
    main = values[: complete_years * periods_per_year].reshape(complete_years, periods_per_year)
    result = list(np.sum(main, axis=1))
    remainder = values[complete_years * periods_per_year :]
    if remainder.size:
        result.append(float(np.sum(remainder)))
    return np.asarray(result, dtype=float)


def simulate_scenario(
    calibration: PDEConstrainedCalibration,
    *,
    years: float = 10.0,
    fertility_multiplier: AgeMultiplier | None = None,
    mortality_multiplier: AgeMultiplier | None = None,
) -> ScenarioResult:
    """Hold fitted period effects fixed and propagate a structural intervention.

    These are conditional scenarios, not unconditional population projections:
    age-specific migration remains zero and future rate multipliers are held at
    their latest smoothed value unless a counterfactual multiplier is supplied.
    """

    if years <= 0.0:
        raise ValueError("years must be positive")
    cfg = calibration.reference
    dt = 1.0 / 12.0
    periods = max(round(years / dt), 1)
    solver = AgeTransportFEM(
        maximum_age=float(cfg.age[-1]),
        nodes=cfg.age.size,
        artificial_diffusion=0.015,
    )
    fertility_effect = _ones if fertility_multiplier is None else fertility_multiplier
    mortality_effect = _ones if mortality_multiplier is None else mortality_multiplier
    latest_fertility_scale = float(calibration.fertility_scale[-1])
    latest_mortality_scale = float(calibration.mortality_scale[-1])

    def fertility(age: FloatArray, _period: int) -> FloatArray:
        return (
            calibration.reference.fertility.rate(age)
            * latest_fertility_scale
            * fertility_effect(age)
        )

    def mortality_male(age: FloatArray, _period: int) -> FloatArray:
        return (
            calibration.reference.mortality_male.rate(age)
            * latest_mortality_scale
            * mortality_effect(age)
        )

    def mortality_female(age: FloatArray, _period: int) -> FloatArray:
        return (
            calibration.reference.mortality_female.rate(age)
            * latest_mortality_scale
            * mortality_effect(age)
        )

    simulation = TwoSexDemographicModel(solver).simulate(
        initial_male=calibration.simulation.states_male[-1],
        initial_female=calibration.simulation.states_female[-1],
        fertility_rate=fertility,
        mortality_male=mortality_male,
        mortality_female=mortality_female,
        periods=periods,
        dt=dt,
    )
    population_total = np.trapezoid(
        simulation.states_male + simulation.states_female,
        solver.age,
        axis=1,
    )
    return ScenarioResult(
        simulation=simulation,
        age=solver.age,
        population_total=np.asarray(population_total, dtype=float),
        annual_births=_annual_aggregate(simulation.predicted_births, 12),
        annual_deaths=_annual_aggregate(simulation.predicted_deaths, 12),
    )
