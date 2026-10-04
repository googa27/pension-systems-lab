"""Two-sex McKendrick--von Foerster model built on the FEM transport solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chile_demographic_pde.pde.fem import AgeTransportFEM

FloatArray = NDArray[np.float64]
RateFunction = Callable[[FloatArray, int], FloatArray]


@dataclass(frozen=True, slots=True, eq=False)
class SimulationResult:
    __hash__ = None  # type: ignore[assignment]

    states_male: FloatArray
    states_female: FloatArray
    predicted_births: FloatArray
    predicted_deaths_male: FloatArray
    predicted_deaths_female: FloatArray

    @property
    def predicted_deaths(self) -> FloatArray:
        return self.predicted_deaths_male + self.predicted_deaths_female


class TwoSexDemographicModel:
    """Propagate male and female age densities with an endogenous birth boundary."""

    def __init__(self, solver: AgeTransportFEM, sex_ratio_male: float = 0.5053) -> None:
        if not 0.0 < sex_ratio_male < 1.0:
            raise ValueError("sex_ratio_male must be in (0, 1)")
        self.solver = solver
        self.sex_ratio_male = sex_ratio_male

    def simulate(
        self,
        initial_male: ArrayLike,
        initial_female: ArrayLike,
        fertility_rate: RateFunction,
        mortality_male: RateFunction,
        mortality_female: RateFunction,
        periods: int,
        dt: float,
        migration_male: RateFunction | None = None,
        migration_female: RateFunction | None = None,
    ) -> SimulationResult:
        male = np.asarray(initial_male, dtype=float)
        female = np.asarray(initial_female, dtype=float)
        if male.shape != self.solver.age.shape or female.shape != male.shape:
            raise ValueError("Initial states must match the FEM grid")
        states_male = [male.copy()]
        states_female = [female.copy()]
        births = np.zeros(periods, dtype=float)
        deaths_male = np.zeros(periods, dtype=float)
        deaths_female = np.zeros(periods, dtype=float)
        age = self.solver.age

        for period in range(periods):
            fertility = np.maximum(fertility_rate(age, period), 0.0)
            mu_male = np.maximum(mortality_male(age, period), 0.0)
            mu_female = np.maximum(mortality_female(age, period), 0.0)
            annual_birth_flow = float(np.trapezoid(fertility * female, age))
            births[period] = dt * annual_birth_flow
            deaths_male[period] = dt * float(np.trapezoid(mu_male * male, age))
            deaths_female[period] = dt * float(np.trapezoid(mu_female * female, age))
            source_male = (
                np.zeros_like(age) if migration_male is None else migration_male(age, period)
            )
            source_female = (
                np.zeros_like(age) if migration_female is None else migration_female(age, period)
            )
            male = self.solver.step(
                male,
                mortality=mu_male,
                inflow=self.sex_ratio_male * annual_birth_flow,
                dt=dt,
                source=source_male,
            )
            female = self.solver.step(
                female,
                mortality=mu_female,
                inflow=(1.0 - self.sex_ratio_male) * annual_birth_flow,
                dt=dt,
                source=source_female,
            )
            states_male.append(male.copy())
            states_female.append(female.copy())

        return SimulationResult(
            states_male=np.stack(states_male),
            states_female=np.stack(states_female),
            predicted_births=births,
            predicted_deaths_male=deaths_male,
            predicted_deaths_female=deaths_female,
        )
