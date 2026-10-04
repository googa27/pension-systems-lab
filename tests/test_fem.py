import numpy as np

from chile_demographic_pde.pde.fem import AgeTransportFEM


def test_fem_step_is_positive_and_advects_mass() -> None:
    solver = AgeTransportFEM(maximum_age=20.0, nodes=81, artificial_diffusion=0.01)
    age = solver.age
    initial = np.exp(-0.5 * ((age - 5.0) / 1.2) ** 2) * 1000.0
    next_state = solver.step(
        initial,
        mortality=np.zeros_like(age),
        inflow=0.0,
        dt=0.25,
    )
    assert np.min(next_state) > -1e-7
    initial_mean = np.trapezoid(age * initial, age) / np.trapezoid(initial, age)
    next_mean = np.trapezoid(age * next_state, age) / np.trapezoid(next_state, age)
    assert next_mean > initial_mean
    assert abs((next_mean - initial_mean) - 0.25) < 0.12
