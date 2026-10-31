import numpy as np
import pytest

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


@pytest.mark.parametrize("field", ["state", "mortality", "source", "inflow", "dt"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_fem_rejects_nonfinite_inputs(field: str, invalid: float) -> None:
    solver = AgeTransportFEM(maximum_age=2.0, nodes=5)
    kwargs = {
        "state": np.ones(5),
        "mortality": np.zeros(5),
        "source": 0.0,
        "inflow": 1.0,
        "dt": 0.1,
    }
    kwargs[field] = invalid if field in {"source", "inflow", "dt"} else np.full(5, invalid)
    with pytest.raises(ValueError, match="finite"):
        solver.step(**kwargs)


@pytest.mark.parametrize("parameter", ["maximum_age", "artificial_diffusion"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_fem_rejects_nonfinite_configuration(parameter: str, invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        AgeTransportFEM(**{parameter: invalid})


def test_fem_preserves_constant_steady_profile_with_nonzero_inflow() -> None:
    solver = AgeTransportFEM(maximum_age=2.0, nodes=21)
    state = np.full(solver.age.shape, 3.0)
    for _ in range(10):
        state = solver.step(state, np.zeros_like(state), inflow=3.0, dt=0.1)
    assert state[0] == 3.0
    np.testing.assert_allclose(state, 3.0, atol=2e-13)
    assert np.trapezoid(state, solver.age) == pytest.approx(6.0)


def test_fem_converges_over_fixed_time_to_mortality_characteristics() -> None:
    errors = []
    for elements in (40, 80, 160):
        solver = AgeTransportFEM(maximum_age=10.0, nodes=elements + 1, artificial_diffusion=0.0)
        age = solver.age
        state = np.exp(-(((age - 4.0) / 0.8) ** 2))
        dt = 1.0 / elements
        for _ in range(elements):
            state = solver.step(state, np.full_like(age, 0.1), inflow=0.0, dt=dt)
        exact = np.exp(-(((age - 5.0) / 0.8) ** 2)) * np.exp(-0.1)
        errors.append(np.sqrt(np.trapezoid((state - exact) ** 2, age)))
        assert state[0] == 0.0
    assert errors[1] < 0.65 * errors[0]
    assert errors[2] < 0.65 * errors[1]
