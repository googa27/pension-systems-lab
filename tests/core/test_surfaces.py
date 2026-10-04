import numpy as np
import pytest
import xarray as xr

from chile_demographic_pde.core.errors import (
    PopulationBasisError,
    SurfaceAlignmentError,
    UnitMismatchError,
)
from chile_demographic_pde.core.semantics import PopulationBasis, Unit
from chile_demographic_pde.core.surfaces import (
    ExpectedCountSurface,
    ExposureSurface,
    RateSurface,
)


def _rate(values: list[float], ages: list[int]) -> RateSurface:
    return RateSurface(
        xr.DataArray(values, dims=("age",), coords={"age": ages}),
        unit=Unit.PER_YEAR,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )


def test_rate_addition_and_exposure_multiplication_preserve_semantics() -> None:
    all_cause = _rate([0.01, 0.02], [40, 41]) + _rate([0.03, 0.04], [40, 41])
    np.testing.assert_allclose(all_cause.data.values, [0.04, 0.06])

    exposure = ExposureSurface(
        xr.DataArray([100.0, 200.0], dims=("age",), coords={"age": [40, 41]}),
        unit=Unit.PERSON_YEAR,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )
    expected = all_cause * exposure
    assert expected.unit is Unit.PERSON
    np.testing.assert_allclose(expected.data.values, [4.0, 12.0])


def test_rate_operators_reject_implicit_coordinate_and_unit_alignment() -> None:
    left = _rate([0.01, 0.02], [40, 41])
    shifted = _rate([0.01, 0.02], [41, 42])
    with pytest.raises(SurfaceAlignmentError, match="coordinates"):
        _ = left + shifted

    monthly = RateSurface(
        left.data,
        unit=Unit.PER_MONTH,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )
    with pytest.raises(UnitMismatchError):
        _ = left + monthly


def test_rate_exposure_multiplication_rejects_unconverted_time_units() -> None:
    monthly_rate = RateSurface(
        xr.DataArray([0.01], dims=("age",), coords={"age": [40]}),
        unit=Unit.PER_MONTH,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )
    yearly_exposure = ExposureSurface(
        xr.DataArray([100.0], dims=("age",), coords={"age": [40]}),
        unit=Unit.PERSON_YEAR,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )

    with pytest.raises(UnitMismatchError, match="identical time units"):
        _ = monthly_rate * yearly_exposure


def test_rate_addition_rejects_different_auxiliary_coordinate_values() -> None:
    left = _rate([0.01, 0.02], [40, 41]).data.assign_coords(
        source=("age", ["estimated", "estimated"])
    )
    right = _rate([0.03, 0.04], [40, 41]).data.assign_coords(
        source=("age", ["observed", "observed"])
    )

    with pytest.raises(SurfaceAlignmentError, match="coordinates"):
        _ = RateSurface(left, Unit.PER_YEAR, PopulationBasis.RESIDENT_ESTIMATE) + RateSurface(
            right, Unit.PER_YEAR, PopulationBasis.RESIDENT_ESTIMATE
        )


def test_rate_addition_rejects_missing_auxiliary_coordinates() -> None:
    left = _rate([0.01, 0.02], [40, 41])
    right = RateSurface(
        _rate([0.03, 0.04], [40, 41]).data.assign_coords(sex="female"),
        Unit.PER_YEAR,
        PopulationBasis.RESIDENT_ESTIMATE,
    )

    with pytest.raises(SurfaceAlignmentError, match="coordinates"):
        _ = left + right


def test_rate_operators_reject_mixed_population_bases() -> None:
    resident = _rate([0.01, 0.02], [40, 41])
    census = RateSurface(
        resident.data,
        unit=Unit.PER_YEAR,
        population_basis=PopulationBasis.CENSUS_ENUMERATED,
    )
    with pytest.raises(PopulationBasisError):
        _ = resident + census


def test_exposure_times_rate_is_not_a_supported_operation() -> None:
    exposure = ExposureSurface(
        xr.DataArray([100.0], dims=("age",), coords={"age": [40]}),
        unit=Unit.PERSON_YEAR,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )
    with pytest.raises(TypeError):
        _ = exposure * _rate([0.01], [40])


def test_rate_surface_is_callable_at_labeled_coordinates() -> None:
    surface = _rate([0.01, 0.03], [40, 42])
    evaluated = surface(age=[41])
    np.testing.assert_allclose(evaluated.values, [0.02])


@pytest.mark.parametrize(
    ("left", "equal", "different"),
    [
        (
            ExpectedCountSurface(xr.DataArray([1.0, 2.0], dims=("age",), coords={"age": [40, 41]})),
            ExpectedCountSurface(xr.DataArray([1.0, 2.0], dims=("age",), coords={"age": [40, 41]})),
            ExpectedCountSurface(xr.DataArray([1.0, 3.0], dims=("age",), coords={"age": [40, 41]})),
        ),
        (
            ExposureSurface(
                xr.DataArray([1.0, 2.0], dims=("age",), coords={"age": [40, 41]}),
                Unit.PERSON_YEAR,
                PopulationBasis.RESIDENT_ESTIMATE,
            ),
            ExposureSurface(
                xr.DataArray([1.0, 2.0], dims=("age",), coords={"age": [40, 41]}),
                Unit.PERSON_YEAR,
                PopulationBasis.RESIDENT_ESTIMATE,
            ),
            ExposureSurface(
                xr.DataArray([1.0, 3.0], dims=("age",), coords={"age": [40, 41]}),
                Unit.PERSON_YEAR,
                PopulationBasis.RESIDENT_ESTIMATE,
            ),
        ),
        (
            _rate([0.01, 0.02], [40, 41]),
            _rate([0.01, 0.02], [40, 41]),
            _rate([0.01, 0.03], [40, 41]),
        ),
    ],
)
def test_surface_value_equality_is_xarray_aware_and_unhashable(
    left: ExpectedCountSurface | ExposureSurface | RateSurface,
    equal: ExpectedCountSurface | ExposureSurface | RateSurface,
    different: ExpectedCountSurface | ExposureSurface | RateSurface,
) -> None:
    assert left == equal
    assert left != different
    assert left.__eq__(object()) is NotImplemented
    assert type(left).__hash__ is None
    with pytest.raises(TypeError):
        hash(left)
