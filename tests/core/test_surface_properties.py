from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from chile_demographic_pde.core.errors import (
    PopulationBasisError,
    SurfaceAlignmentError,
    UnitMismatchError,
)
from chile_demographic_pde.core.semantics import PopulationBasis, Unit
from chile_demographic_pde.core.surfaces import ExposureSurface, RateSurface
from chile_demographic_pde.data.observation import (
    AgeBin,
    ObservationOperator,
    p1_bin_integral_operator,
)

PROPERTY_SETTINGS = settings(max_examples=30, deadline=None)
FINITE = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
NONNEGATIVE = st.floats(
    min_value=0.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=32,
)
RATE_VECTORS = st.lists(NONNEGATIVE, min_size=2, max_size=12)


def _rate(
    values: list[float] | np.ndarray,
    *,
    ages: np.ndarray | None = None,
    unit: Unit = Unit.PER_YEAR,
    basis: PopulationBasis = PopulationBasis.RESIDENT_ESTIMATE,
) -> RateSurface:
    coordinates = np.arange(len(values), dtype=float) if ages is None else ages
    return RateSurface(
        xr.DataArray(values, dims=("age",), coords={"age": coordinates}),
        unit=unit,
        population_basis=basis,
    )


def _exposure(
    values: list[float] | np.ndarray,
    *,
    ages: np.ndarray | None = None,
    basis: PopulationBasis = PopulationBasis.RESIDENT_ESTIMATE,
) -> ExposureSurface:
    coordinates = np.arange(len(values), dtype=float) if ages is None else ages
    return ExposureSurface(
        xr.DataArray(values, dims=("age",), coords={"age": coordinates}),
        unit=Unit.PERSON_YEAR,
        population_basis=basis,
    )


@st.composite
def _three_vectors(
    draw: st.DrawFn,
) -> tuple[list[float], list[float], list[float]]:
    rows = draw(
        st.lists(
            st.tuples(NONNEGATIVE, NONNEGATIVE, NONNEGATIVE),
            min_size=2,
            max_size=12,
        )
    )
    first, second, third = zip(*rows, strict=True)
    return list(first), list(second), list(third)


@st.composite
def _partition(
    draw: st.DrawFn,
) -> tuple[np.ndarray, np.ndarray]:
    widths = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=8))
    nodes = np.concatenate(([0.0], np.cumsum(widths, dtype=float)))
    values = np.asarray(
        draw(st.lists(NONNEGATIVE, min_size=len(nodes), max_size=len(nodes))),
        dtype=float,
    )
    return nodes, values


@st.composite
def _operator_chain(
    draw: st.DrawFn,
) -> tuple[ObservationOperator, ObservationOperator, ObservationOperator]:
    a_size = draw(st.integers(min_value=1, max_value=4))
    b_size = draw(st.integers(min_value=1, max_value=4))
    c_size = draw(st.integers(min_value=1, max_value=4))
    d_size = draw(st.integers(min_value=1, max_value=4))

    def matrix(rows: int, columns: int) -> np.ndarray:
        values = draw(st.lists(FINITE, min_size=rows * columns, max_size=rows * columns))
        return np.asarray(values, dtype=float).reshape(rows, columns)

    a_labels = tuple(f"a{index}" for index in range(a_size))
    b_labels = tuple(f"b{index}" for index in range(b_size))
    c_labels = tuple(f"c{index}" for index in range(c_size))
    d_labels = tuple(f"d{index}" for index in range(d_size))
    first = ObservationOperator(
        xr.DataArray(
            matrix(a_size, b_size),
            dims=("a", "b"),
            coords={"a": list(a_labels), "b": list(b_labels)},
        ),
        "b",
        "a",
        a_labels,
    )
    second = ObservationOperator(
        xr.DataArray(
            matrix(b_size, c_size),
            dims=("b", "c"),
            coords={"b": list(b_labels), "c": list(c_labels)},
        ),
        "c",
        "b",
        b_labels,
    )
    third = ObservationOperator(
        xr.DataArray(
            matrix(c_size, d_size),
            dims=("c", "d"),
            coords={"c": list(c_labels), "d": list(d_labels)},
        ),
        "d",
        "c",
        c_labels,
    )
    return first, second, third


@PROPERTY_SETTINGS
@given(_three_vectors())
def test_rate_addition_is_commutative(
    vectors: tuple[list[float], list[float], list[float]],
) -> None:
    first, second, _ = (_rate(values) for values in vectors)

    np.testing.assert_allclose((first + second).data.values, (second + first).data.values)


@PROPERTY_SETTINGS
@given(_three_vectors())
def test_rate_addition_is_associative(
    vectors: tuple[list[float], list[float], list[float]],
) -> None:
    first, second, third = (_rate(values) for values in vectors)

    np.testing.assert_allclose(
        ((first + second) + third).data.values,
        (first + (second + third)).data.values,
        rtol=1e-12,
        atol=1e-12,
    )


@PROPERTY_SETTINGS
@given(RATE_VECTORS)
def test_rate_scalar_identity(values: list[float]) -> None:
    rate = _rate(values)

    assert rate * 1.0 == rate
    assert 1.0 * rate == rate


@PROPERTY_SETTINGS
@given(RATE_VECTORS, NONNEGATIVE)
def test_rate_scalar_multiplication_is_distributive(
    values: list[float],
    scalar: float,
) -> None:
    rate = _rate(values)

    np.testing.assert_allclose(
        ((rate + rate) * scalar).data.values,
        (rate * scalar + rate * scalar).data.values,
        rtol=1e-12,
        atol=1e-12,
    )


@PROPERTY_SETTINGS
@given(
    st.lists(
        st.tuples(NONNEGATIVE, NONNEGATIVE),
        min_size=2,
        max_size=12,
    ),
    NONNEGATIVE,
)
def test_rate_times_exposure_respects_scaling(
    rows: list[tuple[float, float]],
    scalar: float,
) -> None:
    rate_values, exposure_values = zip(*rows, strict=True)
    rate = _rate(list(rate_values))
    exposure = _exposure(list(exposure_values))

    scaled_rate_product = (rate * scalar) * exposure
    scaled_exposure_product = rate * _exposure(np.asarray(exposure_values) * scalar)
    np.testing.assert_allclose(
        scaled_rate_product.data.values,
        scaled_exposure_product.data.values,
        rtol=1e-12,
        atol=1e-12,
    )


@PROPERTY_SETTINGS
@given(_partition())
def test_p1_partition_conserves_total_mass(
    partition: tuple[np.ndarray, np.ndarray],
) -> None:
    nodes, values = partition
    bins = [
        AgeBin(float(lower), float(upper), f"bin-{index}")
        for index, (lower, upper) in enumerate(pairwise(nodes))
    ]
    observed = p1_bin_integral_operator(nodes, bins) @ _rate(values, ages=nodes)

    np.testing.assert_allclose(
        observed.sum().item(),
        np.trapezoid(values, nodes),
        rtol=1e-12,
        atol=1e-12,
    )


@PROPERTY_SETTINGS
@given(
    st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=8),
    NONNEGATIVE,
    NONNEGATIVE,
)
def test_p1_operator_is_exact_for_affine_fields(
    widths: list[int],
    slope: float,
    intercept: float,
) -> None:
    nodes = np.concatenate(([0.0], np.cumsum(widths, dtype=float)))
    bins = [
        AgeBin(float(lower), float(upper), f"bin-{index}")
        for index, (lower, upper) in enumerate(pairwise(nodes))
    ]
    observed = p1_bin_integral_operator(nodes, bins) @ _rate(
        slope * nodes + intercept,
        ages=nodes,
    )
    primitive = 0.5 * slope * nodes**2 + intercept * nodes

    np.testing.assert_allclose(
        observed.values,
        np.diff(primitive),
        rtol=1e-11,
        atol=1e-10,
    )


@PROPERTY_SETTINGS
@given(_operator_chain())
def test_labeled_operator_composition_is_associative(
    chain: tuple[ObservationOperator, ObservationOperator, ObservationOperator],
) -> None:
    first, second, third = chain
    left = (first @ second) @ third
    right = first @ (second @ third)

    assert isinstance(left, ObservationOperator)
    assert isinstance(right, ObservationOperator)
    assert left.input_dimension == right.input_dimension == "d"
    assert left.output_dimension == right.output_dimension == "a"
    assert left.output_labels == right.output_labels
    assert left.matrix.dims == right.matrix.dims
    assert left.matrix.indexes["a"].equals(right.matrix.indexes["a"])
    assert left.matrix.indexes["d"].equals(right.matrix.indexes["d"])
    np.testing.assert_allclose(
        left.matrix.values,
        right.matrix.values,
        rtol=1e-11,
        atol=1e-10,
    )


@PROPERTY_SETTINGS
@given(RATE_VECTORS, st.integers(min_value=1, max_value=100))
def test_coordinate_mismatches_never_broadcast(
    values: list[float],
    shift: int,
) -> None:
    rate = _rate(values)
    shifted_ages = np.arange(len(values), dtype=float) + shift
    shifted_rate = _rate(values, ages=shifted_ages)
    shifted_exposure = _exposure(np.abs(values), ages=shifted_ages)

    with pytest.raises(SurfaceAlignmentError):
        _ = rate + shifted_rate
    with pytest.raises(SurfaceAlignmentError):
        _ = rate * shifted_exposure


@PROPERTY_SETTINGS
@given(RATE_VECTORS)
def test_unit_and_population_basis_mismatches_are_rejected(
    values: list[float],
) -> None:
    yearly = _rate(values)
    monthly = _rate(values, unit=Unit.PER_MONTH)
    census_rate = _rate(values, basis=PopulationBasis.CENSUS_ENUMERATED)
    census_exposure = _exposure(np.abs(values), basis=PopulationBasis.CENSUS_ENUMERATED)

    with pytest.raises(UnitMismatchError):
        _ = yearly + monthly
    with pytest.raises(UnitMismatchError):
        _ = monthly * _exposure(np.abs(values))
    with pytest.raises(PopulationBasisError):
        _ = yearly + census_rate
    with pytest.raises(PopulationBasisError):
        _ = yearly * census_exposure


@PROPERTY_SETTINGS
@given(st.one_of(st.none(), st.text(max_size=8), st.binary(max_size=8)))
def test_algebraic_dunders_return_not_implemented_for_foreign_operands(
    foreign: object,
) -> None:
    rate = _rate([0.1, 0.2])
    operator = p1_bin_integral_operator(
        [0.0, 1.0],
        [AgeBin(0.0, 1.0, "all")],
    )

    assert rate.__add__(foreign) is NotImplemented
    assert rate.__mul__(foreign) is NotImplemented
    assert rate.__rmul__(foreign) is NotImplemented
    assert rate.__eq__(foreign) is NotImplemented
    assert operator.__matmul__(foreign) is NotImplemented
    assert operator.__eq__(foreign) is NotImplemented


def test_rate_times_exposure_is_not_commutatively_guessed() -> None:
    rate = _rate([0.1])
    exposure = _exposure([10.0])

    assert (rate * exposure).data.item() == 1.0
    with pytest.raises(TypeError):
        _ = exposure * rate
