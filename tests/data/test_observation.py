import numpy as np
import pytest
import xarray as xr

from chile_demographic_pde.core.errors import ObservationOperatorError
from chile_demographic_pde.core.semantics import PopulationBasis, Unit
from chile_demographic_pde.core.surfaces import RateSurface
from chile_demographic_pde.data.observation import (
    AgeBin,
    ObservationOperator,
    p1_bin_integral_operator,
)


def _operator() -> ObservationOperator:
    return p1_bin_integral_operator(
        np.array([0.0, 1.0, 2.0]),
        [AgeBin(0.0, 1.0, "a"), AgeBin(1.0, 2.0, "b")],
    )


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (4.999, False),
        (5, True),
        (9.999, True),
        (10, False),
        ("7", False),
        (None, False),
    ],
)
def test_age_bin_membership_is_half_open(age: object, expected: bool) -> None:
    assert (age in AgeBin(5.0, 10.0, "5-9")) is expected


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (84.999, False),
        (85, True),
        (1_000_000, True),
        (object(), False),
    ],
)
def test_open_age_bin_membership_has_no_upper_endpoint(
    age: object,
    expected: bool,
) -> None:
    assert (age in AgeBin(85.0, None, "85+")) is expected


@pytest.mark.parametrize("age", [False, True])
def test_age_bin_membership_rejects_boolean_operands(age: bool) -> None:
    assert age not in AgeBin(0.0, 2.0, "0-1")


@pytest.mark.parametrize("age", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize(
    "age_bin",
    [
        AgeBin(0.0, 100.0, "finite"),
        AgeBin(85.0, None, "open"),
    ],
)
def test_age_bin_membership_rejects_nonfinite_numeric_ages(
    age: float,
    age_bin: AgeBin,
) -> None:
    assert age not in age_bin


def test_p1_operator_matmul_is_exact_for_affine_field() -> None:
    ages = np.linspace(0.0, 10.0, 21)
    surface = RateSurface(
        xr.DataArray(2.0 + 3.0 * ages, dims=("age",), coords={"age": ages}),
        unit=Unit.PER_YEAR,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )
    operator = p1_bin_integral_operator(
        ages,
        [AgeBin(0.0, 2.5, "0-2"), AgeBin(2.5, 7.5, "2-7")],
    )
    observed = operator @ surface
    expected = np.array(
        [
            2.0 * 2.5 + 1.5 * 2.5**2,
            (2.0 * 7.5 + 1.5 * 7.5**2) - (2.0 * 2.5 + 1.5 * 2.5**2),
        ]
    )
    np.testing.assert_allclose(observed.values, expected, rtol=1e-12, atol=1e-12)


def test_open_bin_requires_an_explicit_domain_end() -> None:
    ages = np.linspace(0.0, 105.0, 106)
    with pytest.raises(ObservationOperatorError, match="open-ended"):
        p1_bin_integral_operator(ages, [AgeBin(85.0, None, "85+")])

    operator = p1_bin_integral_operator(
        ages,
        [AgeBin(85.0, None, "85+")],
        open_bin_end=105.0,
    )
    assert operator.output_labels == ("85+",)


def test_observation_operators_compose_with_matmul() -> None:
    ages = np.array([0.0, 1.0, 2.0])
    fine = p1_bin_integral_operator(
        ages,
        [AgeBin(0.0, 1.0, "a"), AgeBin(1.0, 2.0, "b")],
    )
    coarse_matrix = xr.DataArray(
        [[1.0, 1.0]],
        dims=("coarse_bin", "age_bin"),
        coords={"coarse_bin": ["all"], "age_bin": ["a", "b"]},
    )
    coarse = ObservationOperator(coarse_matrix, "age_bin", "coarse_bin", ("all",))
    combined = coarse @ fine
    assert isinstance(combined, ObservationOperator)
    np.testing.assert_allclose(combined.matrix.values, [[0.5, 1.0, 0.5]])


def test_observation_operator_equality_is_xarray_aware_and_unhashable() -> None:
    operator = _operator()
    equal = ObservationOperator(
        operator.matrix.copy(deep=True),
        operator.input_dimension,
        operator.output_dimension,
        operator.output_labels,
    )
    different_values = ObservationOperator(
        operator.matrix + 1.0,
        operator.input_dimension,
        operator.output_dimension,
        operator.output_labels,
    )
    renamed = ObservationOperator(
        operator.matrix.rename({"age_bin": "group", "age": "node"}),
        "node",
        "group",
        operator.output_labels,
    )

    assert operator == equal
    assert operator != different_values
    assert operator != renamed
    assert operator.__eq__(object()) is NotImplemented
    with pytest.raises(TypeError):
        hash(operator)


@pytest.mark.parametrize(
    "nodes",
    [
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, 1.0, np.inf]),
        np.array([-np.inf, 1.0, 2.0]),
    ],
)
def test_p1_operator_rejects_nonfinite_nodes(nodes: np.ndarray) -> None:
    with pytest.raises(ObservationOperatorError, match="finite"):
        p1_bin_integral_operator(nodes, [AgeBin(0.0, 1.0, "a")])


@pytest.mark.parametrize("open_bin_end", [np.nan, np.inf, -np.inf])
def test_open_bin_rejects_nonfinite_domain_end(open_bin_end: float) -> None:
    with pytest.raises(ObservationOperatorError, match="finite"):
        p1_bin_integral_operator(
            np.array([0.0, 1.0, 2.0]),
            [AgeBin(1.0, None, "1+")],
            open_bin_end=open_bin_end,
        )


@pytest.mark.parametrize(
    ("matrix", "input_dimension", "output_dimension", "output_labels"),
    [
        (
            xr.DataArray(
                [[0.5, 0.5]],
                dims=("age", "age_bin"),
                coords={"age": ["a"], "age_bin": [0.0, 1.0]},
            ),
            "age",
            "age_bin",
            ("a",),
        ),
        (
            xr.DataArray(
                [[0.5, np.nan]],
                dims=("age_bin", "age"),
                coords={"age_bin": ["a"], "age": [0.0, 1.0]},
            ),
            "age",
            "age_bin",
            ("a",),
        ),
        (
            xr.DataArray(
                [[0.5, 0.5]],
                dims=("age_bin", "age"),
                coords={"age_bin": ["a"]},
            ),
            "age",
            "age_bin",
            ("a",),
        ),
        (
            xr.DataArray(
                [[0.5, 0.5]],
                dims=("age_bin", "age"),
                coords={"age_bin": ["a"], "age": [0.0, 0.0]},
            ),
            "age",
            "age_bin",
            ("a",),
        ),
        (
            xr.DataArray(
                [[0.5, 0.5]],
                dims=("age_bin", "age"),
                coords={"age_bin": ["a"], "age": [0.0, 1.0]},
            ),
            "age",
            "age_bin",
            ("different",),
        ),
    ],
)
def test_observation_operator_rejects_invalid_matrix_contracts(
    matrix: xr.DataArray,
    input_dimension: str,
    output_dimension: str,
    output_labels: tuple[str, ...],
) -> None:
    with pytest.raises(ObservationOperatorError):
        ObservationOperator(
            matrix,
            input_dimension,
            output_dimension,
            output_labels,
        )


def test_observation_operator_requires_distinct_dimension_names() -> None:
    matrix = xr.DataArray(
        [[1.0]],
        dims=("age_bin", "age"),
        coords={"age_bin": ["a"], "age": [0.0]},
    )
    with pytest.raises(ObservationOperatorError, match="distinct"):
        ObservationOperator(matrix, "age_bin", "age_bin", ("a",))


def test_surface_application_rejects_output_dimension_collision() -> None:
    operator = _operator()
    surface = RateSurface(
        xr.DataArray(
            np.ones((3, 2)),
            dims=("age", "age_bin"),
            coords={"age": [0.0, 1.0, 2.0], "age_bin": ["a", "b"]},
        ),
        unit=Unit.PER_YEAR,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )

    with pytest.raises(ObservationOperatorError, match="shared dimensions"):
        operator @ surface


def test_surface_application_preserves_noncolliding_dimensions() -> None:
    operator = _operator()
    surface = RateSurface(
        xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            dims=("age", "sex"),
            coords={"age": [0.0, 1.0, 2.0], "sex": ["female", "male"]},
        ),
        unit=Unit.PER_YEAR,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )

    observed = operator @ surface

    assert observed.dims == ("age_bin", "sex")
    assert observed.get_index("sex").tolist() == ["female", "male"]
    np.testing.assert_allclose(observed.values, [[2.0, 3.0], [4.0, 5.0]])


def test_surface_application_rejects_input_coordinate_mismatch() -> None:
    operator = _operator()
    surface = RateSurface(
        xr.DataArray(
            np.ones(3),
            dims=("age",),
            coords={"age": [0.0, 1.0, 3.0]},
        ),
        unit=Unit.PER_YEAR,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
    )

    with pytest.raises(ObservationOperatorError, match="coordinates do not match"):
        operator @ surface


def test_composition_rejects_noncontracted_shared_dimension() -> None:
    inner = ObservationOperator(
        xr.DataArray(
            [[1.0, 0.0], [0.0, 1.0]],
            dims=("age_bin", "coarse_bin"),
            coords={"age_bin": ["a", "b"], "coarse_bin": ["x", "y"]},
        ),
        "coarse_bin",
        "age_bin",
        ("a", "b"),
    )
    outer = ObservationOperator(
        xr.DataArray(
            [[1.0, 1.0], [0.0, 1.0]],
            dims=("coarse_bin", "age_bin"),
            coords={"coarse_bin": ["x", "y"], "age_bin": ["a", "b"]},
        ),
        "age_bin",
        "coarse_bin",
        ("x", "y"),
    )

    with pytest.raises(ObservationOperatorError, match="shared dimensions"):
        outer @ inner


def test_composition_rejects_intermediate_coordinate_mismatch() -> None:
    inner = _operator()
    outer = ObservationOperator(
        xr.DataArray(
            [[1.0, 1.0]],
            dims=("coarse_bin", "age_bin"),
            coords={"coarse_bin": ["all"], "age_bin": ["a", "different"]},
        ),
        "age_bin",
        "coarse_bin",
        ("all",),
    )

    with pytest.raises(ObservationOperatorError, match="coordinates do not match"):
        outer @ inner


def test_matmul_with_foreign_operand_returns_not_implemented() -> None:
    operator = _operator()

    assert operator.__matmul__(object()) is NotImplemented
    with pytest.raises(TypeError):
        operator @ object()
