from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from chile_demographic_pde.core.errors import ReconstructionError
from chile_demographic_pde.data import reconstruction
from chile_demographic_pde.data.age_groups import censo_age_bins
from chile_demographic_pde.data.loaders import load_project_data
from chile_demographic_pde.data.observation import AgeBin, p1_bin_integral_operator
from chile_demographic_pde.data.reconstruction import reconstruct_grouped_density


def test_censo_tail_remains_open_until_the_caller_resolves_it() -> None:
    data = load_project_data(Path("data/processed"))
    bins = censo_age_bins(data.population_age)

    assert bins[-1].label == "85 o más"
    assert bins[-1].upper is None
    operator = p1_bin_integral_operator(
        np.linspace(0.0, 105.0, 211),
        bins,
        open_bin_end=105.0,
    )
    assert operator.output_labels[-1] == "85 o más"


def test_constrained_reconstruction_preserves_every_finite_bin() -> None:
    nodes = np.linspace(0.0, 10.0, 21)
    bins = [
        AgeBin(0.0, 2.0, "0-1"),
        AgeBin(2.0, 5.0, "2-4"),
        AgeBin(5.0, 10.0, "5-9"),
    ]
    counts = np.array([100.0, 240.0, 160.0])

    result = reconstruct_grouped_density(nodes, bins, counts)

    assert result.diagnostics.converged
    assert result.diagnostics.max_absolute_bin_error < 1e-7
    assert result.diagnostics.max_relative_bin_error < 1e-9
    assert result.diagnostics.minimum_normalized_density >= 0.0
    assert result.diagnostics.minimum_density >= 0.0
    assert result.diagnostics.max_normalized_constraint_violation < 1e-12
    assert result.diagnostics.local_curvature_indicator >= 0.0
    assert np.min(result.density.values) >= -1e-10
    np.testing.assert_allclose(result.reconstructed_bins.values, counts, atol=1e-7)
    np.testing.assert_allclose(
        result.absolute_residuals.values,
        np.abs(result.residuals.values),
    )
    np.testing.assert_allclose(
        result.relative_residuals.values,
        np.abs(result.residuals.values) / np.maximum(counts, 1.0),
    )


def test_open_tail_is_rejected_without_explicit_endpoint() -> None:
    nodes = np.linspace(0.0, 105.0, 106)

    with pytest.raises(ReconstructionError, match="open_bin_end"):
        reconstruct_grouped_density(
            nodes,
            [AgeBin(85.0, None, "85+")],
            np.array([1_000.0]),
        )


def test_reconstruction_is_scale_equivariant_and_grid_stable() -> None:
    bins = [AgeBin(0.0, 5.0, "0-4"), AgeBin(5.0, 10.0, "5-9")]
    counts = np.array([100.0, 60.0])

    coarse = reconstruct_grouped_density(np.linspace(0, 10, 21), bins, counts)
    scaled = reconstruct_grouped_density(np.linspace(0, 10, 21), bins, 1_000 * counts)
    fine = reconstruct_grouped_density(np.linspace(0, 10, 41), bins, counts)

    np.testing.assert_allclose(
        scaled.density.values,
        1_000 * coarse.density.values,
        rtol=1e-7,
        atol=1e-7,
    )
    fine_on_coarse = fine.density.interp(age=coarse.density["age"])
    np.testing.assert_allclose(fine_on_coarse, coarse.density, rtol=0.08, atol=0.08)


def test_grid_comparison_matches_an_explicit_refined_reconstruction() -> None:
    nodes = np.linspace(0.0, 10.0, 11)
    bins = [
        AgeBin(0.0, 2.0, "0-1"),
        AgeBin(2.0, 5.0, "2-4"),
        AgeBin(5.0, 10.0, "5-9"),
    ]
    counts = np.array([100.0, 240.0, 160.0])
    coarse = reconstruct_grouped_density(nodes, bins, counts)
    comparison = reconstruction.compare_reconstruction_grids(
        nodes,
        bins,
        counts,
        coarse_result=coarse,
        refinement_factor=2,
    )
    refined_nodes = np.linspace(0.0, 10.0, 21)
    refined = reconstruct_grouped_density(refined_nodes, bins, counts)
    refined_on_coarse = refined.density.interp(age=coarse.density["age"])
    differences = np.abs(refined_on_coarse.values - coarse.density.values)
    expected_absolute = float(np.max(differences))
    expected_scale = max(
        float(np.max(np.abs(coarse.density.values))),
        float(np.max(np.abs(refined_on_coarse.values))),
    )

    assert comparison.coarse_node_count == 11
    assert comparison.refined_node_count == 21
    assert comparison.refinement_factor == 2
    assert comparison.max_absolute_density_difference == pytest.approx(expected_absolute)
    assert comparison.density_scale == pytest.approx(expected_scale)
    assert comparison.relative_linf_density_difference == pytest.approx(
        expected_absolute / expected_scale
    )
    assert comparison.max_absolute_density_difference > 0.0
    assert np.all(
        np.isfinite(
            [
                comparison.max_absolute_density_difference,
                comparison.density_scale,
                comparison.relative_linf_density_difference,
            ]
        )
    )


def _coarse_result_fixture() -> tuple[
    np.ndarray,
    list[AgeBin],
    np.ndarray,
    reconstruction.ReconstructionResult,
]:
    nodes = np.linspace(0.0, 2.0, 5)
    bins = [AgeBin(0.0, 1.0, "0"), AgeBin(1.0, 2.0, "1")]
    counts = np.array([4.0, 2.0])
    result = reconstruct_grouped_density(nodes, bins, counts)
    return nodes, bins, counts, result


def test_grid_comparison_rejects_wrong_coarse_result_type() -> None:
    nodes, bins, counts, _result = _coarse_result_fixture()

    with pytest.raises(ReconstructionError, match="coarse_result") as error:
        reconstruction.compare_reconstruction_grids(
            nodes,
            bins,
            counts,
            coarse_result=object(),  # type: ignore[arg-type]
        )

    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    "invalid_density",
    ["nan", "negative", "wrong_shape", "wrong_ages", "wrong_dimension", "forged"],
)
def test_grid_comparison_recomputes_and_validates_coarse_density(
    invalid_density: str,
) -> None:
    nodes, bins, counts, result = _coarse_result_fixture()
    density = result.density.copy(deep=True)
    if invalid_density == "nan":
        density.values[0] = np.nan
    elif invalid_density == "negative":
        density.values[0] = -1.0
    elif invalid_density == "wrong_shape":
        density = density.isel(age=slice(None, -1))
    elif invalid_density == "wrong_ages":
        density = density.assign_coords(age=density["age"].values + 0.1)
    elif invalid_density == "wrong_dimension":
        density = density.rename({"age": "node"})
    elif invalid_density == "forged":
        density = density + 1.0
    invalid = replace(result, density=density)

    with pytest.raises(ReconstructionError, match="coarse_result") as error:
        reconstruction.compare_reconstruction_grids(
            nodes,
            bins,
            counts,
            coarse_result=invalid,
        )

    assert error.value.__cause__ is not None


def test_grid_comparison_rejects_unconverged_coarse_diagnostics() -> None:
    nodes, bins, counts, result = _coarse_result_fixture()
    invalid = replace(
        result,
        diagnostics=replace(result.diagnostics, converged=False),
    )

    with pytest.raises(ReconstructionError, match="coarse_result") as error:
        reconstruction.compare_reconstruction_grids(
            nodes,
            bins,
            counts,
            coarse_result=invalid,
        )

    assert error.value.__cause__ is not None


def test_grid_comparison_validates_stored_coarse_bin_labels() -> None:
    nodes, bins, counts, result = _coarse_result_fixture()
    invalid = replace(
        result,
        reconstructed_bins=result.reconstructed_bins.assign_coords(age_bin=["wrong-0", "wrong-1"]),
    )

    with pytest.raises(ReconstructionError, match="coarse_result") as error:
        reconstruction.compare_reconstruction_grids(
            nodes,
            bins,
            counts,
            coarse_result=invalid,
        )

    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (np.array([]), "nonempty"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, np.inf]), "finite"),
        (np.array([1.0, -1.0]), "nonnegative"),
        (np.array([[1.0, 2.0]]), "vector matching bins"),
        (np.array([1.0]), "vector matching bins"),
    ],
)
def test_reconstruction_rejects_invalid_grouped_totals(
    counts: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ReconstructionError, match=message):
        reconstruct_grouped_density(
            np.linspace(0.0, 2.0, 5),
            [AgeBin(0.0, 1.0, "0"), AgeBin(1.0, 2.0, "1")],
            counts,
        )


@pytest.mark.parametrize(
    "nodes",
    [
        np.array([0.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([0.0, 2.0, 1.0]),
        np.array([0.0, np.nan, 2.0]),
        np.array([0.0, np.inf, 2.0]),
    ],
)
def test_reconstruction_rejects_invalid_age_nodes(nodes: np.ndarray) -> None:
    with pytest.raises(ReconstructionError, match="finite strictly increasing"):
        reconstruct_grouped_density(
            nodes,
            [AgeBin(0.0, 1.0, "0")],
            np.array([1.0]),
        )


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.nan, np.inf])
def test_reconstruction_rejects_invalid_tolerance(tolerance: float) -> None:
    with pytest.raises(ReconstructionError, match="tolerance"):
        reconstruct_grouped_density(
            np.linspace(0.0, 1.0, 3),
            [AgeBin(0.0, 1.0, "0")],
            np.array([1.0]),
            tolerance=tolerance,
        )


@pytest.mark.parametrize("tolerance", ["bad", object(), np.array([1e-7])])
def test_reconstruction_wraps_malformed_tolerance(tolerance: object) -> None:
    with pytest.raises(ReconstructionError, match="tolerance") as error:
        reconstruct_grouped_density(
            np.linspace(0.0, 1.0, 3),
            [AgeBin(0.0, 1.0, "0")],
            np.array([1.0]),
            tolerance=tolerance,  # type: ignore[arg-type]
        )

    assert error.value.__cause__ is not None


@pytest.mark.parametrize("open_bin_end", [np.nan, np.inf, -np.inf, 85.0])
def test_reconstruction_rejects_invalid_open_bin_resolution(open_bin_end: float) -> None:
    with pytest.raises(ReconstructionError, match="open_bin_end"):
        reconstruct_grouped_density(
            np.linspace(0.0, 105.0, 106),
            [AgeBin(85.0, None, "85+")],
            np.array([1_000.0]),
            open_bin_end=open_bin_end,
        )


@pytest.mark.parametrize("open_bin_end", ["bad", object(), np.array([105.0])])
def test_reconstruction_wraps_malformed_open_bin_end(open_bin_end: object) -> None:
    with pytest.raises(ReconstructionError, match="open_bin_end") as error:
        reconstruct_grouped_density(
            np.linspace(0.0, 105.0, 106),
            [AgeBin(85.0, None, "85+")],
            np.array([1_000.0]),
            open_bin_end=open_bin_end,  # type: ignore[arg-type]
        )

    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    ("solution", "objective", "message"),
    [
        (np.array([-1.0, 2.5, 0.0]), 1.0, "nonnegative"),
        (np.array([np.nan, 2.0, 0.0]), 1.0, "finite solution"),
        (np.array([0.0, 2.0, 0.0]), np.nan, "finite objective"),
    ],
)
def test_reconstruction_rejects_invalid_successful_optimizer_results(
    monkeypatch: pytest.MonkeyPatch,
    solution: np.ndarray,
    objective: float,
    message: str,
) -> None:
    def invalid_minimize(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            message="claimed success",
            x=solution,
            fun=objective,
        )

    monkeypatch.setattr(reconstruction, "minimize", invalid_minimize)

    with pytest.raises(ReconstructionError, match=message) as error:
        reconstruct_grouped_density(
            np.array([0.0, 0.5, 1.0]),
            [AgeBin(0.0, 1.0, "all")],
            np.array([1.0]),
        )

    assert error.value.__cause__ is not None


def test_reconstruction_wraps_observation_operator_failures() -> None:
    with pytest.raises(ReconstructionError, match="observation operator") as error:
        reconstruct_grouped_density(
            np.linspace(0.0, 1.0, 3),
            [AgeBin(0.0, 2.0, "outside")],
            np.array([1.0]),
        )

    assert error.value.__cause__ is not None


def test_reconstruction_result_has_exact_value_equality_and_is_unhashable() -> None:
    result = reconstruct_grouped_density(
        np.linspace(0.0, 2.0, 5),
        [AgeBin(0.0, 1.0, "0"), AgeBin(1.0, 2.0, "1")],
        np.array([4.0, 2.0]),
    )
    equal = replace(
        result,
        density=result.density.copy(deep=True),
        reconstructed_bins=result.reconstructed_bins.copy(deep=True),
        residuals=result.residuals.copy(deep=True),
        absolute_residuals=result.absolute_residuals.copy(deep=True),
        relative_residuals=result.relative_residuals.copy(deep=True),
    )
    different = replace(result, density=result.density + 1.0)

    assert result == equal
    assert result != different
    assert result.__eq__(object()) is NotImplemented
    with pytest.raises(TypeError):
        hash(result)


def test_density_metadata_keeps_censo_enumeration_distinct_from_resident_state() -> None:
    result = reconstruct_grouped_density(
        np.linspace(0.0, 2.0, 5),
        [AgeBin(0.0, 1.0, "0"), AgeBin(1.0, 2.0, "1")],
        np.array([4.0, 2.0]),
    )

    assert result.density.attrs["population_basis"] == "censo_enumerated"
    assert "descriptive" in result.density.attrs["interpretation"].lower()
    assert "resident" in result.density.attrs["interpretation"].lower()
