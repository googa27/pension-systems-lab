from __future__ import annotations

from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKED_IN_NOTEBOOK = PROJECT_ROOT / "notebooks" / "chile_demographic_pde.ipynb"


def _notebook() -> nbformat.NotebookNode:
    return nbformat.read(CHECKED_IN_NOTEBOOK, as_version=4)


def _code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def _all_sources(notebook: nbformat.NotebookNode) -> str:
    return "\n".join(str(cell.source) for cell in notebook.cells)


def test_checked_in_notebook_is_cleared_for_publication() -> None:
    code_cells = _code_cells(_notebook())

    assert code_cells
    assert all(cell.execution_count is None for cell in code_cells)
    assert all(cell.outputs == [] for cell in code_cells)


def test_notebook_keeps_current_observation_matrix_api() -> None:
    code_sources = "\n".join(cell.source for cell in _code_cells(_notebook()))

    assert "chile_demographic_pde.observation" not in code_sources
    assert "PiecewiseLinearBinIntegrator" not in code_sources
    assert (
        "from chile_demographic_pde.data.observation import p1_bin_integral_operator"
    ) in code_sources
    assert "fertility_operator.matrix.values @ (" in code_sources
    assert "mortality_operator.matrix.values @ (" in code_sources


def test_censo_section_leads_with_grouped_counts_and_labels_the_curve() -> None:
    notebook = _notebook()
    sources = _all_sources(notebook)
    title = "Censo 2024 observed age-bin counts and resident-population state"

    assert title in sources
    assert "Regularized continuous-age reconstruction of Censo 2024 population" not in sources
    censo_cells = [
        cell for cell in _code_cells(notebook) if f'ax.set_title("{title}")' in cell.source
    ]
    assert len(censo_cells) == 1
    censo_source = str(censo_cells[0].source)
    assert "population_total" in censo_source
    assert "bin_widths" in censo_source
    assert "ax.bar(" in censo_source
    assert censo_source.index("ax.bar(") < censo_source.index("ax.plot(")
    assert "Official Censo model-domain count density" in censo_source
    assert "Descriptive exact-bin overlay — not observed single-age data" in censo_source


def test_censo_section_displays_exact_bin_and_grid_diagnostics() -> None:
    sources = _all_sources(_notebook())

    assert "censo_reconstruction_male.diagnostics" in sources
    assert "censo_reconstruction_female.diagnostics" in sources
    assert "max_absolute_bin_error" in sources
    assert "max_relative_bin_error" in sources
    assert "local_curvature_indicator" in sources
    assert "grid_sensitivity" not in sources
    assert "censo_grid_comparison_male" in sources
    assert "censo_grid_comparison_female" in sources
    assert "max_absolute_density_difference" in sources
    assert "relative_linf_density_difference" in sources
    assert "coarse_node_count" in sources
    assert "refined_node_count" in sources
    assert "model-domain interval $[85,105]$" in sources
    assert "display(censo_reconstruction_diagnostics)" in sources
    assert "not observed single-age data" in sources
    assert "not resident-population estimates" in sources
