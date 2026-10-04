"""Execute the research notebook in-place using the locked uv environment."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

from chile_demographic_pde.reporting import write_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "chile_demographic_pde.ipynb"
MODEL_SUMMARY = PROJECT_ROOT / "reports" / "model_summary.json"
RESULTS = PROJECT_ROOT / "reports" / "RESULTS.md"


def main(
    *,
    project_root: Path = PROJECT_ROOT,
    notebook_path: Path = NOTEBOOK,
    summary_path: Path = MODEL_SUMMARY,
    results_path: Path = RESULTS,
) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(project_root)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, notebook_path)
    write_results(summary_path, results_path)
    print(f"Executed {notebook_path.relative_to(project_root)}")
    print(f"Built {results_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
