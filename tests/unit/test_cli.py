from __future__ import annotations

from chile_demographic_pde.cli import capabilities_result, demo_result


def test_capabilities_manifest_is_public_and_deterministic() -> None:
    result = capabilities_result()
    assert result.exit_code == 0
    payload = result.payload
    assert payload["cli"] == "pension-lab"
    assert payload["import_name"] == "chile_demographic_pde"
    assert payload["data_policy"]["bundled_inputs"] == "curated_public_csv_only"
    assert payload["data_policy"]["private_data"] is False
    assert payload["data_policy"]["raw_excel_workbooks_bundled"] is False
    assert payload["status"] == "research_prototype_not_official_projection"


def test_demo_runs_actual_two_sex_fem_smoke() -> None:
    result = demo_result()
    assert result.exit_code == 0
    payload = result.payload
    assert payload["scenario"] == "small_two_sex_fem_smoke"
    assert payload["age_nodes"] == 13
    assert payload["periods"] == 4
    assert payload["nonnegative"] is True
    assert payload["births_total"] > 0.0
    assert payload["deaths_total"] > 0.0
