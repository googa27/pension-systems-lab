from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from chile_demographic_pde.cli import verify_data_result

ROOT = Path(__file__).resolve().parents[2]


def test_verify_data_json_contract() -> None:
    result = verify_data_result(ROOT / "data" / "processed")
    assert result.exit_code == 0
    payload = result.payload
    assert payload["status"] == "ok"
    assert payload["observed"]["births_age_total"] == 12_512
    assert payload["observed"]["deaths_age_total"] == 10_041
    assert payload["observed"]["censo_population_total"] == 18_480_432
    assert len(payload["files"]) == 4
    assert all(len(item["sha256"]) == 64 for item in payload["files"])


def test_public_manifest_hashes_every_curated_csv() -> None:
    manifest = json.loads((ROOT / "data" / "PUBLIC_MANIFEST.json").read_text())
    records = {item["path"]: item for item in manifest["files"]}
    csv_paths = sorted(
        path.relative_to(ROOT).as_posix() for path in (ROOT / "data" / "processed").glob("*.csv")
    )

    assert sorted(records) == csv_paths
    assert manifest["private_or_person_level_data_included"] is False
    assert manifest["raw_workbooks_or_legacy_notebook_outputs_included"] is False
    assert manifest["retrieval_dates_verified"] is False
    assert manifest["licensing_verified"] is False
    assert manifest["licensing_status"] == "not_verified"

    for relative_path in csv_paths:
        path = ROOT / relative_path
        record = records[relative_path]
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["bytes"] == path.stat().st_size
        assert record["retrieval_date"] is None
        assert record["licensing_status"] == "not_verified"
        assert record["source_urls"]


def test_verify_data_rejects_hash_drift_without_aggregate_drift(tmp_path: Path) -> None:
    import shutil

    data = tmp_path / "data"
    shutil.copytree(ROOT / "data" / "processed", data / "processed")
    shutil.copy2(ROOT / "data" / "PUBLIC_MANIFEST.json", data / "PUBLIC_MANIFEST.json")
    csv = data / "processed" / "censo2024_chile_population_by_age_sex.csv"
    csv.write_bytes(csv.read_bytes() + b"\n")
    result = verify_data_result(data / "processed")
    assert result.exit_code == 1
    assert result.payload["checks"]["censo_population_total"] is True
    assert result.payload["checks"]["manifest_integrity"] is False


def test_module_cli_verify_data_emits_json() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "chile_demographic_pde.cli", "verify-data"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["private_data"] is False
