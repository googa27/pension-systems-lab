"""Deterministic public command line interface for the pension systems lab."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from chile_demographic_pde import __version__
from chile_demographic_pde.data.loaders import load_project_data
from chile_demographic_pde.pde.fem import AgeTransportFEM
from chile_demographic_pde.pde.model import TwoSexDemographicModel

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SITE_PACKAGES_ROOT = Path(__file__).resolve().parents[1]


def _default_data_dir() -> Path:
    """Find curated CSVs from a source checkout or an installed wheel."""

    for root in (_SOURCE_ROOT, _SITE_PACKAGES_ROOT, Path.cwd()):
        candidate = root / "data" / "processed"
        if candidate.is_dir():
            return candidate
    return _SOURCE_ROOT / "data" / "processed"


DATA_DIR = _default_data_dir()
ROOT = DATA_DIR.parents[1]


@dataclass(frozen=True, slots=True)
class CliResult:
    """Exit code plus JSON-serializable payload."""

    exit_code: int
    payload: dict[str, Any]


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def capabilities_result() -> CliResult:
    """Return the public capability manifest without executing heavy workflows."""

    payload: dict[str, Any] = {
        "package": "chile-demographic-pde",
        "import_name": "chile_demographic_pde",
        "cli": "pension-lab",
        "version": __version__,
        "status": "research_prototype_not_official_projection",
        "commands": ["capabilities", "demo", "verify-data", "pension-demo"],
        "data_policy": {
            "bundled_inputs": "curated_public_csv_only",
            "private_data": False,
            "raw_excel_workbooks_bundled": False,
        },
        "models": [
            "two_sex_mckendrick_von_foerster_fem",
            "grouped_poisson_spline_rates",
            "penalized_poisson_trends",
            "typed_demographic_cube",
            "public_afp_operations_optional_integration",
        ],
    }
    return CliResult(0, payload)


def demo_result() -> CliResult:
    """Run a tiny deterministic two-sex FEM scenario for smoke testing."""

    solver = AgeTransportFEM(maximum_age=6.0, nodes=13, artificial_diffusion=0.002)
    age = solver.age
    initial_male = 950.0 * np.exp(-0.5 * ((age - 2.8) / 1.25) ** 2) + 45.0
    initial_female = 980.0 * np.exp(-0.5 * ((age - 2.6) / 1.20) ** 2) + 48.0

    def fertility(grid: np.ndarray, _period: int) -> np.ndarray:
        return 0.030 * np.exp(-0.5 * ((grid - 2.4) / 0.80) ** 2)

    def mortality_male(grid: np.ndarray, _period: int) -> np.ndarray:
        return 0.010 + 0.004 * grid

    def mortality_female(grid: np.ndarray, _period: int) -> np.ndarray:
        return 0.008 + 0.003 * grid

    result = TwoSexDemographicModel(solver, sex_ratio_male=0.5053).simulate(
        initial_male=initial_male,
        initial_female=initial_female,
        fertility_rate=fertility,
        mortality_male=mortality_male,
        mortality_female=mortality_female,
        periods=4,
        dt=0.25,
    )
    payload: dict[str, Any] = {
        "scenario": "small_two_sex_fem_smoke",
        "age_nodes": int(age.size),
        "periods": 4,
        "dt_years": 0.25,
        "births_total": round(float(result.predicted_births.sum()), 6),
        "deaths_total": round(float(result.predicted_deaths.sum()), 6),
        "final_male_total": round(float(np.trapezoid(result.states_male[-1], age)), 6),
        "final_female_total": round(float(np.trapezoid(result.states_female[-1], age)), 6),
        "nonnegative": bool(
            np.min(result.states_male) >= 0.0 and np.min(result.states_female) >= 0.0
        ),
    }
    return CliResult(0, payload)


def verify_data_result(data_dir: Path = DATA_DIR) -> CliResult:
    """Validate bundled curated public CSVs and return a checksum manifest."""

    data = load_project_data(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    expected = {
        "births_age_total": 12_512,
        "deaths_age_total": 10_041,
        "censo_population_total": 18_480_432,
        "monthly_rows": 16,
        "april_2026_births_total": 12_542,
    }
    observed = {
        "births_age_total": int(data.births_age["births_total"].sum()),
        "deaths_age_total": int(data.deaths_age["deaths_total"].sum()),
        "censo_population_total": int(data.population_age["population_total"].sum()),
        "monthly_rows": len(data.monthly),
        "april_2026_births_total": int(
            data.monthly.loc[data.monthly["period"] == "2026-04", "births_total"].iloc[0]
        ),
    }
    checks = {key: observed[key] == expected[key] for key in expected}
    manifest_path = data_dir.parent / "PUBLIC_MANIFEST.json"
    manifest_checks: dict[str, bool] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        records = {Path(record["path"]).name: record for record in manifest["files"]}
        manifest_checks["file_set"] = set(records) == {path.name for path in files}
        for path in files:
            record = records.get(path.name, {})
            manifest_checks[path.name] = (
                record.get("sha256") == _sha256(path) and record.get("bytes") == path.stat().st_size
            )
    else:
        manifest_checks["manifest_present"] = False
    checks["manifest_integrity"] = all(manifest_checks.values())
    payload: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "data_dir": data_dir.relative_to(ROOT).as_posix()
        if data_dir.is_relative_to(ROOT)
        else data_dir.as_posix(),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix()
                if path.is_relative_to(ROOT)
                else path.as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "manifest_checks": manifest_checks,
        "private_data": False,
        "raw_excel_workbooks_bundled": False,
    }
    return CliResult(0 if all(checks.values()) else 1, payload)


def pension_demo_result() -> CliResult:
    """Invoke the optional public ``afp_operations`` demo when installed."""

    try:
        module = importlib.import_module("afp_operations")
    except ModuleNotFoundError:
        return CliResult(
            0,
            {
                "status": "optional_dependency_not_installed",
                "dependency": "afp_operations",
                "installation": "Install the public afp-operations package separately; see docs/ECOSYSTEM.md",
                "message": "Install the public afp_operations package to run its pension demo.",
            },
        )

    for attribute in ("public_demo", "demo", "run_demo"):
        candidate = getattr(module, attribute, None)
        if callable(candidate):
            value = candidate()
            return CliResult(
                0,
                {
                    "status": "ok",
                    "dependency": "afp_operations",
                    "entrypoint": f"afp_operations.{attribute}",
                    "result": value,
                },
            )
    return CliResult(
        1,
        {
            "status": "missing_public_demo",
            "dependency": "afp_operations",
            "message": "Imported afp_operations but found no callable public_demo, demo, or run_demo.",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pension-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("capabilities", help="Print deterministic public capability JSON.")
    subparsers.add_parser("demo", help="Run a tiny deterministic two-sex FEM demo.")
    subparsers.add_parser("verify-data", help="Validate bundled curated public CSVs and checksums.")
    subparsers.add_parser(
        "pension-demo", help="Invoke optional public afp_operations demo if installed."
    )
    return parser


def run(argv: list[str] | None = None) -> CliResult:
    args = build_parser().parse_args(argv)
    if args.command == "capabilities":
        return capabilities_result()
    if args.command == "demo":
        return demo_result()
    if args.command == "verify-data":
        return verify_data_result()
    if args.command == "pension-demo":
        return pension_demo_result()
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    result = run(argv)
    _emit(result.payload)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
