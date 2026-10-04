"""Print source-table checksums and verify the curated official totals."""

from __future__ import annotations

import hashlib
from pathlib import Path

from chile_demographic_pde.data.loaders import load_project_data

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    data = load_project_data(ROOT / "data" / "processed")
    assert int(data.births_age["births_total"].sum()) == 12_512
    assert int(data.deaths_age["deaths_total"].sum()) == 10_041
    assert int(data.population_age["population_total"].sum()) == 18_480_432
    for path in sorted((ROOT / "data").rglob("*")):
        if path.is_file():
            print(f"{sha256(path)}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
