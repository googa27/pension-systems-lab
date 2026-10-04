from pathlib import Path

from chile_demographic_pde.data.loaders import load_project_data


def test_official_data_totals() -> None:
    data = load_project_data(Path("data/processed"))
    assert int(data.births_age["births_total"].sum()) == 12_512  # excludes unspecified 30
    assert int(data.deaths_age["deaths_total"].sum()) == 10_041
    assert int(data.population_age["population_total"].sum()) == 18_480_432
    assert len(data.monthly) == 16
    assert (
        int(data.monthly.loc[data.monthly["period"] == "2026-04", "births_total"].iloc[0]) == 12_542
    )
