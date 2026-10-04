"""Adapters for official Chilean demographic releases."""

from chile_demographic_pde.countries.chile.census import (
    CuratedDomainRelease,
    CuratedTableContract,
    canonical_curated_table_bytes,
    normalize_censo_grouped_curated,
)
from chile_demographic_pde.countries.chile.deis import (
    DeisBirthArchiveId,
    DeisBirthAssetContract,
    DeisBirthRelease,
    DeisBirthReleaseControls,
    aggregate_deis_births,
    iter_normalize_deis_births,
    normalize_deis_births,
    normalize_deis_deaths,
)
from chile_demographic_pde.countries.chile.population import (
    normalize_ine_population_curated,
)
from chile_demographic_pde.countries.chile.vital_statistics import (
    IneBirthReleaseKind,
    compare_deis_births_to_ine_controls,
    normalize_ine_birth_controls_curated,
)

__all__ = [
    "CuratedDomainRelease",
    "CuratedTableContract",
    "DeisBirthArchiveId",
    "DeisBirthAssetContract",
    "DeisBirthRelease",
    "DeisBirthReleaseControls",
    "IneBirthReleaseKind",
    "aggregate_deis_births",
    "canonical_curated_table_bytes",
    "compare_deis_births_to_ine_controls",
    "iter_normalize_deis_births",
    "normalize_censo_grouped_curated",
    "normalize_deis_births",
    "normalize_deis_deaths",
    "normalize_ine_birth_controls_curated",
    "normalize_ine_population_curated",
]
