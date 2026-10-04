from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from chile_demographic_pde.core.provenance import SourceRef
from chile_demographic_pde.countries.chile.deis import (
    DeisDeathRelease,
    DeisReleaseControls,
    normalize_deis_deaths,
)
from chile_demographic_pde.data.acquisition import AcquiredAsset
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
    verify_release_assets,
)


def _official_release(
    *,
    environment_name: str,
    key: str,
    url: str,
    sha256: str,
    vintage: str,
    provisional: bool,
    released_at: date,
    observation_start: date,
    observation_cutoff: date,
    member: str,
    controls: DeisReleaseControls,
) -> DeisDeathRelease:
    supplied = os.getenv(environment_name)
    if not supplied:
        pytest.skip(
            f"Set {environment_name} to the checksum-pinned official DEIS archive "
            "to run this no-network integration test."
        )
    path = Path(supplied)
    retrieved_at = datetime(2026, 7, 19, tzinfo=UTC)
    acquired = AcquiredAsset(
        key=key,
        path=path,
        sha256=sha256,
        retrieved_at=retrieved_at,
    )
    manifest = ReleaseManifest.create(
        schema_version="deis-death-release-v1",
        assets=(
            RequiredReleaseAsset(
                source_key=key,
                sha256=sha256,
                available_at=retrieved_at,
                released_at=released_at,
                release_missingness_reason=None,
            ),
        ),
        primary_fact_source_key=key,
        identity_registry=IdentityRegistry(),
    )
    source = SourceRef(
        source_key=key,
        name=f"DEIS deaths {vintage}",
        url=url,
        release_date=released_at,
        release_missingness_reason=None,
        retrieved_at=retrieved_at,
        sha256=sha256,
        vintage=vintage,
        provisional=provisional,
    )
    return DeisDeathRelease(
        assets=verify_release_assets(manifest, (acquired,)),
        source_key=key,
        sources=(source,),
        released_at=released_at,
        observation_start=observation_start,
        observation_cutoff=observation_cutoff,
        archive_member=member,
        controls=controls,
    )


def test_preliminary_7_july_2026_archive_matches_pinned_controls() -> None:
    release = _official_release(
        environment_name="DEIS_PRELIM_20260707_ARCHIVE",
        key="deis_deaths_preliminary_20260707",
        url=(
            "https://repositoriodeis.minsal.cl/DatosAbiertos/VITALES/"
            "DEFUNCIONES_FUENTE_DEIS_2024_2026_07072026.zip"
        ),
        sha256="22df2d0cc12c57cd351fff7a5412643f8193bda993796c1f06f9aa75293e1cf2",
        vintage="2026-07-07",
        provisional=True,
        released_at=date(2026, 7, 7),
        observation_start=date(2024, 1, 1),
        observation_cutoff=date(2026, 7, 4),
        member="DEFUNCIONES_FUENTE_DEIS_2024_2026_07072026.csv",
        controls=DeisReleaseControls(
            raw_rows=315_896,
            counts_by_year=((2024, 126_928), (2025, 126_497), (2026, 62_471)),
            counts_by_sex=(
                ("Hombre", 164_352),
                ("Indeterminado", 10),
                ("Mujer", 151_534),
            ),
            counts_by_age_type=(
                ("0", 16),
                ("1", 313_618),
                ("2", 542),
                ("3", 844),
                ("4", 876),
            ),
            age_99_year_deaths=1_654,
            nonblank_external_causes=111_659,
            short_numeric_communes=156_643,
            archive_size=11_133_453,
            member_size=95_658_993,
        ),
    )
    result = normalize_deis_deaths(
        release,
        identity_registry=IdentityRegistry(),
    )
    assert result.audit.normalized_total == 315_896
    assert result.audit.meaningful_external_causes == 19_603
    assert result.audit.blank_communes == 49
    assert result.audit.missing_code_communes == 25


def test_final_1990_2023_archive_matches_pinned_identity_and_2023_control() -> None:
    release = _official_release(
        environment_name="DEIS_FINAL_1990_2023_ARCHIVE",
        key="deis_deaths_final_1990_2023",
        url=(
            "https://repositoriodeis.minsal.cl/DatosAbiertos/VITALES/"
            "DEFUNCIONES_FUENTE_DEIS_1990_2023_CIFRAS_OFICIALES.zip"
        ),
        sha256="311b56533a60cc5b554575de3cec6575063aa1a1ce17d25c6017a2895cdd60e7",
        vintage="1990-2023-final",
        provisional=False,
        released_at=date(2025, 1, 1),
        observation_start=date(1990, 1, 1),
        observation_cutoff=date(2023, 12, 31),
        member="DEFUNCIONES_FUENTE_DEIS_1990_2023_CIFRAS_OFICIALES.csv",
        controls=DeisReleaseControls(
            counts_by_year=((2023, 122_218),),
            archive_size=93_202_577,
            member_size=869_415_741,
        ),
    )
    result = normalize_deis_deaths(
        release,
        identity_registry=IdentityRegistry(),
    )
    assert result.audit.counts_by_year[-1] == (2023, 122_218)
    assert result.audit.missing_occurrence_dates == 323
