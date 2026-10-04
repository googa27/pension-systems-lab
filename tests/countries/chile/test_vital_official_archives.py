from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from chile_demographic_pde.core.provenance import SourceRef
from chile_demographic_pde.countries.chile.deis import (
    DeisBirthArchiveId,
    DeisBirthRelease,
    DeisBirthReleaseControls,
    normalize_deis_births,
)
from chile_demographic_pde.data.acquisition import AcquiredAsset
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
    verify_release_assets,
)

ANNUAL_COUNTS = {
    1992: 279_060,
    1993: 275_828,
    1994: 273_775,
    1995: 265_932,
    1996: 264_793,
    1997: 259_995,
    1998: 257_133,
    1999: 250_610,
    2000: 248_893,
    2001: 246_116,
    2002: 238_981,
    2003: 234_486,
    2004: 230_352,
    2005: 230_831,
    2006: 231_383,
    2007: 240_569,
    2008: 246_581,
    2009: 252_240,
    2010: 250_643,
    2011: 247_358,
    2012: 243_635,
    2013: 242_005,
    2014: 250_997,
    2015: 244_670,
    2016: 231_749,
    2017: 219_186,
    2018: 221_731,
    2019: 210_188,
    2020: 194_978,
    2021: 177_273,
    2022: 189_303,
    2023: 174_057,
}


def _release(
    *,
    environment_name: str,
    archive_id: DeisBirthArchiveId,
    sha256: str,
    released_at: date,
    start_year: int,
    end_year: int,
    archive_size: int,
    member_size: int,
    raw_rows: int,
    **control_overrides: object,
) -> DeisBirthRelease:
    supplied = os.getenv(environment_name)
    if not supplied:
        pytest.skip(
            f"Set {environment_name} to the exact pinned local DEIS birth "
            "archive; this integration test never downloads."
        )
    path = Path(supplied)
    member = f"Serie_Nacimientos_{archive_id.value}.csv"
    key = f"deis_births_{archive_id.value}"
    retrieved_at = datetime(2026, 7, 19, tzinfo=UTC)
    registry = IdentityRegistry()
    manifest = ReleaseManifest.create(
        schema_version="deis-birth-release-v1",
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
        identity_registry=registry,
    )
    source = SourceRef(
        source_key=key,
        name=f"DEIS births {archive_id.value}",
        url=("https://repositoriodeis.minsal.cl/DatosAbiertos/VITALES/NACIMIENTOS/"),
        release_date=released_at,
        release_missingness_reason=None,
        retrieved_at=retrieved_at,
        sha256=sha256,
        vintage=f"{archive_id.value}-final",
        provisional=False,
    )
    controls = DeisBirthReleaseControls(
        raw_rows=raw_rows,
        counts_by_year=tuple(
            (year, ANNUAL_COUNTS[year]) for year in range(start_year, end_year + 1)
        ),
        archive_size=archive_size,
        member_size=member_size,
        **control_overrides,
    )
    acquired = AcquiredAsset(
        key=key,
        path=path,
        sha256=sha256,
        retrieved_at=retrieved_at,
    )
    return DeisBirthRelease(
        assets=verify_release_assets(manifest, (acquired,)),
        source_key=key,
        sources=(source,),
        released_at=released_at,
        observation_start=date(start_year, 1, 1),
        observation_end=date(end_year, 12, 31),
        archive_id=archive_id,
        archive_member=member,
        geography_vintage="dpa-2018",
        registration_inclusion_cutoff="registered_through_following_march_31",
        controls=controls,
    )


def _counting_sidecar_sink() -> tuple[Callable[[pa.Table], None], list[int]]:
    state = [0, 0]

    def consume(table: pa.Table) -> None:
        row_count = table.num_rows
        assert row_count > 0
        row_numbers = table["archive_row_number"]
        first = row_numbers[0].as_py()
        last = row_numbers[row_count - 1].as_py()
        assert first == state[1] + 1
        assert last == state[1] + row_count
        state[0] += row_count
        state[1] = last

    return consume, state


def test_official_birth_archive_1992_2000_matches_pinned_controls() -> None:
    release = _release(
        environment_name="DEIS_BIRTHS_1992_2000_ARCHIVE",
        archive_id=DeisBirthArchiveId.PROFILE_1992_2000,
        sha256="0ddd16b46007995b73f63c3a081c7a39cffb9c7a1ed4591b71a75b3a5a3b355c",
        released_at=date(2025, 10, 23),
        start_year=1992,
        end_year=2000,
        archive_size=18_594_440,
        member_size=255_087_750,
        raw_rows=2_376_019,
        blank_birth_months=51,
        unknown_maternal_age_groups=4,
    )
    result = normalize_deis_births(
        release,
        identity_registry=IdentityRegistry(),
    )
    assert int(result.normalized["value"].sum()) == 2_376_019
    assert result.domain_batch.manifest.release_id == (release.assets.manifest.release_id)


def test_official_birth_archive_2001_2019_matches_pinned_controls() -> None:
    release = _release(
        environment_name="DEIS_BIRTHS_2001_2019_ARCHIVE",
        archive_id=DeisBirthArchiveId.PROFILE_2001_2019,
        sha256="239465b82672f0f869c93d9a15121942eb50195ded583c9297580fb36c6d1a6b",
        released_at=date(2026, 4, 17),
        start_year=2001,
        end_year=2019,
        archive_size=38_057_769,
        member_size=526_690_938,
        raw_rows=4_513_701,
        unknown_maternal_age_groups=1_205,
        unknown_residence_regions=1,
        lowercase_nationality_codes=8,
    )
    sink, state = _counting_sidecar_sink()
    result = normalize_deis_births(
        release,
        identity_registry=IdentityRegistry(),
        raw_feature_sink=sink,
    )
    assert int(result.normalized["value"].sum()) == 4_513_701
    assert state == [4_513_701, 4_513_701]
    assert result.domain_batch.manifest.release_id == (release.assets.manifest.release_id)


def test_official_birth_archive_2020_2023_matches_pinned_controls() -> None:
    release = _release(
        environment_name="DEIS_BIRTHS_2020_2023_ARCHIVE",
        archive_id=DeisBirthArchiveId.PROFILE_2020_2023,
        sha256="66c30a91d6cf470e8fe7571d734d7f04fe97cc62b107453a60106e3fa1ce7b5d",
        released_at=date(2026, 5, 4),
        start_year=2020,
        end_year=2023,
        archive_size=7_663_674,
        member_size=79_837_738,
        raw_rows=735_611,
        counts_by_sex=(("1", 375_094), ("2", 360_450), ("9", 67)),
        distinct_known_residence_regions=16,
    )
    result = normalize_deis_births(
        release,
        identity_registry=IdentityRegistry(),
    )
    assert int(result.normalized["value"].sum()) == 735_611
    assert len(result.normalized) == 19_227
    assert result.domain_batch.manifest.release_id == (release.assets.manifest.release_id)
