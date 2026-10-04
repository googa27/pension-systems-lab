from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.csv
import pytest

from chile_demographic_pde.core.errors import (
    ChecksumMismatchError,
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.core.provenance import SourceRef
from chile_demographic_pde.countries.chile.deis import (
    DEIS_DEATH_COLUMNS,
    DeisAgeGrain,
    DeisAggregation,
    DeisCauseGrain,
    DeisDeathRelease,
    DeisGeographyGrain,
    DeisNormalizationResult,
    DeisReleaseControls,
    DeisTemporalGrain,
    normalize_deis_deaths,
)
from chile_demographic_pde.data.acquisition import AcquiredAsset
from chile_demographic_pde.data.domain_schemas import (
    DemographicObservationSchema,
    validate_demographic_observations,
)
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
    VerifiedReleaseAssets,
    verify_release_assets,
)

EXPECTED_DEIS_DEATH_COLUMNS = (
    "AÑO",
    "FECHA_DEF",
    "SEXO_NOMBRE",
    "EDAD_TIPO",
    "EDAD_CANT",
    "COD_COMUNA",
    "COMUNA",
    "NOMBRE_REGION",
    "DIAG1",
    "CAPITULO_DIAG1",
    "GLOSA_CAPITULO_DIAG1",
    "CODIGO_GRUPO_DIAG1",
    "GLOSA_GRUPO_DIAG1",
    "CODIGO_CATEGORIA_DIAG1",
    "GLOSA_CATEGORIA_DIAG1",
    "CODIGO_SUBCATEGORIA_DIAG1",
    "GLOSA_SUBCATEGORIA_DIAG1",
    "DIAG2",
    "CAPITULO_DIAG2",
    "GLOSA_CAPITULO_DIAG2",
    "CODIGO_GRUPO_DIAG2",
    "GLOSA_GRUPO_DIAG2",
    "CODIGO_CATEGORIA_DIAG2",
    "GLOSA_CATEGORIA_DIAG2",
    "CODIGO_SUBCATEGORIA_DIAG2",
    "GLOSA_SUBCATEGORIA_DIAG2",
    "LUGAR_DEFUNCION",
)
DEFAULT_AGGREGATION = DeisAggregation()
EXPECTED_DEMOGRAPHIC_COLUMNS = tuple(DemographicObservationSchema.to_schema().columns)
FOREIGN_DOMAIN_COLUMNS = {
    "currency",
    "tenor_years",
    "station",
    "pollutant",
    "facility",
    "health_service",
    "mortality_table_id",
    "pension_type",
}
_RETRIEVED_AT = datetime(2025, 1, 16, tzinfo=UTC)


def _row(**updates: str) -> dict[str, str]:
    row = dict.fromkeys(EXPECTED_DEIS_DEATH_COLUMNS, "")
    row.update(
        {
            "AÑO": "2024",
            "FECHA_DEF": "2024-02-03",
            "SEXO_NOMBRE": "Mujer",
            "EDAD_TIPO": "1",
            "EDAD_CANT": "99",
            "COD_COMUNA": "8111",
            "COMUNA": "Tomé",
            "NOMBRE_REGION": "Del Bíobío",
            "DIAG1": "I64X",
            "CAPITULO_DIAG1": "I00-I99",
            "CODIGO_GRUPO_DIAG1": "I60-I69",
            "CODIGO_CATEGORIA_DIAG1": "I64",
            "CODIGO_SUBCATEGORIA_DIAG1": "I64X",
            "LUGAR_DEFUNCION": "Casa habitación",
        }
    )
    row.update(updates)
    return row


def _csv_bytes(
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...] = EXPECTED_DEIS_DEATH_COLUMNS,
) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=list(columns),
        delimiter=";",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("windows-1252")


def _write_zip(
    path: Path,
    member: str,
    payload: bytes,
    *,
    sibling: tuple[str, bytes] | None = None,
    compression: int = zipfile.ZIP_DEFLATED,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr(member, payload)
        if sibling is not None:
            archive.writestr(*sibling)


def _archive_path(tmp_path: Path, key: str) -> Path:
    return tmp_path / f"{key}.zip"


def _attest_archive(
    path: Path,
    *,
    key: str,
    provisional: bool,
    released_at: date,
    observation_start: date,
    observation_cutoff: date,
    archive_member: str,
    controls: DeisReleaseControls,
) -> DeisDeathRelease:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    vintage = "test-preliminary" if provisional else "test-final"
    manifest = ReleaseManifest.create(
        schema_version="deis-death-release-v1",
        assets=(
            RequiredReleaseAsset(
                source_key=key,
                sha256=digest,
                available_at=_RETRIEVED_AT,
                released_at=released_at,
                release_missingness_reason=None,
            ),
        ),
        primary_fact_source_key=key,
        identity_registry=IdentityRegistry(),
    )
    source = SourceRef(
        source_key=key,
        name="DEIS deaths test fixture",
        url="https://example.test/deis",
        release_date=released_at,
        release_missingness_reason=None,
        retrieved_at=_RETRIEVED_AT,
        sha256=digest,
        vintage=vintage,
        provisional=provisional,
    )
    acquired = AcquiredAsset(
        key=key,
        path=path,
        sha256=digest,
        retrieved_at=_RETRIEVED_AT,
    )
    return DeisDeathRelease(
        assets=verify_release_assets(manifest, (acquired,)),
        source_key=key,
        sources=(source,),
        released_at=released_at,
        observation_start=observation_start,
        observation_cutoff=observation_cutoff,
        archive_member=archive_member,
        controls=controls,
    )


def _release_for(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...] = EXPECTED_DEIS_DEATH_COLUMNS,
    controls: DeisReleaseControls | None = None,
    provisional: bool = True,
    member: str = "deaths.csv",
    requested_member: str | None = None,
    sibling: tuple[str, bytes] | None = None,
    observation_start: date = date(2024, 1, 1),
    observation_cutoff: date = date(2024, 12, 31),
    released_at: date = date(2025, 1, 15),
    key: str = "deis_deaths_test",
) -> DeisDeathRelease:
    path = _archive_path(tmp_path, key)
    _write_zip(path, member, _csv_bytes(rows, columns=columns), sibling=sibling)
    return _attest_archive(
        path,
        key=key,
        provisional=provisional,
        released_at=released_at,
        observation_start=observation_start,
        observation_cutoff=observation_cutoff,
        archive_member=requested_member or member,
        controls=controls or DeisReleaseControls(),
    )


def _normalize(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    aggregation: DeisAggregation = DEFAULT_AGGREGATION,
    **release_options: object,
) -> DeisNormalizationResult:
    return normalize_deis_deaths(
        _release_for(tmp_path, rows, **release_options),
        aggregation=aggregation,
        identity_registry=IdentityRegistry(),
    )


def test_public_api_exposes_exact_immutable_raw_contract() -> None:
    assert DEIS_DEATH_COLUMNS == EXPECTED_DEIS_DEATH_COLUMNS
    assert isinstance(DEIS_DEATH_COLUMNS, tuple)
    with pytest.raises(FrozenInstanceError):
        DeisAggregation().by_sex = False  # type: ignore[misc]
    assert hash(DeisAggregation())


def test_deis_release_exposes_verified_assets_and_no_legacy_paths(
    tmp_path: Path,
) -> None:
    release = _release_for(tmp_path, [_row()])

    assert isinstance(release.assets, VerifiedReleaseAssets)
    assert release.source_key == release.assets.manifest.primary_fact_source_key
    assert not hasattr(release, "asset")
    assert not hasattr(release, "acquired")
    assert not hasattr(release.assets, "paths")
    assert not hasattr(release.assets, "acquired_assets")
    assert tuple(source.source_key for source in release.sources) == (release.source_key,)
    assert release.sources[0].sha256 == release.assets.manifest.primary_fact_asset.sha256
    assert (
        release.sources[0].retrieved_at == release.assets.manifest.primary_fact_asset.available_at
    )


def test_deis_release_rejects_source_or_provenance_manifest_mismatch(
    tmp_path: Path,
) -> None:
    release = _release_for(tmp_path, [_row()])
    with pytest.raises(DataContractError, match=r"source|primary|manifest"):
        replace(release, source_key="unknown_source")

    mismatched_source = release.sources[0].model_copy(update={"sha256": "f" * 64})
    with pytest.raises(DataContractError, match=r"source|manifest|SHA"):
        replace(release, sources=(mismatched_source,))


def test_deis_release_requires_provenance_for_every_manifest_member(
    tmp_path: Path,
) -> None:
    release = _release_for(tmp_path, [_row()])
    archive_path = _archive_path(tmp_path, "deis_deaths_test")
    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    sidecar_path = tmp_path / "reviewed-sidecar.json"
    sidecar_path.write_bytes(b'{"reviewed":true}')
    sidecar_digest = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    sidecar_available_at = _RETRIEVED_AT + timedelta(days=1)
    manifest = ReleaseManifest.create(
        schema_version="deis-death-release-v1",
        assets=(
            RequiredReleaseAsset(
                source_key=release.source_key,
                sha256=archive_digest,
                available_at=_RETRIEVED_AT,
                released_at=release.released_at,
                release_missingness_reason=None,
            ),
            RequiredReleaseAsset(
                source_key="reviewed_sidecar",
                sha256=sidecar_digest,
                available_at=sidecar_available_at,
                released_at=release.released_at,
                release_missingness_reason=None,
            ),
        ),
        primary_fact_source_key=release.source_key,
        identity_registry=IdentityRegistry(),
    )
    verified = verify_release_assets(
        manifest,
        (
            AcquiredAsset(
                key=release.source_key,
                path=archive_path,
                sha256=archive_digest,
                retrieved_at=_RETRIEVED_AT,
            ),
            AcquiredAsset(
                key="reviewed_sidecar",
                path=sidecar_path,
                sha256=sidecar_digest,
                retrieved_at=sidecar_available_at,
            ),
        ),
    )
    sidecar_source = SourceRef(
        source_key="reviewed_sidecar",
        name="Reviewed DEIS sidecar",
        url="https://example.test/deis/reviewed-sidecar",
        release_date=release.released_at,
        release_missingness_reason=None,
        retrieved_at=sidecar_available_at,
        sha256=sidecar_digest,
        vintage="test-preliminary",
        provisional=True,
    )

    with pytest.raises(DataContractError, match=r"source|manifest|asset"):
        replace(release, assets=verified)
    complete = replace(
        release,
        assets=verified,
        sources=(*release.sources, sidecar_source),
    )
    assert complete.assets.manifest.available_at == sidecar_available_at
    result = normalize_deis_deaths(
        complete,
        identity_registry=IdentityRegistry(),
    )
    assert {source.source_key for source in result.provenance.sources} == {
        release.source_key,
        "reviewed_sidecar",
    }
    assert result.provenance.available_at == sidecar_available_at


def test_deis_normalizer_rejects_raw_frame_and_invalid_registry(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        normalize_deis_deaths(
            pd.DataFrame(),  # type: ignore[arg-type]
            identity_registry=IdentityRegistry(),
        )
    with pytest.raises(DataContractError, match=r"registry|Identity"):
        normalize_deis_deaths(
            _release_for(tmp_path, [_row()]),
            identity_registry=object(),  # type: ignore[arg-type]
        )


def test_fact_identity_is_stable_across_release_revisions(
    tmp_path: Path,
) -> None:
    first = _normalize(tmp_path, [_row()])
    second = _normalize(
        tmp_path,
        [_row()],
        sibling=("nonfact-review-note.txt", b"revision two"),
    )

    assert set(first.observations["fact_id"]) == set(second.observations["fact_id"])
    assert set(first.observations["release_id"]) != set(second.observations["release_id"])
    assert set(first.observations["observation_id"]) != set(second.observations["observation_id"])


def test_windows_1252_semicolon_stream_preserves_accents_across_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import chile_demographic_pde.countries.chile.deis as deis

    monkeypatch.setattr(deis, "_ARROW_BLOCK_SIZE", 1 << 10)
    rows = [
        _row(
            FECHA_DEF=f"2024-02-{day:02d}",
            COD_COMUNA="16206",
            COMUNA="Ránquil",
            NOMBRE_REGION="De Ñuble",
        )
        for day in range(1, 20)
    ]

    result = _normalize(tmp_path, rows)

    assert result.observations["value"].sum() == 19
    assert set(result.observations["region"]) == {"CL-16"}
    assert result.audit.arrow_batches > 1
    assert "De Ñuble" in result.audit.raw_region_labels


@pytest.mark.parametrize(
    "columns",
    [
        EXPECTED_DEIS_DEATH_COLUMNS[:-1],
        (*EXPECTED_DEIS_DEATH_COLUMNS, "EXTRA"),
        (
            EXPECTED_DEIS_DEATH_COLUMNS[1],
            EXPECTED_DEIS_DEATH_COLUMNS[0],
            *EXPECTED_DEIS_DEATH_COLUMNS[2:],
        ),
        (
            EXPECTED_DEIS_DEATH_COLUMNS[0],
            EXPECTED_DEIS_DEATH_COLUMNS[0],
            *EXPECTED_DEIS_DEATH_COLUMNS[2:],
        ),
    ],
)
def test_header_fails_closed_when_missing_extra_reordered_or_duplicated(
    tmp_path: Path, columns: tuple[str, ...]
) -> None:
    with pytest.raises(DataContractError, match="header"):
        _normalize(tmp_path, [_row()], columns=columns)


def test_missing_exact_member_does_not_fall_back_to_sibling(tmp_path: Path) -> None:
    release = _release_for(
        tmp_path,
        [_row()],
        requested_member="requested-vintage.csv",
        sibling=("older-final.csv", _csv_bytes([_row()])),
    )

    with pytest.raises(MissingOfficialDataError, match="requested-vintage"):
        normalize_deis_deaths(
            release,
            identity_registry=IdentityRegistry(),
        )


def test_checksum_is_recomputed_and_compared_at_ingestion(tmp_path: Path) -> None:
    release = _release_for(tmp_path, [_row()])
    path = _archive_path(tmp_path, "deis_deaths_test")
    path.write_bytes(path.read_bytes() + b"changed")

    with pytest.raises(ChecksumMismatchError):
        normalize_deis_deaths(
            release,
            identity_registry=IdentityRegistry(),
        )


@pytest.mark.parametrize("unsafe_state", ["missing", "symlink"])
def test_verified_boundary_rejects_missing_or_unsafe_bytes_without_path_disclosure(
    tmp_path: Path,
    unsafe_state: str,
) -> None:
    release = _release_for(tmp_path, [_row()])
    path = _archive_path(tmp_path, "deis_deaths_test")
    path.unlink()
    if unsafe_state == "symlink":
        target = tmp_path / "unattested.zip"
        target.write_bytes(b"unattested")
        path.symlink_to(target)

    with pytest.raises(MissingOfficialDataError) as captured:
        normalize_deis_deaths(
            release,
            identity_registry=IdentityRegistry(),
        )
    assert str(path) not in str(captured.value)
    assert str(path) not in repr(captured.value)


def test_zip_parser_consumes_only_the_verified_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original_open = VerifiedReleaseAssets.open

    def tracked_open(
        self: VerifiedReleaseAssets,
        source_key: str,
    ):
        calls.append(source_key)
        return original_open(self, source_key)

    monkeypatch.setattr(VerifiedReleaseAssets, "open", tracked_open)
    release = _release_for(tmp_path, [_row()])

    result = normalize_deis_deaths(
        release,
        identity_registry=IdentityRegistry(),
    )

    assert calls == [release.source_key]
    assert result.audit.raw_rows == 1


def test_bad_zip_and_crc_are_data_contract_failures(tmp_path: Path) -> None:
    bad_zip_path = _archive_path(tmp_path, "deis_deaths_test")
    bad_zip_path.write_bytes(b"not a zip")
    release = _attest_archive(
        bad_zip_path,
        key="deis_deaths_test",
        provisional=True,
        released_at=date(2025, 1, 15),
        observation_start=date(2024, 1, 1),
        observation_cutoff=date(2024, 12, 31),
        archive_member="deaths.csv",
        controls=DeisReleaseControls(),
    )
    with pytest.raises(DataContractError, match="ZIP"):
        normalize_deis_deaths(
            release,
            identity_registry=IdentityRegistry(),
        )

    crc_path = _archive_path(tmp_path, "crc_test")
    payload = _csv_bytes([_row()])
    _write_zip(
        crc_path,
        "deaths.csv",
        payload,
        compression=zipfile.ZIP_STORED,
    )
    with zipfile.ZipFile(crc_path) as archive:
        info = archive.getinfo("deaths.csv")
        payload_offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    corrupted = bytearray(crc_path.read_bytes())
    corrupted[payload_offset + len(payload) - 2] ^= 1
    crc_path.write_bytes(corrupted)
    crc_release = _attest_archive(
        crc_path,
        key="crc_test",
        provisional=True,
        released_at=date(2025, 1, 15),
        observation_start=date(2024, 1, 1),
        observation_cutoff=date(2024, 12, 31),
        archive_member="deaths.csv",
        controls=DeisReleaseControls(),
    )
    with pytest.raises(DataContractError, match="CRC"):
        normalize_deis_deaths(
            crc_release,
            identity_registry=IdentityRegistry(),
        )


def test_archive_and_member_sizes_are_release_controls(tmp_path: Path) -> None:
    release = _release_for(tmp_path, [_row()])
    path = _archive_path(tmp_path, "deis_deaths_test")
    archive_size = path.stat().st_size
    with zipfile.ZipFile(path) as archive:
        member_size = archive.getinfo(release.archive_member).file_size

    normalize_deis_deaths(
        replace(
            release,
            controls=DeisReleaseControls(
                archive_size=archive_size,
                member_size=member_size,
            ),
        ),
        identity_registry=IdentityRegistry(),
    )
    with pytest.raises(DataContractError, match="archive_size"):
        normalize_deis_deaths(
            replace(
                release,
                controls=DeisReleaseControls(archive_size=archive_size + 1),
            ),
            identity_registry=IdentityRegistry(),
        )
    with pytest.raises(DataContractError, match="member_size"):
        normalize_deis_deaths(
            replace(
                release,
                controls=DeisReleaseControls(member_size=member_size + 1),
            ),
            identity_registry=IdentityRegistry(),
        )


def test_streaming_path_never_uses_prohibited_whole_file_apis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def prohibited(*args: object, **kwargs: object) -> None:
        raise AssertionError("whole-file API was called")

    monkeypatch.setattr(pd, "read_csv", prohibited)
    monkeypatch.setattr(pd, "concat", prohibited)
    monkeypatch.setattr(zipfile.ZipFile, "read", prohibited)
    monkeypatch.setattr(pyarrow.csv, "read_csv", prohibited)

    result = _normalize(tmp_path, [_row()])

    assert result.audit.raw_rows == 1


@pytest.mark.parametrize(
    ("grain", "expected_start", "expected_end"),
    [
        (DeisTemporalGrain.DAY, "2024-02-03", "2024-02-04"),
        (DeisTemporalGrain.MONTH, "2024-02-01", "2024-03-01"),
        (DeisTemporalGrain.YEAR, "2024-01-01", "2025-01-01"),
    ],
)
def test_occurrence_periods_are_half_open(
    tmp_path: Path,
    grain: DeisTemporalGrain,
    expected_start: str,
    expected_end: str,
) -> None:
    result = _normalize(
        tmp_path,
        [_row()],
        aggregation=DeisAggregation(temporal_grain=grain),
        provisional=False,
    )

    row = result.observations.iloc[0]
    assert row["period_start"] == date.fromisoformat(expected_start)
    assert row["period_end"] == date.fromisoformat(expected_end)
    assert type(row["period_start"]) is date
    assert type(row["period_end"]) is date
    assert row["date_basis"] == "death_occurrence"


def test_preliminary_last_bucket_is_clipped_to_declared_cutoff(tmp_path: Path) -> None:
    result = _normalize(
        tmp_path,
        [_row(FECHA_DEF="2024-07-04")],
        observation_cutoff=date(2024, 7, 7),
    )

    assert result.observations.iloc[0]["period_end"] == date(2024, 7, 8)


def test_blank_official_occurrence_date_is_preserved_as_unknown_within_year(
    tmp_path: Path,
) -> None:
    rows = [_row(FECHA_DEF=""), _row(FECHA_DEF="2024-02-03")]

    result = _normalize(
        tmp_path,
        rows,
        aggregation=DeisAggregation(temporal_grain=DeisTemporalGrain.DAY),
    )

    unknown = result.observations.loc[
        result.observations["date_basis"].eq("death_occurrence_year_when_date_missing")
    ]
    assert unknown["value"].sum() == 1
    assert unknown.iloc[0]["period_start"] == date(2024, 1, 1)
    assert unknown.iloc[0]["period_end"] == date(2025, 1, 1)
    assert "occurrence_date" in json.loads(unknown.iloc[0]["missingness_reason"])


def test_provenance_range_combines_known_dates_with_unknown_date_year_bounds(
    tmp_path: Path,
) -> None:
    rows = [
        _row(AÑO="2023", FECHA_DEF="2023-06-01"),
        _row(AÑO="2024", FECHA_DEF=""),
    ]
    result = _normalize(
        tmp_path,
        rows,
        observation_start=date(2023, 1, 1),
        observation_cutoff=date(2024, 7, 7),
    )
    assert result.provenance.observation_start == date(2023, 6, 1)
    assert result.provenance.observation_end == date(2024, 7, 8)

    earlier_unknown = _normalize(
        tmp_path,
        [
            _row(AÑO="2023", FECHA_DEF=""),
            _row(AÑO="2024", FECHA_DEF="2024-06-01"),
        ],
        observation_start=date(2023, 1, 1),
        observation_cutoff=date(2024, 7, 7),
        key="earlier_unknown",
    )
    assert earlier_unknown.provenance.observation_start == date(2023, 1, 1)
    assert earlier_unknown.provenance.observation_end == date(2024, 6, 2)


def test_blank_date_bucket_in_incomplete_year_is_clipped_to_cutoff(
    tmp_path: Path,
) -> None:
    result = _normalize(
        tmp_path,
        [_row(FECHA_DEF="")],
        observation_cutoff=date(2024, 7, 7),
        aggregation=DeisAggregation(temporal_grain=DeisTemporalGrain.MONTH),
    )

    row = result.observations.iloc[0]
    assert row["period_start"] == date(2024, 1, 1)
    assert row["period_end"] == date(2024, 7, 8)


def test_nonblank_bad_date_and_year_date_conflict_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="FECHA_DEF"):
        _normalize(tmp_path, [_row(FECHA_DEF="03/02/2024")])
    with pytest.raises(DataContractError, match="AÑO"):
        _normalize(tmp_path, [_row(AÑO="2023", FECHA_DEF="2024-02-03")])


@pytest.mark.parametrize(
    ("age_type", "quantity", "source_unit", "lower", "upper"),
    [
        ("1", "99", "completed_years", 99.0, 100.0),
        ("2", "11", "completed_months", 0.0, 1.0),
        ("3", "31", "completed_days", 0.0, 1.0),
        ("4", "24", "completed_hours", 0.0, 1.0),
    ],
)
def test_source_reported_age_preserves_exact_unit_and_quantity(
    tmp_path: Path,
    age_type: str,
    quantity: str,
    source_unit: str,
    lower: float,
    upper: float,
) -> None:
    result = _normalize(
        tmp_path,
        [_row(EDAD_TIPO=age_type, EDAD_CANT=quantity)],
        aggregation=DeisAggregation(age_grain=DeisAgeGrain.SOURCE_REPORTED),
    )

    row = result.observations.iloc[0]
    assert row["age_measure_unit"] == source_unit
    assert row["age_quantity"] == float(quantity)
    assert row["age_lower"] == lower
    assert row["age_upper"] == upper


def test_completed_year_collapses_subyear_ages_to_zero_and_keeps_unknown(
    tmp_path: Path,
) -> None:
    rows = [
        _row(EDAD_TIPO="2", EDAD_CANT="3"),
        _row(EDAD_TIPO="0", EDAD_CANT="99", SEXO_NOMBRE="Hombre"),
        _row(EDAD_TIPO="", EDAD_CANT="2", SEXO_NOMBRE="Indeterminado"),
        _row(EDAD_TIPO="1", EDAD_CANT="", FECHA_DEF=""),
    ]

    result = _normalize(tmp_path, rows)

    infant = result.observations.loc[result.observations["sex"].eq("female")].iloc[0]
    assert (infant["age_lower"], infant["age_upper"], infant["age_quantity"]) == (
        0.0,
        1.0,
        0.0,
    )
    unknown = result.observations.loc[
        result.observations["age_measure_unit"].eq("source_age_unknown")
    ]
    assert unknown["age_lower"].isna().all()
    assert unknown["age_upper"].isna().all()
    assert unknown["age_quantity"].isna().all()
    missing_quantity = result.observations.loc[
        result.observations["sex"].eq("female")
        & result.observations["date_basis"].eq("death_occurrence_year_when_date_missing")
    ]
    assert missing_quantity["age_lower"].isna().all()


def test_age_all_collapses_age_dimensions_without_dropping_events(tmp_path: Path) -> None:
    result = _normalize(
        tmp_path,
        [_row(), _row(EDAD_TIPO="2", EDAD_CANT="4")],
        aggregation=DeisAggregation(age_grain=DeisAgeGrain.ALL),
    )

    assert result.observations["value"].sum() == 2
    assert result.observations["age_lower"].isna().all()
    assert result.observations["age_measure_unit"].isna().all()


@pytest.mark.parametrize(
    ("age_type", "quantity"),
    [
        ("1", "-1"),
        ("1", "1.5"),
        ("2", "12"),
        ("3", "32"),
        ("4", "25"),
        ("5", "1"),
    ],
)
def test_nonblank_invalid_age_fields_fail_closed(
    tmp_path: Path, age_type: str, quantity: str
) -> None:
    with pytest.raises(DataContractError, match="EDAD"):
        _normalize(tmp_path, [_row(EDAD_TIPO=age_type, EDAD_CANT=quantity)])


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("Hombre", "male"),
        ("Mujer", "female"),
        ("Indeterminado", "indeterminate"),
    ],
)
def test_sex_mapping_is_exact_and_indeterminate_is_valid(
    tmp_path: Path, raw: str, canonical: str
) -> None:
    result = _normalize(tmp_path, [_row(SEXO_NOMBRE=raw)])
    assert result.observations.iloc[0]["sex"] == canonical
    assert result.observations.iloc[0]["sex_role"] == "decedent"


def test_unknown_sex_fails_and_collapsed_sex_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="SEXO_NOMBRE"):
        _normalize(tmp_path, [_row(SEXO_NOMBRE="Otro")])

    result = _normalize(
        tmp_path,
        [_row(SEXO_NOMBRE="Hombre"), _row(SEXO_NOMBRE="Mujer")],
        aggregation=DeisAggregation(by_sex=False),
    )
    assert set(result.observations["sex"]) == {"all"}
    assert result.observations["value"].sum() == 2


@pytest.mark.parametrize(
    ("raw_code", "expected"),
    [("8111", "08111"), ("16206", "16206")],
)
def test_commune_codes_are_read_as_text_and_left_padded(
    tmp_path: Path, raw_code: str, expected: str
) -> None:
    result = _normalize(
        tmp_path,
        [
            _row(
                COD_COMUNA=raw_code,
                NOMBRE_REGION=("De Ñuble" if raw_code == "16206" else "Del Bíobío"),
            )
        ],
        aggregation=DeisAggregation(geography_grain=DeisGeographyGrain.COMMUNE),
    )
    assert result.observations.iloc[0]["commune"] == expected
    assert (
        result.observations.iloc[0]["geography_basis"]
        == "decedent_usual_residence_within_chile_occurrence_universe"
    )
    assert result.observations.iloc[0]["geography_vintage"] == "dpa-2019"


@pytest.mark.parametrize("raw_code", ["", "99999"])
def test_verified_missing_communes_remain_explicit_unknowns(tmp_path: Path, raw_code: str) -> None:
    result = _normalize(
        tmp_path,
        [_row(COD_COMUNA=raw_code, COMUNA="", NOMBRE_REGION="")],
        aggregation=DeisAggregation(geography_grain=DeisGeographyGrain.COMMUNE),
    )
    row = result.observations.iloc[0]
    assert pd.isna(row["commune"])
    assert row["region"] == "unknown"
    missingness = json.loads(row["missingness_reason"])
    assert missingness["residence_region"] == "blank_or_99999_source_commune_and_region"
    assert missingness["residence_commune"] == "blank_or_99999_source_commune"


@pytest.mark.parametrize("raw_code", ["", "99999"])
def test_missing_commune_with_known_region_keeps_region_and_commune_missingness(
    tmp_path: Path,
    raw_code: str,
) -> None:
    result = _normalize(
        tmp_path,
        [
            _row(
                COD_COMUNA=raw_code,
                COMUNA="Ignorada" if raw_code == "99999" else "",
                NOMBRE_REGION="De Ñuble",
            )
        ],
        aggregation=DeisAggregation(geography_grain=DeisGeographyGrain.COMMUNE),
    )

    row = result.observations.iloc[0]
    assert row["region"] == "CL-16"
    assert pd.isna(row["commune"])
    missingness = json.loads(row["missingness_reason"])
    assert missingness["residence_commune"] == "blank_or_99999_source_commune"
    assert "residence_region" not in missingness


@pytest.mark.parametrize(
    ("commune", "region_label", "expected_region"),
    [
        ("11101", "De Aisén del Gral. C. Ibáñez del Campo", "CL-11"),
        ("6101", "Del Libertador B. O'Higgins", "CL-06"),
    ],
)
def test_live_abbreviated_region_labels_cross_check_against_commune(
    tmp_path: Path,
    commune: str,
    region_label: str,
    expected_region: str,
) -> None:
    result = _normalize(
        tmp_path,
        [_row(COD_COMUNA=commune, NOMBRE_REGION=region_label)],
        aggregation=DeisAggregation(geography_grain=DeisGeographyGrain.COMMUNE),
    )

    assert result.observations.iloc[0]["region"] == expected_region


@pytest.mark.parametrize("raw_code", ["12A4", "123456", "-1"])
def test_invalid_nonmissing_commune_codes_fail_closed(tmp_path: Path, raw_code: str) -> None:
    with pytest.raises(DataContractError, match="COD_COMUNA"):
        _normalize(tmp_path, [_row(COD_COMUNA=raw_code)])


def test_national_aggregation_uses_occurrence_universe_not_resident_basis(
    tmp_path: Path,
) -> None:
    result = _normalize(
        tmp_path,
        [_row()],
        aggregation=DeisAggregation(geography_grain=DeisGeographyGrain.NATIONAL),
    )
    row = result.observations.iloc[0]
    assert row["region"] == "CL"
    assert pd.isna(row["commune"])
    assert row["geography_basis"] == "death_occurrence_territory_chile"
    assert row["population_basis"] == "not_applicable"
    assert row["unit"] == "event"


def test_icd_transition_is_occurrence_year_based_and_conflicts_fail(
    tmp_path: Path,
) -> None:
    result_1996 = _normalize(
        tmp_path,
        [_row(AÑO="1996", FECHA_DEF="1996-03-01", DIAG1="8690")],
        provisional=False,
        observation_start=date(1996, 1, 1),
        observation_cutoff=date(1996, 12, 31),
        released_at=date(1997, 6, 1),
        key="deis_1996",
    )
    result_1997 = _normalize(
        tmp_path,
        [_row(AÑO="1997", FECHA_DEF="1997-03-01", DIAG1="T290")],
        provisional=False,
        observation_start=date(1997, 1, 1),
        observation_cutoff=date(1997, 12, 31),
        released_at=date(1998, 6, 1),
        key="deis_1997",
    )
    assert set(result_1996.observations["cause_code_system"]) == {"ICD-9"}
    assert set(result_1997.observations["cause_code_system"]) == {"ICD-10"}

    with pytest.raises(DataContractError, match="ICD"):
        _normalize(
            tmp_path,
            [_row(AÑO="1996", FECHA_DEF="1996-03-01", DIAG1="A123")],
            provisional=False,
            observation_start=date(1996, 1, 1),
            observation_cutoff=date(1996, 12, 31),
            released_at=date(1997, 6, 1),
            key="conflict_1996",
        )
    with pytest.raises(DataContractError, match="ICD"):
        _normalize(
            tmp_path,
            [_row(AÑO="1997", FECHA_DEF="1997-03-01", DIAG1="1234")],
            provisional=False,
            observation_start=date(1997, 1, 1),
            observation_cutoff=date(1997, 12, 31),
            released_at=date(1998, 6, 1),
            key="conflict_1997",
        )


def test_historical_source_code_anomalies_are_preserved_and_audited(
    tmp_path: Path,
) -> None:
    result = _normalize(
        tmp_path,
        [_row(AÑO="1996", FECHA_DEF="1996-03-01", DIAG1="402-")],
        provisional=False,
        observation_start=date(1996, 1, 1),
        observation_cutoff=date(1996, 12, 31),
        released_at=date(1997, 6, 1),
        key="anomaly_1996",
        aggregation=DeisAggregation(cause_grain=DeisCauseGrain.UNDERLYING_CODE),
    )
    assert set(result.observations["cause"]) == {"402-"}
    assert result.audit.anomalous_underlying_codes == 1


@pytest.mark.parametrize(
    ("cause_grain", "expected"),
    [
        (DeisCauseGrain.ALL, "all_causes"),
        (DeisCauseGrain.UNDERLYING_CODE, "I64X"),
        (DeisCauseGrain.CHAPTER, "I00-I99"),
        (DeisCauseGrain.GROUP, "I60-I69"),
        (DeisCauseGrain.CATEGORY, "I64"),
        (DeisCauseGrain.SUBCATEGORY, "I64X"),
    ],
)
def test_each_cause_grain_returns_one_nonduplicating_view(
    tmp_path: Path, cause_grain: DeisCauseGrain, expected: str
) -> None:
    result = _normalize(
        tmp_path,
        [_row(DIAG2="W19X")],
        aggregation=DeisAggregation(
            cause_grain=cause_grain,
            retain_external_cause=True,
        ),
    )
    assert result.observations["value"].sum() == 1
    assert result.observations.iloc[0]["cause"] == expected
    expected_role = "all_cause" if cause_grain is DeisCauseGrain.ALL else "underlying"
    assert result.observations.iloc[0]["cause_role"] == expected_role
    assert result.observations.iloc[0]["external_cause"] == "W19X"


def test_blank_external_cause_means_not_applicable_not_unknown_underlying(
    tmp_path: Path,
) -> None:
    result = _normalize(
        tmp_path,
        [_row(DIAG2=""), _row(DIAG2="W19X", SEXO_NOMBRE="Hombre")],
        aggregation=DeisAggregation(retain_external_cause=True),
    )
    assert result.observations["value"].sum() == 2
    assert set(result.observations["external_cause"]) == {
        "not_applicable",
        "W19X",
    }
    assert set(result.observations["cause"]) == {"all_causes"}


def test_whitespace_external_cause_is_not_applicable_but_audited_raw(
    tmp_path: Path,
) -> None:
    result = _normalize(
        tmp_path,
        [_row(DIAG2="    ")],
        aggregation=DeisAggregation(retain_external_cause=True),
    )
    assert set(result.observations["external_cause"]) == {"not_applicable"}
    assert result.audit.raw_nonempty_external_causes == 1
    assert result.audit.meaningful_external_causes == 0


def test_blank_underlying_code_is_preserved_as_unknown_for_specific_view(
    tmp_path: Path,
) -> None:
    result = _normalize(
        tmp_path,
        [_row(DIAG1="")],
        aggregation=DeisAggregation(cause_grain=DeisCauseGrain.UNDERLYING_CODE),
    )
    assert result.observations.iloc[0]["cause"] == "unknown"
    assert "underlying_cause" in json.loads(result.observations.iloc[0]["missingness_reason"])


def test_exact_release_controls_pass_and_one_row_perturbation_fails(
    tmp_path: Path,
) -> None:
    rows = [
        _row(),
        _row(
            AÑO="2025",
            FECHA_DEF="2025-01-02",
            SEXO_NOMBRE="Hombre",
            EDAD_TIPO="2",
            EDAD_CANT="3",
            COD_COMUNA="16206",
            NOMBRE_REGION="De Ñuble",
            DIAG2="W19X",
        ),
        _row(
            AÑO="2025",
            FECHA_DEF="2025-01-03",
            SEXO_NOMBRE="Indeterminado",
            EDAD_TIPO="0",
            EDAD_CANT="0",
            COD_COMUNA="",
            NOMBRE_REGION="",
        ),
    ]
    controls = DeisReleaseControls(
        raw_rows=3,
        counts_by_year=((2024, 1), (2025, 2)),
        counts_by_sex=(("Hombre", 1), ("Indeterminado", 1), ("Mujer", 1)),
        counts_by_age_type=(("0", 1), ("1", 1), ("2", 1)),
        age_99_year_deaths=1,
        nonblank_external_causes=1,
        short_numeric_communes=1,
    )
    result = _normalize(
        tmp_path,
        rows,
        controls=controls,
        observation_cutoff=date(2025, 12, 31),
        released_at=date(2026, 1, 15),
    )
    assert result.audit.raw_rows == 3
    assert result.audit.counts_by_year == ((2024, 1), (2025, 2))

    with pytest.raises(DataContractError, match="control"):
        _normalize(
            tmp_path,
            rows[:-1],
            controls=controls,
            observation_cutoff=date(2025, 12, 31),
            released_at=date(2026, 1, 15),
            key="perturbed",
        )


def test_control_key_duplicates_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        DeisReleaseControls(counts_by_year=((2024, 1), (2024, 2)))
    with pytest.raises(ValueError, match="duplicate"):
        DeisReleaseControls(counts_by_age_type=(("", 1), ("", 2)))


def test_pair_controls_may_pin_a_verified_subset_of_years(tmp_path: Path) -> None:
    result = _normalize(
        tmp_path,
        [_row(), _row(AÑO="2025", FECHA_DEF="2025-01-02")],
        controls=DeisReleaseControls(counts_by_year=((2025, 1),)),
        observation_cutoff=date(2025, 12, 31),
        released_at=date(2026, 1, 15),
    )
    assert result.audit.counts_by_year == ((2024, 1), (2025, 1))


def test_region_label_must_match_region_derived_from_commune(tmp_path: Path) -> None:
    with pytest.raises(DataContractError, match="region"):
        _normalize(
            tmp_path,
            [_row(COD_COMUNA="16206", NOMBRE_REGION="De Tarapacá")],
        )


def test_release_identity_and_observation_coverage_fail_closed(tmp_path: Path) -> None:
    release = _release_for(tmp_path, [_row()])
    with pytest.raises(DataContractError, match=r"source|key|primary"):
        replace(release, source_key="other")
    with pytest.raises(DataContractError, match="coverage"):
        normalize_deis_deaths(
            replace(release, observation_cutoff=date(2024, 1, 31)),
            identity_registry=IdentityRegistry(),
        )


def test_scratch_resources_are_removed_on_success_and_failure(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    release = _release_for(tmp_path, [_row()])

    normalize_deis_deaths(
        release,
        scratch_directory=scratch,
        identity_registry=IdentityRegistry(),
    )
    assert list(scratch.iterdir()) == []

    bad_release = _release_for(tmp_path, [_row(SEXO_NOMBRE="invalid")], key="bad")
    with pytest.raises(DataContractError):
        normalize_deis_deaths(
            bad_release,
            scratch_directory=scratch,
            identity_registry=IdentityRegistry(),
        )
    assert list(scratch.iterdir()) == []


def test_result_is_canonical_valid_and_audit_reconciles_to_observations(
    tmp_path: Path,
) -> None:
    aggregation = DeisAggregation(
        temporal_grain=DeisTemporalGrain.MONTH,
        age_grain=DeisAgeGrain.SOURCE_REPORTED,
        geography_grain=DeisGeographyGrain.COMMUNE,
        cause_grain=DeisCauseGrain.UNDERLYING_CODE,
        retain_external_cause=True,
    )
    result = _normalize(tmp_path, [_row()], aggregation=aggregation)

    assert tuple(result.observations.columns) == EXPECTED_DEMOGRAPHIC_COLUMNS
    assert FOREIGN_DOMAIN_COLUMNS.isdisjoint(result.observations.columns)
    pd.testing.assert_frame_equal(
        validate_demographic_observations(result.observations),
        result.observations,
    )
    pd.testing.assert_frame_equal(
        result.observations,
        result.batch.observations,
    )
    assert set(result.observations["domain"]) == {"demographic"}
    assert set(result.observations["observation_role"]) == {"observed_fact"}
    assert set(result.observations["parity_scope"]) == {"not_applicable"}
    assert set(result.observations["population_basis"]) == {"not_applicable"}
    assert set(result.observations["transformation_id"]) == {"deis-death-events-to-counts-v1"}
    assert set(result.observations["transformation_version"]) == {"v1"}
    assert set(result.observations["release_id"]) == {result.batch.manifest.release_id.value}
    assert result.provenance == result.batch.provenance
    assert result.batch.identities.require(result.batch.manifest.release_id.value)
    for identifier in result.observations["fact_id"]:
        assert result.batch.identities.require(identifier)
    for identifier in result.observations["observation_id"]:
        assert result.batch.identities.require(identifier)
    assert result.audit.normalized_total == result.audit.raw_rows == 1
    assert result.audit.normalized_rows == len(result.observations)
    assert result.provenance.sources[0].sha256 == result.audit.sha256
    assert result.provenance.sources[0].vintage == "test-preliminary"
    assert result.provenance.dimensions == (
        "occurrence_month",
        "source_reported_age",
        "decedent_sex",
        "residence_commune",
        "underlying_code",
        "external_cause",
    )
    assert "no registration" in result.provenance.missingness_reason
    assert (
        json.loads(result.observations.iloc[0]["missingness_reason"])["registration_window"]
        == "not_available_in_current_deis_file"
    )

    leaked = result.observations
    leaked.loc[:, "value"] = -999.0
    assert (result.observations["value"] >= 0.0).all()
    assert (result.batch.observations["value"] >= 0.0).all()


def test_result_repr_is_concise_and_does_not_render_dataframe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _normalize(tmp_path, [_row()])

    def prohibited(self: pd.DataFrame) -> str:
        raise AssertionError("DataFrame repr must not be called")

    monkeypatch.setattr(pd.DataFrame, "__repr__", prohibited)
    rendered = repr(result)

    assert rendered.startswith("DeisNormalizationResult(")
    assert "rows=1" in rendered
    assert len(rendered) < 180


@pytest.mark.parametrize("memory_limit_bytes", [0, -1, "512MB"])
def test_memory_limit_must_be_positive(tmp_path: Path, memory_limit_bytes: object) -> None:
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        normalize_deis_deaths(
            _release_for(tmp_path, [_row()]),
            memory_limit_bytes=memory_limit_bytes,  # type: ignore[arg-type]
            identity_registry=IdentityRegistry(),
        )
