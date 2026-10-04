from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.csv
import pytest

import chile_demographic_pde.countries.chile.vital_statistics as vital_statistics
from chile_demographic_pde.core.errors import (
    ChecksumMismatchError,
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.countries.chile.census import (
    CuratedDomainRelease,
    CuratedTableContract,
    canonical_curated_table_bytes,
)
from chile_demographic_pde.countries.chile.deis import (
    DEIS_BIRTH_COLUMNS_25,
    DEIS_BIRTH_COLUMNS_29,
    DeisBirthArchiveId,
    DeisBirthAssetContract,
    DeisBirthRelease,
    DeisBirthReleaseControls,
    aggregate_deis_births,
    iter_normalize_deis_births,
    normalize_deis_births,
)
from chile_demographic_pde.countries.chile.vital_statistics import (
    INE_BIRTH_CONTROL_COLUMNS,
    IneBirthReleaseKind,
    compare_deis_births_to_ine_controls,
)
from chile_demographic_pde.data.acquisition import AcquiredAsset
from chile_demographic_pde.data.domain_schemas import (
    DEMOGRAPHIC_OBSERVATION_KEY,
    DemographicObservationBatch,
    DemographicObservationSchema,
    validate_demographic_observations,
)
from chile_demographic_pde.data.envelope import (
    NormalizedDomainBatch,
    assign_observation_ids,
)
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
    VerifiedReleaseAssets,
    verify_release_assets,
)
from chile_demographic_pde.data.roles import ObservationDomain

EXPECTED_BIRTH_COLUMNS_25 = (
    "MES_NAC",
    "ANO_NAC",
    "SEXO",
    "TIPO_PARTO",
    "TIPO_ATEN",
    "PARTO_LOCAL",
    "SEMANAS",
    "RANGO_PESO",
    "TALLA",
    "GRUPO_ETARIO_PADRE",
    "CURSO_PADRE",
    "NIVEL_PADRE",
    "ACTIV_PADRE",
    "OCUPA_PADRE",
    "CATEG_PADRE",
    "GRUPO_ETARIO_MADRE",
    "EST_CIV_MADRE",
    "CURSO_MADRE",
    "NIVEL_MADRE",
    "ACTIV_MADRE",
    "OCUPA_MADRE",
    "CATEG_MADRE",
    "NACIONALIDAD_MADRE",
    "REGION_RESIDENCIA",
    "GLOSA_REGION_RESIDENCIA",
)

EXPECTED_BIRTH_COLUMNS_29 = (
    *EXPECTED_BIRTH_COLUMNS_25[:22],
    "HIJ_VIVOS",
    "HIJ_FALL",
    "HIJ_MORT",
    "HIJ_TOTAL",
    *EXPECTED_BIRTH_COLUMNS_25[22:],
)
EXPECTED_DEMOGRAPHIC_COLUMNS = tuple(DemographicObservationSchema.to_schema().columns)
_RETRIEVED_AT = datetime(2026, 7, 19, tzinfo=UTC)
_INE_CONTROL_CONTRACT = CuratedTableContract(
    contract_id="ine-birth-controls-curated-v1",
    columns=(
        "year",
        "month",
        "sex",
        "region",
        "geography_vintage",
        "births",
    ),
    logical_types=(
        "integer",
        "integer",
        "string",
        "string",
        "string",
        "integer",
    ),
)


def _birth_row(
    *,
    profile: int = 29,
    **updates: str,
) -> dict[str, str]:
    columns = EXPECTED_BIRTH_COLUMNS_29 if profile == 29 else EXPECTED_BIRTH_COLUMNS_25
    row = dict.fromkeys(columns, "")
    row.update(
        {
            "MES_NAC": "1",
            "ANO_NAC": "2023",
            "SEXO": "2",
            "GRUPO_ETARIO_MADRE": "20 A 24 AÑOS",
            "NACIONALIDAD_MADRE": "E",
            "REGION_RESIDENCIA": "8",
            "GLOSA_REGION_RESIDENCIA": "Del Bíobío",
        }
    )
    if profile == 29:
        row.update(
            {
                "HIJ_VIVOS": "02",
                "HIJ_FALL": "",
                "HIJ_MORT": "01",
                "HIJ_TOTAL": "03",
            }
        )
    row.update(updates)
    return row


def _birth_csv_bytes(
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...],
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
    return text.getvalue().encode("utf-8-sig")


def _birth_release_for(
    tmp_path: Path,
    rows: list[dict[str, str]],
    *,
    archive_id: DeisBirthArchiveId = DeisBirthArchiveId.PROFILE_2001_2019,
    columns: tuple[str, ...] | None = None,
    controls: DeisBirthReleaseControls | None = None,
    member: str = "births.csv",
    requested_member: str | None = None,
    sibling_csv: tuple[str, bytes] | None = None,
    sibling: tuple[str, bytes] | None = None,
    key: str = "deis_births_test",
    released_at: date = date(2025, 3, 31),
) -> DeisBirthRelease:
    selected_columns = columns or archive_id.expected_columns
    path = tmp_path / f"{key}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            member,
            _birth_csv_bytes(rows, columns=selected_columns),
        )
        if sibling_csv is not None:
            archive.writestr(*sibling_csv)
        if sibling is not None:
            archive.writestr(*sibling)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    registry = IdentityRegistry()
    manifest = ReleaseManifest.create(
        schema_version="deis-birth-release-v1",
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
        identity_registry=registry,
    )
    acquired = AcquiredAsset(
        key=key,
        path=path,
        sha256=digest,
        retrieved_at=_RETRIEVED_AT,
    )
    source = SourceRef(
        source_key=key,
        name="DEIS births test fixture",
        url="https://example.test/deis-births",
        release_date=released_at,
        release_missingness_reason=None,
        retrieved_at=_RETRIEVED_AT,
        sha256=digest,
        vintage=f"test-{archive_id.value}",
        provisional=False,
    )
    return DeisBirthRelease(
        assets=verify_release_assets(manifest, (acquired,)),
        source_key=key,
        sources=(source,),
        released_at=released_at,
        observation_start=date(1992, 1, 1),
        observation_end=date(2023, 12, 31),
        archive_id=archive_id,
        archive_member=requested_member or member,
        geography_vintage="dpa-2018",
        registration_inclusion_cutoff="registered_through_following_march_31",
        controls=controls or DeisBirthReleaseControls(),
        asset_contract=DeisBirthAssetContract.LOCAL_PROFILE_FIXTURE,
    )


def _ine_birth_controls(
    *,
    year: object = 2023,
    month: object = pd.NA,
    sex: object = "all",
    region: object = "CL",
    geography_vintage: object = pd.NA,
    births: object = 174_057,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "year": [year],
            "month": [month],
            "sex": [sex],
            "region": [region],
            "geography_vintage": [geography_vintage],
            "births": [births],
        }
    )
    if type(year) is int:
        frame["year"] = frame["year"].astype("int64")
    if month is pd.NA or month is None:
        frame["month"] = pd.Series([pd.NA], dtype="Int64")
    elif type(month) is int:
        frame["month"] = frame["month"].astype("int64")
    if type(births) is int:
        frame["births"] = frame["births"].astype("int64")
    return frame.loc[:, list(_INE_CONTROL_CONTRACT.columns)]


def _ine_birth_release(
    frame: pd.DataFrame,
    kind: IneBirthReleaseKind = IneBirthReleaseKind.FINAL_ANNUAL,
    *,
    contract: CuratedTableContract = _INE_CONTROL_CONTRACT,
) -> tuple[CuratedDomainRelease, IdentityRegistry]:
    provisional = kind is not IneBirthReleaseKind.FINAL_ANNUAL
    source_key = f"ine_births_{kind.value}"
    digest = hashlib.sha256(canonical_curated_table_bytes(frame, contract)).hexdigest()
    registry = IdentityRegistry()
    manifest = ReleaseManifest.create(
        schema_version="ine-birth-controls-curated-v1",
        assets=(
            RequiredReleaseAsset(
                source_key=source_key,
                sha256=digest,
                available_at=_RETRIEVED_AT,
                released_at=date(2025, 3, 31),
                release_missingness_reason=None,
            ),
        ),
        primary_fact_source_key=source_key,
        identity_registry=registry,
    )
    source = SourceRef(
        source_key=source_key,
        name=source_key,
        url="https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales",
        release_date=date(2025, 3, 31),
        release_missingness_reason=None,
        retrieved_at=_RETRIEVED_AT,
        sha256=digest,
        vintage=f"2023-{kind.value}",
        provisional=provisional,
    )
    provenance = VariableProvenance(
        sources=(source,),
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=date(2023, 1, 1),
        observation_end=date(2024, 1, 1),
        dimensions=("birth_occurrence_period", "newborn_sex", "maternal_region"),
        aggregation_rules=("preserve the published INE birth total as a diagnostic control",),
        missingness_reason="not_applicable_observed",
    )
    return (
        CuratedDomainRelease(
            manifest=manifest,
            provenance=provenance,
            vintage=source.vintage,
            curated_source_key=source_key,
            table_contract=contract,
        ),
        registry,
    )


def _normalize_ine(
    frame: pd.DataFrame | None = None,
    *,
    kind: IneBirthReleaseKind = IneBirthReleaseKind.FINAL_ANNUAL,
) -> DemographicObservationBatch:
    selected = _ine_birth_controls() if frame is None else frame
    release, registry = _ine_birth_release(selected, kind)
    normalizer = getattr(
        vital_statistics,
        "normalize_ine_birth_controls_curated",
        None,
    )
    if normalizer is None:
        pytest.fail("normalize_ine_birth_controls_curated is not implemented")
    return cast(
        DemographicObservationBatch,
        normalizer(
            selected,
            release=release,
            identity_registry=registry,
        ),
    )


def _derived_deis_control_grain_batch(
    controls: DemographicObservationBatch,
    *,
    value: int | None = None,
    updates: dict[str, object] | None = None,
) -> DemographicObservationBatch:
    """Build a valid event-derived aggregate at the published control grain."""

    observations = controls.observations
    payload = observations.drop(columns=["fact_id", "observation_id", "release_id"]).copy(deep=True)
    for column in DEMOGRAPHIC_OBSERVATION_KEY:
        payload[column] = pd.Series(
            (None if pd.isna(item) else item for item in payload[column]),
            index=payload.index,
            dtype=object,
        )
    payload["age_open"] = payload["age_open"].astype(bool)
    payload.loc[:, "observation_role"] = "observed_fact"
    payload.loc[:, "parity_scope"] = "unknown_order"
    payload.loc[:, "vintage"] = "2023-event-derived"
    payload.loc[:, "provisional"] = False
    payload.loc[:, "transformation_id"] = "deis-birth-national-sum-v1"
    payload.loc[:, "transformation_version"] = "v1"
    payload.loc[:, "aggregation_rule"] = (
        "sum released DEIS birth events to the published INE control grain"
    )
    if value is not None:
        payload.loc[:, "value"] = value
    for column, replacement in (updates or {}).items():
        payload.loc[:, column] = replacement

    digest = hashlib.sha256(
        repr(
            payload.loc[
                :,
                [
                    "period_start",
                    "period_end",
                    "sex",
                    "region",
                    "geography_vintage",
                    "value",
                ],
            ].to_dict(orient="records")
        ).encode()
    ).hexdigest()
    registry = IdentityRegistry()
    manifest = ReleaseManifest.create(
        schema_version="deis-birth-comparison-test-v1",
        assets=(
            RequiredReleaseAsset(
                source_key="deis_births_comparison",
                sha256=digest,
                available_at=_RETRIEVED_AT,
                released_at=date(2025, 3, 31),
                release_missingness_reason=None,
            ),
        ),
        primary_fact_source_key="deis_births_comparison",
        identity_registry=registry,
    )
    source = SourceRef(
        source_key="deis_births_comparison",
        name="Derived DEIS birth comparison",
        url="https://example.test/deis-births-comparison",
        release_date=date(2025, 3, 31),
        release_missingness_reason=None,
        retrieved_at=_RETRIEVED_AT,
        sha256=digest,
        vintage="2023-event-derived",
        provisional=False,
    )
    provenance = VariableProvenance(
        sources=(source,),
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=controls.provenance.observation_start,
        observation_end=controls.provenance.observation_end,
        dimensions=controls.provenance.dimensions,
        aggregation_rules=("sum released DEIS birth events to the published INE control grain",),
        missingness_reason=controls.provenance.missingness_reason,
    )
    assigned = assign_observation_ids(
        payload,
        domain=ObservationDomain.DEMOGRAPHIC,
        domain_key=DEMOGRAPHIC_OBSERVATION_KEY,
        source_identity={
            "publisher": "Departamento de Estadisticas e Informacion de Salud",
            "dataset": "birth_microdata_derived_control_grain",
        },
        manifest=manifest,
        registry=registry,
    )
    validated = validate_demographic_observations(assigned)
    return NormalizedDomainBatch.create(
        manifest=manifest,
        observations=validated,
        provenance=provenance,
        identities=registry.snapshot(),
    )


def _rebuild_demographic_batch(
    batch: DemographicObservationBatch,
    *,
    updates: dict[str, object],
) -> DemographicObservationBatch:
    payload = batch.observations.drop(columns=["fact_id", "observation_id", "release_id"]).copy(
        deep=True
    )
    for column in DEMOGRAPHIC_OBSERVATION_KEY:
        payload[column] = pd.Series(
            (None if pd.isna(item) else item for item in payload[column]),
            index=payload.index,
            dtype=object,
        )
    for column, replacement in updates.items():
        payload.loc[:, column] = replacement
    payload["age_open"] = payload["age_open"].astype(bool)
    registry = IdentityRegistry()
    assigned = assign_observation_ids(
        payload,
        domain=ObservationDomain.DEMOGRAPHIC,
        domain_key=DEMOGRAPHIC_OBSERVATION_KEY,
        source_identity={
            "publisher": "test",
            "dataset": "rebuilt_demographic_batch",
        },
        manifest=batch.manifest,
        registry=registry,
    )
    return NormalizedDomainBatch.create(
        manifest=batch.manifest,
        observations=validate_demographic_observations(assigned),
        provenance=batch.provenance,
        identities=registry.snapshot(),
    )


def test_chile_package_exports_birth_and_control_public_entrypoints() -> None:
    import chile_demographic_pde.countries.chile as chile

    expected = {
        "DeisBirthArchiveId",
        "DeisBirthAssetContract",
        "DeisBirthRelease",
        "DeisBirthReleaseControls",
        "IneBirthReleaseKind",
        "aggregate_deis_births",
        "compare_deis_births_to_ine_controls",
        "iter_normalize_deis_births",
        "normalize_deis_births",
        "normalize_ine_birth_controls_curated",
    }
    assert expected.issubset(chile.__all__)
    assert all(callable(getattr(chile, name)) for name in expected)
    assert not hasattr(chile, "normalize_ine_birth_controls")


def test_birth_archive_profiles_are_exact_ordered_immutable_contracts() -> None:
    assert DEIS_BIRTH_COLUMNS_25 == EXPECTED_BIRTH_COLUMNS_25
    assert DEIS_BIRTH_COLUMNS_29 == EXPECTED_BIRTH_COLUMNS_29
    assert isinstance(DEIS_BIRTH_COLUMNS_25, tuple)
    assert isinstance(DEIS_BIRTH_COLUMNS_29, tuple)
    assert DeisBirthArchiveId.PROFILE_1992_2000.expected_columns == (EXPECTED_BIRTH_COLUMNS_25)
    assert DeisBirthArchiveId.PROFILE_2001_2019.expected_columns == (EXPECTED_BIRTH_COLUMNS_29)
    assert DeisBirthArchiveId.PROFILE_2020_2023.expected_columns == (EXPECTED_BIRTH_COLUMNS_25)


def test_birth_archive_ids_bind_exact_official_asset_and_member_identity() -> None:
    expected = {
        DeisBirthArchiveId.PROFILE_1992_2000: (
            "deis_births_1992_2000",
            "Serie_Nacimientos_1992_2000.zip",
            "Serie_Nacimientos_1992_2000.csv",
        ),
        DeisBirthArchiveId.PROFILE_2001_2019: (
            "deis_births_2001_2019",
            "Serie_Nacimientos_2001_2019.zip",
            "Serie_Nacimientos_2001_2019.csv",
        ),
        DeisBirthArchiveId.PROFILE_2020_2023: (
            "deis_births_2020_2023",
            "Serie_Nacimientos_2020_2023.zip",
            "Serie_Nacimientos_2020_2023.csv",
        ),
    }

    for archive_id, identity in expected.items():
        assert (
            archive_id.expected_asset_key,
            archive_id.expected_filename,
            archive_id.expected_member,
        ) == identity


@pytest.mark.parametrize(
    ("archive_id", "wrong_id"),
    [
        (
            DeisBirthArchiveId.PROFILE_1992_2000,
            DeisBirthArchiveId.PROFILE_2020_2023,
        ),
        (
            DeisBirthArchiveId.PROFILE_2020_2023,
            DeisBirthArchiveId.PROFILE_1992_2000,
        ),
    ],
)
def test_two_25_column_official_archives_cannot_be_cross_paired(
    tmp_path: Path,
    archive_id: DeisBirthArchiveId,
    wrong_id: DeisBirthArchiveId,
) -> None:
    local = _birth_release_for(
        tmp_path,
        [_birth_row(profile=25)],
        archive_id=archive_id,
        key=f"cross_paired_25_{archive_id.value}",
    )

    with pytest.raises(ValueError, match=r"archive_id.*source key"):
        replace(
            local,
            archive_member=wrong_id.expected_member,
            asset_contract=DeisBirthAssetContract.PINNED_OFFICIAL,
        )


def test_birth_release_exposes_verified_assets_and_no_legacy_paths(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(tmp_path, [_birth_row()])

    assert isinstance(release.assets, VerifiedReleaseAssets)
    assert release.source_key == release.assets.manifest.primary_fact_source_key
    assert not hasattr(release, "asset")
    assert not hasattr(release, "acquired")
    assert not hasattr(release.assets, "paths")
    assert not hasattr(release.assets, "acquired_assets")
    assert tuple(source.source_key for source in release.sources) == (release.source_key,)
    selected = release.assets.manifest.primary_fact_asset
    assert release.sources[0].sha256 == selected.sha256
    assert release.sources[0].retrieved_at == selected.available_at


def test_birth_release_rejects_source_or_provenance_manifest_mismatch(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(tmp_path, [_birth_row()])

    with pytest.raises(DataContractError, match=r"source|primary|manifest"):
        replace(release, source_key="different_asset")
    mismatched_source = release.sources[0].model_copy(update={"sha256": "f" * 64})
    with pytest.raises(DataContractError, match=r"source|manifest|SHA"):
        replace(release, sources=(mismatched_source,))


def test_official_birth_member_is_exact_but_local_fixture_contract_is_explicit(
    tmp_path: Path,
) -> None:
    local = _birth_release_for(tmp_path, [_birth_row()])
    assert local.asset_contract is DeisBirthAssetContract.LOCAL_PROFILE_FIXTURE

    with pytest.raises(ValueError, match=r"source key|archive member"):
        replace(
            local,
            archive_member="births.csv",
            asset_contract=DeisBirthAssetContract.PINNED_OFFICIAL,
        )


def test_birth_release_and_streaming_api_are_public_and_immutable() -> None:
    assert DeisBirthRelease.__dataclass_params__.frozen
    assert callable(iter_normalize_deis_births)
    assert issubclass(FrozenInstanceError, AttributeError)


@pytest.mark.parametrize(
    ("archive_id", "expected_profile"),
    [
        (DeisBirthArchiveId.PROFILE_1992_2000, 25),
        (DeisBirthArchiveId.PROFILE_2001_2019, 29),
        (DeisBirthArchiveId.PROFILE_2020_2023, 25),
    ],
)
def test_archive_identity_selects_one_schema_profile(
    archive_id: DeisBirthArchiveId,
    expected_profile: int,
) -> None:
    assert len(archive_id.expected_columns) == expected_profile


@pytest.mark.parametrize(
    "columns",
    [
        EXPECTED_BIRTH_COLUMNS_29[:-1],
        (*EXPECTED_BIRTH_COLUMNS_29, "EXTRA"),
        (
            EXPECTED_BIRTH_COLUMNS_29[1],
            EXPECTED_BIRTH_COLUMNS_29[0],
            *EXPECTED_BIRTH_COLUMNS_29[2:],
        ),
    ],
)
def test_birth_header_fails_closed_on_missing_extra_or_reordered_columns(
    tmp_path: Path,
    columns: tuple[str, ...],
) -> None:
    release = _birth_release_for(
        tmp_path,
        [_birth_row()],
        columns=columns,
    )

    with pytest.raises(DataContractError, match=r"header|profile|schema"):
        next(
            iter_normalize_deis_births(
                release,
                identity_registry=IdentityRegistry(),
            )
        )


def test_birth_archive_requires_exact_member_and_rejects_extra_csv(
    tmp_path: Path,
) -> None:
    missing = _birth_release_for(
        tmp_path,
        [_birth_row()],
        requested_member="absent.csv",
        sibling_csv=("other.csv", b"x\n"),
        key="missing_member",
    )
    with pytest.raises(MissingOfficialDataError, match=r"absent\.csv"):
        next(
            iter_normalize_deis_births(
                missing,
                identity_registry=IdentityRegistry(),
            )
        )

    extra = _birth_release_for(
        tmp_path,
        [_birth_row()],
        sibling_csv=("unexpected.csv", b"x\n"),
        key="extra_member",
    )
    with pytest.raises(DataContractError, match="CSV member"):
        next(
            iter_normalize_deis_births(
                extra,
                identity_registry=IdentityRegistry(),
            )
        )


def test_release_controls_and_crc_are_verified_before_first_batch(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(
        tmp_path,
        [_birth_row()],
        controls=DeisBirthReleaseControls(raw_rows=2),
    )

    with pytest.raises(DataContractError, match="raw_rows"):
        next(
            iter_normalize_deis_births(
                release,
                identity_registry=IdentityRegistry(),
            )
        )


def test_birth_archive_member_and_missing_dimension_controls(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(
        tmp_path,
        [
            _birth_row(
                MES_NAC="",
                GRUPO_ETARIO_MADRE="NO ESPECIFICADO",
                REGION_RESIDENCIA="99",
                GLOSA_REGION_RESIDENCIA="Ignorada",
            )
        ],
        key="extended_controls",
    )
    with release.assets.open(release.source_key) as stream:
        archive_size = os.fstat(stream.fileno()).st_size
        with zipfile.ZipFile(stream) as archive:
            member_size = archive.getinfo(release.archive_member).file_size
    controls = DeisBirthReleaseControls(
        raw_rows=1,
        archive_size=archive_size,
        member_size=member_size,
        blank_birth_months=1,
        unknown_maternal_age_groups=1,
        unknown_residence_regions=1,
        distinct_known_residence_regions=0,
    )

    [batch] = list(
        iter_normalize_deis_births(
            replace(release, controls=controls),
            identity_registry=IdentityRegistry(),
        )
    )
    assert batch.audit.archive_size == controls.archive_size
    assert batch.audit.member_size == controls.member_size
    assert batch.audit.blank_birth_months == 1
    assert batch.audit.unknown_maternal_age_groups == 1
    assert batch.audit.unknown_residence_regions == 1
    assert batch.audit.distinct_known_residence_regions == 0

    with pytest.raises(DataContractError, match="blank_birth_months"):
        next(
            iter_normalize_deis_births(
                replace(
                    release,
                    controls=replace(controls, blank_birth_months=2),
                ),
                identity_registry=IdentityRegistry(),
            )
        )


def test_streaming_path_uses_multiple_batches_and_no_whole_file_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _birth_row(
            ANO_NAC="2023",
            MES_NAC=str(month),
            HIJ_VIVOS=f"{month:02d}",
        )
        for month in range(1, 13)
    ]
    release = _birth_release_for(tmp_path, rows)

    def prohibited(*args: object, **kwargs: object) -> object:
        raise AssertionError("whole-file CSV/ZIP APIs are prohibited")

    monkeypatch.setattr(pyarrow.csv, "read_csv", prohibited)
    monkeypatch.setattr(zipfile.ZipFile, "read", prohibited)
    batches = list(
        iter_normalize_deis_births(
            release,
            identity_registry=IdentityRegistry(),
            block_size=256,
        )
    )

    assert len(batches) > 1
    assert sum(len(batch.normalized) for batch in batches) == 12
    assert sum(batch.audit.raw_rows for batch in batches[:1]) == 12


def test_birth_parser_consumes_only_verified_descriptor(
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
    release = _birth_release_for(
        tmp_path,
        [_birth_row(profile=25)],
        archive_id=DeisBirthArchiveId.PROFILE_2020_2023,
    )

    [batch] = list(
        iter_normalize_deis_births(
            release,
            identity_registry=IdentityRegistry(),
        )
    )

    assert calls == [release.source_key]
    assert batch.audit.raw_rows == 1


def test_birth_fact_identity_is_stable_across_release_revisions(
    tmp_path: Path,
) -> None:
    first_release = _birth_release_for(
        tmp_path,
        [_birth_row(profile=25)],
        archive_id=DeisBirthArchiveId.PROFILE_2020_2023,
        key="stable_birth_fact",
    )
    first = normalize_deis_births(
        first_release,
        identity_registry=IdentityRegistry(),
    )
    second_release = _birth_release_for(
        tmp_path,
        [_birth_row(profile=25)],
        archive_id=DeisBirthArchiveId.PROFILE_2020_2023,
        key="stable_birth_fact",
        sibling=("review-note.txt", b"release revision two"),
    )
    second = normalize_deis_births(
        second_release,
        identity_registry=IdentityRegistry(),
    )

    assert set(first.normalized["fact_id"]) == set(second.normalized["fact_id"])
    assert set(first.normalized["release_id"]) != set(second.normalized["release_id"])
    assert set(first.normalized["observation_id"]) != set(second.normalized["observation_id"])


def test_profile_29_normalizes_to_strict_demographic_batch_and_preserves_hij_sidecar(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(
        tmp_path,
        [_birth_row(NACIONALIDAD_MADRE="e")],
        controls=DeisBirthReleaseControls(
            raw_rows=1,
            counts_by_year=((2023, 1),),
            counts_by_sex=(("2", 1),),
            lowercase_nationality_codes=1,
        ),
    )

    registry = IdentityRegistry()
    [batch] = list(
        iter_normalize_deis_births(
            release,
            identity_registry=registry,
        )
    )
    row = batch.normalized.iloc[0]
    assert tuple(batch.normalized.columns) == EXPECTED_DEMOGRAPHIC_COLUMNS
    pd.testing.assert_frame_equal(
        validate_demographic_observations(batch.normalized),
        batch.normalized,
    )
    pd.testing.assert_frame_equal(
        batch.normalized,
        batch.domain_batch.observations,
    )
    assert row["domain"] == "demographic"
    assert row["observation_role"] == "observed_fact"
    assert row["parity_scope"] == "unknown_order"
    assert row["release_id"] == release.assets.manifest.release_id.value
    assert batch.domain_batch.identities.require(row["fact_id"])
    assert batch.domain_batch.identities.require(row["observation_id"])
    assert row["variable"] == "births"
    assert row["semantic_kind"] == "count"
    assert row["unit"] == "event"
    assert row["value"] == 1
    assert row["period_start"] == date(2023, 1, 1)
    assert row["period_end"] == date(2023, 2, 1)
    assert row["date_basis"] == "birth_occurrence_month"
    assert row["age_role"] == "mother"
    assert row["age_measure_unit"] == "grouped_completed_years"
    assert pd.isna(row["age_quantity"])
    assert (row["age_lower"], row["age_upper"], row["age_open"]) == (
        20.0,
        25.0,
        False,
    )
    assert row["sex"] == "female"
    assert row["sex_role"] == "newborn"
    assert row["region"] == "CL-08"
    assert row["geography_basis"] == "maternal_residence"
    assert row["geography_vintage"] == "dpa-2018"
    assert row["nationality"] == "E"
    assert row["nationality_role"] == "mother"
    assert pd.isna(row["birth_order"])
    assert pd.isna(row["registration_start"])
    assert pd.isna(row["registration_end"])
    missingness = json.loads(row["missingness_reason"])
    assert set(
        [
            "birth_day",
            "registration_window",
            "residence_commune",
            "parity",
            "birth_order",
        ]
    ).issubset(missingness)
    assert batch.raw_event_features is not None
    assert batch.raw_event_features.to_pylist() == [
        {
            "archive_id": "2001_2019",
            "archive_row_number": 1,
            "HIJ_VIVOS": "02",
            "HIJ_FALL": "",
            "HIJ_MORT": "01",
            "HIJ_TOTAL": "03",
        }
    ]
    assert batch.audit.lowercase_nationality_codes == 1
    assert "not parity or birth order" in " ".join(batch.provenance.notes)
    assert batch.provenance.observation_start == release.observation_start
    assert batch.provenance.observation_end == date(2024, 1, 1)
    leaked = batch.normalized
    leaked.loc[:, "value"] = -999.0
    assert (batch.normalized["value"] >= 0.0).all()
    assert (batch.domain_batch.observations["value"] >= 0.0).all()


def test_event_batch_sums_distinct_source_rows_with_same_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chile_demographic_pde.countries.chile.deis_births as deis_births

    release = _birth_release_for(
        tmp_path,
        [_birth_row(), _birth_row()],
        controls=DeisBirthReleaseControls(raw_rows=2),
        key="duplicate_dimensions",
    )
    normalization_calls = 0
    original_normalize = deis_births._normalize_birth_record

    def count_normalizations(
        row: dict[str, str],
        birth_release: DeisBirthRelease,
    ) -> dict[str, object]:
        nonlocal normalization_calls
        normalization_calls += 1
        return original_normalize(row, birth_release)

    monkeypatch.setattr(
        deis_births,
        "_normalize_birth_record",
        count_normalizations,
    )

    [batch] = list(
        iter_normalize_deis_births(
            release,
            identity_registry=IdentityRegistry(),
        )
    )

    assert normalization_calls == 1
    assert len(batch.normalized) == 1
    assert batch.normalized["value"].tolist() == [2]
    assert batch.raw_event_features is not None
    assert batch.raw_event_features["archive_row_number"].to_pylist() == [1, 2]


def test_global_birth_aggregation_is_unique_across_stream_batches(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(
        tmp_path,
        [_birth_row() for _ in range(6)],
        controls=DeisBirthReleaseControls(raw_rows=6),
        key="global_duplicate_dimensions",
    )
    stream_registry = IdentityRegistry()
    streamed = list(
        iter_normalize_deis_births(
            release,
            identity_registry=stream_registry,
            block_size=256,
        )
    )
    assert len(streamed) > 1
    sidecars: list[pa.Table] = []
    aggregated = aggregate_deis_births(
        streamed,
        identity_registry=stream_registry,
        raw_feature_sink=sidecars.append,
    )
    convenience_sidecars: list[pa.Table] = []
    convenient = normalize_deis_births(
        release,
        identity_registry=IdentityRegistry(),
        block_size=256,
        raw_feature_sink=convenience_sidecars.append,
    )

    assert isinstance(DEMOGRAPHIC_OBSERVATION_KEY, tuple)
    assert len(aggregated.normalized) == 1
    assert aggregated.normalized.iloc[0]["value"] == 6
    assert aggregated.raw_event_features is None
    assert [
        row_number for table in sidecars for row_number in table["archive_row_number"].to_pylist()
    ] == list(range(1, 7))
    assert [
        row_number
        for table in convenience_sidecars
        for row_number in table["archive_row_number"].to_pylist()
    ] == list(range(1, 7))
    with pytest.raises(DataContractError, match="raw_feature_sink"):
        aggregate_deis_births(
            streamed,
            identity_registry=IdentityRegistry(),
        )
    with pytest.raises(DataContractError, match="raw_feature_sink"):
        normalize_deis_births(
            release,
            identity_registry=IdentityRegistry(),
            block_size=256,
        )
    pd.testing.assert_frame_equal(aggregated.normalized, convenient.normalized)
    pd.testing.assert_frame_equal(
        aggregated.normalized,
        aggregated.domain_batch.observations,
    )
    assert aggregated.provenance.transformation_ids[-1] == "deis-birth-record-sum-v1"


def test_stream_shards_and_global_aggregate_have_distinct_identities_and_metadata(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(
        tmp_path,
        [_birth_row(profile=25) for _ in range(12)],
        archive_id=DeisBirthArchiveId.PROFILE_2020_2023,
        controls=DeisBirthReleaseControls(raw_rows=12),
        key="stream_identity_partition",
    )
    registry = IdentityRegistry()
    streamed = list(
        iter_normalize_deis_births(
            release,
            identity_registry=registry,
            block_size=256,
        )
    )
    assert len(streamed) > 1
    aggregated = aggregate_deis_births(
        streamed,
        identity_registry=registry,
    )

    shard_rows = pd.DataFrame.from_records(
        [
            batch.normalized.iloc[0]
            .loc[
                [
                    "fact_id",
                    "observation_id",
                    "value",
                    "transformation_id",
                ]
            ]
            .to_dict()
            for batch in streamed
        ]
    )
    aggregate_row = aggregated.normalized.iloc[0]
    assert not shard_rows["observation_id"].duplicated().any()
    assert aggregate_row["observation_id"] not in set(shard_rows["observation_id"])
    assert aggregate_row["fact_id"] not in set(shard_rows["fact_id"])
    assert set(shard_rows["transformation_id"]) == {"deis-birth-stream-shard"}
    assert aggregate_row["transformation_id"] == "deis-birth-record-sum"
    assert "globally sum" in aggregate_row["aggregation_rule"]
    assert int(shard_rows["value"].sum()) == int(aggregate_row["value"]) == 12


def test_birth_batch_carries_immutable_complete_release_identity(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(tmp_path, [_birth_row()])
    [batch] = list(
        iter_normalize_deis_births(
            release,
            identity_registry=IdentityRegistry(),
        )
    )

    assert batch.release == release
    assert batch.release.archive_id is DeisBirthArchiveId.PROFILE_2001_2019
    with pytest.raises(FrozenInstanceError):
        batch.release = release  # type: ignore[misc]


def test_birth_aggregation_rejects_any_audit_provenance_or_release_change(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(
        tmp_path,
        [_birth_row(MES_NAC=str(month)) for month in range(1, 7)],
        controls=DeisBirthReleaseControls(raw_rows=6),
        key="incompatible_batch_identity",
    )
    registry = IdentityRegistry()
    streamed = list(
        iter_normalize_deis_births(
            release,
            identity_registry=registry,
            block_size=256,
        )
    )
    assert len(streamed) > 1
    reference = streamed[0]
    candidate = streamed[1]
    variants = (
        replace(
            candidate,
            audit=replace(
                candidate.audit,
                counts_by_sex=((*candidate.audit.counts_by_sex, ("changed", 0))),
            ),
        ),
        replace(
            candidate,
            domain_batch=NormalizedDomainBatch.create(
                manifest=candidate.domain_batch.manifest,
                observations=candidate.domain_batch.observations,
                provenance=candidate.provenance.model_copy(
                    update={"notes": (*candidate.provenance.notes, "changed")}
                ),
                identities=candidate.domain_batch.identities,
            ),
        ),
        replace(
            candidate,
            release=replace(
                candidate.release,
                archive_id=DeisBirthArchiveId.PROFILE_1992_2000,
            ),
        ),
        replace(
            candidate,
            release=replace(
                candidate.release,
                geography_vintage="changed",
            ),
        ),
    )

    for changed in variants:
        with pytest.raises(DataContractError, match="incompatible"):
            aggregate_deis_births(
                [reference, changed, *streamed[2:]],
                identity_registry=registry,
                raw_feature_sink=lambda _table: None,
            )


def test_profile_25_has_no_hij_sidecar_and_blank_month_is_annual_unknown(
    tmp_path: Path,
) -> None:
    release = _birth_release_for(
        tmp_path,
        [
            _birth_row(
                profile=25,
                ANO_NAC="1992",
                MES_NAC="",
                NACIONALIDAD_MADRE="",
                GRUPO_ETARIO_MADRE="MENORES 15 AÑO",
            )
        ],
        archive_id=DeisBirthArchiveId.PROFILE_1992_2000,
    )

    [batch] = list(
        iter_normalize_deis_births(
            release,
            identity_registry=IdentityRegistry(),
        )
    )
    row = batch.normalized.iloc[0]
    assert batch.raw_event_features is None
    assert row["period_start"] == date(1992, 1, 1)
    assert row["period_end"] == date(1993, 1, 1)
    assert row["date_basis"] == "birth_occurrence_year_when_month_missing"
    assert (row["age_lower"], row["age_upper"], row["age_open"]) == (
        0.0,
        15.0,
        False,
    )
    assert pd.isna(row["nationality"])
    missingness = json.loads(row["missingness_reason"])
    assert missingness["birth_month"] == "missing_in_source_year_only"
    assert missingness["maternal_nationality"] == "missing_in_source"


@pytest.mark.parametrize(
    ("source_group", "expected_lower", "expected_upper", "expected_open"),
    [
        ("MENORES 15 AÑO", 0.0, 15.0, False),
        ("MENORES 15 AÑOS", 0.0, 15.0, False),
        ("15 A 19 AÑOS", 15.0, 20.0, False),
        ("45 A 49 AÑOS", 45.0, 50.0, False),
        ("50 O MAS AÑOS", 50.0, None, True),
        ("NO ESPECIFICADO", None, None, False),
    ],
)
def test_maternal_age_groups_are_intervals_never_exact_ages(
    tmp_path: Path,
    source_group: str,
    expected_lower: float | None,
    expected_upper: float | None,
    expected_open: bool,
) -> None:
    release = _birth_release_for(
        tmp_path,
        [_birth_row(GRUPO_ETARIO_MADRE=source_group)],
        key=f"age_{abs(hash(source_group))}",
    )
    [batch] = list(
        iter_normalize_deis_births(
            release,
            identity_registry=IdentityRegistry(),
        )
    )
    row = batch.normalized.iloc[0]
    if expected_lower is None:
        assert pd.isna(row["age_lower"])
    else:
        assert row["age_lower"] == expected_lower
    if expected_upper is None:
        assert pd.isna(row["age_upper"])
    else:
        assert row["age_upper"] == expected_upper
    assert row["age_open"] == expected_open
    assert pd.isna(row["age_quantity"])


@pytest.mark.parametrize(
    ("source_sex", "normalized"),
    [("1", "male"), ("2", "female"), ("9", "indeterminate")],
)
def test_newborn_sex_codes_are_exact(
    tmp_path: Path,
    source_sex: str,
    normalized: str,
) -> None:
    release = _birth_release_for(
        tmp_path,
        [_birth_row(SEXO=source_sex)],
        key=f"sex_{source_sex}",
    )
    [batch] = list(
        iter_normalize_deis_births(
            release,
            identity_registry=IdentityRegistry(),
        )
    )
    assert batch.normalized.iloc[0]["sex"] == normalized


def test_bad_sex_and_region_label_conflict_fail_closed(tmp_path: Path) -> None:
    bad_sex = _birth_release_for(
        tmp_path,
        [_birth_row(SEXO="3")],
        key="bad_sex",
    )
    with pytest.raises(DataContractError, match="SEXO"):
        next(
            iter_normalize_deis_births(
                bad_sex,
                identity_registry=IdentityRegistry(),
            )
        )

    bad_region = _birth_release_for(
        tmp_path,
        [_birth_row(REGION_RESIDENCIA="8", GLOSA_REGION_RESIDENCIA="De Ñuble")],
        key="bad_region",
    )
    with pytest.raises(DataContractError, match="region"):
        next(
            iter_normalize_deis_births(
                bad_region,
                identity_registry=IdentityRegistry(),
            )
        )


def test_verified_unknown_region_99_remains_explicit_unknown(tmp_path: Path) -> None:
    release = _birth_release_for(
        tmp_path,
        [_birth_row(REGION_RESIDENCIA="99", GLOSA_REGION_RESIDENCIA="Ignorada")],
    )
    [batch] = list(
        iter_normalize_deis_births(
            release,
            identity_registry=IdentityRegistry(),
        )
    )
    row = batch.normalized.iloc[0]
    assert row["region"] == "unknown"
    assert "maternal_residence" in row["missingness_reason"]


def test_ine_birth_control_contract_is_exact_and_release_is_immutable() -> None:
    assert INE_BIRTH_CONTROL_COLUMNS == (
        "year",
        "month",
        "sex",
        "region",
        "geography_vintage",
        "births",
    )
    assert isinstance(INE_BIRTH_CONTROL_COLUMNS, tuple)
    frame = _ine_birth_controls()
    release, _ = _ine_birth_release(frame)
    with pytest.raises(FrozenInstanceError):
        release.vintage = "changed"  # type: ignore[misc]

    normalizer = getattr(
        vital_statistics,
        "normalize_ine_birth_controls_curated",
        None,
    )
    if normalizer is None:
        pytest.fail("normalize_ine_birth_controls_curated is not implemented")
    with pytest.raises(DataContractError, match=r"exact columns|contract"):
        normalizer(
            frame.rename(columns={"births": "value"}),
            release=release,
            identity_registry=IdentityRegistry(),
        )


def test_birth_sources_have_explicit_distinct_parity_scope(
    tmp_path: Path,
) -> None:
    deis_release = _birth_release_for(
        tmp_path,
        [_birth_row(profile=25)],
        archive_id=DeisBirthArchiveId.PROFILE_2020_2023,
    )
    deis = next(
        iter_normalize_deis_births(
            deis_release,
            identity_registry=IdentityRegistry(),
        )
    )
    controls = _normalize_ine()

    assert set(deis.normalized["parity_scope"]) == {"unknown_order"}
    assert set(deis.normalized["observation_role"]) == {"observed_fact"}
    assert set(controls.observations["parity_scope"]) == {"all_orders"}
    assert set(controls.observations["observation_role"]) == {"diagnostic_control"}


@pytest.mark.parametrize(
    ("kind", "month", "period_start", "period_end", "provisional"),
    [
        (
            IneBirthReleaseKind.FINAL_ANNUAL,
            pd.NA,
            date(2023, 1, 1),
            date(2024, 1, 1),
            False,
        ),
        (
            IneBirthReleaseKind.PROVISIONAL_ANNUAL,
            pd.NA,
            date(2023, 1, 1),
            date(2024, 1, 1),
            True,
        ),
        (
            IneBirthReleaseKind.COYUNTURAL_MONTHLY,
            2,
            date(2023, 2, 1),
            date(2023, 3, 1),
            True,
        ),
    ],
)
def test_ine_curated_period_grain_derives_checked_release_kind(
    kind: IneBirthReleaseKind,
    month: object,
    period_start: date,
    period_end: date,
    provisional: bool,
) -> None:
    normalized = _normalize_ine(
        _ine_birth_controls(month=month),
        kind=kind,
    )
    row = normalized.observations.iloc[0]

    assert row["period_start"] == period_start
    assert row["period_end"] == period_end
    assert row["date_basis"] == (
        "birth_occurrence_month"
        if kind is IneBirthReleaseKind.COYUNTURAL_MONTHLY
        else "birth_occurrence_year"
    )
    assert row["sex_role"] == "newborn"
    assert row["geography_basis"] == "maternal_residence"
    assert pd.isna(row["geography_vintage"])
    assert row["provisional"] is provisional or row["provisional"] == provisional
    assert row["parity_scope"] == "all_orders"
    assert row["observation_role"] == "diagnostic_control"
    assert normalized.provenance.observation_start == date(2023, 1, 1)
    assert normalized.provenance.observation_end == date(2024, 1, 1)


def test_ine_birth_controls_require_exact_attested_curated_bytes() -> None:
    frame = _ine_birth_controls()
    release, registry = _ine_birth_release(frame)
    normalizer = getattr(
        vital_statistics,
        "normalize_ine_birth_controls_curated",
        None,
    )
    if normalizer is None:
        pytest.fail("normalize_ine_birth_controls_curated is not implemented")

    normalized = normalizer(
        frame,
        release=release,
        identity_registry=registry,
    )
    assert int(normalized.observations["value"].sum()) == 174_057

    changed = frame.copy(deep=True)
    changed.loc[0, "births"] = 174_058
    with pytest.raises(ChecksumMismatchError):
        normalizer(
            changed,
            release=release,
            identity_registry=IdentityRegistry(),
        )

    wrong_dtype = frame.copy(deep=True)
    wrong_dtype["year"] = wrong_dtype["year"].astype("float64")
    with pytest.raises(DataContractError, match=r"dtype|logical"):
        normalizer(
            wrong_dtype,
            release=release,
            identity_registry=IdentityRegistry(),
        )


def test_ine_mapper_rejects_a_self_consistent_substitute_physical_contract() -> None:
    frame = _ine_birth_controls()
    frame["month"] = frame["month"].astype(object)
    substitute = CuratedTableContract(
        contract_id="ine-birth-controls-curated-v2",
        columns=_INE_CONTROL_CONTRACT.columns,
        logical_types=(
            "integer",
            "string",
            "string",
            "string",
            "string",
            "integer",
        ),
    )
    release, registry = _ine_birth_release(
        frame,
        contract=substitute,
    )
    normalizer = getattr(
        vital_statistics,
        "normalize_ine_birth_controls_curated",
        None,
    )
    if normalizer is None:
        pytest.fail("normalize_ine_birth_controls_curated is not implemented")

    with pytest.raises(DataContractError, match=r"physical|contract"):
        normalizer(
            frame,
            release=release,
            identity_registry=registry,
        )


def test_ine_regional_controls_require_explicit_geography_vintage() -> None:
    missing = _ine_birth_controls(region="CL-08")
    missing_release, missing_registry = _ine_birth_release(missing)
    normalizer = getattr(
        vital_statistics,
        "normalize_ine_birth_controls_curated",
        None,
    )
    if normalizer is None:
        pytest.fail("normalize_ine_birth_controls_curated is not implemented")
    with pytest.raises(DataContractError, match="geography_vintage"):
        normalizer(
            missing,
            release=missing_release,
            identity_registry=missing_registry,
        )

    explicit = _ine_birth_controls(
        region="CL-08",
        geography_vintage="dpa-2018",
    )
    explicit_release, explicit_registry = _ine_birth_release(explicit)
    normalized = normalizer(
        explicit,
        release=explicit_release,
        identity_registry=explicit_registry,
    )
    assert normalized.observations["geography_vintage"].tolist() == ["dpa-2018"]


@pytest.mark.parametrize("bad_births", [-1, 1.5, True, "174057"])
def test_ine_birth_counts_must_be_nonnegative_integers(
    bad_births: object,
) -> None:
    invalid = _ine_birth_controls(births=bad_births)
    if bad_births == -1:
        release, registry = _ine_birth_release(invalid)
        normalizer = getattr(
            vital_statistics,
            "normalize_ine_birth_controls_curated",
            None,
        )
        if normalizer is None:
            pytest.fail("normalize_ine_birth_controls_curated is not implemented")
        with pytest.raises(DataContractError, match=r"integer|nonnegative"):
            normalizer(
                invalid,
                release=release,
                identity_registry=registry,
            )
        return
    with pytest.raises(DataContractError, match=r"logical|dtype|cell"):
        canonical_curated_table_bytes(
            invalid,
            _INE_CONTROL_CONTRACT,
        )


def test_ine_birth_release_kind_rejects_mixed_or_final_monthly_rows() -> None:
    annual = _ine_birth_controls(month=pd.NA)
    monthly = _ine_birth_controls(month=2)
    mixed = pd.concat([annual, monthly], ignore_index=True)
    mixed["month"] = mixed["month"].astype("Int64")
    mixed_release, mixed_registry = _ine_birth_release(
        mixed,
        IneBirthReleaseKind.PROVISIONAL_ANNUAL,
    )
    normalizer = getattr(
        vital_statistics,
        "normalize_ine_birth_controls_curated",
        None,
    )
    if normalizer is None:
        pytest.fail("normalize_ine_birth_controls_curated is not implemented")
    with pytest.raises(DataContractError, match=r"mix|month|grain"):
        normalizer(
            mixed,
            release=mixed_release,
            identity_registry=mixed_registry,
        )

    final_monthly = _ine_birth_controls(month=2)
    final_release, final_registry = _ine_birth_release(
        final_monthly,
        IneBirthReleaseKind.FINAL_ANNUAL,
    )
    with pytest.raises(DataContractError, match=r"month|provisional"):
        normalizer(
            final_monthly,
            release=final_release,
            identity_registry=final_registry,
        )


def test_ine_birth_control_keys_must_be_unique() -> None:
    duplicated = pd.concat(
        [_ine_birth_controls(), _ine_birth_controls()],
        ignore_index=True,
    )
    duplicated["month"] = duplicated["month"].astype("Int64")
    release, registry = _ine_birth_release(duplicated)
    normalizer = getattr(
        vital_statistics,
        "normalize_ine_birth_controls_curated",
        None,
    )
    if normalizer is None:
        pytest.fail("normalize_ine_birth_controls_curated is not implemented")
    with pytest.raises(DataContractError, match="duplicate"):
        normalizer(
            duplicated,
            release=release,
            identity_registry=registry,
        )


def test_2023_final_control_reconciles_diagnostically_without_mutation() -> None:
    controls = _normalize_ine(
        _ine_birth_controls(births=174_057),
    )
    deis = _derived_deis_control_grain_batch(controls)
    deis_before = deis.observations
    controls_before = controls.observations

    diagnostic = compare_deis_births_to_ine_controls(
        deis,
        controls,
        control_release_kind=IneBirthReleaseKind.FINAL_ANNUAL,
    )

    assert diagnostic["deis_value"].tolist() == [174_057]
    assert diagnostic["control_value"].tolist() == [174_057]
    assert diagnostic["difference"].tolist() == [0]
    assert diagnostic["status"].tolist() == ["reconciled_final"]
    pd.testing.assert_frame_equal(deis.observations, deis_before)
    pd.testing.assert_frame_equal(controls.observations, controls_before)


def test_final_control_difference_is_reported_without_correction() -> None:
    controls = _normalize_ine(_ine_birth_controls(births=174_057))
    deis = _derived_deis_control_grain_batch(controls, value=174_050)

    diagnostic = compare_deis_births_to_ine_controls(
        deis,
        controls,
        control_release_kind=IneBirthReleaseKind.FINAL_ANNUAL,
    )

    assert diagnostic["difference"].tolist() == [-7]
    assert diagnostic["status"].tolist() == ["different_final"]
    assert controls.observations["value"].tolist() == [174_057]
    assert deis.observations["value"].tolist() == [174_050]


def test_ine_comparator_rejects_partial_key_reconciliation() -> None:
    controls = _normalize_ine()
    deis = _derived_deis_control_grain_batch(
        controls,
        updates={
            "region": "CL-01",
            "geography_vintage": "dpa-2018",
        },
    )

    with pytest.raises(DataContractError, match="unmatched"):
        compare_deis_births_to_ine_controls(
            deis,
            controls,
            control_release_kind=IneBirthReleaseKind.FINAL_ANNUAL,
        )


@pytest.mark.parametrize(
    "specific_grain",
    [
        {
            "age_lower": 20.0,
            "age_upper": 25.0,
            "age_open": False,
            "age_measure_unit": "grouped_completed_years",
        },
        {"nationality": "E"},
        {"birth_order": 1, "parity_scope": "known_order"},
        {"population_basis": "pensioner"},
        {"cause": "maternal_cause", "cause_role": "underlying"},
    ],
    ids=[
        "maternal-age-specific",
        "maternal-nationality-specific",
        "birth-order-specific",
        "non-not-applicable-population-basis",
        "cause-specific",
    ],
)
def test_ine_comparator_requires_exact_aggregate_control_grain(
    specific_grain: dict[str, object],
) -> None:
    controls = _normalize_ine()
    deis = _derived_deis_control_grain_batch(
        controls,
        updates=specific_grain,
    )

    with pytest.raises(DataContractError, match=r"unmatched|parity"):
        compare_deis_births_to_ine_controls(
            deis,
            controls,
            control_release_kind=IneBirthReleaseKind.FINAL_ANNUAL,
        )


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("period_end", date(2023, 12, 1), "period"),
        ("geography_basis", "registration_place", "geography"),
        ("sex_role", "mother", "sex"),
    ],
)
def test_ine_comparator_rejects_incompatible_concepts(
    column: str,
    replacement: object,
    message: str,
) -> None:
    controls = _normalize_ine()
    deis = _derived_deis_control_grain_batch(
        controls,
        updates={column: replacement},
    )

    with pytest.raises(DataContractError, match=message):
        compare_deis_births_to_ine_controls(
            deis,
            controls,
            control_release_kind=IneBirthReleaseKind.FINAL_ANNUAL,
        )


@pytest.mark.parametrize(
    ("kind", "month", "expected_status"),
    [
        (
            IneBirthReleaseKind.PROVISIONAL_ANNUAL,
            pd.NA,
            "matched_provisional",
        ),
        (
            IneBirthReleaseKind.COYUNTURAL_MONTHLY,
            1,
            "matched_coyuntural",
        ),
    ],
)
def test_nonfinal_ine_controls_never_satisfy_final_truth(
    kind: IneBirthReleaseKind,
    month: object,
    expected_status: str,
) -> None:
    controls = _normalize_ine(
        _ine_birth_controls(month=month),
        kind=kind,
    )
    deis = _derived_deis_control_grain_batch(controls)

    diagnostic = compare_deis_births_to_ine_controls(
        deis,
        controls,
        control_release_kind=kind,
    )

    assert diagnostic["status"].tolist() == [expected_status]
    assert "reconciled_final" not in diagnostic["status"].tolist()
    assert controls.observations["provisional"].tolist() == [True]


def test_comparator_rejects_explicit_release_kind_mismatch() -> None:
    controls = _normalize_ine(
        _ine_birth_controls(month=2),
        kind=IneBirthReleaseKind.COYUNTURAL_MONTHLY,
    )
    deis = _derived_deis_control_grain_batch(controls)

    with pytest.raises(DataContractError, match=r"kind|month|grain"):
        compare_deis_births_to_ine_controls(
            deis,
            controls,
            control_release_kind=IneBirthReleaseKind.PROVISIONAL_ANNUAL,
        )


def test_comparator_validates_calendar_interval_shape_not_only_date_basis() -> None:
    annual = _normalize_ine()
    mislabeled_month = _rebuild_demographic_batch(
        annual,
        updates={"period_end": date(2023, 2, 1)},
    )
    deis = _derived_deis_control_grain_batch(mislabeled_month)

    with pytest.raises(DataContractError, match=r"calendar|interval|grain"):
        compare_deis_births_to_ine_controls(
            deis,
            mislabeled_month,
            control_release_kind=IneBirthReleaseKind.FINAL_ANNUAL,
        )
