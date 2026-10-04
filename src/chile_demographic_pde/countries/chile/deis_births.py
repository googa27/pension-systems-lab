"""Two-pass streaming normalization of official DEIS birth microdata."""

from __future__ import annotations

import csv
import json
import os
import re
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Final, cast

import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv
from pandera.errors import SchemaErrors
from pydantic import ValidationError

from chile_demographic_pde.core.errors import (
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.countries.chile._text import normalize_spanish_label
from chile_demographic_pde.countries.chile.geography import (
    normalize_region_label,
    region_code_for_label,
)
from chile_demographic_pde.data.domain_schemas import (
    DEMOGRAPHIC_OBSERVATION_KEY,
    DemographicObservationBatch,
    validate_demographic_observations,
)
from chile_demographic_pde.data.envelope import (
    ENVELOPE_COLUMNS,
    NormalizedDomainBatch,
    assign_observation_ids,
)
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import VerifiedReleaseAssets
from chile_demographic_pde.data.roles import ObservationDomain

DEIS_BIRTH_COLUMNS_25: Final = (
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
DEIS_BIRTH_COLUMNS_29: Final = (
    *DEIS_BIRTH_COLUMNS_25[:22],
    "HIJ_VIVOS",
    "HIJ_FALL",
    "HIJ_MORT",
    "HIJ_TOTAL",
    *DEIS_BIRTH_COLUMNS_25[22:],
)
_HIJ_COLUMNS: Final = ("HIJ_VIVOS", "HIJ_FALL", "HIJ_MORT", "HIJ_TOTAL")
_BIRTH_IDENTITY_COLUMNS: Final = (
    "ANO_NAC",
    "MES_NAC",
    "SEXO",
    "GRUPO_ETARIO_MADRE",
    "NACIONALIDAD_MADRE",
    "REGION_RESIDENCIA",
    "GLOSA_REGION_RESIDENCIA",
)
_SEX_CODES: Final = {"1": "male", "2": "female", "9": "indeterminate"}
_NATIONALITY_CODES: Final = frozenset({"C", "E", "N"})
_INTEGER_TEXT = re.compile(r"^[0-9]+$")
_CLOSED_MATERNAL_AGE = re.compile(r"^(?P<lower>[0-9]+) a (?P<upper>[0-9]+) anos$")
_TRANSFORMATION_ID: Final = "deis-birth-event-normalization-v1"
_TRANSFORMATION_VERSION: Final = "v1"
_STREAM_TRANSFORMATION_ID: Final = "deis-birth-stream-shard"
_GLOBAL_TRANSFORMATION_ID: Final = "deis-birth-record-sum"
_GLOBAL_PROVENANCE_TRANSFORMATION_ID: Final = "deis-birth-record-sum-v1"
_GLOBAL_AGGREGATION_RULE: Final = (
    "globally sum DEIS birth records over the complete canonical demographic key"
)
_BASE_DIMENSION_MISSINGNESS: Final = {
    "birth_day": "not_published",
    "birth_order": "not_published",
    "parity": "not_published",
    "registration_window": "release_inclusion_metadata_only",
    "residence_commune": "not_published",
}
_DEMOGRAPHIC_DIMENSION_COLUMNS: Final = (
    "age_lower",
    "age_upper",
    "age_open",
    "sex",
    "region",
    "cause",
    "date_basis",
    "age_role",
    "age_measure_unit",
    "age_quantity",
    "sex_role",
    "commune",
    "geography_basis",
    "geography_vintage",
    "cause_code_system",
    "cause_role",
    "external_cause",
    "registration_start",
    "registration_end",
    "birth_order",
    "nationality",
    "nationality_role",
    "country_of_birth",
    "education_level",
    "residence_five_years_ago",
    "migration_status",
)
_BIRTH_PAYLOAD_COLUMNS: Final = (
    *(
        column
        for column in ENVELOPE_COLUMNS
        if column not in {"fact_id", "observation_id", "release_id"}
    ),
    *_DEMOGRAPHIC_DIMENSION_COLUMNS,
)
_NULLABLE_PAYLOAD_COLUMNS: Final = (
    "release_missingness_reason",
    "source_status_code",
    "age_lower",
    "age_upper",
    "age_quantity",
    "commune",
    "cause_code_system",
    "cause_role",
    "external_cause",
    "registration_start",
    "registration_end",
    "birth_order",
    "nationality",
    "country_of_birth",
    "education_level",
    "residence_five_years_ago",
    "migration_status",
)


class DeisBirthArchiveId(StrEnum):
    """Exact DEIS birth archive identity and its bound raw schema profile."""

    PROFILE_1992_2000 = "1992_2000"
    PROFILE_2001_2019 = "2001_2019"
    PROFILE_2020_2023 = "2020_2023"

    @property
    def expected_columns(self) -> tuple[str, ...]:
        """Return the immutable ordered columns published by this archive."""

        if self is DeisBirthArchiveId.PROFILE_2001_2019:
            return DEIS_BIRTH_COLUMNS_29
        return DEIS_BIRTH_COLUMNS_25

    @property
    def expected_asset_key(self) -> str:
        """Return the exact official catalog key bound to this archive."""

        return f"deis_births_{self.value}"

    @property
    def expected_filename(self) -> str:
        """Return the exact official ZIP basename bound to this archive."""

        return f"Serie_Nacimientos_{self.value}.zip"

    @property
    def expected_member(self) -> str:
        """Return the exact official CSV member bound to this archive."""

        return f"Serie_Nacimientos_{self.value}.csv"


class DeisBirthAssetContract(StrEnum):
    """Whether a release is a pinned official asset or a local profile fixture."""

    PINNED_OFFICIAL = "pinned_official"
    LOCAL_PROFILE_FIXTURE = "local_profile_fixture"


@dataclass(frozen=True, slots=True)
class DeisBirthReleaseControls:
    """Optional source-level totals checked before the first batch is yielded."""

    raw_rows: int | None = None
    counts_by_year: tuple[tuple[int, int], ...] = ()
    counts_by_sex: tuple[tuple[str, int], ...] = ()
    lowercase_nationality_codes: int | None = None
    blank_birth_months: int | None = None
    unknown_maternal_age_groups: int | None = None
    unknown_residence_regions: int | None = None
    distinct_known_residence_regions: int | None = None
    archive_size: int | None = None
    member_size: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("raw_rows", self.raw_rows),
            ("lowercase_nationality_codes", self.lowercase_nationality_codes),
            ("blank_birth_months", self.blank_birth_months),
            ("unknown_maternal_age_groups", self.unknown_maternal_age_groups),
            ("unknown_residence_regions", self.unknown_residence_regions),
            (
                "distinct_known_residence_regions",
                self.distinct_known_residence_regions,
            ),
            ("archive_size", self.archive_size),
            ("member_size", self.member_size),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name, pairs in (
            ("counts_by_year", self.counts_by_year),
            ("counts_by_sex", self.counts_by_sex),
        ):
            keys = [key for key, _ in pairs]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{name} contains duplicate control keys.")
            if any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for _, count in pairs
            ):
                raise ValueError(f"{name} counts must be nonnegative integers.")


@dataclass(frozen=True, slots=True)
class DeisBirthRelease:
    """Verified archive and release metadata for one DEIS birth series."""

    assets: VerifiedReleaseAssets
    source_key: str
    sources: tuple[SourceRef, ...]
    released_at: date
    observation_start: date
    observation_end: date
    archive_id: DeisBirthArchiveId
    archive_member: str
    geography_vintage: str
    registration_inclusion_cutoff: str
    controls: DeisBirthReleaseControls = field(default_factory=DeisBirthReleaseControls)
    asset_contract: DeisBirthAssetContract = DeisBirthAssetContract.PINNED_OFFICIAL

    def __post_init__(self) -> None:
        if not isinstance(self.assets, VerifiedReleaseAssets):
            raise DataContractError(
                "DEIS birth release assets must be a VerifiedReleaseAssets boundary."
            )
        manifest = self.assets.manifest
        if (
            not isinstance(self.source_key, str)
            or self.source_key != manifest.primary_fact_source_key
        ):
            raise DataContractError(
                "DEIS birth source key must equal the manifest primary fact source."
            )
        if not isinstance(self.sources, tuple) or not self.sources:
            raise DataContractError("DEIS birth sources must be a nonempty immutable tuple.")
        validated_sources: list[SourceRef] = []
        for source in self.sources:
            try:
                validated_sources.append(
                    SourceRef.model_validate(source.model_dump(mode="python", warnings="none"))
                )
            except (AttributeError, TypeError, ValueError, ValidationError):
                raise DataContractError(
                    "DEIS birth source provenance violates its typed contract."
                ) from None
        sources_by_key = {source.source_key: source for source in validated_sources}
        if len(sources_by_key) != len(validated_sources):
            raise DataContractError("DEIS birth source provenance keys must be unique.")
        manifest_keys = {asset.source_key for asset in manifest.assets}
        if set(sources_by_key) != manifest_keys:
            raise DataContractError(
                "DEIS birth source provenance must cover every manifest asset exactly."
            )
        for asset in manifest.assets:
            source = sources_by_key[asset.source_key]
            if source.sha256 != asset.sha256:
                raise DataContractError("DEIS birth source SHA-256 must match its manifest asset.")
            if source.retrieved_at != asset.available_at:
                raise DataContractError(
                    "DEIS birth source retrieval time must match its manifest asset."
                )
            if (
                source.release_date != asset.released_at
                or source.release_missingness_reason != asset.release_missingness_reason
            ):
                raise DataContractError(
                    "DEIS birth source publisher release metadata must match its manifest asset."
                )
        object.__setattr__(
            self,
            "sources",
            tuple(sources_by_key[asset.source_key] for asset in manifest.assets),
        )
        if any(
            type(value) is not date
            for value in (
                self.released_at,
                self.observation_start,
                self.observation_end,
            )
        ):
            raise DataContractError(
                "DEIS birth release and observation bounds must be built-in dates."
            )
        if sources_by_key[self.source_key].release_date != self.released_at:
            raise DataContractError(
                "DEIS birth released_at must equal the selected source release date."
            )
        member = PurePosixPath(self.archive_member)
        if not self.archive_member.strip() or member.is_absolute() or ".." in member.parts:
            raise ValueError("archive_member must be a safe nonblank ZIP member path.")
        if self.observation_start > self.observation_end:
            raise ValueError("observation_start must not follow observation_end.")
        if not isinstance(self.archive_id, DeisBirthArchiveId):
            raise TypeError("archive_id must be a DeisBirthArchiveId.")
        if not isinstance(self.asset_contract, DeisBirthAssetContract):
            raise TypeError("asset_contract must be a DeisBirthAssetContract.")
        if self.asset_contract is DeisBirthAssetContract.PINNED_OFFICIAL:
            if self.source_key != self.archive_id.expected_asset_key:
                raise ValueError(
                    "DEIS birth archive_id and official source key must match their exact binding."
                )
            if self.archive_member != self.archive_id.expected_member:
                raise ValueError(
                    "DEIS birth official release requires the exact archive member "
                    f"{self.archive_id.expected_member!r}."
                )
        if not self.geography_vintage.strip():
            raise ValueError("geography_vintage must be nonblank.")
        if not self.registration_inclusion_cutoff.strip():
            raise ValueError("registration_inclusion_cutoff must be nonblank.")


@dataclass(frozen=True, slots=True)
class DeisBirthIngestionAudit:
    """Immutable first-pass controls and source diagnostics."""

    raw_rows: int
    arrow_batches: int
    counts_by_year: tuple[tuple[int, int], ...]
    counts_by_sex: tuple[tuple[str, int], ...]
    lowercase_nationality_codes: int
    blank_birth_months: int
    unknown_maternal_age_groups: int
    unknown_residence_regions: int
    distinct_known_residence_regions: int
    sha256: str
    archive_size: int
    member_size: int


@dataclass(frozen=True, slots=True, eq=False)
class DeisBirthBatch:
    """One bounded normalized Arrow batch and its optional lossless sidecar."""

    domain_batch: DemographicObservationBatch
    raw_event_features: pa.Table | None
    audit: DeisBirthIngestionAudit
    release: DeisBirthRelease

    @property
    def normalized(self) -> pd.DataFrame:
        """Return an owned copy of the strict demographic observations."""

        return self.domain_batch.observations

    @property
    def provenance(self) -> VariableProvenance:
        """Expose the exact provenance carried by the immutable domain batch."""

        return self.domain_batch.provenance


def iter_normalize_deis_births(
    release: DeisBirthRelease,
    *,
    identity_registry: IdentityRegistry,
    block_size: int = 1 << 20,
) -> Iterator[DeisBirthBatch]:
    """Validate one archive fully, then yield bounded normalized birth batches."""

    if not isinstance(release, DeisBirthRelease):
        raise TypeError("release must be a DeisBirthRelease.")
    if not isinstance(identity_registry, IdentityRegistry):
        raise DataContractError("DEIS birth normalization requires an IdentityRegistry.")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    selected_asset = release.assets.manifest.asset(release.source_key)
    try:
        with release.assets.open(release.source_key) as stream:
            archive_size = os.fstat(stream.fileno()).st_size
            with zipfile.ZipFile(stream) as archive:
                info = _exact_birth_member(archive, release.archive_member)
                audit = _scan_authoritative_source(
                    archive,
                    info,
                    release=release,
                    block_size=block_size,
                    observed_sha256=selected_asset.sha256,
                    archive_size=archive_size,
                    member_size=info.file_size,
                )
                _validate_controls(release.controls, audit)
                provenance = _birth_provenance(release)
                archive_row_number = 0
                for raw_batch in _record_batches(
                    archive,
                    info,
                    expected_columns=release.archive_id.expected_columns,
                    block_size=block_size,
                ):
                    first_row_number = archive_row_number + 1
                    archive_row_number += raw_batch.num_rows
                    payload = _stream_partition_payload(
                        _normalize_batch(raw_batch, release),
                        first_row_number=first_row_number,
                        last_row_number=archive_row_number,
                    )
                    yield DeisBirthBatch(
                        domain_batch=_finalize_birth_batch(
                            payload,
                            release=release,
                            provenance=provenance,
                            identity_registry=identity_registry,
                        ),
                        raw_event_features=_raw_feature_sidecar(
                            raw_batch,
                            release.archive_id,
                            first_row_number=first_row_number,
                        ),
                        audit=audit,
                        release=release,
                    )
                if archive_row_number != audit.raw_rows:
                    raise DataContractError(
                        "DEIS birth second pass does not reconcile to the verified row count."
                    )
    except zipfile.BadZipFile as error:
        detail = "CRC" if "CRC" in str(error).upper() else "ZIP"
        raise DataContractError(f"DEIS birth {detail} validation failed: {error}") from error


def aggregate_deis_births(
    batches: Iterable[DeisBirthBatch],
    *,
    identity_registry: IdentityRegistry,
    raw_feature_sink: Callable[[pa.Table], None] | None = None,
) -> DeisBirthBatch:
    """Incrementally form one globally unique canonical birth-count cube."""

    if not isinstance(identity_registry, IdentityRegistry):
        raise DataContractError("DEIS birth aggregation requires an IdentityRegistry.")
    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise DataContractError("Cannot aggregate an empty DEIS birth stream.") from error
    if first.raw_event_features is not None and raw_feature_sink is None:
        raise DataContractError(
            "A raw_feature_sink is required to preserve the 29-profile HIJ sidecar."
        )
    if first.raw_event_features is not None and raw_feature_sink is not None:
        raw_feature_sink(first.raw_event_features)
    accumulated: dict[tuple[object, ...], dict[str, object]] = {}
    _accumulate_birth_counts(
        accumulated,
        _global_aggregate_payload(first.normalized),
    )
    for batch in iterator:
        _require_compatible_birth_batch(first, batch)
        if batch.raw_event_features is not None:
            if raw_feature_sink is None:
                raise DataContractError(
                    "A raw_feature_sink is required to preserve every HIJ sidecar."
                )
            raw_feature_sink(batch.raw_event_features)
        _accumulate_birth_counts(
            accumulated,
            _global_aggregate_payload(batch.normalized),
        )
    normalized = _aggregate_normalized_frame(
        pd.DataFrame.from_records(
            list(accumulated.values()),
            columns=_BIRTH_PAYLOAD_COLUMNS,
        )
    )
    if int(normalized["value"].sum()) != first.audit.raw_rows:
        raise DataContractError(
            "Globally aggregated DEIS birth counts do not reconcile to raw rows."
        )
    provenance = first.provenance.model_copy(
        update={
            "aggregation_rules": (
                *first.provenance.aggregation_rules,
                _GLOBAL_AGGREGATION_RULE,
            ),
            "transformation_ids": (
                *first.provenance.transformation_ids,
                _GLOBAL_PROVENANCE_TRANSFORMATION_ID,
            ),
        }
    )
    return DeisBirthBatch(
        domain_batch=_finalize_birth_batch(
            normalized,
            release=first.release,
            provenance=provenance,
            identity_registry=identity_registry,
        ),
        raw_event_features=None,
        audit=first.audit,
        release=first.release,
    )


def _accumulate_birth_counts(
    accumulated: dict[tuple[object, ...], dict[str, object]],
    frame: pd.DataFrame,
) -> None:
    records = cast(list[dict[str, object]], frame.to_dict(orient="records"))
    for record in records:
        key = tuple(
            _stable_key_scalar(record[column], column=column)
            for column in DEMOGRAPHIC_OBSERVATION_KEY
        )
        existing = accumulated.get(key)
        if existing is None:
            accumulated[key] = record
        else:
            existing["value"] = int(cast(int, existing["value"])) + int(cast(int, record["value"]))


def _stable_key_scalar(value: object, *, column: str) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return ("__missing__", column)
    if isinstance(value, float) and value != value:
        return ("__missing__", column)
    return value


def normalize_deis_births(
    release: DeisBirthRelease,
    *,
    identity_registry: IdentityRegistry,
    block_size: int = 1 << 20,
    raw_feature_sink: Callable[[pa.Table], None] | None = None,
) -> DeisBirthBatch:
    """Return one notebook-ready globally unique cube from a verified release."""

    if release.archive_id is DeisBirthArchiveId.PROFILE_2001_2019 and raw_feature_sink is None:
        raise DataContractError(
            "A raw_feature_sink is required for the 29-profile DEIS birth archive."
        )
    return aggregate_deis_births(
        iter_normalize_deis_births(
            release,
            identity_registry=identity_registry,
            block_size=block_size,
        ),
        identity_registry=identity_registry,
        raw_feature_sink=raw_feature_sink,
    )


def _require_compatible_birth_batch(
    reference: DeisBirthBatch,
    candidate: DeisBirthBatch,
) -> None:
    if (
        candidate.audit != reference.audit
        or candidate.provenance != reference.provenance
        or candidate.release != reference.release
        or (candidate.raw_event_features is None) != (reference.raw_event_features is None)
    ):
        raise DataContractError(
            "DEIS birth aggregation cannot mix incompatible profiles or vintages."
        )


def _exact_birth_member(
    archive: zipfile.ZipFile,
    requested_member: str,
) -> zipfile.ZipInfo:
    matching = [info for info in archive.infolist() if info.filename == requested_member]
    if not matching:
        raise MissingOfficialDataError(
            f"Exact DEIS birth archive member {requested_member!r} is absent."
        )
    if len(matching) != 1:
        raise DataContractError(f"DEIS birth archive duplicates member {requested_member!r}.")
    csv_members = [info for info in archive.infolist() if info.filename.lower().endswith(".csv")]
    if len(csv_members) != 1 or csv_members[0].filename != requested_member:
        raise DataContractError(
            "DEIS birth archive must contain one exact authoritative CSV member."
        )
    return matching[0]


def _record_batches(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    expected_columns: tuple[str, ...],
    block_size: int,
) -> Iterator[pa.RecordBatch]:
    try:
        with archive.open(info) as member:
            raw_header = member.readline()
            _validate_header(raw_header, expected_columns)
            reader = arrow_csv.open_csv(  # type: ignore[attr-defined]
                member,
                read_options=arrow_csv.ReadOptions(  # type: ignore[attr-defined]
                    block_size=block_size,
                    column_names=list(expected_columns),
                    encoding="utf8",
                ),
                parse_options=arrow_csv.ParseOptions(  # type: ignore[attr-defined]
                    delimiter=";"
                ),
                convert_options=arrow_csv.ConvertOptions(  # type: ignore[attr-defined]
                    column_types={column: pa.string() for column in expected_columns},
                    strings_can_be_null=False,
                ),
            )
            if tuple(reader.schema.names) != expected_columns:
                raise DataContractError("DEIS birth Arrow schema does not match the bound profile.")
            yield from reader
    except zipfile.BadZipFile as error:
        raise DataContractError(f"DEIS birth ZIP CRC validation failed: {error}") from error
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
        raise DataContractError(f"DEIS birth streaming schema/encoding failed: {error}") from error


def _validate_header(
    raw_header: bytes,
    expected_columns: tuple[str, ...],
) -> None:
    if not raw_header:
        raise DataContractError("DEIS birth CSV header is absent.")
    try:
        decoded = raw_header.decode("utf-8-sig").rstrip("\r\n")
        parsed = tuple(next(csv.reader([decoded], delimiter=";", strict=True)))
    except (UnicodeDecodeError, csv.Error) as error:
        raise DataContractError(f"DEIS birth CSV header encoding is invalid: {error}") from error
    if parsed != expected_columns:
        raise DataContractError("DEIS birth CSV header does not match the archive schema profile.")


def _scan_authoritative_source(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    release: DeisBirthRelease,
    block_size: int,
    observed_sha256: str,
    archive_size: int,
    member_size: int,
) -> DeisBirthIngestionAudit:
    rows = 0
    arrow_batches = 0
    years: Counter[int] = Counter()
    sexes: Counter[str] = Counter()
    lowercase_nationality_codes = 0
    blank_birth_months = 0
    unknown_maternal_age_groups = 0
    unknown_residence_regions = 0
    known_residence_regions: set[str] = set()
    for batch in _record_batches(
        archive,
        info,
        expected_columns=release.archive_id.expected_columns,
        block_size=block_size,
    ):
        arrow_batches += 1
        frame = _birth_identity_frame(batch)
        rows += len(frame)
        for source_year, count in (
            frame["ANO_NAC"]
            .value_counts(
                dropna=False,
                sort=False,
            )
            .items()
        ):
            year = _parse_birth_year(str(source_year), release)
            years[year] += int(count)
        for source_month, count in (
            frame["MES_NAC"]
            .value_counts(
                dropna=False,
                sort=False,
            )
            .items()
        ):
            month = _parse_birth_month(str(source_month))
            blank_birth_months += int(count) if month is None else 0
        for source_sex, count in (
            frame["SEXO"]
            .value_counts(
                dropna=False,
                sort=False,
            )
            .items()
        ):
            raw_sex = str(source_sex)
            _normalize_newborn_sex(raw_sex)
            sexes[raw_sex.strip()] += int(count)
        for source_age, count in (
            frame["GRUPO_ETARIO_MADRE"]
            .value_counts(
                dropna=False,
                sort=False,
            )
            .items()
        ):
            raw_age = str(source_age)
            _parse_maternal_age(raw_age)
            unknown_maternal_age_groups += int(count) * int(
                normalize_spanish_label(
                    raw_age,
                    field="DEIS maternal age group",
                )
                == "no especificado"
            )
        for source_nationality, count in (
            frame["NACIONALIDAD_MADRE"].value_counts(dropna=False, sort=False).items()
        ):
            raw_nationality = str(source_nationality)
            _normalize_nationality(raw_nationality)
            lowercase_nationality_codes += int(count) * int(raw_nationality.strip() == "e")
        region_counts = (
            frame.groupby(
                ["REGION_RESIDENCIA", "GLOSA_REGION_RESIDENCIA"],
                dropna=False,
                sort=False,
            )
            .size()
            .reset_index(name="_event_count")
        )
        region_records = cast(
            list[dict[str, object]],
            region_counts.to_dict(orient="records"),
        )
        for region_record in region_records:
            source_region = region_record["REGION_RESIDENCIA"]
            source_label = region_record["GLOSA_REGION_RESIDENCIA"]
            count = int(cast(int, region_record["_event_count"]))
            region = _normalize_birth_region(
                str(source_region),
                str(source_label),
            )
            unknown_residence_regions += int(count) * int(region == "unknown")
            if region != "unknown":
                known_residence_regions.add(region)
    if rows == 0:
        raise DataContractError("DEIS birth member contains no event rows.")
    return DeisBirthIngestionAudit(
        raw_rows=rows,
        arrow_batches=arrow_batches,
        counts_by_year=tuple(sorted(years.items())),
        counts_by_sex=tuple(sorted(sexes.items())),
        lowercase_nationality_codes=lowercase_nationality_codes,
        blank_birth_months=blank_birth_months,
        unknown_maternal_age_groups=unknown_maternal_age_groups,
        unknown_residence_regions=unknown_residence_regions,
        distinct_known_residence_regions=len(known_residence_regions),
        sha256=observed_sha256,
        archive_size=archive_size,
        member_size=member_size,
    )


def _validate_controls(
    controls: DeisBirthReleaseControls,
    audit: DeisBirthIngestionAudit,
) -> None:
    _check_scalar_control("raw_rows", controls.raw_rows, audit.raw_rows)
    _check_pair_control("counts_by_year", controls.counts_by_year, audit.counts_by_year)
    _check_pair_control("counts_by_sex", controls.counts_by_sex, audit.counts_by_sex)
    _check_scalar_control(
        "lowercase_nationality_codes",
        controls.lowercase_nationality_codes,
        audit.lowercase_nationality_codes,
    )
    _check_scalar_control(
        "blank_birth_months",
        controls.blank_birth_months,
        audit.blank_birth_months,
    )
    _check_scalar_control(
        "unknown_maternal_age_groups",
        controls.unknown_maternal_age_groups,
        audit.unknown_maternal_age_groups,
    )
    _check_scalar_control(
        "unknown_residence_regions",
        controls.unknown_residence_regions,
        audit.unknown_residence_regions,
    )
    _check_scalar_control(
        "distinct_known_residence_regions",
        controls.distinct_known_residence_regions,
        audit.distinct_known_residence_regions,
    )
    _check_scalar_control(
        "archive_size",
        controls.archive_size,
        audit.archive_size,
    )
    _check_scalar_control(
        "member_size",
        controls.member_size,
        audit.member_size,
    )


def _check_scalar_control(
    name: str,
    expected: int | None,
    actual: int,
) -> None:
    if expected is not None and expected != actual:
        raise DataContractError(
            f"DEIS birth release control {name} observed {actual}; expected {expected}."
        )


def _check_pair_control(
    name: str,
    expected: tuple[tuple[object, int], ...],
    actual: tuple[tuple[object, int], ...],
) -> None:
    actual_mapping = dict(actual)
    if expected and any(actual_mapping.get(key) != count for key, count in expected):
        raise DataContractError(
            f"DEIS birth release control {name} observed {actual!r}; expected {expected!r}."
        )


def _normalize_batch(
    batch: pa.RecordBatch,
    release: DeisBirthRelease,
) -> pd.DataFrame:
    raw_groups = (
        _birth_identity_frame(batch)
        .groupby(
            list(_BIRTH_IDENTITY_COLUMNS),
            dropna=False,
            sort=False,
        )
        .size()
        .reset_index(name="_event_count")
    )
    records: list[dict[str, object]] = []
    grouped_records = cast(
        list[dict[str, object]],
        raw_groups.to_dict(orient="records"),
    )
    for raw_group in grouped_records:
        event_count = int(cast(int, raw_group.pop("_event_count")))
        normalized = _normalize_birth_record(
            cast(dict[str, str], raw_group),
            release,
        )
        normalized["value"] = event_count
        records.append(normalized)
    frame = pd.DataFrame(records, columns=_BIRTH_PAYLOAD_COLUMNS)
    return _aggregate_normalized_frame(frame)


def _birth_identity_frame(batch: pa.RecordBatch) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        batch.select(_BIRTH_IDENTITY_COLUMNS).to_pandas(),
    )


def _aggregate_normalized_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != _BIRTH_PAYLOAD_COLUMNS:
        raise DataContractError(
            "DEIS birth payload columns must exactly match the demographic contract."
        )
    accumulated: dict[tuple[object, ...], dict[str, object]] = {}
    _accumulate_birth_counts(accumulated, frame)
    return _canonicalize_payload_nulls(
        pd.DataFrame.from_records(
            list(accumulated.values()),
            columns=_BIRTH_PAYLOAD_COLUMNS,
        )
    )


def _stream_partition_payload(
    frame: pd.DataFrame,
    *,
    first_row_number: int,
    last_row_number: int,
) -> pd.DataFrame:
    """Identify one bounded transport shard without claiming a global fact."""

    if first_row_number < 1 or last_row_number < first_row_number:
        raise DataContractError("DEIS birth stream partition row bounds are invalid.")
    partition = frame.copy(deep=True)
    partition.loc[:, "transformation_id"] = _STREAM_TRANSFORMATION_ID
    partition.loc[:, "transformation_version"] = _TRANSFORMATION_VERSION
    partition.loc[:, "aggregation_rule"] = (
        "partial DEIS birth stream shard over inclusive archive rows "
        f"{first_row_number}-{last_row_number}; requires global aggregation"
    )
    return partition


def _global_aggregate_payload(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop shard identities and assign release-wide aggregation semantics."""

    missing = [column for column in _BIRTH_PAYLOAD_COLUMNS if column not in frame]
    if missing:
        raise DataContractError(
            f"DEIS birth stream batch is missing demographic payload columns: {missing!r}."
        )
    payload = _canonicalize_payload_nulls(frame.loc[:, _BIRTH_PAYLOAD_COLUMNS])
    payload.loc[:, "transformation_id"] = _GLOBAL_TRANSFORMATION_ID
    payload.loc[:, "transformation_version"] = _TRANSFORMATION_VERSION
    payload.loc[:, "aggregation_rule"] = _GLOBAL_AGGREGATION_RULE
    return payload


def _canonicalize_payload_nulls(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.copy(deep=True)
    for column in _NULLABLE_PAYLOAD_COLUMNS:
        canonical[column] = pd.Series(
            (
                None
                if value is None
                or value is pd.NA
                or value is pd.NaT
                or (isinstance(value, float) and value != value)
                else value
                for value in canonical[column]
            ),
            index=canonical.index,
            dtype=object,
        )
    return canonical.loc[:, _BIRTH_PAYLOAD_COLUMNS]


def _normalize_birth_record(
    row: dict[str, str],
    release: DeisBirthRelease,
) -> dict[str, object]:
    year, month = _validate_occurrence(row, release)
    age_lower, age_upper, age_open = _parse_maternal_age(row["GRUPO_ETARIO_MADRE"])
    sex = _normalize_newborn_sex(row["SEXO"])
    region = _normalize_birth_region(
        row["REGION_RESIDENCIA"],
        row["GLOSA_REGION_RESIDENCIA"],
    )
    nationality = _normalize_nationality(row["NACIONALIDAD_MADRE"])
    if month is None:
        period_start = date(year, 1, 1)
        period_end = date(year + 1, 1, 1)
        date_basis = "birth_occurrence_year_when_month_missing"
    else:
        period_start = date(year, month, 1)
        period_end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        date_basis = "birth_occurrence_month"
    missingness = dict(_BASE_DIMENSION_MISSINGNESS)
    if month is None:
        missingness["birth_month"] = "missing_in_source_year_only"
    if age_lower is None:
        missingness["maternal_age"] = "source_group_unknown"
    if region == "unknown":
        missingness["maternal_residence"] = "source_code_99"
    if nationality is None:
        missingness["maternal_nationality"] = "missing_in_source"
    manifest = release.assets.manifest
    primary = manifest.primary_fact_asset
    selected_source = _selected_source(release)
    return {
        "domain": ObservationDomain.DEMOGRAPHIC.value,
        "variable": "births",
        "value": 1,
        "semantic_kind": "count",
        "unit": "event",
        "population_basis": "not_applicable",
        "observation_role": "observed_fact",
        "parity_scope": "unknown_order",
        "period_start": period_start,
        "period_end": period_end,
        "source_key": primary.source_key,
        "vintage": selected_source.vintage,
        "released_at": primary.released_at,
        "release_missingness_reason": (
            None
            if primary.release_missingness_reason is None
            else primary.release_missingness_reason.value
        ),
        "available_at": manifest.available_at,
        "provisional": selected_source.provisional,
        "transformation_id": _TRANSFORMATION_ID,
        "transformation_version": _TRANSFORMATION_VERSION,
        "aggregation_rule": "one released DEIS birth record represented as one event",
        "missingness_reason": json.dumps(
            missingness,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "source_status_code": None,
        "age_lower": age_lower,
        "age_upper": age_upper,
        "age_open": age_open,
        "sex": sex,
        "region": region,
        "cause": "not_applicable",
        "date_basis": date_basis,
        "age_role": "mother",
        "age_measure_unit": "grouped_completed_years",
        "age_quantity": None,
        "sex_role": "newborn",
        "commune": None,
        "geography_basis": "maternal_residence",
        "geography_vintage": release.geography_vintage,
        "cause_code_system": None,
        "cause_role": None,
        "external_cause": None,
        "registration_start": pd.NaT,
        "registration_end": pd.NaT,
        "birth_order": None,
        "nationality": nationality,
        "nationality_role": "mother",
        "country_of_birth": None,
        "education_level": None,
        "residence_five_years_ago": None,
        "migration_status": None,
    }


def _validate_occurrence(
    row: dict[str, str],
    release: DeisBirthRelease,
) -> tuple[int, int | None]:
    return (
        _parse_birth_year(row["ANO_NAC"], release),
        _parse_birth_month(row["MES_NAC"]),
    )


def _parse_birth_year(raw_value: str, release: DeisBirthRelease) -> int:
    raw_year = raw_value.strip()
    if not _INTEGER_TEXT.fullmatch(raw_year):
        raise DataContractError("DEIS ANO_NAC must be an integer year.")
    year = int(raw_year)
    if year < release.observation_start.year or year > release.observation_end.year:
        raise DataContractError("DEIS ANO_NAC falls outside release coverage.")
    return year


def _parse_birth_month(raw_value: str) -> int | None:
    raw_month = raw_value.strip()
    if raw_month == "":
        return None
    if not _INTEGER_TEXT.fullmatch(raw_month):
        raise DataContractError("DEIS MES_NAC must be blank or an integer month.")
    month = int(raw_month)
    if not 1 <= month <= 12:
        raise DataContractError("DEIS MES_NAC must lie from 1 through 12.")
    return month


def _parse_maternal_age(
    source_group: str,
) -> tuple[float | None, float | None, bool]:
    try:
        label = normalize_spanish_label(
            source_group,
            field="DEIS maternal age group",
        )
    except ValueError as error:
        raise DataContractError(str(error)) from error
    if label in {"menores 15 ano", "menores 15 anos"}:
        return 0.0, 15.0, False
    if label == "50 o mas anos":
        return 50.0, None, True
    if label == "no especificado":
        return None, None, False
    match = _CLOSED_MATERNAL_AGE.fullmatch(label)
    if match is None:
        raise DataContractError(f"Unknown DEIS maternal age group {source_group!r}.")
    lower = int(match.group("lower"))
    upper_inclusive = int(match.group("upper"))
    if lower not in range(15, 50, 5) or upper_inclusive != lower + 4:
        raise DataContractError(f"Invalid DEIS maternal age group {source_group!r}.")
    return float(lower), float(upper_inclusive + 1), False


def _normalize_newborn_sex(source_sex: str) -> str:
    canonical = _SEX_CODES.get(source_sex.strip())
    if canonical is None:
        raise DataContractError(f"Unknown DEIS newborn SEXO code {source_sex!r}.")
    return canonical


def _normalize_birth_region(
    source_code: str,
    source_label: str,
) -> str:
    code = source_code.strip()
    normalized_label = normalize_region_label(source_label)
    if code == "99":
        if normalized_label not in {"", "ignorada"}:
            raise DataContractError("DEIS maternal residence region code 99 conflicts with label.")
        return "unknown"
    if not _INTEGER_TEXT.fullmatch(code):
        raise DataContractError(f"Unknown DEIS maternal residence region code {source_code!r}.")
    numeric = int(code)
    if not 1 <= numeric <= 16:
        raise DataContractError(f"Unknown DEIS maternal residence region code {source_code!r}.")
    expected = f"{numeric:02d}"
    if region_code_for_label(source_label) != expected:
        raise DataContractError(
            f"DEIS maternal residence region label {source_label!r} conflicts "
            f"with code {source_code!r}."
        )
    return f"CL-{expected}"


def _normalize_nationality(source_code: str) -> str | None:
    code = source_code.strip()
    if code == "":
        return None
    canonical = "E" if code == "e" else code
    if canonical not in _NATIONALITY_CODES:
        raise DataContractError(f"Unknown DEIS maternal nationality code {source_code!r}.")
    return canonical


def _raw_feature_sidecar(
    batch: pa.RecordBatch,
    archive_id: DeisBirthArchiveId,
    *,
    first_row_number: int,
) -> pa.Table | None:
    if archive_id is not DeisBirthArchiveId.PROFILE_2001_2019:
        return None
    return pa.table(
        {
            "archive_id": pa.array(
                [archive_id.value] * batch.num_rows,
                type=pa.string(),
            ),
            "archive_row_number": pa.array(
                range(first_row_number, first_row_number + batch.num_rows),
                type=pa.int64(),
            ),
            **{
                column: batch.column(batch.schema.get_field_index(column))
                for column in _HIJ_COLUMNS
            },
        }
    )


def _selected_source(release: DeisBirthRelease) -> SourceRef:
    return next(source for source in release.sources if source.source_key == release.source_key)


def _birth_provenance(release: DeisBirthRelease) -> VariableProvenance:
    notes = [
        "Universe: released DEIS live-birth event records.",
        "Birth day, residence commune, event-level registration date, parity, "
        "and birth order are not published.",
        "Registration inclusion cutoff is release metadata, not a row timestamp: "
        f"{release.registration_inclusion_cutoff}.",
    ]
    if release.archive_id is DeisBirthArchiveId.PROFILE_2001_2019:
        notes.append(
            "HIJ_VIVOS, HIJ_FALL, HIJ_MORT, and HIJ_TOTAL are preserved "
            "losslessly and are not parity or birth order."
        )
    manifest = release.assets.manifest
    return VariableProvenance(
        sources=release.sources,
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=release.observation_start,
        observation_end=release.observation_end + timedelta(days=1),
        dimensions=(
            "birth_occurrence_month_or_unknown_within_year",
            "grouped_maternal_age",
            "newborn_sex",
            "maternal_residence_region",
            "maternal_nationality",
        ),
        aggregation_rules=(
            "one released DEIS birth record represented as one event",
            "bounded stream shards are partial transport materializations "
            "identified by inclusive archive row bounds",
        ),
        missingness_reason=(
            "birth day, residence commune, event-level registration date, "
            "parity, and birth order are not published"
        ),
        transformation_ids=(
            _TRANSFORMATION_ID,
            _STREAM_TRANSFORMATION_ID,
        ),
        notes=tuple(notes),
    )


def _finalize_birth_batch(
    payload: pd.DataFrame,
    *,
    release: DeisBirthRelease,
    provenance: VariableProvenance,
    identity_registry: IdentityRegistry,
) -> DemographicObservationBatch:
    if not isinstance(identity_registry, IdentityRegistry):
        raise DataContractError("DEIS birth normalization requires an IdentityRegistry.")
    assigned = assign_observation_ids(
        payload.copy(deep=True),
        domain=ObservationDomain.DEMOGRAPHIC,
        domain_key=DEMOGRAPHIC_OBSERVATION_KEY,
        source_identity={
            "publisher": "Departamento de Estadisticas e Informacion de Salud",
            "dataset": "birth_microdata",
        },
        manifest=release.assets.manifest,
        registry=identity_registry,
    )
    try:
        validated = validate_demographic_observations(assigned)
    except SchemaErrors as error:
        raise DataContractError(
            f"Normalized DEIS birth observations violate the demographic schema: {error}"
        ) from error
    return NormalizedDomainBatch.create(
        manifest=release.assets.manifest,
        observations=validated,
        provenance=provenance,
        identities=identity_registry.snapshot(),
    )
