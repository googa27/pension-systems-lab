"""Bounded-memory normalization of official DEIS death microdata."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final

import duckdb
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
from chile_demographic_pde.countries.chile.deis_births import (
    DEIS_BIRTH_COLUMNS_25 as DEIS_BIRTH_COLUMNS_25,
)
from chile_demographic_pde.countries.chile.deis_births import (
    DEIS_BIRTH_COLUMNS_29 as DEIS_BIRTH_COLUMNS_29,
)
from chile_demographic_pde.countries.chile.deis_births import (
    DeisBirthArchiveId as DeisBirthArchiveId,
)
from chile_demographic_pde.countries.chile.deis_births import (
    DeisBirthAssetContract as DeisBirthAssetContract,
)
from chile_demographic_pde.countries.chile.deis_births import (
    DeisBirthBatch as DeisBirthBatch,
)
from chile_demographic_pde.countries.chile.deis_births import (
    DeisBirthRelease as DeisBirthRelease,
)
from chile_demographic_pde.countries.chile.deis_births import (
    DeisBirthReleaseControls as DeisBirthReleaseControls,
)
from chile_demographic_pde.countries.chile.deis_births import (
    aggregate_deis_births as aggregate_deis_births,
)
from chile_demographic_pde.countries.chile.deis_births import (
    iter_normalize_deis_births as iter_normalize_deis_births,
)
from chile_demographic_pde.countries.chile.deis_births import (
    normalize_deis_births as normalize_deis_births,
)
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
    NormalizedDomainBatch,
    assign_observation_ids,
)
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import VerifiedReleaseAssets
from chile_demographic_pde.data.roles import ObservationDomain

DEIS_DEATH_COLUMNS: Final = (
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

_STAGED_COLUMNS: Final = (
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
    "CODIGO_GRUPO_DIAG1",
    "CODIGO_CATEGORIA_DIAG1",
    "CODIGO_SUBCATEGORIA_DIAG1",
    "DIAG2",
)
_ARROW_BLOCK_SIZE = 1 << 20
_TRANSFORMATION_ID: Final = "deis-death-events-to-counts-v1"
_REGISTRATION_LIMITATION: Final = (
    "no registration window is available in the current DEIS merged file"
)
_DIMENSION_MISSINGNESS_BASE: Final = {"registration_window": "not_available_in_current_deis_file"}
_SEX_MAP: Final = {
    "Hombre": "male",
    "Mujer": "female",
    "Indeterminado": "indeterminate",
}
_AGE_UNITS: Final = {
    "1": "completed_years",
    "2": "completed_months",
    "3": "completed_days",
    "4": "completed_hours",
    "0": "source_age_unknown",
}
_VALID_REGION_PREFIXES: Final = frozenset(f"{number:02d}" for number in range(1, 17))
_CAUSE_COLUMN: Final = {
    "underlying_code": "DIAG1",
    "chapter": "CAPITULO_DIAG1",
    "group": "CODIGO_GRUPO_DIAG1",
    "category": "CODIGO_CATEGORIA_DIAG1",
    "subcategory": "CODIGO_SUBCATEGORIA_DIAG1",
}
_SORT_COLUMNS: Final = (
    "period_start",
    "period_end",
    "date_basis",
    "age_lower",
    "age_upper",
    "age_measure_unit",
    "age_quantity",
    "sex",
    "region",
    "commune",
    "cause_code_system",
    "cause",
    "external_cause",
)
_INTEGER_TEXT = re.compile(r"^\d+$")


class DeisTemporalGrain(StrEnum):
    """Occurrence-time resolution requested from the event file."""

    DAY = "day"
    MONTH = "month"
    YEAR = "year"


class DeisAgeGrain(StrEnum):
    """Age resolution requested from the source-reported age fields."""

    SOURCE_REPORTED = "source_reported"
    COMPLETED_YEAR = "completed_year"
    ALL = "all"


class DeisGeographyGrain(StrEnum):
    """Geographic resolution of the normalized death count."""

    NATIONAL = "national"
    REGION = "region"
    COMMUNE = "commune"


class DeisCauseGrain(StrEnum):
    """Published underlying-cause hierarchy selected for one result."""

    ALL = "all"
    UNDERLYING_CODE = "underlying_code"
    CHAPTER = "chapter"
    GROUP = "group"
    CATEGORY = "category"
    SUBCATEGORY = "subcategory"


@dataclass(frozen=True, slots=True)
class DeisAggregation:
    """Immutable grouping request suitable for use as a cache key."""

    temporal_grain: DeisTemporalGrain = DeisTemporalGrain.YEAR
    age_grain: DeisAgeGrain = DeisAgeGrain.COMPLETED_YEAR
    geography_grain: DeisGeographyGrain = DeisGeographyGrain.REGION
    cause_grain: DeisCauseGrain = DeisCauseGrain.ALL
    by_sex: bool = True
    retain_external_cause: bool = False

    def __post_init__(self) -> None:
        enum_fields = (
            ("temporal_grain", self.temporal_grain, DeisTemporalGrain),
            ("age_grain", self.age_grain, DeisAgeGrain),
            ("geography_grain", self.geography_grain, DeisGeographyGrain),
            ("cause_grain", self.cause_grain, DeisCauseGrain),
        )
        for name, value, enum_type in enum_fields:
            if not isinstance(value, enum_type):
                raise TypeError(f"{name} must be a {enum_type.__name__}.")
        if not isinstance(self.by_sex, bool):
            raise TypeError("by_sex must be bool.")
        if not isinstance(self.retain_external_cause, bool):
            raise TypeError("retain_external_cause must be bool.")


_DEFAULT_AGGREGATION: Final = DeisAggregation()


@dataclass(frozen=True, slots=True)
class DeisReleaseControls:
    """Optional release-internal totals used to detect upstream changes."""

    raw_rows: int | None = None
    counts_by_year: tuple[tuple[int, int], ...] = ()
    counts_by_sex: tuple[tuple[str, int], ...] = ()
    counts_by_age_type: tuple[tuple[str, int], ...] = ()
    age_99_year_deaths: int | None = None
    nonblank_external_causes: int | None = None
    short_numeric_communes: int | None = None
    archive_size: int | None = None
    member_size: int | None = None

    def __post_init__(self) -> None:
        scalar_controls = (
            self.raw_rows,
            self.age_99_year_deaths,
            self.nonblank_external_causes,
            self.short_numeric_communes,
            self.archive_size,
            self.member_size,
        )
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
            for value in scalar_controls
        ):
            raise ValueError("DEIS scalar release controls must be nonnegative integers.")
        for name, pairs in (
            ("counts_by_year", self.counts_by_year),
            ("counts_by_sex", self.counts_by_sex),
            ("counts_by_age_type", self.counts_by_age_type),
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
class DeisDeathRelease:
    """Exact official archive identity, coverage, and validation controls."""

    assets: VerifiedReleaseAssets
    source_key: str
    sources: tuple[SourceRef, ...]
    released_at: date
    observation_start: date
    observation_cutoff: date
    archive_member: str
    controls: DeisReleaseControls

    def __post_init__(self) -> None:
        if not isinstance(self.assets, VerifiedReleaseAssets):
            raise DataContractError("DEIS release assets must be a VerifiedReleaseAssets boundary.")
        manifest = self.assets.manifest
        if (
            not isinstance(self.source_key, str)
            or self.source_key != manifest.primary_fact_source_key
        ):
            raise DataContractError("DEIS source key must equal the manifest primary fact source.")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise DataContractError("DEIS sources must be a nonempty immutable tuple.")
        validated_sources: list[SourceRef] = []
        for source in self.sources:
            try:
                validated_sources.append(
                    SourceRef.model_validate(source.model_dump(mode="python", warnings="none"))
                )
            except (AttributeError, TypeError, ValueError, ValidationError):
                raise DataContractError(
                    "DEIS source provenance violates the typed source contract."
                ) from None
        sources_by_key = {source.source_key: source for source in validated_sources}
        if len(sources_by_key) != len(validated_sources):
            raise DataContractError("DEIS source provenance keys must be unique.")
        manifest_keys = {asset.source_key for asset in manifest.assets}
        if set(sources_by_key) != manifest_keys:
            raise DataContractError(
                "DEIS source provenance must cover every manifest asset exactly."
            )
        for asset in manifest.assets:
            source = sources_by_key[asset.source_key]
            if source.sha256 != asset.sha256:
                raise DataContractError("DEIS source SHA-256 must match its manifest asset.")
            if source.retrieved_at != asset.available_at:
                raise DataContractError("DEIS source retrieval time must match its manifest asset.")
            if (
                source.release_date != asset.released_at
                or source.release_missingness_reason != asset.release_missingness_reason
            ):
                raise DataContractError(
                    "DEIS source publisher release metadata must match its manifest asset."
                )
        ordered_sources = tuple(sources_by_key[asset.source_key] for asset in manifest.assets)
        object.__setattr__(self, "sources", ordered_sources)

        date_fields = (
            ("released_at", self.released_at),
            ("observation_start", self.observation_start),
            ("observation_cutoff", self.observation_cutoff),
        )
        if any(type(value) is not date for _, value in date_fields):
            raise DataContractError("DEIS release and observation bounds must be built-in dates.")
        if sources_by_key[self.source_key].release_date != self.released_at:
            raise DataContractError("DEIS released_at must equal the selected source release date.")
        if not isinstance(self.archive_member, str):
            raise ValueError("archive_member must be a string.")
        member = PurePosixPath(self.archive_member)
        if not self.archive_member.strip() or member.is_absolute() or ".." in member.parts:
            raise ValueError("archive_member must be a safe nonblank ZIP member path.")
        if self.observation_start > self.observation_cutoff:
            raise ValueError("observation_start must not follow observation_cutoff.")
        if self.released_at < self.observation_cutoff:
            raise ValueError("released_at must not precede the observation cutoff.")
        selected_source = sources_by_key[self.source_key]
        if selected_source.provisional and not selected_source.vintage.strip():
            raise ValueError("A preliminary DEIS release requires a nonblank vintage.")
        if not isinstance(self.controls, DeisReleaseControls):
            raise DataContractError("DEIS release controls must be DeisReleaseControls.")


@dataclass(frozen=True, slots=True)
class DeisIngestionAudit:
    """Small, immutable reconciliation record for one streamed ingestion."""

    sha256: str
    archive_size: int
    member_size: int
    raw_rows: int
    arrow_batches: int
    normalized_rows: int
    normalized_total: int
    counts_by_year: tuple[tuple[int, int], ...]
    counts_by_sex: tuple[tuple[str, int], ...]
    counts_by_age_type: tuple[tuple[str, int], ...]
    age_99_year_deaths: int
    raw_nonempty_external_causes: int
    meaningful_external_causes: int
    short_numeric_communes: int
    blank_communes: int
    missing_code_communes: int
    missing_occurrence_dates: int
    missing_age_types: int
    anomalous_underlying_codes: int
    anomalous_external_codes: int
    actual_occurrence_start: date
    actual_occurrence_end: date
    raw_region_labels: tuple[str, ...]
    database_bytes: int


@dataclass(slots=True, eq=False)
class DeisNormalizationResult:
    """Normalized aggregate together with authoritative provenance and audit."""

    batch: DemographicObservationBatch
    audit: DeisIngestionAudit

    @property
    def observations(self) -> pd.DataFrame:
        """Return an owned view of the immutable normalized batch."""

        return self.batch.observations

    @property
    def provenance(self) -> VariableProvenance:
        """Expose the exact provenance stored by the normalized batch."""

        return self.batch.provenance

    def __repr__(self) -> str:
        primary = self.batch.manifest.primary_fact_source_key
        source = next(source for source in self.provenance.sources if source.source_key == primary)
        return (
            f"{type(self).__name__}(rows={self.audit.normalized_rows}, "
            f"total={self.audit.normalized_total}, "
            f"vintage={source.vintage!r})"
        )


def normalize_deis_deaths(
    release: DeisDeathRelease,
    *,
    aggregation: DeisAggregation = _DEFAULT_AGGREGATION,
    scratch_directory: Path | None = None,
    memory_limit_bytes: int = 512 << 20,
    identity_registry: IdentityRegistry,
) -> DeisNormalizationResult:
    """Stream, validate, and aggregate one exact DEIS death archive."""

    if not isinstance(release, DeisDeathRelease):
        raise TypeError("release must be a DeisDeathRelease.")
    if not isinstance(identity_registry, IdentityRegistry):
        raise DataContractError("DEIS normalization requires an IdentityRegistry.")
    if (
        isinstance(memory_limit_bytes, bool)
        or not isinstance(memory_limit_bytes, int)
        or memory_limit_bytes <= 0
    ):
        raise ValueError("memory_limit_bytes must be a positive integer.")
    if not isinstance(aggregation, DeisAggregation):
        raise TypeError("aggregation must be a DeisAggregation.")
    scratch_root = None if scratch_directory is None else Path(scratch_directory)
    if scratch_root is not None and not scratch_root.is_dir():
        raise ValueError("scratch_directory must be an existing directory.")

    selected_asset = release.assets.manifest.asset(release.source_key)
    try:
        with release.assets.open(release.source_key) as stream:
            archive_size = os.fstat(stream.fileno()).st_size
            _check_scalar_control(
                "archive_size",
                release.controls.archive_size,
                archive_size,
            )
            with zipfile.ZipFile(stream) as archive:
                matching = [
                    info for info in archive.infolist() if info.filename == release.archive_member
                ]
                if not matching:
                    raise MissingOfficialDataError(
                        f"Exact DEIS archive member "
                        f"{release.archive_member!r} is absent; sibling or "
                        "older releases are not substitutes."
                    )
                if len(matching) != 1:
                    raise DataContractError(
                        f"DEIS archive contains duplicate exact member {release.archive_member!r}."
                    )
                info = matching[0]
                _check_scalar_control(
                    "member_size",
                    release.controls.member_size,
                    info.file_size,
                )
                _validate_header(archive, info)
                with tempfile.TemporaryDirectory(
                    prefix="deis-deaths-",
                    dir=scratch_root,
                ) as temporary:
                    return _stream_and_normalize(
                        archive=archive,
                        info=info,
                        release=release,
                        aggregation=aggregation,
                        temporary=Path(temporary),
                        memory_limit_bytes=memory_limit_bytes,
                        identity_registry=identity_registry,
                        observed_sha256=selected_asset.sha256,
                        archive_size=archive_size,
                    )
    except MissingOfficialDataError:
        raise
    except zipfile.BadZipFile as error:
        detail = "CRC" if "CRC" in str(error).upper() else "ZIP"
        raise DataContractError(
            f"DEIS {detail} validation failed for verified source {release.source_key!r}: {error}"
        ) from error


def _validate_header(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
    try:
        with archive.open(info) as member:
            raw_header = member.readline()
    except zipfile.BadZipFile as error:
        raise DataContractError(f"DEIS ZIP header/CRC validation failed: {error}") from error
    if not raw_header:
        raise DataContractError("DEIS death CSV header is absent.")
    try:
        decoded = raw_header.decode("windows-1252").rstrip("\r\n")
        parsed = next(csv.reader([decoded], delimiter=";", strict=True))
    except (UnicodeDecodeError, csv.Error) as error:
        raise DataContractError(f"DEIS death CSV header is malformed: {error}") from error
    if tuple(parsed) != DEIS_DEATH_COLUMNS:
        raise DataContractError(
            "DEIS death CSV header must match the exact ordered 27-column contract."
        )


def _stream_and_normalize(
    *,
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    release: DeisDeathRelease,
    aggregation: DeisAggregation,
    temporary: Path,
    memory_limit_bytes: int,
    identity_registry: IdentityRegistry,
    observed_sha256: str,
    archive_size: int,
) -> DeisNormalizationResult:
    database_path = temporary / "deis.duckdb"
    spill_path = temporary / "spill"
    spill_path.mkdir()
    connection = duckdb.connect(
        str(database_path),
        config={
            "memory_limit": f"{memory_limit_bytes}B",
            "temp_directory": str(spill_path),
        },
    )
    arrow_batches = 0
    try:
        _create_raw_table(connection)
        try:
            with archive.open(info) as member:
                reader = arrow_csv.open_csv(  # type: ignore[attr-defined]
                    member,
                    read_options=arrow_csv.ReadOptions(  # type: ignore[attr-defined]
                        block_size=_ARROW_BLOCK_SIZE,
                        encoding="windows-1252",
                    ),
                    parse_options=arrow_csv.ParseOptions(  # type: ignore[attr-defined]
                        delimiter=";"
                    ),
                    convert_options=arrow_csv.ConvertOptions(  # type: ignore[attr-defined]
                        column_types={column: pa.string() for column in DEIS_DEATH_COLUMNS},
                        strings_can_be_null=False,
                    ),
                )
                if tuple(reader.schema.names) != DEIS_DEATH_COLUMNS:
                    raise DataContractError(
                        "Arrow reader schema differs from the exact DEIS header."
                    )
                for batch in reader:
                    arrow_batches += 1
                    selected = batch.select(_STAGED_COLUMNS)
                    connection.register("_deis_batch", selected)
                    try:
                        connection.execute(
                            f"INSERT INTO raw_deaths SELECT "
                            f"{_quoted_columns(_STAGED_COLUMNS)} FROM _deis_batch"
                        )
                    finally:
                        connection.unregister("_deis_batch")
        except zipfile.BadZipFile as error:
            raise DataContractError(
                f"DEIS ZIP CRC validation failed while consuming the member: {error}"
            ) from error
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
            raise DataContractError(f"DEIS streaming CSV contract failed: {error}") from error

        actual = _validate_and_audit_raw(connection, release)
        _validate_release_controls(release.controls, actual)
        region_rows = _install_region_lookup(connection)
        aggregate = connection.execute(_aggregation_sql(aggregation, release)).fetchdf()
        if aggregate.empty:
            raise DataContractError("DEIS archive contains no normalizable death events.")
        normalized_payload = _complete_normalized_frame(
            aggregate,
            release=release,
            aggregation=aggregation,
        )
        normalized_total = int(normalized_payload["value"].sum())
        if normalized_total != actual.raw_rows:
            raise DataContractError(
                "DEIS normalized aggregate does not reconcile to raw event rows."
            )
        connection.execute("CHECKPOINT")
        database_bytes = sum(file.stat().st_size for file in temporary.rglob("*") if file.is_file())
        audit = DeisIngestionAudit(
            sha256=observed_sha256,
            archive_size=archive_size,
            member_size=info.file_size,
            raw_rows=actual.raw_rows,
            arrow_batches=arrow_batches,
            normalized_rows=len(normalized_payload),
            normalized_total=normalized_total,
            counts_by_year=actual.counts_by_year,
            counts_by_sex=actual.counts_by_sex,
            counts_by_age_type=actual.counts_by_age_type,
            age_99_year_deaths=actual.age_99_year_deaths,
            raw_nonempty_external_causes=actual.raw_nonempty_external_causes,
            meaningful_external_causes=actual.meaningful_external_causes,
            short_numeric_communes=actual.short_numeric_communes,
            blank_communes=actual.blank_communes,
            missing_code_communes=actual.missing_code_communes,
            missing_occurrence_dates=actual.missing_occurrence_dates,
            missing_age_types=actual.missing_age_types,
            anomalous_underlying_codes=actual.anomalous_underlying_codes,
            anomalous_external_codes=actual.anomalous_external_codes,
            actual_occurrence_start=actual.actual_occurrence_start,
            actual_occurrence_end=actual.actual_occurrence_end,
            raw_region_labels=tuple(sorted({label for _, label, _ in region_rows if label})),
            database_bytes=database_bytes,
        )
        provenance = _provenance(
            release=release,
            aggregation=aggregation,
            actual=actual,
        )
        assigned = assign_observation_ids(
            normalized_payload,
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=DEMOGRAPHIC_OBSERVATION_KEY,
            source_identity={
                "publisher": "Departamento de Estadisticas e Informacion de Salud",
                "dataset": "death_microdata",
                "source_contract": release.source_key,
            },
            manifest=release.assets.manifest,
            registry=identity_registry,
        )
        try:
            validated = validate_demographic_observations(assigned)
        except SchemaErrors as error:
            raise DataContractError(
                f"Normalized DEIS observations violate the demographic schema: {error}"
            ) from error
        batch = NormalizedDomainBatch.create(
            manifest=release.assets.manifest,
            observations=validated,
            provenance=provenance,
            identities=identity_registry.snapshot(),
        )
        return DeisNormalizationResult(
            batch=batch,
            audit=audit,
        )
    finally:
        connection.close()


def _create_raw_table(connection: duckdb.DuckDBPyConnection) -> None:
    declarations = ", ".join(f'"{column}" VARCHAR' for column in _STAGED_COLUMNS)
    connection.execute(f"CREATE TABLE raw_deaths ({declarations})")


@dataclass(frozen=True, slots=True)
class _RawAudit:
    raw_rows: int
    counts_by_year: tuple[tuple[int, int], ...]
    counts_by_sex: tuple[tuple[str, int], ...]
    counts_by_age_type: tuple[tuple[str, int], ...]
    age_99_year_deaths: int
    raw_nonempty_external_causes: int
    meaningful_external_causes: int
    short_numeric_communes: int
    blank_communes: int
    missing_code_communes: int
    missing_occurrence_dates: int
    missing_age_types: int
    anomalous_underlying_codes: int
    anomalous_external_codes: int
    actual_occurrence_start: date
    actual_occurrence_end: date


def _validate_and_audit_raw(
    connection: duckdb.DuckDBPyConnection,
    release: DeisDeathRelease,
) -> _RawAudit:
    raw_rows = _scalar(connection, "SELECT count(*) FROM raw_deaths")
    if raw_rows == 0:
        raise DataContractError("DEIS death member contains a header but no events.")
    validations = (
        (
            "AÑO",
            """SELECT count(*) FROM raw_deaths
               WHERE NOT regexp_full_match(trim("AÑO"), '[0-9]{4}')""",
        ),
        (
            "FECHA_DEF",
            """SELECT count(*) FROM raw_deaths
               WHERE trim("FECHA_DEF") <> ''
                 AND try_strptime(trim("FECHA_DEF"), '%Y-%m-%d') IS NULL""",
        ),
        (
            "AÑO/FECHA_DEF",
            """SELECT count(*) FROM raw_deaths
               WHERE trim("FECHA_DEF") <> ''
                 AND CAST("AÑO" AS INTEGER)
                     <> year(CAST(trim("FECHA_DEF") AS DATE))""",
        ),
        (
            "SEXO_NOMBRE",
            """SELECT count(*) FROM raw_deaths
               WHERE "SEXO_NOMBRE" NOT IN ('Hombre', 'Mujer', 'Indeterminado')""",
        ),
        (
            "EDAD_TIPO",
            """SELECT count(*) FROM raw_deaths
               WHERE trim("EDAD_TIPO") NOT IN ('', '0', '1', '2', '3', '4')""",
        ),
        (
            "EDAD_CANT",
            """SELECT count(*) FROM raw_deaths
               WHERE trim("EDAD_CANT") <> ''
                 AND (
                   NOT regexp_full_match(trim("EDAD_CANT"), '[0-9]+')
                   OR (trim("EDAD_TIPO") = '2'
                       AND try_cast(trim("EDAD_CANT") AS INTEGER) > 11)
                   OR (trim("EDAD_TIPO") = '3'
                       AND try_cast(trim("EDAD_CANT") AS INTEGER) > 31)
                   OR (trim("EDAD_TIPO") = '4'
                       AND try_cast(trim("EDAD_CANT") AS INTEGER) > 24)
                 )""",
        ),
        (
            "COD_COMUNA",
            """SELECT count(*) FROM raw_deaths
               WHERE trim("COD_COMUNA") NOT IN ('', '99999')
                 AND NOT regexp_full_match(trim("COD_COMUNA"), '[0-9]{1,5}')""",
        ),
        (
            "ICD classifier",
            """SELECT count(*) FROM raw_deaths
               WHERE trim("DIAG1") <> ''
                 AND (
                   (CAST("AÑO" AS INTEGER) <= 1996
                    AND regexp_matches(upper(trim("DIAG1")), '^[A-DF-UW-Z]'))
                   OR
                   (CAST("AÑO" AS INTEGER) >= 1997
                    AND regexp_full_match(trim("DIAG1"), '[0-9]{3,5}'))
                 )""",
        ),
    )
    for label, query in validations:
        failures = _scalar(connection, query)
        if failures:
            raise DataContractError(f"DEIS {label} contract failed for {failures} event row(s).")

    start_literal = release.observation_start.isoformat()
    cutoff_literal = release.observation_cutoff.isoformat()
    outside = _scalar(
        connection,
        f"""SELECT count(*) FROM raw_deaths
            WHERE (
              trim("FECHA_DEF") <> ''
              AND CAST(trim("FECHA_DEF") AS DATE)
                  NOT BETWEEN DATE '{start_literal}' AND DATE '{cutoff_literal}'
            ) OR (
              trim("FECHA_DEF") = ''
              AND CAST("AÑO" AS INTEGER)
                  NOT BETWEEN {release.observation_start.year}
                          AND {release.observation_cutoff.year}
            )""",
    )
    if outside:
        raise DataContractError(f"DEIS observation coverage excludes {outside} event row(s).")

    counts_by_year = tuple(
        (int(year), int(count))
        for year, count in connection.execute(
            'SELECT CAST("AÑO" AS INTEGER), count(*) FROM raw_deaths GROUP BY 1 ORDER BY 1'
        ).fetchall()
    )
    counts_by_sex = tuple(
        (str(sex), int(count))
        for sex, count in connection.execute(
            'SELECT "SEXO_NOMBRE", count(*) FROM raw_deaths GROUP BY 1 ORDER BY 1'
        ).fetchall()
    )
    counts_by_age_type = tuple(
        (str(age_type), int(count))
        for age_type, count in connection.execute(
            'SELECT trim("EDAD_TIPO"), count(*) FROM raw_deaths GROUP BY 1 ORDER BY 1'
        ).fetchall()
    )
    missing_dates = _scalar(
        connection,
        """SELECT count(*) FROM raw_deaths WHERE trim("FECHA_DEF") = ''""",
    )
    occurrence_bounds = connection.execute(
        """SELECT
             min(CASE WHEN trim("FECHA_DEF") <> ''
                      THEN CAST(trim("FECHA_DEF") AS DATE) END),
             max(CASE WHEN trim("FECHA_DEF") <> ''
                      THEN CAST(trim("FECHA_DEF") AS DATE) END),
             min(CAST("AÑO" AS INTEGER)),
             max(CAST("AÑO" AS INTEGER)),
             min(CASE WHEN trim("FECHA_DEF") = ''
                      THEN CAST("AÑO" AS INTEGER) END),
             max(CASE WHEN trim("FECHA_DEF") = ''
                      THEN CAST("AÑO" AS INTEGER) END)
           FROM raw_deaths"""
    ).fetchone()
    if occurrence_bounds is None:
        raise DataContractError("DEIS occurrence-range audit returned no result.")
    (
        known_start,
        known_end,
        min_year,
        max_year,
        blank_min_year,
        blank_max_year,
    ) = occurrence_bounds
    start_candidates = [
        date.fromisoformat(str(known_start))
        if known_start is not None
        else date(int(min_year), 1, 1),
    ]
    if blank_min_year is not None:
        start_candidates.append(date(int(blank_min_year), 1, 1))
    end_candidates = [
        date.fromisoformat(str(known_end))
        if known_end is not None
        else min(date(int(max_year), 12, 31), release.observation_cutoff),
    ]
    if blank_max_year is not None:
        end_candidates.append(
            min(
                date(int(blank_max_year), 12, 31),
                release.observation_cutoff,
            )
        )
    actual_start = min(start_candidates)
    actual_end = max(end_candidates)
    return _RawAudit(
        raw_rows=raw_rows,
        counts_by_year=counts_by_year,
        counts_by_sex=counts_by_sex,
        counts_by_age_type=counts_by_age_type,
        age_99_year_deaths=_scalar(
            connection,
            """SELECT count(*) FROM raw_deaths
               WHERE trim("EDAD_TIPO") = '1' AND trim("EDAD_CANT") = '99'""",
        ),
        raw_nonempty_external_causes=_scalar(
            connection,
            """SELECT count(*) FROM raw_deaths WHERE "DIAG2" <> ''""",
        ),
        meaningful_external_causes=_scalar(
            connection,
            """SELECT count(*) FROM raw_deaths WHERE trim("DIAG2") <> ''""",
        ),
        short_numeric_communes=_scalar(
            connection,
            """SELECT count(*) FROM raw_deaths
               WHERE regexp_full_match(trim("COD_COMUNA"), '[0-9]+')
                 AND length(trim("COD_COMUNA")) < 5""",
        ),
        blank_communes=_scalar(
            connection,
            """SELECT count(*) FROM raw_deaths WHERE trim("COD_COMUNA") = ''""",
        ),
        missing_code_communes=_scalar(
            connection,
            """SELECT count(*) FROM raw_deaths WHERE trim("COD_COMUNA") = '99999'""",
        ),
        missing_occurrence_dates=missing_dates,
        missing_age_types=_scalar(
            connection,
            """SELECT count(*) FROM raw_deaths WHERE trim("EDAD_TIPO") = ''""",
        ),
        anomalous_underlying_codes=_scalar(
            connection,
            """SELECT count(*) FROM raw_deaths
               WHERE trim("DIAG1") <> ''
                 AND (
                   (CAST("AÑO" AS INTEGER) <= 1996
                    AND NOT regexp_full_match(
                        upper(trim("DIAG1")), '(?:[0-9]{3,5}|[EV][0-9]{2,4})'
                    ))
                   OR
                   (CAST("AÑO" AS INTEGER) >= 1997
                    AND NOT regexp_full_match(
                        upper(trim("DIAG1")), '[A-Z][0-9]{2}[0-9X]'
                    ))
                 )""",
        ),
        anomalous_external_codes=_scalar(
            connection,
            """SELECT count(*) FROM raw_deaths
               WHERE trim("DIAG2") <> ''
                 AND upper(trim("DIAG2")) NOT IN ('XXX', 'XXX1', 'XXX2', 'X')
                 AND (
                   (CAST("AÑO" AS INTEGER) <= 1996
                    AND NOT regexp_full_match(
                        upper(trim("DIAG2")), '(?:[0-9]{3,5}|[EV][0-9]{2,4})'
                    ))
                   OR
                   (CAST("AÑO" AS INTEGER) >= 1997
                    AND NOT regexp_full_match(
                        upper(trim("DIAG2")), '[A-Z][0-9]{2}[0-9X]'
                    ))
                 )""",
        ),
        actual_occurrence_start=actual_start,
        actual_occurrence_end=actual_end,
    )


def _validate_release_controls(
    controls: DeisReleaseControls,
    actual: _RawAudit,
) -> None:
    _check_scalar_control("raw_rows", controls.raw_rows, actual.raw_rows)
    _check_pair_control("counts_by_year", controls.counts_by_year, actual.counts_by_year)
    _check_pair_control("counts_by_sex", controls.counts_by_sex, actual.counts_by_sex)
    _check_pair_control(
        "counts_by_age_type",
        controls.counts_by_age_type,
        actual.counts_by_age_type,
    )
    _check_scalar_control(
        "age_99_year_deaths",
        controls.age_99_year_deaths,
        actual.age_99_year_deaths,
    )
    _check_scalar_control(
        "nonblank_external_causes",
        controls.nonblank_external_causes,
        actual.raw_nonempty_external_causes,
    )
    _check_scalar_control(
        "short_numeric_communes",
        controls.short_numeric_communes,
        actual.short_numeric_communes,
    )


def _check_scalar_control(
    name: str,
    expected: int | None,
    actual: int,
) -> None:
    if expected is not None and actual != expected:
        raise DataContractError(
            f"DEIS release control {name} observed {actual}; expected {expected}."
        )


def _check_pair_control(
    name: str,
    expected: tuple[tuple[object, int], ...],
    actual: tuple[tuple[object, int], ...],
) -> None:
    actual_mapping = dict(actual)
    if expected and any(actual_mapping.get(key) != value for key, value in expected):
        raise DataContractError(
            f"DEIS release control {name} observed {actual!r}; expected {expected!r}."
        )


def _install_region_lookup(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[tuple[str, str, str], ...]:
    raw_pairs = connection.execute(
        """SELECT DISTINCT "COD_COMUNA", "NOMBRE_REGION"
           FROM raw_deaths ORDER BY 1, 2"""
    ).fetchall()
    rows = tuple(
        (str(code), str(label), _canonical_region(str(code), str(label)))
        for code, label in raw_pairs
    )
    connection.execute(
        """CREATE TABLE region_lookup (
             raw_commune VARCHAR,
             raw_region VARCHAR,
             canonical_region VARCHAR
           )"""
    )
    connection.executemany(
        "INSERT INTO region_lookup VALUES (?, ?, ?)",
        rows,
    )
    return rows


def _canonical_region(raw_commune: str, raw_region: str) -> str:
    code = raw_commune.strip()
    region_label = normalize_region_label(raw_region)
    label_code = region_code_for_label(raw_region)
    if code in {"", "99999"}:
        if region_label in {"", "ignorada"}:
            return "unknown"
        if label_code is None:
            raise DataContractError(f"Unknown DEIS residence region label {raw_region!r}.")
        return f"CL-{label_code}"
    if not _INTEGER_TEXT.fullmatch(code) or len(code) > 5:
        raise DataContractError(f"Invalid DEIS COD_COMUNA {raw_commune!r}.")
    padded = code.zfill(5)
    prefix = padded[:2]
    if prefix not in _VALID_REGION_PREFIXES or padded == "00000":
        raise DataContractError(f"DEIS COD_COMUNA {raw_commune!r} has no valid DPA-2019 region.")
    if label_code != prefix:
        raise DataContractError(
            f"DEIS residence region {raw_region!r} conflicts with commune {raw_commune!r}."
        )
    return f"CL-{prefix}"


def _selected_source(release: DeisDeathRelease) -> SourceRef:
    for source in release.sources:
        if source.source_key == release.source_key:
            return source
    raise DataContractError("DEIS selected source is absent from release provenance.")


def _aggregation_sql(
    aggregation: DeisAggregation,
    release: DeisDeathRelease,
) -> str:
    period_start, nominal_end = _period_expressions(aggregation.temporal_grain)
    cutoff_plus_one = (release.observation_cutoff + timedelta(days=1)).isoformat()
    period_end = (
        f"least({nominal_end}, DATE '{cutoff_plus_one}')"
        if _selected_source(release).provisional
        else nominal_end
    )
    age_lower, age_upper, age_unit, age_quantity = _age_expressions(aggregation.age_grain)
    sex = (
        """CASE "SEXO_NOMBRE"
             WHEN 'Hombre' THEN 'male'
             WHEN 'Mujer' THEN 'female'
             ELSE 'indeterminate'
           END"""
        if aggregation.by_sex
        else "'all'"
    )
    if aggregation.geography_grain is DeisGeographyGrain.NATIONAL:
        region = "'CL'"
        commune = "NULL::VARCHAR"
    else:
        region = "canonical_region"
        commune = (
            """CASE
                 WHEN trim("COD_COMUNA") IN ('', '99999') THEN NULL
                 ELSE lpad(trim("COD_COMUNA"), 5, '0')
               END"""
            if aggregation.geography_grain is DeisGeographyGrain.COMMUNE
            else "NULL::VARCHAR"
        )
    if aggregation.cause_grain is DeisCauseGrain.ALL:
        cause = "'all_causes'"
    else:
        cause_column = _CAUSE_COLUMN[aggregation.cause_grain.value]
        cause = f"""CASE WHEN trim("{cause_column}") = '' THEN 'unknown'
                     ELSE trim("{cause_column}") END"""
    external = (
        """CASE
             WHEN trim("DIAG2") = '' THEN 'not_applicable'
             WHEN upper(trim("DIAG2")) IN ('XXX', 'XXX1', 'XXX2', 'X')
               THEN 'unknown_not_coded'
             ELSE trim("DIAG2")
           END"""
        if aggregation.retain_external_cause
        else "NULL::VARCHAR"
    )
    return f"""
        WITH typed AS (
          SELECT raw_deaths.*, canonical_region,
                 CAST("AÑO" AS INTEGER) AS event_year,
                 CASE WHEN trim("FECHA_DEF") = '' THEN NULL
                      ELSE CAST(trim("FECHA_DEF") AS DATE) END AS event_date
          FROM raw_deaths
          JOIN region_lookup
            ON raw_deaths."COD_COMUNA" = region_lookup.raw_commune
           AND raw_deaths."NOMBRE_REGION" = region_lookup.raw_region
        ),
        canonical AS (
          SELECT
            CASE WHEN event_date IS NULL
                 THEN 'death_occurrence_year_when_date_missing'
                 ELSE 'death_occurrence' END AS date_basis,
            {period_start} AS period_start,
            {period_end} AS period_end,
            {age_lower} AS age_lower,
            {age_upper} AS age_upper,
            FALSE AS age_open,
            {age_unit} AS age_measure_unit,
            {age_quantity} AS age_quantity,
            {sex} AS sex,
            {region} AS region,
            {commune} AS commune,
            CASE WHEN event_year <= 1996 THEN 'ICD-9' ELSE 'ICD-10' END
              AS cause_code_system,
            {cause} AS cause,
            {external} AS external_cause
          FROM typed
        )
        SELECT *, count(*)::BIGINT AS value
        FROM canonical
        GROUP BY ALL
    """


def _period_expressions(
    grain: DeisTemporalGrain,
) -> tuple[str, str]:
    if grain is DeisTemporalGrain.DAY:
        known_start = "event_date"
        known_end = "event_date + INTERVAL 1 DAY"
    elif grain is DeisTemporalGrain.MONTH:
        known_start = "date_trunc('month', event_date)::DATE"
        known_end = "date_trunc('month', event_date)::DATE + INTERVAL 1 MONTH"
    else:
        known_start = "make_date(event_year, 1, 1)"
        known_end = "make_date(event_year + 1, 1, 1)"
    start = f"CASE WHEN event_date IS NULL THEN make_date(event_year, 1, 1) ELSE {known_start} END"
    end = f"CASE WHEN event_date IS NULL THEN make_date(event_year + 1, 1, 1) ELSE {known_end} END"
    return start, end


def _age_expressions(
    grain: DeisAgeGrain,
) -> tuple[str, str, str, str]:
    quantity = 'try_cast(trim("EDAD_CANT") AS INTEGER)'
    if grain is DeisAgeGrain.ALL:
        return (
            "NULL::DOUBLE",
            "NULL::DOUBLE",
            "NULL::VARCHAR",
            "NULL::DOUBLE",
        )
    if grain is DeisAgeGrain.SOURCE_REPORTED:
        lower = f"""CASE
          WHEN trim("EDAD_TIPO") = '1' AND trim("EDAD_CANT") <> ''
            THEN {quantity}::DOUBLE
          WHEN trim("EDAD_TIPO") IN ('2', '3', '4')
               AND trim("EDAD_CANT") <> '' THEN 0.0
          ELSE NULL END"""
        upper = f"""CASE
          WHEN trim("EDAD_TIPO") = '1' AND trim("EDAD_CANT") <> ''
            THEN ({quantity} + 1)::DOUBLE
          WHEN trim("EDAD_TIPO") IN ('2', '3', '4')
               AND trim("EDAD_CANT") <> '' THEN 1.0
          ELSE NULL END"""
        unit = """CASE
          WHEN trim("EDAD_TIPO") = '1' THEN 'completed_years'
          WHEN trim("EDAD_TIPO") = '2' THEN 'completed_months'
          WHEN trim("EDAD_TIPO") = '3' THEN 'completed_days'
          WHEN trim("EDAD_TIPO") = '4' THEN 'completed_hours'
          ELSE 'source_age_unknown' END"""
        source_quantity = f"""CASE
          WHEN trim("EDAD_TIPO") IN ('1', '2', '3', '4')
               AND trim("EDAD_CANT") <> '' THEN {quantity}::DOUBLE
          ELSE NULL END"""
        return lower, upper, unit, source_quantity
    lower = f"""CASE
      WHEN trim("EDAD_TIPO") = '1' AND trim("EDAD_CANT") <> ''
        THEN {quantity}::DOUBLE
      WHEN trim("EDAD_TIPO") IN ('2', '3', '4')
           AND trim("EDAD_CANT") <> '' THEN 0.0
      ELSE NULL END"""
    upper = f"""CASE
      WHEN trim("EDAD_TIPO") = '1' AND trim("EDAD_CANT") <> ''
        THEN ({quantity} + 1)::DOUBLE
      WHEN trim("EDAD_TIPO") IN ('2', '3', '4')
           AND trim("EDAD_CANT") <> '' THEN 1.0
      ELSE NULL END"""
    unit = """CASE
      WHEN (
        trim("EDAD_TIPO") = '1' AND trim("EDAD_CANT") <> ''
      ) OR (
        trim("EDAD_TIPO") IN ('2', '3', '4') AND trim("EDAD_CANT") <> ''
      ) THEN 'completed_years'
      ELSE 'source_age_unknown' END"""
    completed_quantity = f"""CASE
      WHEN trim("EDAD_TIPO") = '1' AND trim("EDAD_CANT") <> ''
        THEN {quantity}::DOUBLE
      WHEN trim("EDAD_TIPO") IN ('2', '3', '4')
           AND trim("EDAD_CANT") <> '' THEN 0.0
      ELSE NULL END"""
    return lower, upper, unit, completed_quantity


def _complete_normalized_frame(
    aggregate: pd.DataFrame,
    *,
    release: DeisDeathRelease,
    aggregation: DeisAggregation,
) -> pd.DataFrame:
    aggregate = aggregate.sort_values(
        list(_SORT_COLUMNS),
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    geography_basis = (
        "death_occurrence_territory_chile"
        if aggregation.geography_grain is DeisGeographyGrain.NATIONAL
        else "decedent_usual_residence_within_chile_occurrence_universe"
    )
    geography_vintage = (
        None if aggregation.geography_grain is DeisGeographyGrain.NATIONAL else "dpa-2019"
    )
    aggregation_rule = _aggregation_rule(aggregation)
    manifest = release.assets.manifest
    primary = manifest.primary_fact_asset
    selected_source = _selected_source(release)
    missingness = aggregate.apply(
        lambda row: _dimension_missingness(row, aggregation),
        axis=1,
    )
    row_count = len(aggregate)
    payload = pd.DataFrame(
        {
            "domain": ObservationDomain.DEMOGRAPHIC.value,
            "variable": "deaths",
            "value": aggregate["value"],
            "semantic_kind": "count",
            "unit": "event",
            "population_basis": "not_applicable",
            "observation_role": "observed_fact",
            "parity_scope": "not_applicable",
            "period_start": _built_in_dates(aggregate["period_start"]),
            "period_end": _built_in_dates(aggregate["period_end"]),
            "source_key": primary.source_key,
            "vintage": selected_source.vintage,
            "released_at": pd.Series(
                [primary.released_at] * row_count,
                dtype=object,
            ),
            "release_missingness_reason": pd.Series(
                [
                    (
                        None
                        if primary.release_missingness_reason is None
                        else primary.release_missingness_reason.value
                    )
                ]
                * row_count,
                dtype=object,
            ),
            "available_at": pd.Series(
                [manifest.available_at] * row_count,
                dtype=object,
            ),
            "provisional": selected_source.provisional,
            "transformation_id": _TRANSFORMATION_ID,
            "transformation_version": "v1",
            "aggregation_rule": aggregation_rule,
            "missingness_reason": missingness,
            "source_status_code": pd.Series([None] * row_count, dtype=object),
            "age_lower": _nullable_numbers(aggregate["age_lower"]),
            "age_upper": _nullable_numbers(aggregate["age_upper"]),
            "age_open": aggregate["age_open"].astype(bool),
            "sex": aggregate["sex"],
            "region": aggregate["region"],
            "cause": aggregate["cause"],
            "date_basis": aggregate["date_basis"],
            "age_role": "decedent",
            "age_measure_unit": _nullable_text(aggregate["age_measure_unit"]),
            "age_quantity": _nullable_numbers(aggregate["age_quantity"]),
            "sex_role": "decedent",
            "commune": _nullable_text(aggregate["commune"]),
            "geography_basis": geography_basis,
            "geography_vintage": pd.Series(
                [geography_vintage] * row_count,
                dtype=object,
            ),
            "cause_code_system": aggregate["cause_code_system"],
            "cause_role": (
                "all_cause" if aggregation.cause_grain is DeisCauseGrain.ALL else "underlying"
            ),
            "external_cause": _nullable_text(aggregate["external_cause"]),
            "registration_start": pd.Series(
                [None] * row_count,
                dtype=object,
            ),
            "registration_end": pd.Series(
                [None] * row_count,
                dtype=object,
            ),
            "birth_order": pd.Series([None] * row_count, dtype=object),
            "nationality": pd.Series([None] * row_count, dtype=object),
            "nationality_role": pd.Series(
                [None] * row_count,
                dtype=object,
            ),
            "country_of_birth": pd.Series(
                [None] * row_count,
                dtype=object,
            ),
            "education_level": pd.Series(
                [None] * row_count,
                dtype=object,
            ),
            "residence_five_years_ago": pd.Series(
                [None] * row_count,
                dtype=object,
            ),
            "migration_status": pd.Series(
                [None] * row_count,
                dtype=object,
            ),
        },
        index=aggregate.index,
    )
    return payload


def _built_in_dates(series: pd.Series) -> pd.Series:
    return pd.Series(
        [pd.Timestamp(value).date() for value in series],
        index=series.index,
        dtype=object,
    )


def _nullable_numbers(series: pd.Series) -> pd.Series:
    return pd.Series(
        [None if pd.isna(value) else float(value) for value in series],
        index=series.index,
        dtype=object,
    )


def _nullable_text(series: pd.Series) -> pd.Series:
    return pd.Series(
        [None if pd.isna(value) else str(value) for value in series],
        index=series.index,
        dtype=object,
    )


def _dimension_missingness(
    row: pd.Series,
    aggregation: DeisAggregation,
) -> str:
    reasons = dict(_DIMENSION_MISSINGNESS_BASE)
    if row["date_basis"] == "death_occurrence_year_when_date_missing":
        reasons["occurrence_date"] = "missing_in_source_year_only"
    if (
        aggregation.age_grain is not DeisAgeGrain.ALL
        and row["age_measure_unit"] == "source_age_unknown"
    ):
        reasons["decedent_age"] = "source_age_unknown"
    if (
        aggregation.age_grain is not DeisAgeGrain.ALL
        and pd.isna(row["age_quantity"])
        and row["age_measure_unit"] != "source_age_unknown"
    ):
        reasons["decedent_age_quantity"] = "missing_in_source"
    if (
        aggregation.geography_grain is not DeisGeographyGrain.NATIONAL
        and row["region"] == "unknown"
    ):
        reasons["residence_region"] = "blank_or_99999_source_commune_and_region"
    if aggregation.geography_grain is DeisGeographyGrain.COMMUNE and pd.isna(row["commune"]):
        reasons["residence_commune"] = "blank_or_99999_source_commune"
    if row["cause"] == "unknown":
        reasons["underlying_cause"] = "missing_at_requested_source_hierarchy"
    if row["external_cause"] == "unknown_not_coded":
        reasons["external_cause"] = "published_placeholder_not_coded"
    return json.dumps(reasons, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _aggregation_rule(aggregation: DeisAggregation) -> str:
    sex = "decedent sex" if aggregation.by_sex else "sex collapsed to all"
    external = (
        "external cause retained as a dimension"
        if aggregation.retain_external_cause
        else "external cause collapsed"
    )
    return (
        "each DEIS death event counted exactly once; "
        f"occurrence_time={aggregation.temporal_grain.value}; "
        f"age={aggregation.age_grain.value}; {sex}; "
        f"geography={aggregation.geography_grain.value}; "
        f"underlying_cause={aggregation.cause_grain.value}; {external}; "
        "preliminary terminal buckets clipped to the declared occurrence cutoff"
    )


def _provenance(
    *,
    release: DeisDeathRelease,
    aggregation: DeisAggregation,
    actual: _RawAudit,
) -> VariableProvenance:
    dimensions = (
        f"occurrence_{aggregation.temporal_grain.value}",
        {
            DeisAgeGrain.SOURCE_REPORTED: "source_reported_age",
            DeisAgeGrain.COMPLETED_YEAR: "completed_year_age",
            DeisAgeGrain.ALL: "age_collapsed",
        }[aggregation.age_grain],
        "decedent_sex" if aggregation.by_sex else "sex_collapsed",
        {
            DeisGeographyGrain.NATIONAL: "occurrence_territory_national",
            DeisGeographyGrain.REGION: "residence_region",
            DeisGeographyGrain.COMMUNE: "residence_commune",
        }[aggregation.geography_grain],
        (
            "all_underlying_causes"
            if aggregation.cause_grain is DeisCauseGrain.ALL
            else aggregation.cause_grain.value
        ),
        ("external_cause" if aggregation.retain_external_cause else "external_cause_collapsed"),
    )
    manifest = release.assets.manifest
    return VariableProvenance(
        sources=release.sources,
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=actual.actual_occurrence_start,
        observation_end=actual.actual_occurrence_end + timedelta(days=1),
        dimensions=dimensions,
        aggregation_rules=(_aggregation_rule(aggregation),),
        missingness_reason=_REGISTRATION_LIMITATION,
        transformation_ids=(_TRANSFORMATION_ID,),
        notes=(
            "Universe: deaths occurring in Chilean territory.",
            "Geography fields describe decedent usual residence; canonical region "
            "codes are cross-checked against the raw DEIS label.",
            "Residence commune codes are left-padded to five digits under DPA 2019.",
            "Underlying-cause system is ICD-9 through 1996 and ICD-10 from 1997.",
            "Published irregular cause strings are preserved and counted in the audit.",
            "Blank occurrence dates are retained as explicit unknown-within-year buckets.",
            _REGISTRATION_LIMITATION,
        ),
    )


def _scalar(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> int:
    value = connection.execute(query).fetchone()
    if value is None:
        raise DataContractError("DEIS audit query returned no result.")
    return int(value[0])


def _quoted_columns(columns: tuple[str, ...]) -> str:
    return ", ".join(f'"{column}"' for column in columns)
