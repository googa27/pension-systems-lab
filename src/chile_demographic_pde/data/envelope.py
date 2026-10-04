"""Shared observation metadata and canonical one-release batch boundaries."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from typing import Final, cast

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.typing import Series
from pydantic import ValidationError

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.core.provenance import VariableProvenance
from chile_demographic_pde.core.semantics import SEMANTIC_COMPATIBILITY
from chile_demographic_pde.data.identity import (
    CanonicalIdentity,
    FactIdFactory,
    IdentityRegistry,
    IdentityRegistrySnapshot,
    ObservationIdFactory,
    TransformationIdentity,
)
from chile_demographic_pde.data.releases import ReleaseManifest
from chile_demographic_pde.data.roles import (
    ObservationDomain,
    ObservationRole,
    ParityScope,
    ReleaseMissingnessReason,
)

ENVELOPE_COLUMNS: Final = (
    "fact_id",
    "observation_id",
    "release_id",
    "domain",
    "variable",
    "value",
    "semantic_kind",
    "unit",
    "population_basis",
    "observation_role",
    "parity_scope",
    "period_start",
    "period_end",
    "source_key",
    "vintage",
    "released_at",
    "release_missingness_reason",
    "available_at",
    "provisional",
    "transformation_id",
    "transformation_version",
    "aggregation_rule",
    "missingness_reason",
    "source_status_code",
)

_ID_COLUMNS: Final = ("fact_id", "observation_id", "release_id")
_FACT_ID = re.compile(r"^cldemopde:fact:v1:[0-9a-f]{64}$")
_OBSERVATION_ID = re.compile(r"^cldemopde:observation:v1:[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^cldemopde:release:v1:[0-9a-f]{64}$")
_SOURCE_KEY = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_SEMANTIC_TRIPLES: Final = frozenset(
    (kind.value, unit.value, basis.value)
    for kind, unit_contracts in SEMANTIC_COMPATIBILITY.items()
    for unit, bases in unit_contracts.items()
    for basis in bases
)
_DOMAINS: Final = frozenset(item.value for item in ObservationDomain)
_ROLES: Final = frozenset(item.value for item in ObservationRole)
_PARITY_SCOPES: Final = frozenset(item.value for item in ParityScope)
_RELEASE_REASONS: Final = frozenset(item.value for item in ReleaseMissingnessReason)
_NONBLANK_COLUMNS: Final = (
    "fact_id",
    "observation_id",
    "release_id",
    "domain",
    "variable",
    "semantic_kind",
    "unit",
    "population_basis",
    "observation_role",
    "parity_scope",
    "source_key",
    "vintage",
    "aggregation_rule",
    "missingness_reason",
)


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    return False


def _is_date_not_datetime(value: object) -> bool:
    return isinstance(value, date) and not isinstance(value, datetime)


def _is_utc_datetime(value: object) -> bool:
    if not isinstance(value, datetime) or value.tzinfo is None:
        return False
    offset = value.utcoffset()
    return offset is not None and offset.total_seconds() == 0


def _matches(pattern: re.Pattern[str], value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


class ObservationEnvelopeSchema(pa.DataFrameModel):
    """Strict common fields shared by every isolated observation domain."""

    fact_id: Series[str] = pa.Field(str_matches=r"^cldemopde:fact:v1:[0-9a-f]{64}$")
    observation_id: Series[str] = pa.Field(str_matches=r"^cldemopde:observation:v1:[0-9a-f]{64}$")
    release_id: Series[str] = pa.Field(str_matches=r"^cldemopde:release:v1:[0-9a-f]{64}$")
    domain: Series[str] = pa.Field(isin=sorted(_DOMAINS))
    variable: Series[str]
    value: Series[float] = pa.Field(nullable=True, coerce=True)
    semantic_kind: Series[str]
    unit: Series[str]
    population_basis: Series[str]
    observation_role: Series[str] = pa.Field(isin=sorted(_ROLES))
    parity_scope: Series[str] = pa.Field(isin=sorted(_PARITY_SCOPES))
    period_start: Series[object]
    period_end: Series[object]
    source_key: Series[str] = pa.Field(str_matches=r"^[a-z0-9][a-z0-9_]*$")
    vintage: Series[str]
    released_at: Series[object] = pa.Field(nullable=True)
    release_missingness_reason: Series[str] = pa.Field(nullable=True)
    available_at: Series[object]
    provisional: Series[bool]
    transformation_id: Series[str] = pa.Field(nullable=True)
    transformation_version: Series[str] = pa.Field(nullable=True)
    aggregation_rule: Series[str]
    missingness_reason: Series[str]
    source_status_code: Series[str] = pa.Field(nullable=True)

    @pa.dataframe_check(ignore_na=False)
    def is_nonempty(cls, frame: pd.DataFrame) -> bool:
        return not frame.empty

    @pa.dataframe_check(ignore_na=False)
    def identifiers_are_versioned(cls, frame: pd.DataFrame) -> pd.Series[bool]:
        return pd.Series(
            [
                _matches(_FACT_ID, fact)
                and _matches(_OBSERVATION_ID, observation)
                and _matches(_RELEASE_ID, release)
                for fact, observation, release in frame.loc[:, list(_ID_COLUMNS)].itertuples(
                    index=False, name=None
                )
            ],
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def enums_and_semantics_are_closed(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series[bool]:
        triples = frame.loc[:, ["semantic_kind", "unit", "population_basis"]].itertuples(
            index=False, name=None
        )
        return pd.Series(
            (
                domain in _DOMAINS
                and role in _ROLES
                and parity in _PARITY_SCOPES
                and triple in _SEMANTIC_TRIPLES
                for domain, role, parity, triple in zip(
                    frame["domain"],
                    frame["observation_role"],
                    frame["parity_scope"],
                    triples,
                    strict=True,
                )
            ),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def values_are_finite_or_explicitly_missing(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series[bool]:
        value = frame["value"]
        finite_or_missing = value.isna() | np.isfinite(value)
        explicitly_missing = value.notna() | (
            frame["missingness_reason"].str.contains(r"\S", regex=True, na=False)
            & frame["missingness_reason"].ne("not_applicable_observed")
        )
        return pd.Series(
            finite_or_missing & explicitly_missing,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def periods_are_half_open(cls, frame: pd.DataFrame) -> pd.Series[bool]:
        return pd.Series(
            [
                _is_date_not_datetime(start) and _is_date_not_datetime(end) and start < end
                for start, end in frame.loc[:, ["period_start", "period_end"]].itertuples(
                    index=False, name=None
                )
            ],
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def available_at_is_utc(cls, frame: pd.DataFrame) -> pd.Series[bool]:
        return frame["available_at"].map(_is_utc_datetime).astype(bool)

    @pa.dataframe_check(ignore_na=False)
    def release_date_has_exact_reason_xor(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series[bool]:
        return pd.Series(
            [
                (_is_date_not_datetime(released) and _is_missing(reason))
                or (
                    _is_missing(released) and isinstance(reason, str) and reason in _RELEASE_REASONS
                )
                for released, reason in frame.loc[
                    :, ["released_at", "release_missingness_reason"]
                ].itertuples(index=False, name=None)
            ],
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def transformations_are_paired(cls, frame: pd.DataFrame) -> pd.Series[bool]:
        return pd.Series(
            [
                (_is_missing(transformation_id) and _is_missing(transformation_version))
                or (
                    isinstance(transformation_id, str)
                    and bool(transformation_id.strip())
                    and isinstance(transformation_version, str)
                    and bool(transformation_version.strip())
                )
                for transformation_id, transformation_version in frame.loc[
                    :, ["transformation_id", "transformation_version"]
                ].itertuples(index=False, name=None)
            ],
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def required_text_is_nonblank(cls, frame: pd.DataFrame) -> pd.Series[bool]:
        valid = pd.Series(True, index=frame.index, dtype=bool)
        for column in _NONBLANK_COLUMNS:
            valid &= frame[column].map(lambda value: isinstance(value, str) and bool(value.strip()))
        valid &= frame["source_key"].map(lambda value: _matches(_SOURCE_KEY, value))
        valid &= frame["source_status_code"].map(
            lambda value: _is_missing(value) or (isinstance(value, str) and bool(value.strip()))
        )
        return valid

    class Config:
        strict = False
        coerce = False
        unique_column_names = True


def _validate_complete_payload(frame: pd.DataFrame) -> pd.DataFrame:
    expected = tuple(column for column in ENVELOPE_COLUMNS if column not in _ID_COLUMNS)
    missing = [column for column in expected if column not in frame]
    if missing:
        raise DataContractError(f"Observation payload is missing envelope fields: {missing!r}.")
    return frame.copy(deep=True)


def _identity_atom(value: object) -> object:
    if _is_missing(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, (str, bool, int, float, date, datetime)):
        return value
    raise DataContractError(
        "Domain identity fields must use canonical scalar atoms; "
        f"received {type(value).__qualname__}."
    )


def _require_preidentity_contract(
    frame: pd.DataFrame,
    *,
    domain: ObservationDomain,
) -> None:
    if not frame.columns.is_unique:
        raise DataContractError("Observation payload column names must be unique.")
    if frame.empty:
        raise DataContractError("Observation payload must contain at least one row.")
    if not frame["domain"].eq(domain.value).all():
        raise DataContractError("Observation payload domain must equal the typed domain argument.")
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        triple = (
            row["semantic_kind"],
            row["unit"],
            row["population_basis"],
        )
        semantic_fields_are_strings = all(isinstance(item, str) for item in triple)
        role = row["observation_role"]
        parity = row["parity_scope"]
        value = row["value"]
        reason = row["missingness_reason"]
        transformation_id = row["transformation_id"]
        transformation_version = row["transformation_version"]
        released_at = row["released_at"]
        release_reason = row["release_missingness_reason"]
        required_text = (
            "release_id",
            "domain",
            "variable",
            "semantic_kind",
            "unit",
            "population_basis",
            "observation_role",
            "parity_scope",
            "source_key",
            "vintage",
            "aggregation_rule",
            "missingness_reason",
        )
        valid = (
            _matches(_RELEASE_ID, row["release_id"])
            and isinstance(role, str)
            and role in _ROLES
            and isinstance(parity, str)
            and parity in _PARITY_SCOPES
            and semantic_fields_are_strings
            and triple in _SEMANTIC_TRIPLES
            and (
                _is_missing(value)
                or (
                    isinstance(value, Real)
                    and not isinstance(value, bool)
                    and np.isfinite(float(value))
                )
            )
            and (
                not _is_missing(value)
                or (
                    isinstance(reason, str)
                    and bool(reason.strip())
                    and reason != "not_applicable_observed"
                )
            )
            and _is_date_not_datetime(row["period_start"])
            and _is_date_not_datetime(row["period_end"])
            and row["period_start"] < row["period_end"]
            and _is_utc_datetime(row["available_at"])
            and (
                (_is_date_not_datetime(released_at) and _is_missing(release_reason))
                or (
                    _is_missing(released_at)
                    and isinstance(release_reason, str)
                    and release_reason in _RELEASE_REASONS
                )
            )
            and (
                (_is_missing(transformation_id) and _is_missing(transformation_version))
                or (
                    isinstance(transformation_id, str)
                    and bool(transformation_id.strip())
                    and isinstance(transformation_version, str)
                    and bool(transformation_version.strip())
                )
            )
            and all(
                isinstance(row[column], str) and bool(row[column].strip())
                for column in required_text
            )
            and _matches(_SOURCE_KEY, row["source_key"])
            and (
                _is_missing(row["source_status_code"])
                or (
                    isinstance(row["source_status_code"], str)
                    and bool(row["source_status_code"].strip())
                )
            )
            and isinstance(row["provisional"], (bool, np.bool_))
        )
        if not valid:
            raise DataContractError(
                "Observation payload violates the envelope contract before "
                f"identity assignment at row ordinal {ordinal}."
            )


def _preflight_identity_records(
    registry: IdentityRegistry,
    identities: tuple[CanonicalIdentity, ...],
) -> None:
    known = {record.value: record.canonical_payload for record in registry.snapshot().records}
    for identity in identities:
        existing = known.get(identity.value)
        if existing is not None and existing != identity.canonical_payload:
            raise DataContractError(f"Canonical identity collision for {identity.value}.")
        known[identity.value] = identity.canonical_payload


def assign_observation_ids(
    frame: pd.DataFrame,
    *,
    domain: ObservationDomain,
    domain_key: tuple[str, ...],
    source_identity: Mapping[str, object],
    manifest: ReleaseManifest,
    registry: IdentityRegistry,
) -> pd.DataFrame:
    """Assign canonical IDs after validating a complete source/domain payload."""

    if not isinstance(frame, pd.DataFrame):
        raise DataContractError("Observation identity assignment requires a DataFrame.")
    supplied_ids = [column for column in _ID_COLUMNS if column in frame]
    if supplied_ids:
        raise DataContractError(
            "Observation ID columns must be absent before identity assignment; "
            f"caller supplied {supplied_ids!r}."
        )
    if not isinstance(domain, ObservationDomain):
        raise DataContractError("domain must be an ObservationDomain.")
    if (
        not isinstance(domain_key, tuple)
        or not domain_key
        or len(set(domain_key)) != len(domain_key)
        or any(not isinstance(column, str) or not column for column in domain_key)
    ):
        raise DataContractError("domain_key must be a nonempty tuple of unique column names.")
    if not isinstance(source_identity, Mapping) or not source_identity:
        raise DataContractError("source_identity must be a nonempty mapping.")
    if not isinstance(manifest, ReleaseManifest):
        raise DataContractError("manifest must be a ReleaseManifest.")
    if not isinstance(registry, IdentityRegistry):
        raise DataContractError("registry must be an IdentityRegistry.")

    working = _validate_complete_payload(frame)
    missing_key_columns = [column for column in domain_key if column not in working]
    if missing_key_columns:
        raise DataContractError(f"Domain identity key columns are absent: {missing_key_columns!r}.")

    primary = manifest.primary_fact_asset
    working["release_id"] = manifest.release_id.value
    working["available_at"] = pd.Series(
        [manifest.available_at] * len(working),
        index=working.index,
        dtype=object,
    )
    working["source_key"] = primary.source_key
    working["released_at"] = pd.Series(
        [primary.released_at] * len(working),
        index=working.index,
        dtype=object,
    )
    working["release_missingness_reason"] = (
        None
        if primary.release_missingness_reason is None
        else primary.release_missingness_reason.value
    )
    try:
        _require_preidentity_contract(working, domain=domain)
    except DataContractError:
        raise
    except (TypeError, ValueError, OverflowError):
        raise DataContractError(
            "Observation payload violates the envelope contract before identity assignment."
        ) from None
    for column in domain_key:
        if (
            working[column]
            .map(lambda value: isinstance(value, (float, np.floating)) and np.isnan(float(value)))
            .any()
        ):
            raise DataContractError(
                "Domain identity key columns require explicit null atoms; "
                f"column {column!r} contains a noncanonical NaN."
            )
    fact_ids = []
    observation_ids = []
    for _, row in working.iterrows():
        key_fields = {column: _identity_atom(row[column]) for column in domain_key}
        fact_id = FactIdFactory.v1(domain, source_identity, key_fields)
        transformation = TransformationIdentity(
            None if _is_missing(row["transformation_id"]) else cast(str, row["transformation_id"]),
            None
            if _is_missing(row["transformation_version"])
            else cast(str, row["transformation_version"]),
        )
        observation_id = ObservationIdFactory.v1(
            fact_id,
            manifest.release_id,
            cast(str, row["vintage"]),
            transformation,
            manifest.available_at,
        )
        fact_ids.append(fact_id)
        observation_ids.append(observation_id)

    identities: tuple[CanonicalIdentity, ...] = (
        manifest.release_id,
        *fact_ids,
        *observation_ids,
    )
    _preflight_identity_records(registry, identities)
    working.loc[:, "fact_id"] = [identity.value for identity in fact_ids]
    working.loc[:, "observation_id"] = [identity.value for identity in observation_ids]
    ordered = working.loc[
        :,
        [
            *ENVELOPE_COLUMNS,
            *(column for column in working if column not in ENVELOPE_COLUMNS),
        ],
    ]
    try:
        validated = ObservationEnvelopeSchema.validate(
            ordered,
            lazy=True,
            inplace=False,
        )
    except SchemaErrors as error:
        raise DataContractError(
            f"Assigned observations violate the envelope contract: {error}"
        ) from error
    for identity in identities:
        registry.record(identity)
    return validated.copy(deep=True)


def _validate_registry_record(
    identities: IdentityRegistrySnapshot,
    value: str,
) -> bytes:
    record = identities.require(value)
    prefix, _, digest = value.rpartition(":")
    if not prefix or len(digest) != 64:
        raise DataContractError("Canonical identity record has a malformed value.")
    if hashlib.sha256(record.canonical_payload).hexdigest() != digest:
        raise DataContractError(
            f"Canonical identity bytes do not reproduce registered value {value}."
        )
    return record.canonical_payload


def _revalidate_provenance(
    provenance: VariableProvenance,
) -> VariableProvenance:
    """Rebuild a provenance tree instead of trusting an unvalidated model copy."""

    validated: VariableProvenance | None = None
    try:
        payload = provenance.model_dump(mode="python", warnings="none")
        validated = VariableProvenance.model_validate(payload)
    except (ValidationError, TypeError, ValueError):
        pass
    if validated is None:
        raise DataContractError("Batch provenance violates its immutable typed contract.")
    return validated


@dataclass(frozen=True, slots=True, init=False)
class NormalizedDomainBatch[DomainFrameT]:
    """Immutable, deep-copy-safe observations for exactly one release."""

    manifest: ReleaseManifest
    _observations: pd.DataFrame
    provenance: VariableProvenance
    identities: IdentityRegistrySnapshot

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("NormalizedDomainBatch must be created by NormalizedDomainBatch.create().")

    @classmethod
    def create(
        cls,
        *,
        manifest: ReleaseManifest,
        observations: pd.DataFrame,
        provenance: VariableProvenance,
        identities: IdentityRegistrySnapshot,
    ) -> NormalizedDomainBatch[pd.DataFrame]:
        if not isinstance(manifest, ReleaseManifest):
            raise DataContractError("Batch manifest must be a ReleaseManifest.")
        if not isinstance(observations, pd.DataFrame):
            raise DataContractError("Batch observations must be a DataFrame.")
        if not isinstance(provenance, VariableProvenance):
            raise DataContractError("Batch provenance must be a VariableProvenance.")
        provenance = _revalidate_provenance(provenance)
        if not isinstance(identities, IdentityRegistrySnapshot):
            raise DataContractError("Batch identities must be an IdentityRegistrySnapshot.")

        owned = observations.copy(deep=True)
        try:
            validated = ObservationEnvelopeSchema.validate(
                owned,
                lazy=True,
                inplace=False,
            )
        except SchemaErrors as error:
            raise DataContractError(
                f"Batch observations violate the envelope contract: {error}"
            ) from error

        domains = set(validated["domain"])
        if len(domains) != 1:
            raise DataContractError("A normalized domain batch requires exactly one discriminator.")
        release_ids = set(validated["release_id"])
        if release_ids != {manifest.release_id.value}:
            raise DataContractError(
                "Batch observation release IDs must equal the manifest release."
            )
        if not validated["available_at"].map(lambda value: value == manifest.available_at).all():
            raise DataContractError("Batch observation availability must equal the manifest clock.")

        if provenance.release_id != manifest.release_id.value:
            raise DataContractError("Batch provenance release ID must equal the manifest release.")
        if provenance.available_at != manifest.available_at:
            raise DataContractError("Batch provenance availability must equal the manifest clock.")
        manifest_assets = tuple((asset.source_key, asset.sha256) for asset in manifest.assets)
        provenance_assets = tuple(
            sorted((source.source_key, source.sha256) for source in provenance.sources)
        )
        if provenance_assets != manifest_assets:
            raise DataContractError("Batch provenance assets must exactly match the manifest.")
        provenance_by_key = {source.source_key: source for source in provenance.sources}
        if len(provenance_by_key) != len(provenance.sources):
            raise DataContractError("Batch provenance source keys must be unique.")
        for asset in manifest.assets:
            source = provenance_by_key[asset.source_key]
            if (
                source.release_date != asset.released_at
                or source.release_missingness_reason != asset.release_missingness_reason
            ):
                raise DataContractError(
                    "Batch provenance publisher release metadata must match "
                    "each manifest asset independently."
                )
            if source.retrieved_at != asset.available_at:
                raise DataContractError(
                    "Batch provenance source knowledge-time availability must "
                    "match each manifest asset independently."
                )

        primary = manifest.primary_fact_asset
        expected_reason = (
            None
            if primary.release_missingness_reason is None
            else primary.release_missingness_reason.value
        )
        if (
            not validated["source_key"].eq(primary.source_key).all()
            or not validated["released_at"].map(lambda value: value == primary.released_at).all()
            or not validated["release_missingness_reason"]
            .map(
                lambda value: (
                    _is_missing(value) if expected_reason is None else value == expected_reason
                )
            )
            .all()
        ):
            raise DataContractError(
                "Envelope publisher metadata must come from the primary fact asset."
            )

        release_payload = _validate_registry_record(
            identities,
            manifest.release_id.value,
        )
        if release_payload != manifest.release_id.canonical_payload:
            raise DataContractError("Manifest release canonical bytes differ from the registry.")
        for column in ("fact_id", "observation_id"):
            for value in validated[column].unique():
                _validate_registry_record(identities, cast(str, value))

        batch = cast(
            "NormalizedDomainBatch[pd.DataFrame]",
            object.__new__(cls),
        )
        object.__setattr__(batch, "manifest", manifest)
        object.__setattr__(batch, "_observations", validated.copy(deep=True))
        object.__setattr__(batch, "provenance", provenance)
        object.__setattr__(batch, "identities", identities)
        return batch

    @property
    def observations(self) -> pd.DataFrame:
        return self._observations.copy(deep=True)

    @property
    def domain(self) -> ObservationDomain:
        return ObservationDomain(cast(str, self._observations["domain"].iloc[0]))
