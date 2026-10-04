"""Immutable single-domain histories and point-in-time observation selection."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pandas as pd
from pandas.api.types import is_dtype_equal
from pandera.errors import SchemaError, SchemaErrors
from pydantic import ValidationError

from chile_demographic_pde.core.errors import (
    AmbiguousRevisionError,
    DataContractError,
    IdentityCollisionError,
)
from chile_demographic_pde.core.provenance import VariableProvenance
from chile_demographic_pde.data.domain_schemas import (
    DEMOGRAPHIC_OBSERVATION_KEY,
    ENVIRONMENTAL_OBSERVATION_KEY,
    HEALTH_SYSTEM_OBSERVATION_KEY,
    MARKET_QUOTE_KEY,
    PENSION_PRODUCT_STATISTIC_KEY,
    REGULATORY_MORTALITY_KEY,
    REGULATORY_RATE_KEY,
    validate_demographic_observations,
    validate_environmental_observations,
    validate_health_system_observations,
    validate_market_quotes,
    validate_pension_product_statistics,
    validate_regulatory_mortality,
    validate_regulatory_rates,
)
from chile_demographic_pde.data.envelope import NormalizedDomainBatch
from chile_demographic_pde.data.identity import (
    CanonicalIdentityRecord,
    IdentityRegistry,
    IdentityRegistrySnapshot,
)
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
)
from chile_demographic_pde.data.roles import ObservationDomain

_DomainValidator = Callable[[pd.DataFrame], pd.DataFrame]

_DOMAIN_CONTRACTS: dict[
    ObservationDomain,
    tuple[_DomainValidator, tuple[str, ...]],
] = {
    ObservationDomain.DEMOGRAPHIC: (
        validate_demographic_observations,
        DEMOGRAPHIC_OBSERVATION_KEY,
    ),
    ObservationDomain.ENVIRONMENTAL: (
        validate_environmental_observations,
        ENVIRONMENTAL_OBSERVATION_KEY,
    ),
    ObservationDomain.HEALTH_SYSTEM: (
        validate_health_system_observations,
        HEALTH_SYSTEM_OBSERVATION_KEY,
    ),
    ObservationDomain.MARKET_QUOTE: (
        validate_market_quotes,
        MARKET_QUOTE_KEY,
    ),
    ObservationDomain.REGULATORY_RATE: (
        validate_regulatory_rates,
        REGULATORY_RATE_KEY,
    ),
    ObservationDomain.REGULATORY_MORTALITY_TABLE: (
        validate_regulatory_mortality,
        REGULATORY_MORTALITY_KEY,
    ),
    ObservationDomain.PENSION_PRODUCT_STATISTIC: (
        validate_pension_product_statistics,
        PENSION_PRODUCT_STATISTIC_KEY,
    ),
}


def _sanitize_manifest(manifest: object) -> ReleaseManifest:
    """Reconstruct a manifest and verify every factory-derived field."""

    if not isinstance(manifest, ReleaseManifest):
        raise DataContractError("Domain batch manifest is invalid.")
    try:
        schema_version = manifest.schema_version
        original_assets = manifest.assets
        primary_source_key = manifest.primary_fact_source_key
        original_release_id = manifest.release_id
        original_available_at = manifest.available_at
        if type(schema_version) is not str or not schema_version.strip():
            raise TypeError
        if type(original_assets) is not tuple or not original_assets:
            raise TypeError
        rebuilt_assets = tuple(
            RequiredReleaseAsset(
                source_key=asset.source_key,
                sha256=asset.sha256,
                available_at=asset.available_at,
                released_at=asset.released_at,
                release_missingness_reason=asset.release_missingness_reason,
            )
            for asset in original_assets
        )
        registry = IdentityRegistry()
        rebuilt = ReleaseManifest.create(
            schema_version=schema_version,
            assets=rebuilt_assets,
            primary_fact_source_key=primary_source_key,
            identity_registry=registry,
        )
        exact = (
            original_assets == rebuilt.assets
            and original_available_at == rebuilt.available_at
            and original_release_id.value == rebuilt.release_id.value
            and original_release_id.canonical_payload == rebuilt.release_id.canonical_payload
        )
    except (
        AttributeError,
        DataContractError,
        TypeError,
        ValueError,
    ):
        raise DataContractError("Domain batch manifest is invalid.") from None
    if not exact:
        raise DataContractError("Domain batch manifest factory-derived identity is invalid.")
    return rebuilt


def _sanitize_provenance(provenance: object) -> VariableProvenance:
    if not isinstance(provenance, VariableProvenance):
        raise DataContractError("Domain batch provenance is invalid.")
    try:
        payload = provenance.model_dump(mode="python", warnings="none")
        return VariableProvenance.model_validate(payload)
    except (
        AttributeError,
        TypeError,
        ValidationError,
        ValueError,
    ):
        raise DataContractError("Domain batch provenance is invalid.") from None


def _sanitize_identities(identities: object) -> IdentityRegistrySnapshot:
    if not isinstance(identities, IdentityRegistrySnapshot):
        raise DataContractError("Domain batch identity snapshot is invalid.")
    try:
        records = tuple(
            CanonicalIdentityRecord(
                value=record.value,
                canonical_payload=bytes(record.canonical_payload),
            )
            for record in identities.records
        )
        return IdentityRegistrySnapshot(records)
    except IdentityCollisionError:
        raise
    except (
        AttributeError,
        DataContractError,
        TypeError,
        ValueError,
    ):
        raise DataContractError("Domain batch identity snapshot is invalid.") from None


def _sanitize_domain_batch(
    batch: object,
) -> NormalizedDomainBatch[pd.DataFrame]:
    """Reconstruct and strictly validate a nominal one-release domain batch."""

    if not isinstance(batch, NormalizedDomainBatch):
        raise DataContractError("A normalized domain batch is required.")
    try:
        manifest = _sanitize_manifest(batch.manifest)
        provenance = _sanitize_provenance(batch.provenance)
        identities = _sanitize_identities(batch.identities)
        observations = batch.observations
        envelope_batch = NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=provenance,
            identities=identities,
        )
        validator, _ = _DOMAIN_CONTRACTS[envelope_batch.domain]
        validated = validator(envelope_batch.observations)
        return NormalizedDomainBatch.create(
            manifest=manifest,
            observations=validated,
            provenance=provenance,
            identities=identities,
        )
    except IdentityCollisionError:
        raise
    except (
        AttributeError,
        DataContractError,
        KeyError,
        SchemaError,
        SchemaErrors,
        TypeError,
        ValidationError,
        ValueError,
    ):
        raise DataContractError(
            "Domain batch violates its exact domain schema or release contract."
        ) from None


def _merge_identity_snapshots(
    batches: Sequence[NormalizedDomainBatch[pd.DataFrame]],
) -> IdentityRegistrySnapshot:
    return IdentityRegistrySnapshot(
        tuple(record for batch in batches for record in batch.identities.records)
    )


def _require_cross_release_consistency(
    batches: Sequence[NormalizedDomainBatch[pd.DataFrame]],
    *,
    domain: ObservationDomain,
) -> None:
    _, domain_key = _DOMAIN_CONTRACTS[domain]
    observation_rows: dict[str, pd.Series] = {}
    fact_keys: dict[str, pd.Series] = {}
    for batch in batches:
        for _, row in batch.observations.iterrows():
            observation_id = cast(str, row["observation_id"])
            previous_row = observation_rows.get(observation_id)
            if previous_row is not None and not previous_row.equals(row):
                raise DataContractError("One observation identity describes different rows.")
            observation_rows[observation_id] = row.copy(deep=True)

            fact_id = cast(str, row["fact_id"])
            fact_key = row.loc[list(domain_key)].copy(deep=True)
            previous_key = fact_keys.get(fact_id)
            if previous_key is not None and not previous_key.equals(fact_key):
                raise DataContractError("One fact identity describes different domain-key fields.")
            fact_keys[fact_id] = fact_key


@dataclass(frozen=True, slots=True, init=False, eq=False)
class DomainObservationHistory[DomainFrameT: pd.DataFrame]:
    """Validated, deterministic release history for exactly one domain."""

    _batches: tuple[NormalizedDomainBatch[pd.DataFrame], ...]
    _domain: ObservationDomain

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError(
            "DomainObservationHistory must be created by DomainObservationHistory.create()."
        )

    @classmethod
    def create(
        cls,
        batches: Sequence[NormalizedDomainBatch[DomainFrameT]],
    ) -> DomainObservationHistory[DomainFrameT]:
        if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
            raise DataContractError("Domain observation history requires a sequence of batches.")
        if not batches:
            raise DataContractError(
                "Domain observation history requires a nonempty batch sequence."
            )
        sanitized = tuple(_sanitize_domain_batch(batch) for batch in batches)
        domains = {batch.domain for batch in sanitized}
        if len(domains) != 1:
            raise DataContractError("Domain observation history cannot mix domain schemas.")
        domain = next(iter(domains))
        release_ids = tuple(batch.manifest.release_id.value for batch in sanitized)
        if len(set(release_ids)) != len(release_ids):
            raise DataContractError("Domain observation history contains a duplicate release.")
        _merge_identity_snapshots(sanitized)
        _require_cross_release_consistency(sanitized, domain=domain)
        ordered = tuple(
            sorted(
                sanitized,
                key=lambda batch: (
                    batch.manifest.available_at,
                    batch.manifest.release_id.value,
                ),
            )
        )
        history = object.__new__(cls)
        object.__setattr__(history, "_batches", ordered)
        object.__setattr__(history, "_domain", domain)
        return history

    @property
    def batches(
        self,
    ) -> tuple[NormalizedDomainBatch[DomainFrameT], ...]:
        return cast(
            tuple[NormalizedDomainBatch[DomainFrameT], ...],
            self._batches,
        )

    @property
    def domain(self) -> ObservationDomain:
        return self._domain

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(
        self,
    ) -> Iterator[NormalizedDomainBatch[DomainFrameT]]:
        return iter(self.batches)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class CompiledDomainSelection[DomainFrameT: pd.DataFrame]:
    """Immutable multi-release result of one point-in-time selection."""

    _observations: pd.DataFrame
    manifests: tuple[ReleaseManifest, ...]
    provenance: tuple[VariableProvenance, ...]
    identities: IdentityRegistrySnapshot
    domain: ObservationDomain
    as_of: datetime

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("CompiledDomainSelection must be created by latest_as_of().")

    @classmethod
    def _create(
        cls,
        *,
        observations: pd.DataFrame,
        manifests: tuple[ReleaseManifest, ...],
        provenance: tuple[VariableProvenance, ...],
        identities: IdentityRegistrySnapshot,
        domain: ObservationDomain,
        as_of: datetime,
    ) -> CompiledDomainSelection[DomainFrameT]:
        selection = object.__new__(cls)
        object.__setattr__(
            selection,
            "_observations",
            observations.copy(deep=True),
        )
        object.__setattr__(selection, "manifests", manifests)
        object.__setattr__(selection, "provenance", provenance)
        object.__setattr__(selection, "identities", identities)
        object.__setattr__(selection, "domain", domain)
        object.__setattr__(selection, "as_of", as_of)
        return selection

    @property
    def observations(self) -> pd.DataFrame:
        return self._observations.copy(deep=True)

    def __len__(self) -> int:
        return len(self._observations)


def _sanitize_history(
    history: object,
) -> DomainObservationHistory[pd.DataFrame]:
    if not isinstance(history, DomainObservationHistory):
        raise DataContractError("latest_as_of requires a DomainObservationHistory.")
    try:
        original_batches = history.batches
        original_domain = history.domain
        sanitized = DomainObservationHistory.create(original_batches)
        original_release_order = tuple(
            batch.manifest.release_id.value for batch in original_batches
        )
        sanitized_release_order = tuple(
            batch.manifest.release_id.value for batch in sanitized.batches
        )
    except IdentityCollisionError:
        raise
    except (
        AttributeError,
        DataContractError,
        KeyError,
        TypeError,
        ValueError,
    ):
        raise DataContractError("Domain observation history is malformed.") from None
    if original_domain is not sanitized.domain or original_release_order != sanitized_release_order:
        raise DataContractError("Domain observation history is malformed.")
    return sanitized


def _minimal_identity_snapshot(
    batches: Sequence[NormalizedDomainBatch[pd.DataFrame]],
    observations: pd.DataFrame,
) -> IdentityRegistrySnapshot:
    records_by_id = {
        record.value: record for batch in batches for record in batch.identities.records
    }
    selected_ids = {
        *cast(set[str], set(observations["release_id"])),
        *cast(set[str], set(observations["fact_id"])),
        *cast(set[str], set(observations["observation_id"])),
    }
    missing = selected_ids.difference(records_by_id)
    if missing:
        raise DataContractError("Selected observations lack canonical identity evidence.")
    return IdentityRegistrySnapshot(tuple(records_by_id[value] for value in sorted(selected_ids)))


def _concatenate_frames_columnwise(
    frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Concatenate schema-identical frames without pandas block coercion."""

    if not frames:
        raise DataContractError("Point-in-time selection requires at least one eligible frame.")
    reference = frames[0]
    expected_columns = reference.columns
    if not expected_columns.is_unique:
        raise DataContractError("Eligible observation frames require unique columns.")
    for frame in frames[1:]:
        if not frame.columns.equals(expected_columns):
            raise DataContractError("Eligible observation frames have inconsistent columns.")

    result = pd.DataFrame(index=pd.RangeIndex(sum(len(frame) for frame in frames)))
    for column in expected_columns:
        expected_dtype = reference[column].dtype
        columns = [frame[column] for frame in frames]
        if any(not is_dtype_equal(column_data.dtype, expected_dtype) for column_data in columns):
            raise DataContractError("Eligible observation frames have inconsistent dtypes.")
        concatenated = pd.concat(columns, ignore_index=True)
        if not is_dtype_equal(concatenated.dtype, expected_dtype):
            try:
                concatenated = concatenated.astype(
                    expected_dtype,
                    copy=False,
                )
            except (TypeError, ValueError):
                raise DataContractError(
                    "Eligible observation frame dtype was not preserved."
                ) from None
        if not is_dtype_equal(concatenated.dtype, expected_dtype):
            raise DataContractError("Eligible observation frame dtype was not preserved.")
        result[column] = concatenated
    return result


def latest_as_of[DomainFrameT: pd.DataFrame](
    history: DomainObservationHistory[DomainFrameT],
    *,
    as_of: datetime,
) -> CompiledDomainSelection[DomainFrameT]:
    """Select the uniquely latest eligible observation for every stable fact."""

    sanitized_history = _sanitize_history(history)
    if not isinstance(as_of, datetime):
        raise DataContractError("as_of must be a datetime.")
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise DataContractError("as_of must be timezone-aware.")
    cutoff = as_of.astimezone(UTC)

    template = sanitized_history.batches[0].observations.iloc[0:0].copy(deep=True)
    eligible_frames = [
        batch.observations.loc[
            batch.observations["available_at"].map(lambda value: value <= cutoff)
        ]
        for batch in sanitized_history.batches
    ]
    nonempty = [frame for frame in eligible_frames if not frame.empty]
    if not nonempty:
        return CompiledDomainSelection._create(
            observations=template,
            manifests=(),
            provenance=(),
            identities=IdentityRegistrySnapshot(()),
            domain=sanitized_history.domain,
            as_of=cutoff,
        )

    eligible = _concatenate_frames_columnwise(nonempty)
    selected_indices: list[int] = []
    for _, revisions in eligible.groupby("fact_id", sort=True):
        greatest = max(cast(datetime, value) for value in revisions["available_at"])
        latest = revisions.loc[revisions["available_at"].eq(greatest)]
        if latest["observation_id"].nunique(dropna=False) != 1:
            raise AmbiguousRevisionError(
                "A fact has distinct revisions at the same latest availability."
            )
        selected_indices.append(cast(int, latest.index[0]))
    selected = (
        eligible.loc[selected_indices]
        .sort_values(
            ["fact_id", "observation_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    validator, _ = _DOMAIN_CONTRACTS[sanitized_history.domain]
    try:
        selected = validator(selected)
    except (
        DataContractError,
        SchemaError,
        SchemaErrors,
        TypeError,
        ValueError,
    ):
        raise DataContractError(
            "Point-in-time selection violates its exact domain schema."
        ) from None

    contributing_release_ids = set(cast(str, value) for value in selected["release_id"])
    contributing = tuple(
        batch
        for batch in sanitized_history.batches
        if batch.manifest.release_id.value in contributing_release_ids
    )
    manifests = tuple(batch.manifest for batch in contributing)
    provenance = tuple(batch.provenance for batch in contributing)
    identities = _minimal_identity_snapshot(contributing, selected)
    return CompiledDomainSelection._create(
        observations=selected,
        manifests=manifests,
        provenance=provenance,
        identities=identities,
        domain=sanitized_history.domain,
        as_of=cutoff,
    )
