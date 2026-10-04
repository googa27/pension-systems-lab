from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from pydantic import ValidationError

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.data.envelope import (
    ENVELOPE_COLUMNS,
    NormalizedDomainBatch,
    ObservationEnvelopeSchema,
    assign_observation_ids,
)
from chile_demographic_pde.data.identity import (
    CanonicalIdentityRecord,
    IdentityRegistry,
    IdentityRegistrySnapshot,
)
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
)
from chile_demographic_pde.data.roles import (
    ObservationDomain,
    ReleaseMissingnessReason,
)


def _manifest(registry: IdentityRegistry) -> ReleaseManifest:
    return ReleaseManifest.create(
        schema_version="fixture-v1",
        assets=(
            RequiredReleaseAsset(
                source_key="workbook",
                sha256="a" * 64,
                available_at=datetime(2026, 7, 7, 15, tzinfo=UTC),
                released_at=date(2026, 6, 30),
                release_missingness_reason=None,
            ),
            RequiredReleaseAsset(
                source_key="entity_registry",
                sha256="b" * 64,
                available_at=datetime(2026, 7, 8, 15, tzinfo=UTC),
                released_at=None,
                release_missingness_reason=(ReleaseMissingnessReason.PUBLISHER_DATE_NOT_AVAILABLE),
            ),
        ),
        primary_fact_source_key="workbook",
        identity_registry=registry,
    )


def _payload() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "domain": "demographic",
                "variable": "deaths",
                "value": 12.0,
                "semantic_kind": "count",
                "unit": "event",
                "population_basis": "resident_estimate",
                "observation_role": "observed_fact",
                "parity_scope": "not_applicable",
                "period_start": date(2024, 1, 1),
                "period_end": date(2025, 1, 1),
                "source_key": "caller_must_not_be_authoritative",
                "vintage": "2024-final",
                "released_at": date(1900, 1, 1),
                "release_missingness_reason": ("publisher_date_not_available"),
                "available_at": datetime(1900, 1, 1, tzinfo=UTC),
                "provisional": False,
                "transformation_id": "normalize-deaths",
                "transformation_version": "v1",
                "aggregation_rule": "published count",
                "missingness_reason": "not_applicable_observed",
                "source_status_code": None,
            }
        ]
    )


def _provenance(manifest: ReleaseManifest) -> VariableProvenance:
    sources = tuple(
        SourceRef(
            source_key=asset.source_key,
            name=asset.source_key,
            url="https://example.invalid/",
            release_date=asset.released_at,
            release_missingness_reason=asset.release_missingness_reason,
            retrieved_at=asset.available_at,
            sha256=asset.sha256,
            vintage="2024-final",
            provisional=False,
        )
        for asset in manifest.assets
    )
    return VariableProvenance(
        sources=sources,
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=date(2024, 1, 1),
        observation_end=date(2025, 1, 1),
        dimensions=("period",),
        aggregation_rules=("published count",),
        missingness_reason="not_applicable_observed",
    )


def _assigned() -> tuple[
    pd.DataFrame,
    ReleaseManifest,
    IdentityRegistrySnapshot,
]:
    registry = IdentityRegistry()
    manifest = _manifest(registry)
    assigned = assign_observation_ids(
        _payload(),
        domain=ObservationDomain.DEMOGRAPHIC,
        domain_key=("variable", "period_start", "period_end"),
        source_identity={"publisher": "fixture"},
        manifest=manifest,
        registry=registry,
    )
    return assigned, manifest, registry.snapshot()


@pytest.mark.parametrize(
    "nullable_atom",
    [None, pd.NA, pd.NaT],
    ids=["none", "pandas_na", "pandas_nat"],
)
def test_identity_assignment_canonicalizes_nullable_domain_key_atoms(
    nullable_atom: object,
) -> None:
    registry = IdentityRegistry()
    manifest = _manifest(registry)
    reference = _payload().assign(nullable_dimension=None)
    candidate = _payload().assign(nullable_dimension=nullable_atom)

    first = assign_observation_ids(
        reference,
        domain=ObservationDomain.DEMOGRAPHIC,
        domain_key=(
            "variable",
            "period_start",
            "period_end",
            "nullable_dimension",
        ),
        source_identity={"publisher": "fixture"},
        manifest=manifest,
        registry=registry,
    )
    second = assign_observation_ids(
        candidate,
        domain=ObservationDomain.DEMOGRAPHIC,
        domain_key=(
            "variable",
            "period_start",
            "period_end",
            "nullable_dimension",
        ),
        source_identity={"publisher": "fixture"},
        manifest=manifest,
        registry=registry,
    )

    assert first["fact_id"].tolist() == second["fact_id"].tolist()
    assert first["observation_id"].tolist() == second["observation_id"].tolist()


def test_source_release_date_is_nullable_only_with_reason() -> None:
    source = SourceRef(
        source_key="cmf_vtd_xlsx",
        name="CMF VTD",
        url="https://www.cmfchile.cl/",
        release_date=None,
        release_missingness_reason="publisher_date_not_available",
        retrieved_at=datetime(2026, 7, 19, tzinfo=UTC),
        sha256="a" * 64,
        vintage="2026-06",
        provisional=False,
    )

    assert source.release_date is None
    with pytest.raises(ValidationError, match="release"):
        SourceRef.model_validate({**source.model_dump(), "release_missingness_reason": None})
    with pytest.raises(ValidationError, match="release"):
        SourceRef.model_validate(
            {
                **source.model_dump(),
                "release_date": date(2026, 7, 1),
            }
        )


def test_source_and_variable_provenance_require_typed_composite_metadata() -> None:
    source = SourceRef(
        source_key="asset",
        name="asset",
        url="https://example.invalid/",
        release_date=date(2026, 7, 6),
        release_missingness_reason=None,
        retrieved_at=datetime(2026, 7, 7, tzinfo=UTC),
        sha256="a" * 64,
        vintage="v1",
        provisional=False,
    )
    provenance = VariableProvenance(
        sources=(source,),
        release_id="cldemopde:release:v1:" + "b" * 64,
        available_at=datetime(2026, 7, 7, tzinfo=UTC),
        observation_start=date(2026, 7, 1),
        observation_end=date(2026, 8, 1),
        dimensions=("effective_interval",),
        aggregation_rules=("published scalar",),
        missingness_reason="not_applicable_observed",
    )

    assert provenance.available_at.tzinfo is UTC
    with pytest.raises(ValidationError, match="release_id"):
        VariableProvenance.model_validate({**provenance.model_dump(), "release_id": "release-b"})
    with pytest.raises(ValidationError, match="available_at"):
        VariableProvenance.model_validate(
            {
                **provenance.model_dump(),
                "available_at": datetime(2026, 7, 7),
            }
        )


def test_provenance_timestamps_normalize_to_utc_and_fields_are_required() -> None:
    offset = timezone(timedelta(hours=-4))
    source = SourceRef(
        source_key="asset",
        name="asset",
        url="https://example.invalid/",
        release_date=None,
        release_missingness_reason="publisher_date_not_available",
        retrieved_at=datetime(2026, 7, 7, 8, tzinfo=offset),
        sha256="a" * 64,
        vintage="v1",
        provisional=False,
    )
    assert source.retrieved_at == datetime(2026, 7, 7, 12, tzinfo=UTC)
    payload = _provenance(_manifest(IdentityRegistry())).model_dump()
    payload["available_at"] = datetime(2026, 7, 8, 11, tzinfo=offset)
    assert VariableProvenance.model_validate(payload).available_at == datetime(
        2026, 7, 8, 15, tzinfo=UTC
    )

    for field in ("source_key", "release_date", "release_missingness_reason"):
        source_payload = source.model_dump()
        del source_payload[field]
        with pytest.raises(ValidationError, match=field):
            SourceRef.model_validate(source_payload)
    for field in ("release_id", "available_at"):
        provenance_payload = payload.copy()
        del provenance_payload[field]
        with pytest.raises(ValidationError, match=field):
            VariableProvenance.model_validate(provenance_payload)


def test_legacy_not_missing_release_reason_is_rejected() -> None:
    payload = _provenance(_manifest(IdentityRegistry())).sources[1].model_dump()
    payload["release_missingness_reason"] = "not_missing"
    with pytest.raises(ValidationError, match="release_missingness_reason"):
        SourceRef.model_validate(payload)


def test_provenance_observation_interval_is_strictly_half_open() -> None:
    payload = _provenance(_manifest(IdentityRegistry())).model_dump()
    payload["observation_end"] = payload["observation_start"]

    with pytest.raises(ValidationError, match="observation"):
        VariableProvenance.model_validate(payload)


def test_assignment_derives_manifest_and_primary_fact_metadata() -> None:
    assigned, manifest, _ = _assigned()

    assert tuple(assigned.columns[: len(ENVELOPE_COLUMNS)]) == ENVELOPE_COLUMNS
    assert set(assigned["release_id"]) == {manifest.release_id.value}
    assert set(assigned["available_at"]) == {manifest.available_at}
    assert set(assigned["source_key"]) == {"workbook"}
    assert set(assigned["released_at"]) == {date(2026, 6, 30)}
    assert assigned["release_missingness_reason"].isna().all()
    assert assigned.loc[0, "fact_id"].startswith("cldemopde:fact:v1:")
    assert assigned.loc[0, "observation_id"].startswith("cldemopde:observation:v1:")


@pytest.mark.parametrize("identifier", ["fact_id", "observation_id", "release_id"])
def test_assignment_rejects_every_caller_supplied_identity(identifier: str) -> None:
    payload = _payload()
    payload[identifier] = "caller-controlled"
    registry = IdentityRegistry()

    with pytest.raises(DataContractError, match=r"ID|identity"):
        assign_observation_ids(
            payload,
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=("variable", "period_start", "period_end"),
            source_identity={"publisher": "fixture"},
            manifest=_manifest(registry),
            registry=registry,
        )


@pytest.mark.parametrize(
    "domain_key",
    [(), ("variable", "variable"), ("missing_column",)],
)
def test_assignment_rejects_invalid_domain_keys(
    domain_key: tuple[str, ...],
) -> None:
    registry = IdentityRegistry()
    with pytest.raises(DataContractError, match=r"domain_key|key columns"):
        assign_observation_ids(
            _payload(),
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=domain_key,
            source_identity={"publisher": "fixture"},
            manifest=_manifest(registry),
            registry=registry,
        )


def test_assignment_rejects_domain_mismatch_and_noncanonical_key_values() -> None:
    registry = IdentityRegistry()
    payload = _payload()
    payload.loc[0, "domain"] = "environmental"
    with pytest.raises(DataContractError, match="domain"):
        assign_observation_ids(
            payload,
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=("variable",),
            source_identity={"publisher": "fixture"},
            manifest=_manifest(registry),
            registry=registry,
        )

    payload = _payload()
    payload["domain_specific_key"] = float("nan")
    with pytest.raises(DataContractError, match="key"):
        assign_observation_ids(
            payload,
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=("domain_specific_key",),
            source_identity={"publisher": "fixture"},
            manifest=_manifest(registry),
            registry=registry,
        )

    payload = _payload()
    payload["domain_specific_key"] = pd.Series([object()], dtype=object)
    with pytest.raises(
        DataContractError,
        match=r"key|Unsupported|value type|canonical scalar",
    ):
        assign_observation_ids(
            payload,
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=("domain_specific_key",),
            source_identity={"publisher": "fixture"},
            manifest=_manifest(registry),
            registry=registry,
        )


def test_assignment_failure_never_mutates_input_or_registry() -> None:
    registry = IdentityRegistry()
    manifest = _manifest(registry)
    payload = _payload()
    original = payload.copy(deep=True)
    before = registry.snapshot()

    with pytest.raises(
        DataContractError,
        match=r"Unsupported|source_identity|value type",
    ):
        assign_observation_ids(
            payload,
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=("variable",),
            source_identity={"unsupported": object()},
            manifest=manifest,
            registry=registry,
        )

    pd.testing.assert_frame_equal(payload, original)
    assert registry.snapshot() == before

    invalid = _payload()
    invalid.loc[0, "semantic_kind"] = "not_a_kind"
    invalid_original = invalid.copy(deep=True)
    with pytest.raises(DataContractError, match="envelope"):
        assign_observation_ids(
            invalid,
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=("variable",),
            source_identity={"publisher": "fixture"},
            manifest=manifest,
            registry=registry,
        )
    pd.testing.assert_frame_equal(invalid, invalid_original)
    assert registry.snapshot() == before


@pytest.mark.parametrize(
    "column",
    [
        "observation_role",
        "parity_scope",
        "semantic_kind",
        "unit",
        "population_basis",
    ],
)
def test_assignment_sanitizes_unhashable_preidentity_values_atomically(
    column: str,
) -> None:
    registry = IdentityRegistry()
    manifest = _manifest(registry)
    payload = _payload()
    payload.at[0, column] = ["caller", "controlled"]
    expected = payload.copy(deep=True)
    before = registry.snapshot()

    with pytest.raises(DataContractError, match="envelope"):
        assign_observation_ids(
            payload,
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=("variable",),
            source_identity={"publisher": "fixture"},
            manifest=manifest,
            registry=registry,
        )

    pd.testing.assert_frame_equal(payload, expected)
    assert registry.snapshot() == before


def test_preidentity_error_never_discloses_caller_index() -> None:
    registry = IdentityRegistry()
    payload = _payload()
    payload.index = pd.Index(["TOP_SECRET_CALLER_INDEX"])
    payload.loc[:, "semantic_kind"] = "invalid"

    with pytest.raises(DataContractError, match="envelope") as captured:
        assign_observation_ids(
            payload,
            domain=ObservationDomain.DEMOGRAPHIC,
            domain_key=("variable",),
            source_identity={"publisher": "fixture"},
            manifest=_manifest(registry),
            registry=registry,
        )

    assert "TOP_SECRET_CALLER_INDEX" not in str(captured.value)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("domain", "not_a_domain", "domain"),
        ("observation_role", "not_a_role", "observation_role"),
        ("parity_scope", "not_a_scope", "parity_scope"),
        ("period_end", date(2024, 1, 1), "period"),
        ("available_at", datetime(2026, 7, 8), "available_at"),
        ("fact_id", "fact", "fact_id"),
        ("observation_id", "observation", "observation_id"),
        ("release_id", "release", "release_id"),
    ],
)
def test_envelope_rejects_invalid_closed_or_versioned_fields(
    column: str,
    value: object,
    match: str,
) -> None:
    assigned, _, _ = _assigned()
    assigned.loc[0, column] = value

    with pytest.raises(SchemaErrors, match=match):
        ObservationEnvelopeSchema.validate(assigned, lazy=True)


def test_envelope_requires_paired_transformations_and_explicit_missingness() -> None:
    assigned, _, _ = _assigned()
    assigned.loc[0, "transformation_version"] = None
    with pytest.raises(SchemaErrors, match="transformation"):
        ObservationEnvelopeSchema.validate(assigned, lazy=True)

    assigned, _, _ = _assigned()
    assigned.loc[0, "value"] = float("nan")
    assigned.loc[0, "missingness_reason"] = "not_applicable_observed"
    with pytest.raises(SchemaErrors, match="missing"):
        ObservationEnvelopeSchema.validate(assigned, lazy=True)

    assigned.loc[0, "missingness_reason"] = "publisher_cell_blank"
    ObservationEnvelopeSchema.validate(assigned, lazy=True)


def test_envelope_release_date_and_reason_are_exact_xor() -> None:
    assigned, _, _ = _assigned()
    assigned.loc[0, "release_missingness_reason"] = "publisher_date_not_available"
    with pytest.raises(SchemaErrors, match="release"):
        ObservationEnvelopeSchema.validate(assigned, lazy=True)


def test_batch_uses_primary_fact_date_but_preserves_sidecar_date() -> None:
    observations, manifest, identities = _assigned()
    batch = NormalizedDomainBatch.create(
        manifest=manifest,
        observations=observations,
        provenance=_provenance(manifest),
        identities=identities,
    )

    assert set(batch.observations["released_at"]) == {date(2026, 6, 30)}
    sources = {source.source_key: source for source in batch.provenance.sources}
    assert sources["workbook"].release_date == date(2026, 6, 30)
    assert sources["entity_registry"].release_date is None
    assert sources["entity_registry"].release_missingness_reason is not None


def test_batch_revalidates_and_owns_provenance_collections() -> None:
    observations, manifest, identities = _assigned()
    original = _provenance(manifest)
    forged = original.model_copy(
        update={
            "sources": list(original.sources),
            "dimensions": list(original.dimensions),
        }
    )

    batch = NormalizedDomainBatch.create(
        manifest=manifest,
        observations=observations,
        provenance=forged,
        identities=identities,
    )

    assert isinstance(batch.provenance.sources, tuple)
    assert isinstance(batch.provenance.dimensions, tuple)
    assert batch.provenance is not forged


def test_batch_sanitizes_invalid_provenance_source_members() -> None:
    observations, manifest, identities = _assigned()
    forged = _provenance(manifest).model_copy(update={"sources": (object(),)})

    with pytest.raises(DataContractError, match="provenance"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=forged,
            identities=identities,
        )


@pytest.mark.parametrize(
    "update",
    [
        {"source_key": "INVALID SOURCE"},
        {
            "release_date": None,
            "release_missingness_reason": None,
        },
        {"retrieved_at": "not-a-datetime"},
        {"retrieved_at": datetime(2026, 7, 8)},
    ],
)
def test_batch_defensively_revalidates_nested_source_refs(
    update: dict[str, object],
) -> None:
    observations, manifest, identities = _assigned()
    provenance = _provenance(manifest)
    forged_source = provenance.sources[0].model_copy(update=update)
    forged = provenance.model_copy(update={"sources": (forged_source, *provenance.sources[1:])})

    with pytest.raises(DataContractError, match="provenance"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=forged,
            identities=identities,
        )


def test_batch_revalidates_model_constructed_provenance_tree() -> None:
    observations, manifest, identities = _assigned()
    valid = _provenance(manifest)
    source_payload = valid.sources[0].model_dump()
    source_payload["source_key"] = "INVALID SOURCE"
    forged_source = SourceRef.model_construct(**source_payload)
    provenance_payload = valid.model_dump()
    provenance_payload["sources"] = [forged_source, *valid.sources[1:]]
    provenance_payload["dimensions"] = list(valid.dimensions)
    forged = VariableProvenance.model_construct(**provenance_payload)

    with pytest.raises(DataContractError, match="provenance"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=forged,
            identities=identities,
        )


def test_batch_provenance_error_chain_never_retains_secret_values() -> None:
    observations, manifest, identities = _assigned()
    secret = "TOP_S" + "ECRET_PROVENANCE_VALUE"
    valid = _provenance(manifest)
    forged_source = valid.sources[0].model_copy(update={"source_key": secret})
    forged = valid.model_copy(update={"sources": (forged_source, *valid.sources[1:])})

    with pytest.raises(DataContractError, match="provenance") as captured:
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=forged,
            identities=identities,
        )

    current: BaseException | None = captured.value
    while current is not None:
        assert secret not in str(current)
        assert current.__cause__ is None
        next_context = current.__context__
        assert next_context is None
        current = next_context


def test_batch_requires_each_source_knowledge_time_to_match_manifest() -> None:
    observations, manifest, identities = _assigned()
    provenance = _provenance(manifest)
    forged_source = provenance.sources[0].model_copy(
        update={"retrieved_at": datetime(2026, 7, 9, tzinfo=UTC)}
    )
    forged = provenance.model_copy(update={"sources": (forged_source, *provenance.sources[1:])})

    with pytest.raises(DataContractError, match=r"availability|knowledge"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=forged,
            identities=identities,
        )


def test_batch_owns_validated_frame_and_returns_deep_copies() -> None:
    observations, manifest, identities = _assigned()
    batch = NormalizedDomainBatch.create(
        manifest=manifest,
        observations=observations,
        provenance=_provenance(manifest),
        identities=identities,
    )
    expected = batch.observations
    observations.loc[:, "value"] = -999.0
    leaked = batch.observations
    leaked.loc[:, "value"] = 999.0

    pd.testing.assert_frame_equal(batch.observations, expected)
    assert batch.domain is ObservationDomain.DEMOGRAPHIC


def test_batch_requires_exact_manifest_provenance_and_identity_records() -> None:
    observations, manifest, identities = _assigned()
    wrong_provenance = _provenance(manifest).model_copy(
        update={
            "sources": (
                _provenance(manifest).sources[0].model_copy(update={"sha256": "c" * 64}),
                _provenance(manifest).sources[1],
            )
        }
    )
    with pytest.raises(DataContractError, match="manifest"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=wrong_provenance,
            identities=identities,
        )


@pytest.mark.parametrize("field", ["release_id", "available_at"])
def test_batch_rejects_provenance_release_metadata_mismatch(field: str) -> None:
    observations, manifest, identities = _assigned()
    update: dict[str, object] = {
        "release_id": "cldemopde:release:v1:" + "c" * 64,
        "available_at": datetime(2026, 7, 9, tzinfo=UTC),
    }
    provenance = _provenance(manifest).model_copy(update={field: update[field]})
    with pytest.raises(DataContractError, match="provenance"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=provenance,
            identities=identities,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "date"])
def test_batch_rejects_nonexact_provenance_sources(mutation: str) -> None:
    observations, manifest, identities = _assigned()
    provenance = _provenance(manifest)
    sources = list(provenance.sources)
    if mutation == "missing":
        sources.pop()
    elif mutation == "extra":
        sources.append(sources[0].model_copy(update={"source_key": "extra", "sha256": "c" * 64}))
    elif mutation == "duplicate":
        sources.append(sources[0])
    else:
        workbook_index = next(
            index for index, source in enumerate(sources) if source.source_key == "workbook"
        )
        sources[workbook_index] = sources[workbook_index].model_copy(
            update={
                "release_date": None,
                "release_missingness_reason": (
                    ReleaseMissingnessReason.PUBLISHER_DATE_NOT_AVAILABLE
                ),
            }
        )
    changed = provenance.model_copy(update={"sources": tuple(sources)})
    with pytest.raises(DataContractError, match=r"provenance|manifest|source"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=changed,
            identities=identities,
        )


@pytest.mark.parametrize("mutation", ["domain", "release_id", "available_at"])
def test_batch_rejects_mixed_row_release_discriminators(mutation: str) -> None:
    observations, manifest, identities = _assigned()
    observations = pd.concat([observations, observations], ignore_index=True)
    if mutation == "domain":
        observations.loc[1, "domain"] = "environmental"
    elif mutation == "release_id":
        observations.loc[1, "release_id"] = "cldemopde:release:v1:" + "c" * 64
    else:
        observations.loc[1, "available_at"] = datetime(2026, 7, 9, tzinfo=UTC)
    with pytest.raises(
        DataContractError,
        match=r"batch|release|availability|discriminator",
    ):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=_provenance(manifest),
            identities=identities,
        )


@pytest.mark.parametrize("field", ["source_key", "released_at", "reason"])
def test_batch_rejects_envelope_primary_fact_metadata_mismatch(field: str) -> None:
    observations, manifest, identities = _assigned()
    if field == "source_key":
        observations.loc[0, "source_key"] = "entity_registry"
    elif field == "released_at":
        observations.loc[0, "released_at"] = date(2026, 7, 1)
    else:
        observations.loc[0, "released_at"] = None
        observations.loc[0, "release_missingness_reason"] = "publisher_date_not_available"
    with pytest.raises(DataContractError, match="primary fact"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=_provenance(manifest),
            identities=identities,
        )


@pytest.mark.parametrize("identity_kind", ["fact", "observation", "release"])
def test_batch_requires_every_exact_canonical_identity_record(
    identity_kind: str,
) -> None:
    observations, manifest, identities = _assigned()
    value = {
        "fact": observations.loc[0, "fact_id"],
        "observation": observations.loc[0, "observation_id"],
        "release": manifest.release_id.value,
    }[identity_kind]
    missing = IdentityRegistrySnapshot(
        tuple(record for record in identities.records if record.value != value)
    )
    with pytest.raises(DataContractError, match="missing"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=_provenance(manifest),
            identities=missing,
        )

    wrong = IdentityRegistrySnapshot(
        tuple(
            CanonicalIdentityRecord(record.value, b"not canonical")
            if record.value == value
            else record
            for record in identities.records
        )
    )
    with pytest.raises(DataContractError, match=r"Canonical|canonical"):
        NormalizedDomainBatch.create(
            manifest=manifest,
            observations=observations,
            provenance=_provenance(manifest),
            identities=wrong,
        )


def test_envelope_has_no_domain_specific_columns() -> None:
    assert set(ENVELOPE_COLUMNS).isdisjoint(
        {
            "age_lower",
            "station",
            "facility",
            "source_series_id",
            "regulatory_basis",
            "mortality_table_id",
            "pension_type",
        }
    )
