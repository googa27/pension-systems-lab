from __future__ import annotations

import hashlib
import warnings
from datetime import UTC, date, datetime, timedelta, timezone

import pandas as pd
import pytest

from chile_demographic_pde.core.errors import (
    AmbiguousRevisionError,
    DataContractError,
    IdentityCollisionError,
)
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.data.domain_schemas import (
    DEMOGRAPHIC_OBSERVATION_KEY,
    MARKET_QUOTE_KEY,
    DemographicObservationBatch,
    MarketQuoteBatch,
    validate_demographic_observations,
    validate_market_quotes,
)
from chile_demographic_pde.data.envelope import (
    NormalizedDomainBatch,
    assign_observation_ids,
)
from chile_demographic_pde.data.identity import (
    CanonicalIdentityRecord,
    IdentityRegistry,
    IdentityRegistrySnapshot,
)
from chile_demographic_pde.data.model_input import (
    CompiledDomainSelection,
    DomainObservationHistory,
    latest_as_of,
)
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
)
from chile_demographic_pde.data.roles import ObservationDomain


def _release_context(
    *,
    tag: str,
    source_key: str,
    available_at: datetime,
    dimensions: tuple[str, ...],
) -> tuple[ReleaseManifest, VariableProvenance, IdentityRegistry]:
    digest = hashlib.sha256(tag.encode()).hexdigest()
    registry = IdentityRegistry()
    released_at = date(2024, 12, 31)
    manifest = ReleaseManifest.create(
        schema_version="history-fixture-v1",
        assets=(
            RequiredReleaseAsset(
                source_key=source_key,
                sha256=digest,
                available_at=available_at,
                released_at=released_at,
                release_missingness_reason=None,
            ),
        ),
        primary_fact_source_key=source_key,
        identity_registry=registry,
    )
    source = SourceRef(
        source_key=source_key,
        name=f"History fixture {tag}",
        url="https://example.invalid/history",
        release_date=released_at,
        release_missingness_reason=None,
        retrieved_at=available_at,
        sha256=digest,
        vintage=tag,
        provisional=False,
    )
    provenance = VariableProvenance(
        sources=(source,),
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=date(2024, 1, 1),
        observation_end=date(2025, 1, 1),
        dimensions=dimensions,
        aggregation_rules=("preserve fixture observation",),
        missingness_reason="not_applicable_observed",
    )
    return manifest, provenance, registry


def _demographic_batch(
    *,
    tag: str,
    available_at: datetime,
    facts: tuple[tuple[int, float], ...] = ((40, 10.0),),
) -> DemographicObservationBatch:
    manifest, provenance, registry = _release_context(
        tag=tag,
        source_key="demographic_history",
        available_at=available_at,
        dimensions=("age", "sex", "period"),
    )
    records = [
        {
            "domain": "demographic",
            "variable": "deaths",
            "value": value,
            "semantic_kind": "count",
            "unit": "event",
            "population_basis": "resident_estimate",
            "observation_role": "observed_fact",
            "parity_scope": "not_applicable",
            "period_start": date(2024, 1, 1),
            "period_end": date(2025, 1, 1),
            "source_key": manifest.primary_fact_source_key,
            "vintage": tag,
            "released_at": manifest.primary_fact_asset.released_at,
            "release_missingness_reason": None,
            "available_at": manifest.available_at,
            "provisional": False,
            "transformation_id": None,
            "transformation_version": None,
            "aggregation_rule": "published annual count",
            "missingness_reason": "not_applicable_observed",
            "source_status_code": None,
            "age_lower": float(age),
            "age_upper": float(age + 1),
            "age_open": False,
            "sex": "female",
            "region": "CL",
            "cause": "all",
            "date_basis": "occurrence_interval",
            "age_role": "decedent",
            "age_measure_unit": "completed_years",
            "age_quantity": float(age),
            "sex_role": "decedent",
            "commune": None,
            "geography_basis": "usual_residence",
            "geography_vintage": None,
            "cause_code_system": None,
            "cause_role": "all_cause",
            "external_cause": None,
            "registration_start": None,
            "registration_end": None,
            "birth_order": None,
            "nationality": None,
            "nationality_role": None,
            "country_of_birth": None,
            "education_level": None,
            "residence_five_years_ago": None,
            "migration_status": None,
        }
        for age, value in facts
    ]
    assigned = assign_observation_ids(
        pd.DataFrame.from_records(records),
        domain=ObservationDomain.DEMOGRAPHIC,
        domain_key=DEMOGRAPHIC_OBSERVATION_KEY,
        source_identity={
            "publisher": "history_fixture",
            "dataset": "demographic_history",
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


def _market_quote_batch(
    *,
    tag: str,
    available_at: datetime,
) -> MarketQuoteBatch:
    manifest, provenance, registry = _release_context(
        tag=tag,
        source_key="market_history",
        available_at=available_at,
        dimensions=("tenor", "currency", "period"),
    )
    payload = pd.DataFrame.from_records(
        [
            {
                "domain": "market_quote",
                "variable": "official_benchmark_yield",
                "value": 0.035,
                "semantic_kind": "rate",
                "unit": "1 / year",
                "population_basis": "not_applicable",
                "observation_role": "market_quote",
                "parity_scope": "not_applicable",
                "period_start": date(2024, 1, 1),
                "period_end": date(2024, 1, 2),
                "source_key": manifest.primary_fact_source_key,
                "vintage": tag,
                "released_at": manifest.primary_fact_asset.released_at,
                "release_missingness_reason": None,
                "available_at": manifest.available_at,
                "provisional": False,
                "transformation_id": None,
                "transformation_version": None,
                "aggregation_rule": "published quote",
                "missingness_reason": "not_applicable_observed",
                "source_status_code": None,
                "source_series_id": "BCCH_FIXTURE",
                "quote_type": "benchmark_yield",
                "tenor_years": 10.0,
                "currency": "CLP",
                "indexation": "nominal",
                "instrument_type": "bcp_benchmark_bond",
                "compounding": "annual_compounded",
                "day_count": None,
                "date_basis": "published_observation_date",
            }
        ]
    )
    assigned = assign_observation_ids(
        payload,
        domain=ObservationDomain.MARKET_QUOTE,
        domain_key=MARKET_QUOTE_KEY,
        source_identity={
            "publisher": "history_fixture",
            "dataset": "market_history",
        },
        manifest=manifest,
        registry=registry,
    )
    validated = validate_market_quotes(assigned)
    return NormalizedDomainBatch.create(
        manifest=manifest,
        observations=validated,
        provenance=provenance,
        identities=registry.snapshot(),
    )


def _observation_ids(batch: DemographicObservationBatch) -> tuple[str, ...]:
    return tuple(batch.observations["observation_id"])


def test_history_and_selection_are_factory_only() -> None:
    with pytest.raises(TypeError):
        DomainObservationHistory()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        CompiledDomainSelection()  # type: ignore[call-arg]


def test_history_rejects_empty_invalid_and_mixed_domain_inputs() -> None:
    demographic = _demographic_batch(
        tag="demographic",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    market = _market_quote_batch(
        tag="market",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    with pytest.raises(DataContractError, match=r"nonempty|batch"):
        DomainObservationHistory.create(())
    with pytest.raises(DataContractError, match=r"sequence|batch"):
        DomainObservationHistory.create("not batches")  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="batch"):
        DomainObservationHistory.create((object(),))  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="domain"):
        DomainObservationHistory.create((demographic, market))
    with pytest.raises(DataContractError, match="release"):
        DomainObservationHistory.create((demographic, demographic))


def test_history_revalidates_the_exact_domain_schema() -> None:
    batch = _demographic_batch(
        tag="strict-schema",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    observations = batch.observations.assign(foreign_domain_column="invalid")
    forged = NormalizedDomainBatch.create(
        manifest=batch.manifest,
        observations=observations,
        provenance=batch.provenance,
        identities=batch.identities,
    )

    with pytest.raises(DataContractError, match=r"schema|domain"):
        DomainObservationHistory.create((forged,))


def test_history_order_is_deterministic_across_input_permutations() -> None:
    same_time = datetime(2025, 1, 2, tzinfo=UTC)
    early = _demographic_batch(
        tag="early",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    tied_a = _demographic_batch(tag="tied-a", available_at=same_time, facts=((41, 2),))
    tied_b = _demographic_batch(tag="tied-b", available_at=same_time, facts=((42, 3),))

    first = DomainObservationHistory.create((tied_b, early, tied_a))
    second = DomainObservationHistory.create((tied_a, tied_b, early))

    expected_release_ids = tuple(
        batch.manifest.release_id.value
        for batch in sorted(
            (early, tied_a, tied_b),
            key=lambda item: (
                item.manifest.available_at,
                item.manifest.release_id.value,
            ),
        )
    )
    assert tuple(batch.manifest.release_id.value for batch in first.batches) == expected_release_ids
    assert (
        tuple(batch.manifest.release_id.value for batch in second.batches) == expected_release_ids
    )
    assert first.domain is ObservationDomain.DEMOGRAPHIC


def test_latest_as_of_requires_aware_cutoff_and_is_inclusive() -> None:
    available = datetime(2025, 1, 1, tzinfo=UTC)
    batch = _demographic_batch(tag="inclusive", available_at=available)
    history = DomainObservationHistory.create((batch,))

    with pytest.raises(DataContractError, match=r"aware|timezone"):
        latest_as_of(history, as_of=datetime(2025, 1, 1))
    with pytest.raises(DataContractError, match="datetime"):
        latest_as_of(history, as_of="2025-01-01")  # type: ignore[arg-type]

    equivalent_offset = datetime(
        2024,
        12,
        31,
        21,
        tzinfo=timezone(-timedelta(hours=3)),
    )
    selected = latest_as_of(history, as_of=equivalent_offset)

    assert len(selected.observations) == 1
    assert selected.as_of == available
    assert selected.as_of.tzinfo is UTC


def test_latest_as_of_selects_each_stable_fact_independently() -> None:
    first = _demographic_batch(
        tag="first",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
        facts=((40, 10), (41, 20)),
    )
    revision = _demographic_batch(
        tag="revision",
        available_at=datetime(2025, 2, 1, tzinfo=UTC),
        facts=((40, 11),),
    )
    history = DomainObservationHistory.create((revision, first))

    selected = latest_as_of(
        history,
        as_of=datetime(2025, 2, 1, tzinfo=UTC),
    )
    observations = selected.observations.sort_values("age_lower")

    assert observations["age_lower"].tolist() == [40.0, 41.0]
    assert observations["value"].tolist() == [11.0, 20.0]
    assert not observations["fact_id"].duplicated().any()
    first_by_age = first.observations.set_index("age_lower")
    revision_by_age = revision.observations.set_index("age_lower")
    selected_by_age = observations.set_index("age_lower")
    assert selected_by_age.loc[40.0, "fact_id"] == revision_by_age.loc[40.0, "fact_id"]
    assert selected_by_age.loc[41.0, "observation_id"] == first_by_age.loc[41.0, "observation_id"]


def test_tied_distinct_revisions_raise_without_row_order_tiebreaker() -> None:
    available = datetime(2025, 2, 1, tzinfo=UTC)
    left = _demographic_batch(
        tag="tie-left",
        available_at=available,
        facts=((40, 10),),
    )
    right = _demographic_batch(
        tag="tie-right",
        available_at=available,
        facts=((40, 11),),
    )

    for batches in ((left, right), (right, left)):
        history = DomainObservationHistory.create(batches)
        with pytest.raises(AmbiguousRevisionError):
            latest_as_of(history, as_of=available)


def test_selection_contains_exact_contributing_evidence_and_identities() -> None:
    first = _demographic_batch(
        tag="evidence-first",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
        facts=((40, 10), (41, 20)),
    )
    revision = _demographic_batch(
        tag="evidence-revision",
        available_at=datetime(2025, 2, 1, tzinfo=UTC),
        facts=((40, 11),),
    )
    selected = latest_as_of(
        DomainObservationHistory.create((first, revision)),
        as_of=datetime(2025, 3, 1, tzinfo=UTC),
    )
    observations = selected.observations
    release_ids = set(observations["release_id"])

    assert {manifest.release_id.value for manifest in selected.manifests} == (release_ids)
    assert {item.release_id for item in selected.provenance} == release_ids
    expected_identity_ids = (
        release_ids | set(observations["fact_id"]) | set(observations["observation_id"])
    )
    assert {record.value for record in selected.identities.records} == (expected_identity_ids)


def test_empty_as_of_retains_typed_shape_domain_and_cutoff() -> None:
    batch = _demographic_batch(
        tag="future",
        available_at=datetime(2025, 2, 1, tzinfo=UTC),
    )
    cutoff = datetime(2025, 1, 1, tzinfo=UTC)
    selected = latest_as_of(
        DomainObservationHistory.create((batch,)),
        as_of=cutoff,
    )
    template = batch.observations.iloc[0:0]
    observations = selected.observations

    assert observations.empty
    assert tuple(observations.columns) == tuple(template.columns)
    assert observations.dtypes.equals(template.dtypes)
    assert selected.domain is ObservationDomain.DEMOGRAPHIC
    assert selected.as_of == cutoff
    assert selected.manifests == ()
    assert selected.provenance == ()
    assert selected.identities.records == ()
    assert not isinstance(selected, NormalizedDomainBatch)


def test_history_rejects_cross_snapshot_identity_collisions() -> None:
    first = _demographic_batch(
        tag="collision-first",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
        facts=((40, 10),),
    )
    second = _demographic_batch(
        tag="collision-second",
        available_at=datetime(2025, 2, 1, tzinfo=UTC),
        facts=((41, 20),),
    )
    foreign_id = _observation_ids(first)[0]
    poisoned_snapshot = IdentityRegistrySnapshot(
        (
            *second.identities.records,
            CanonicalIdentityRecord(foreign_id, b"different canonical bytes"),
        )
    )
    poisoned = NormalizedDomainBatch.create(
        manifest=second.manifest,
        observations=second.observations,
        provenance=second.provenance,
        identities=poisoned_snapshot,
    )

    with pytest.raises(IdentityCollisionError):
        DomainObservationHistory.create((first, poisoned))


def test_history_rejects_reused_observation_id_with_different_row() -> None:
    first = _demographic_batch(
        tag="reused-first",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
        facts=((40, 10),),
    )
    second = _demographic_batch(
        tag="reused-second",
        available_at=datetime(2025, 2, 1, tzinfo=UTC),
        facts=((40, 11),),
    )
    reused_id = _observation_ids(first)[0]
    reused_record = first.identities.require(reused_id)
    poisoned_snapshot = IdentityRegistrySnapshot((*second.identities.records, reused_record))
    poisoned_observations = second.observations
    poisoned_observations.loc[:, "observation_id"] = reused_id
    poisoned = NormalizedDomainBatch.create(
        manifest=second.manifest,
        observations=poisoned_observations,
        provenance=second.provenance,
        identities=poisoned_snapshot,
    )

    with pytest.raises(DataContractError, match="observation"):
        DomainObservationHistory.create((first, poisoned))


def test_history_and_selection_do_not_leak_mutable_frames() -> None:
    batch = _demographic_batch(
        tag="copy-safe",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    history = DomainObservationHistory.create((batch,))
    selected = latest_as_of(
        history,
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
    )
    expected = selected.observations

    leaked_history = history.batches[0].observations
    leaked_history.loc[:, "value"] = -1.0
    leaked_selection = selected.observations
    leaked_selection.loc[:, "value"] = -2.0

    pd.testing.assert_frame_equal(selected.observations, expected)
    assert history.batches[0].observations["value"].tolist() == [10.0]


def test_history_equality_uses_object_identity_without_comparing_frames() -> None:
    batch = _demographic_batch(
        tag="history-identity-equality",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    first = DomainObservationHistory.create((batch,))
    second = DomainObservationHistory.create((batch,))

    assert first == first
    assert first != second


def test_selection_equality_uses_object_identity_without_comparing_frames() -> None:
    available_at = datetime(2025, 1, 1, tzinfo=UTC)
    batch = _demographic_batch(
        tag="selection-identity-equality",
        available_at=available_at,
    )
    history = DomainObservationHistory.create((batch,))
    first = latest_as_of(history, as_of=available_at)
    second = latest_as_of(history, as_of=available_at)

    assert first == first
    assert first != second


def test_latest_as_of_emits_no_pandas_concat_deprecation_warning() -> None:
    first = _demographic_batch(
        tag="warning-first",
        available_at=datetime(2025, 1, 1, tzinfo=UTC),
        facts=((40, 10), (41, 20)),
    )
    revision = _demographic_batch(
        tag="warning-revision",
        available_at=datetime(2025, 2, 1, tzinfo=UTC),
        facts=((40, 11),),
    )
    history = DomainObservationHistory.create((first, revision))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        selected = latest_as_of(
            history,
            as_of=datetime(2025, 2, 1, tzinfo=UTC),
        )

    assert len(selected) == 2
