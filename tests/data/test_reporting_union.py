from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, date, datetime
from enum import Enum
from types import MappingProxyType
from typing import cast

import pandas as pd
import pytest
from domain_fixtures import (
    demographic_frame,
    environmental_frame,
    health_system_frame,
    market_quote_frame,
    pension_product_frame,
    regulatory_mortality_frame,
    regulatory_rate_frame,
)
from pandera.errors import SchemaErrors

import chile_demographic_pde.data.reporting_union as reporting_union
from chile_demographic_pde.core.errors import (
    DataContractError,
    ReportingProjectionInputError,
)
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.data.domain_schemas import (
    DEMOGRAPHIC_OBSERVATION_KEY,
    ENVIRONMENTAL_OBSERVATION_KEY,
    HEALTH_SYSTEM_OBSERVATION_KEY,
    MARKET_QUOTE_KEY,
    PENSION_PRODUCT_STATISTIC_KEY,
    REGULATORY_MORTALITY_KEY,
    REGULATORY_RATE_KEY,
    DemographicObservationSchema,
    EnvironmentalObservationSchema,
    HealthSystemObservationSchema,
    MarketQuoteSchema,
    PensionProductStatisticSchema,
    RegulatoryMortalityTableObservationSchema,
    RegulatoryRateSchema,
    validate_demographic_observations,
    validate_environmental_observations,
    validate_health_system_observations,
    validate_market_quotes,
    validate_pension_product_statistics,
    validate_regulatory_mortality,
    validate_regulatory_rates,
)
from chile_demographic_pde.data.envelope import (
    ENVELOPE_COLUMNS,
    NormalizedDomainBatch,
    assign_observation_ids,
)
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
)
from chile_demographic_pde.data.reporting_union import (
    REPORTING_DISPLAY_DIMENSIONS,
    REPORTING_LONG_COLUMNS,
    ReportingProjectionMetadata,
    project_long_observations,
    require_demographic_mortality_batch,
    require_domain_batch,
    require_market_quote_batch,
    require_regulatory_rate_batch,
)
from chile_demographic_pde.data.roles import ObservationDomain

type DomainBatch = NormalizedDomainBatch[pd.DataFrame]
type FrameFactory = Callable[..., pd.DataFrame]
type FrameValidator = Callable[[pd.DataFrame], pd.DataFrame]

_RELEASED_AT = date(2025, 6, 30)
_BASE_AVAILABLE_AT = datetime(2025, 7, 1, 12, tzinfo=UTC)
_ID_COLUMNS = ("fact_id", "observation_id", "release_id")


@dataclass(frozen=True, slots=True)
class _DomainSpec:
    domain: ObservationDomain
    factory: FrameFactory
    validator: FrameValidator
    key: tuple[str, ...]
    schema: object
    display_dimensions: tuple[str, ...]


_DOMAIN_SPECS = (
    _DomainSpec(
        ObservationDomain.DEMOGRAPHIC,
        demographic_frame,
        validate_demographic_observations,
        DEMOGRAPHIC_OBSERVATION_KEY,
        DemographicObservationSchema,
        (
            "age_lower",
            "age_upper",
            "age_open",
            "sex",
            "region",
            "commune",
            "cause",
        ),
    ),
    _DomainSpec(
        ObservationDomain.ENVIRONMENTAL,
        environmental_frame,
        validate_environmental_observations,
        ENVIRONMENTAL_OBSERVATION_KEY,
        EnvironmentalObservationSchema,
        (
            "station",
            "region",
            "commune",
            "pollutant",
            "environmental_measure",
        ),
    ),
    _DomainSpec(
        ObservationDomain.HEALTH_SYSTEM,
        health_system_frame,
        validate_health_system_observations,
        HEALTH_SYSTEM_OBSERVATION_KEY,
        HealthSystemObservationSchema,
        (
            "facility",
            "health_service",
            "region",
            "commune",
            "reporting_grain",
        ),
    ),
    _DomainSpec(
        ObservationDomain.MARKET_QUOTE,
        market_quote_frame,
        validate_market_quotes,
        MARKET_QUOTE_KEY,
        MarketQuoteSchema,
        (
            "source_series_id",
            "quote_type",
            "tenor_years",
            "currency",
            "indexation",
            "instrument_type",
        ),
    ),
    _DomainSpec(
        ObservationDomain.REGULATORY_RATE,
        regulatory_rate_frame,
        validate_regulatory_rates,
        REGULATORY_RATE_KEY,
        RegulatoryRateSchema,
        (
            "regulatory_basis",
            "regulatory_use",
            "quote_type",
            "effective_start",
            "effective_end",
            "maximum_effective_end",
            "tenor_years",
            "financial_entity_id",
            "financial_entity_type",
            "currency",
            "indexation",
            "instrument_type",
        ),
    ),
    _DomainSpec(
        ObservationDomain.REGULATORY_MORTALITY_TABLE,
        regulatory_mortality_frame,
        validate_regulatory_mortality,
        REGULATORY_MORTALITY_KEY,
        RegulatoryMortalityTableObservationSchema,
        (
            "regulatory_basis",
            "mortality_table_id",
            "age",
            "sex",
            "population_segment",
            "reference_year",
            "improvement_year",
            "effective_start",
            "effective_end",
            "maximum_effective_end",
        ),
    ),
    _DomainSpec(
        ObservationDomain.PENSION_PRODUCT_STATISTIC,
        pension_product_frame,
        validate_pension_product_statistics,
        PENSION_PRODUCT_STATISTIC_KEY,
        PensionProductStatisticSchema,
        (
            "financial_entity_id",
            "financial_entity_type",
            "pension_type",
            "modality",
            "offer_type",
            "intermediation_type",
            "contract_state",
            "aggregation_grain",
            "product_term_set_id",
        ),
    ),
)
_SPEC_BY_DOMAIN = {spec.domain: spec for spec in _DOMAIN_SPECS}


def _make_batch(
    spec: _DomainSpec,
    *,
    available_at: datetime = _BASE_AVAILABLE_AT,
    frame_overrides: dict[str, object] | None = None,
    release_suffix: str = "first",
) -> DomainBatch:
    frame = spec.factory(**(frame_overrides or {}))
    payload = frame.drop(columns=list(_ID_COLUMNS)).copy(deep=True)
    source_key = f"{spec.domain.value}_{release_suffix}"
    digest = hashlib.sha256(
        f"{spec.domain.value}:{release_suffix}:{available_at.isoformat()}".encode()
    ).hexdigest()
    registry = IdentityRegistry()
    manifest = ReleaseManifest.create(
        schema_version="reporting-union-fixture-v1",
        assets=(
            RequiredReleaseAsset(
                source_key=source_key,
                sha256=digest,
                available_at=available_at,
                released_at=_RELEASED_AT,
                release_missingness_reason=None,
            ),
        ),
        primary_fact_source_key=source_key,
        identity_registry=registry,
    )
    source = SourceRef(
        source_key=source_key,
        name=f"{spec.domain.value} reporting fixture",
        url=f"https://example.invalid/{source_key}",
        release_date=_RELEASED_AT,
        release_missingness_reason=None,
        retrieved_at=available_at,
        sha256=digest,
        vintage=cast(str, payload.iloc[0]["vintage"]),
        provisional=bool(payload.iloc[0]["provisional"]),
    )
    provenance = VariableProvenance(
        sources=(source,),
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=min(payload["period_start"]),
        observation_end=max(payload["period_end"]),
        dimensions=spec.display_dimensions,
        aggregation_rules=(cast(str, payload.iloc[0]["aggregation_rule"]),),
        missingness_reason=cast(str, payload.iloc[0]["missingness_reason"]),
    )
    assigned = assign_observation_ids(
        payload,
        domain=spec.domain,
        domain_key=spec.key,
        source_identity={
            "publisher": "reporting union fixture publisher",
            "dataset": spec.domain.value,
        },
        manifest=manifest,
        registry=registry,
    )
    validated = spec.validator(assigned)
    return NormalizedDomainBatch.create(
        manifest=manifest,
        observations=validated,
        provenance=provenance,
        identities=registry.snapshot(),
    )


def _make_batch_from_shared_manifest(
    spec: _DomainSpec,
    *,
    manifest: ReleaseManifest,
) -> DomainBatch:
    frame = spec.factory()
    payload = frame.drop(columns=list(_ID_COLUMNS)).copy(deep=True)
    registry = IdentityRegistry()
    registry.record(manifest.release_id)
    asset = manifest.primary_fact_asset
    source = SourceRef(
        source_key=asset.source_key,
        name="Shared reporting fixture release",
        url="https://example.invalid/shared-reporting-release",
        release_date=asset.released_at,
        release_missingness_reason=asset.release_missingness_reason,
        retrieved_at=manifest.available_at,
        sha256=asset.sha256,
        vintage=cast(str, payload.iloc[0]["vintage"]),
        provisional=bool(payload.iloc[0]["provisional"]),
    )
    provenance = VariableProvenance(
        sources=(source,),
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=min(payload["period_start"]),
        observation_end=max(payload["period_end"]),
        dimensions=spec.display_dimensions,
        aggregation_rules=(cast(str, payload.iloc[0]["aggregation_rule"]),),
        missingness_reason=cast(str, payload.iloc[0]["missingness_reason"]),
    )
    assigned = assign_observation_ids(
        payload,
        domain=spec.domain,
        domain_key=spec.key,
        source_identity={
            "publisher": "shared reporting fixture publisher",
            "dataset": spec.domain.value,
        },
        manifest=manifest,
        registry=registry,
    )
    return NormalizedDomainBatch.create(
        manifest=manifest,
        observations=spec.validator(assigned),
        provenance=provenance,
        identities=registry.snapshot(),
    )


@pytest.fixture
def all_domain_batches() -> tuple[DomainBatch, ...]:
    return tuple(_make_batch(spec) for spec in _DOMAIN_SPECS)


def test_reporting_contract_has_exact_columns_and_selected_dimensions() -> None:
    assert (
        *ENVELOPE_COLUMNS,
        "projected_domain",
        "display_dimensions",
    ) == REPORTING_LONG_COLUMNS
    assert {
        spec.domain: spec.display_dimensions for spec in _DOMAIN_SPECS
    } == REPORTING_DISPLAY_DIMENSIONS


def test_display_dimensions_public_contract_is_exact_and_immutable() -> None:
    dimensions = reporting_union.DISPLAY_DIMENSIONS

    assert isinstance(dimensions, MappingProxyType)
    assert dimensions == {spec.domain: spec.display_dimensions for spec in _DOMAIN_SPECS}
    with pytest.raises(TypeError):
        dimensions[ObservationDomain.DEMOGRAPHIC] = ()  # type: ignore[index]


def test_all_seven_domains_project_to_deterministic_compact_json(
    all_domain_batches: tuple[DomainBatch, ...],
) -> None:
    projection = project_long_observations(all_domain_batches)
    frame = projection.to_frame()

    assert tuple(frame.columns) == REPORTING_LONG_COLUMNS
    assert set(frame["projected_domain"]) == {domain.value for domain in ObservationDomain}
    assert frame["projected_domain"].equals(frame["domain"])
    assert (
        frame["display_dimensions"]
        .map(lambda value: isinstance(value, str) and "\n" not in value and ": " not in value)
        .all()
    )

    reversed_projection = project_long_observations(tuple(reversed(all_domain_batches)))
    pd.testing.assert_frame_equal(
        projection.to_frame(),
        reversed_projection.to_frame(),
    )


def test_display_json_preserves_explicit_null_dates_booleans_numbers_and_utf8() -> None:
    demographic = _make_batch(
        _SPEC_BY_DOMAIN[ObservationDomain.DEMOGRAPHIC],
        frame_overrides={
            "age_lower": 85.0,
            "age_upper": None,
            "age_open": True,
            "cause": "caída",
            "commune": None,
        },
    )
    regulatory = _make_batch(_SPEC_BY_DOMAIN[ObservationDomain.REGULATORY_RATE])
    frame = project_long_observations((regulatory, demographic)).to_frame()

    demographic_json = frame.loc[
        frame["projected_domain"].eq("demographic"),
        "display_dimensions",
    ].item()
    assert demographic_json == (
        '{"age_lower":85.0,"age_open":true,"age_upper":null,'
        '"cause":"caída","commune":null,"region":"CL","sex":"female"}'
    )
    regulatory_json = frame.loc[
        frame["projected_domain"].eq("regulatory_rate"),
        "display_dimensions",
    ].item()
    assert '"effective_end":"2026-04-01"' in regulatory_json
    assert '"effective_start":"2026-03-01"' in regulatory_json
    assert '"financial_entity_id":null' in regulatory_json
    assert '"tenor_years":10.0' in regulatory_json


def test_display_json_encodes_enum_values() -> None:
    class DisplayLabel(Enum):
        FEMALE = "female"

    assert reporting_union._json_scalar(DisplayLabel.FEMALE) == "female"


def test_projection_metadata_is_frozen_explicitly_lossy_and_auditable(
    all_domain_batches: tuple[DomainBatch, ...],
) -> None:
    projection = project_long_observations(all_domain_batches)
    metadata = projection.metadata
    expected_domains = tuple(sorted(ObservationDomain, key=lambda item: item.value))
    expected_selected = tuple(
        (domain.value, _SPEC_BY_DOMAIN[domain].display_dimensions) for domain in expected_domains
    )
    expected_omitted = tuple(
        (
            domain.value,
            tuple(
                column
                for column in _SPEC_BY_DOMAIN[domain].schema.to_schema().columns
                if column not in ENVELOPE_COLUMNS
                and column not in _SPEC_BY_DOMAIN[domain].display_dimensions
            ),
        )
        for domain in expected_domains
    )

    assert isinstance(metadata, ReportingProjectionMetadata)
    assert metadata.schema_version == "reporting-long-v1"
    assert metadata.lossy is True
    assert metadata.intended_use == "display_only"
    assert metadata.source_domains == expected_domains
    assert metadata.source_release_ids == tuple(
        sorted(batch.manifest.release_id.value for batch in all_domain_batches)
    )
    assert metadata.projected_columns == REPORTING_LONG_COLUMNS
    assert metadata.selected_dimensions == expected_selected
    assert metadata.omitted_dimensions == expected_omitted
    with pytest.raises(FrozenInstanceError):
        metadata.lossy = False  # type: ignore[misc]


def test_projection_metadata_deduplicates_a_release_shared_across_domains() -> None:
    demographic = _make_batch(_SPEC_BY_DOMAIN[ObservationDomain.DEMOGRAPHIC])
    market = _make_batch_from_shared_manifest(
        _SPEC_BY_DOMAIN[ObservationDomain.MARKET_QUOTE],
        manifest=demographic.manifest,
    )

    metadata = project_long_observations((demographic, market)).metadata

    assert metadata.source_domains == (
        ObservationDomain.DEMOGRAPHIC,
        ObservationDomain.MARKET_QUOTE,
    )
    assert metadata.source_release_ids == (demographic.manifest.release_id.value,)


def test_projection_owns_inputs_and_returns_only_deep_copies(
    all_domain_batches: tuple[DomainBatch, ...],
) -> None:
    projection = project_long_observations(all_domain_batches)
    expected = projection.to_frame()

    leaked_input = all_domain_batches[0].observations
    leaked_input.loc[:, "value"] = -999.0
    leaked_projection = projection.to_frame()
    leaked_projection.loc[:, "value"] = -888.0

    pd.testing.assert_frame_equal(projection.to_frame(), expected)
    assert not isinstance(projection, pd.DataFrame)
    assert not isinstance(projection, NormalizedDomainBatch)
    assert not hasattr(projection, "observations")
    assert not hasattr(projection, "manifest")


@pytest.mark.parametrize(
    "bad_input",
    [
        (),
        pd.DataFrame(),
        "demographic",
        (object(),),
    ],
    ids=["empty", "dataframe", "string", "arbitrary-object"],
)
def test_projection_rejects_nonbatch_or_empty_inputs(bad_input: object) -> None:
    with pytest.raises(DataContractError):
        project_long_observations(bad_input)  # type: ignore[arg-type]


def test_projection_rejects_duplicate_observations_and_projection_nesting() -> None:
    batch = _make_batch(_SPEC_BY_DOMAIN[ObservationDomain.DEMOGRAPHIC])
    with pytest.raises(DataContractError, match=r"duplicate|observation"):
        project_long_observations((batch, batch))

    projection = project_long_observations((batch,))
    with pytest.raises(ReportingProjectionInputError):
        project_long_observations((projection,))  # type: ignore[arg-type]


def test_projection_cannot_round_trip_into_any_strict_domain(
    all_domain_batches: tuple[DomainBatch, ...],
) -> None:
    projected = project_long_observations(all_domain_batches).to_frame()

    for spec in _DOMAIN_SPECS:
        with pytest.raises((SchemaErrors, DataContractError)):
            spec.validator(projected)


def test_common_guard_accepts_every_exact_domain_batch(
    all_domain_batches: tuple[DomainBatch, ...],
) -> None:
    for batch in all_domain_batches:
        guarded = require_domain_batch(
            batch,
            consumer="report_builder",
        )

        assert guarded is not batch
        pd.testing.assert_frame_equal(
            guarded.observations,
            batch.observations,
        )


def test_common_guard_returns_a_detached_sanitized_batch() -> None:
    batch = _make_batch(_SPEC_BY_DOMAIN[ObservationDomain.DEMOGRAPHIC])
    expected = batch.observations
    guarded = require_domain_batch(batch, consumer="model_compiler")
    corrupted = batch.observations
    corrupted.loc[:, "value"] = -999.0
    object.__setattr__(batch, "_observations", corrupted)

    assert guarded is not batch
    pd.testing.assert_frame_equal(guarded.observations, expected)


def test_common_guard_rejects_projection_raw_frame_arbitrary_and_forged_batch() -> None:
    demographic = _make_batch(_SPEC_BY_DOMAIN[ObservationDomain.DEMOGRAPHIC])
    projection = project_long_observations((demographic,))

    with pytest.raises(
        ReportingProjectionInputError,
        match="curve_builder",
    ):
        require_domain_batch(projection, consumer="curve_builder")
    for invalid in (demographic.observations, object()):
        with pytest.raises(DataContractError):
            require_domain_batch(invalid, consumer="model_compiler")

    forged = cast(DomainBatch, object.__new__(NormalizedDomainBatch))
    object.__setattr__(forged, "manifest", demographic.manifest)
    object.__setattr__(forged, "provenance", demographic.provenance)
    object.__setattr__(forged, "identities", demographic.identities)
    corrupted = demographic.observations
    corrupted.loc[:, "domain"] = "market_quote"
    object.__setattr__(forged, "_observations", corrupted)
    with pytest.raises(DataContractError):
        require_domain_batch(forged, consumer="model_compiler")


@pytest.mark.parametrize("consumer", ["", "   ", 7, None])
def test_common_guard_requires_a_nonblank_builtin_consumer(
    consumer: object,
) -> None:
    batch = _make_batch(_SPEC_BY_DOMAIN[ObservationDomain.DEMOGRAPHIC])
    with pytest.raises((TypeError, ValueError, DataContractError)):
        require_domain_batch(batch, consumer=consumer)  # type: ignore[arg-type]


def test_named_guards_accept_only_their_exact_schema_and_domain() -> None:
    demographic = _make_batch(_SPEC_BY_DOMAIN[ObservationDomain.DEMOGRAPHIC])
    market = _make_batch(_SPEC_BY_DOMAIN[ObservationDomain.MARKET_QUOTE])
    rate = _make_batch(_SPEC_BY_DOMAIN[ObservationDomain.REGULATORY_RATE])
    regulatory_mortality = _make_batch(
        _SPEC_BY_DOMAIN[ObservationDomain.REGULATORY_MORTALITY_TABLE]
    )

    guarded_demographic = require_demographic_mortality_batch(
        demographic,
        consumer="population_mortality_model",
    )
    guarded_market = require_market_quote_batch(
        market,
        consumer="market_curve_builder",
    )
    guarded_rate = require_regulatory_rate_batch(
        rate,
        consumer="regulatory_curve_builder",
    )
    assert guarded_demographic is not demographic
    assert guarded_market is not market
    assert guarded_rate is not rate

    with pytest.raises(DataContractError, match=r"demographic|regulatory"):
        require_demographic_mortality_batch(
            regulatory_mortality,
            consumer="population_mortality_model",
        )
    with pytest.raises(DataContractError, match=r"market_quote|regulatory_rate"):
        require_market_quote_batch(
            rate,
            consumer="market_curve_builder",
        )
    with pytest.raises(DataContractError, match=r"regulatory_rate|market_quote"):
        require_regulatory_rate_batch(
            market,
            consumer="regulatory_curve_builder",
        )


def test_demographic_guard_does_not_invent_a_mortality_variable_allowlist() -> None:
    population = _make_batch(
        _SPEC_BY_DOMAIN[ObservationDomain.DEMOGRAPHIC],
        frame_overrides={
            "variable": "population",
            "semantic_kind": "population",
            "unit": "person",
            "population_basis": "resident_estimate",
        },
    )

    guarded = require_demographic_mortality_batch(
        population,
        consumer="population_mortality_model",
    )

    assert guarded is not population
    pd.testing.assert_frame_equal(
        guarded.observations,
        population.observations,
    )
