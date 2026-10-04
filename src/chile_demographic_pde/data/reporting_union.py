"""Read-only, explicitly lossy projections across observation domains.

This module is a presentation boundary, not a source-domain normalization
boundary.  Projected rows retain their source envelope and canonical IDs, but
selected domain dimensions are encoded into compact JSON for display.  The
result therefore cannot be passed back into statistical or pricing consumers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Final, cast

import numpy as np
import pandas as pd

from chile_demographic_pde.core.errors import (
    DataContractError,
    ReportingProjectionInputError,
)
from chile_demographic_pde.data.domain_schemas import (
    DemographicObservationSchema,
    EnvironmentalObservationSchema,
    HealthSystemObservationSchema,
    MarketQuoteSchema,
    PensionProductStatisticSchema,
    RegulatoryMortalityTableObservationSchema,
    RegulatoryRateSchema,
)
from chile_demographic_pde.data.envelope import (
    ENVELOPE_COLUMNS,
    NormalizedDomainBatch,
)
from chile_demographic_pde.data.model_input import _sanitize_domain_batch
from chile_demographic_pde.data.roles import ObservationDomain

REPORTING_LONG_COLUMNS: Final = (
    *ENVELOPE_COLUMNS,
    "projected_domain",
    "display_dimensions",
)

DISPLAY_DIMENSIONS: Final[Mapping[ObservationDomain, tuple[str, ...]]] = MappingProxyType(
    {
        ObservationDomain.DEMOGRAPHIC: (
            "age_lower",
            "age_upper",
            "age_open",
            "sex",
            "region",
            "commune",
            "cause",
        ),
        ObservationDomain.ENVIRONMENTAL: (
            "station",
            "region",
            "commune",
            "pollutant",
            "environmental_measure",
        ),
        ObservationDomain.HEALTH_SYSTEM: (
            "facility",
            "health_service",
            "region",
            "commune",
            "reporting_grain",
        ),
        ObservationDomain.MARKET_QUOTE: (
            "source_series_id",
            "quote_type",
            "tenor_years",
            "currency",
            "indexation",
            "instrument_type",
        ),
        ObservationDomain.REGULATORY_RATE: (
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
        ObservationDomain.REGULATORY_MORTALITY_TABLE: (
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
        ObservationDomain.PENSION_PRODUCT_STATISTIC: (
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
    }
)
REPORTING_DISPLAY_DIMENSIONS: Final = DISPLAY_DIMENSIONS

_SCHEMA_BY_DOMAIN: Final = {
    ObservationDomain.DEMOGRAPHIC: DemographicObservationSchema,
    ObservationDomain.ENVIRONMENTAL: EnvironmentalObservationSchema,
    ObservationDomain.HEALTH_SYSTEM: HealthSystemObservationSchema,
    ObservationDomain.MARKET_QUOTE: MarketQuoteSchema,
    ObservationDomain.REGULATORY_RATE: RegulatoryRateSchema,
    ObservationDomain.REGULATORY_MORTALITY_TABLE: (RegulatoryMortalityTableObservationSchema),
    ObservationDomain.PENSION_PRODUCT_STATISTIC: PensionProductStatisticSchema,
}


@dataclass(frozen=True, slots=True)
class ReportingProjectionMetadata:
    """Immutable audit metadata for an intentionally lossy display projection."""

    schema_version: str
    lossy: bool
    intended_use: str
    source_domains: tuple[ObservationDomain, ...]
    source_release_ids: tuple[str, ...]
    projected_columns: tuple[str, ...]
    selected_dimensions: tuple[tuple[str, tuple[str, ...]], ...]
    omitted_dimensions: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ReportingLongProjection:
    """Factory-created reporting frame that exposes deep copies only."""

    _frame: pd.DataFrame
    metadata: ReportingProjectionMetadata

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("ReportingLongProjection must be created by project_long_observations().")

    @classmethod
    def _create(
        cls,
        *,
        frame: pd.DataFrame,
        metadata: ReportingProjectionMetadata,
    ) -> ReportingLongProjection:
        projection = object.__new__(cls)
        object.__setattr__(projection, "_frame", frame.copy(deep=True))
        object.__setattr__(projection, "metadata", metadata)
        return projection

    def to_frame(self) -> pd.DataFrame:
        """Return a caller-owned copy of the reporting rows."""

        return self._frame.copy(deep=True)


def _require_consumer(consumer: object) -> str:
    if type(consumer) is not str or not consumer.strip():
        raise DataContractError("consumer must be a nonblank built-in string.")
    return consumer


def _validated_domain_batch(
    value: object,
    *,
    consumer: str,
) -> NormalizedDomainBatch[pd.DataFrame]:
    consumer_name = _require_consumer(consumer)
    if isinstance(value, ReportingLongProjection):
        raise ReportingProjectionInputError(
            f"{consumer_name} requires an exact source-domain batch; "
            "reporting projections are display-only."
        )
    if not isinstance(value, NormalizedDomainBatch):
        raise DataContractError(f"{consumer_name} requires a NormalizedDomainBatch.")

    return _sanitize_domain_batch(value)


def require_domain_batch(
    value: object,
    *,
    consumer: str,
) -> NormalizedDomainBatch[object]:
    """Require and fully revalidate an exact normalized source-domain batch."""

    return cast(
        "NormalizedDomainBatch[object]",
        _validated_domain_batch(value, consumer=consumer),
    )


def _require_exact_domain(
    value: object,
    *,
    consumer: str,
    expected: ObservationDomain,
) -> NormalizedDomainBatch[pd.DataFrame]:
    sanitized = _validated_domain_batch(value, consumer=consumer)
    if sanitized.domain is not expected:
        raise DataContractError(
            f"{consumer} requires domain {expected.value!r}; received {sanitized.domain.value!r}."
        )
    return sanitized


def require_demographic_mortality_batch(
    value: object,
    *,
    consumer: str,
) -> NormalizedDomainBatch[pd.DataFrame]:
    """Require an exact demographic batch without restricting its variables."""

    return _require_exact_domain(
        value,
        consumer=consumer,
        expected=ObservationDomain.DEMOGRAPHIC,
    )


def require_market_quote_batch(
    value: object,
    *,
    consumer: str,
) -> NormalizedDomainBatch[pd.DataFrame]:
    """Require an exact market-quote batch."""

    return _require_exact_domain(
        value,
        consumer=consumer,
        expected=ObservationDomain.MARKET_QUOTE,
    )


def require_regulatory_rate_batch(
    value: object,
    *,
    consumer: str,
) -> NormalizedDomainBatch[pd.DataFrame]:
    """Require an exact regulatory-rate batch."""

    return _require_exact_domain(
        value,
        consumer=consumer,
        expected=ObservationDomain.REGULATORY_RATE,
    )


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    return False


def _json_scalar(value: object) -> None | bool | int | float | str:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, Enum):
        if not isinstance(value.value, str):
            raise DataContractError("Reporting display-dimension enum values must be strings.")
        return _json_scalar(value.value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise DataContractError("Reporting display dimensions require finite numeric values.")
        return number
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            raise DataContractError(
                "Reporting display dimensions require valid UTF-8 text."
            ) from error
        return value
    raise DataContractError(
        "Reporting display dimensions require explicit null, date, boolean, "
        f"finite-number, or UTF-8 text atoms; received {type(value).__qualname__}."
    )


def _display_json(
    row: pd.Series,
    dimensions: tuple[str, ...],
) -> str:
    payload = {column: _json_scalar(row[column]) for column in dimensions}
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _project_batch(
    batch: NormalizedDomainBatch[pd.DataFrame],
) -> pd.DataFrame:
    domain = batch.domain
    dimensions = REPORTING_DISPLAY_DIMENSIONS[domain]
    source = batch.observations
    projected = source.loc[:, list(ENVELOPE_COLUMNS)].copy(deep=True)
    projected["projected_domain"] = domain.value
    projected["display_dimensions"] = source.apply(
        _display_json,
        axis="columns",
        dimensions=dimensions,
    )
    return projected.loc[:, list(REPORTING_LONG_COLUMNS)]


def _metadata(
    batches: tuple[NormalizedDomainBatch[pd.DataFrame], ...],
) -> ReportingProjectionMetadata:
    source_domains = tuple(sorted({batch.domain for batch in batches}))
    selected_dimensions = tuple(
        (domain.value, REPORTING_DISPLAY_DIMENSIONS[domain]) for domain in source_domains
    )
    omitted_dimensions = tuple(
        (
            domain.value,
            tuple(
                column
                for column in _SCHEMA_BY_DOMAIN[domain].to_schema().columns
                if column not in ENVELOPE_COLUMNS
                and column not in REPORTING_DISPLAY_DIMENSIONS[domain]
            ),
        )
        for domain in source_domains
    )
    return ReportingProjectionMetadata(
        schema_version="reporting-long-v1",
        lossy=True,
        intended_use="display_only",
        source_domains=source_domains,
        source_release_ids=tuple(sorted({batch.manifest.release_id.value for batch in batches})),
        projected_columns=REPORTING_LONG_COLUMNS,
        selected_dimensions=selected_dimensions,
        omitted_dimensions=omitted_dimensions,
    )


def project_long_observations(
    batches: Sequence[NormalizedDomainBatch[object]],
) -> ReportingLongProjection:
    """Project exact source batches into a deterministic display-only union."""

    if (
        isinstance(batches, (str, bytes, pd.DataFrame))
        or not isinstance(batches, Sequence)
        or not batches
    ):
        raise DataContractError(
            "Reporting projection requires a nonempty sequence of domain batches."
        )

    sanitized: list[NormalizedDomainBatch[pd.DataFrame]] = []
    for value in batches:
        if isinstance(value, ReportingLongProjection):
            raise ReportingProjectionInputError("Reporting projections cannot be nested.")
        exact = _validated_domain_batch(
            value,
            consumer="reporting_projection",
        )
        sanitized.append(exact)

    owned_batches = tuple(sanitized)
    frame = pd.concat(
        [_project_batch(batch) for batch in owned_batches],
        ignore_index=True,
        copy=True,
    )
    if frame["observation_id"].duplicated(keep=False).any():
        raise DataContractError("Reporting projection rejects duplicate observation IDs.")

    frame = frame.sort_values(
        by=["projected_domain", "release_id", "observation_id"],
        kind="stable",
        ignore_index=True,
    )
    return ReportingLongProjection._create(
        frame=frame,
        metadata=_metadata(owned_batches),
    )
