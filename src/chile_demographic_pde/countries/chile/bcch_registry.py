"""Independently reviewed Banco Central de Chile series authorizations.

The production registry is deliberately empty.  Raw BCCh identifiers are not
semantic authorizations: a series becomes usable only after an independent
review commits to its complete semantic specification, metadata, and both raw
asset digests.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Final

from chile_demographic_pde.core.errors import (
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.data.domain_schemas import (
    CompoundingConvention,
    DayCountConvention,
    MarketQuoteType,
)
from chile_demographic_pde.data.identity import canonical_identity_bytes

_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_SOURCE_KEY: Final = re.compile(r"[a-z0-9][a-z0-9_]*")
_QUOTE_TYPES: Final = frozenset(item.value for item in MarketQuoteType)
_COMPOUNDING: Final = frozenset(item.value for item in CompoundingConvention)
_DAY_COUNT: Final = frozenset(item.value for item in DayCountConvention)
_SOURCE_UNITS: Final = frozenset({"percent", "CLP", "index_points"})
_FREQUENCIES: Final = frozenset({"DAILY", "MONTHLY", "QUARTERLY", "ANNUAL"})
_SPEC_SCHEMA_VERSION: Final = "bcch-series-spec-v1"
_PRODUCTION_REGISTRY_VERSION: Final = "bcch-reviewed-series-v1"


def _nonblank(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or value != value.strip()
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise DataContractError(f"BCCh {field} must be a canonical nonblank string.")
    return value


def _canonical_date(value: object, *, field: str) -> date:
    if type(value) is not date:
        raise DataContractError(f"BCCh {field} must be a date without a time.")
    return value


def _source_key(value: object, *, field: str) -> str:
    if type(value) is not str or _SOURCE_KEY.fullmatch(value) is None:
        raise DataContractError(f"BCCh {field} must use lowercase snake-case characters.")
    return value


def _sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DataContractError(f"BCCh {field} must be a lowercase SHA-256 digest.")
    return value


def _frequency(value: object, *, field: str = "frequency_code") -> str:
    if type(value) is not str or value not in _FREQUENCIES:
        raise DataContractError(f"BCCh {field} must be DAILY, MONTHLY, QUARTERLY, or ANNUAL.")
    return value


def _is_frequency_aligned(value: date, frequency: str) -> bool:
    if frequency == "DAILY":
        return True
    if value.day != 1:
        return False
    if frequency == "MONTHLY":
        return True
    if frequency == "QUARTERLY":
        return value.month in {1, 4, 7, 10}
    return value.month == 1


def _require_metadata_alignment(
    *,
    first_observation: date,
    last_observation: date,
    frequency_code: str,
) -> None:
    if not _is_frequency_aligned(
        first_observation,
        frequency_code,
    ) or not _is_frequency_aligned(last_observation, frequency_code):
        raise DataContractError(
            "BCCh first/last observation dates must align with their frequency."
        )


@dataclass(frozen=True, slots=True)
class BcchSeriesSpec:
    """Reviewed semantic meaning of one exact BCCh source series."""

    series_id: str
    variable: str
    quote_type: str
    tenor_years: float | None
    currency: str
    indexation: str
    instrument_type: str
    compounding: str | None
    day_count: str | None
    source_unit: str
    canonical_unit: str

    def __post_init__(self) -> None:
        for field in (
            "series_id",
            "variable",
            "currency",
            "indexation",
            "instrument_type",
            "canonical_unit",
        ):
            _nonblank(getattr(self, field), field=field)
        if type(self.quote_type) is not str or self.quote_type not in _QUOTE_TYPES:
            raise DataContractError("BCCh quote_type is not supported.")
        if self.tenor_years is not None and (
            type(self.tenor_years) is not float
            or not math.isfinite(self.tenor_years)
            or self.tenor_years <= 0.0
        ):
            raise DataContractError("BCCh tenor_years must be a positive finite float or None.")
        if self.compounding is not None and (
            type(self.compounding) is not str or self.compounding not in _COMPOUNDING
        ):
            raise DataContractError("BCCh compounding must be a supported convention or None.")
        if self.day_count is not None and (
            type(self.day_count) is not str or self.day_count not in _DAY_COUNT
        ):
            raise DataContractError("BCCh day_count must be a supported convention or None.")
        if type(self.source_unit) is not str or self.source_unit not in _SOURCE_UNITS:
            raise DataContractError("BCCh source_unit is not supported.")
        self._require_exact_semantics()

    def _require_exact_semantics(self) -> None:
        rate_common = (
            self.tenor_years is not None
            and self.source_unit == "percent"
            and self.canonical_unit == "1 / year"
        )
        if self.quote_type == MarketQuoteType.BENCHMARK_YIELD.value:
            instruments = {
                ("CLP", "nominal", "bcp_benchmark_bond"),
                ("UF", "UF", "bcu_btu_benchmark_bond"),
            }
            valid = (
                self.variable == "official_benchmark_yield"
                and rate_common
                and (self.currency, self.indexation, self.instrument_type) in instruments
            )
        elif self.quote_type == MarketQuoteType.SWAP_FIXED_RATE.value:
            instruments = {
                ("CLP", "nominal", "spc_nominal_fixed_swap"),
                ("UF", "UF", "spc_uf_fixed_swap"),
            }
            valid = (
                self.variable == "official_swap_fixed_rate"
                and rate_common
                and (self.currency, self.indexation, self.instrument_type) in instruments
            )
        elif self.quote_type == MarketQuoteType.FIXING.value:
            valid = (
                self.variable == "uf_fixing_clp"
                and self.tenor_years is None
                and self.currency == "CLP"
                and self.indexation == "UF"
                and self.instrument_type == "uf_fixing"
                and self.compounding is None
                and self.day_count is None
                and self.source_unit == "CLP"
                and self.canonical_unit == "CLP"
            )
        elif self.quote_type == MarketQuoteType.PUBLISHED_INDEX.value:
            valid = (
                self.variable == "consumer_price_index"
                and self.tenor_years is None
                and self.currency == "not_applicable"
                and self.indexation == "IPC"
                and self.instrument_type == "ipc_index"
                and self.compounding is None
                and self.day_count is None
                and self.source_unit == "index_points"
                and self.canonical_unit == "dimensionless"
            )
        else:
            valid = (
                self.variable == "consumer_price_change"
                and self.tenor_years is None
                and self.currency == "not_applicable"
                and self.indexation == "IPC"
                and self.instrument_type == "ipc_change"
                and self.compounding is None
                and self.day_count is None
                and self.source_unit == "percent"
                and self.canonical_unit == "percent"
            )
        if not valid:
            raise DataContractError(
                "BCCh variable and instrument fields violate the exact "
                f"{self.quote_type} semantic mapping."
            )


def _bcch_spec_payload(spec: BcchSeriesSpec) -> dict[str, object]:
    if not isinstance(spec, BcchSeriesSpec):
        raise DataContractError("BCCh spec digest requires a BcchSeriesSpec.")
    return {
        "schema_version": _SPEC_SCHEMA_VERSION,
        "series_id": spec.series_id,
        "variable": spec.variable,
        "quote_type": spec.quote_type,
        "tenor_years": spec.tenor_years,
        "currency": spec.currency,
        "indexation": spec.indexation,
        "instrument_type": spec.instrument_type,
        "compounding": spec.compounding,
        "day_count": spec.day_count,
        "source_unit": spec.source_unit,
        "canonical_unit": spec.canonical_unit,
    }


def bcch_spec_digest(spec: BcchSeriesSpec) -> str:
    """Return the v1 typed identity digest of every semantic spec field."""

    return hashlib.sha256(canonical_identity_bytes(_bcch_spec_payload(spec))).hexdigest()


@dataclass(frozen=True, slots=True)
class BcchRawContract:
    """Exact parser contract independently committed by a reviewed entry."""

    series_id: str
    frequency_code: str
    spanish_title: str
    english_title: str
    first_observation: date
    last_observation: date
    updated_on: date
    created_on: date
    search_series_source_key: str
    search_series_sha256: str

    def __post_init__(self) -> None:
        for field in ("series_id", "spanish_title", "english_title"):
            _nonblank(getattr(self, field), field=field)
        _frequency(self.frequency_code)
        for field in (
            "first_observation",
            "last_observation",
            "updated_on",
            "created_on",
        ):
            _canonical_date(getattr(self, field), field=field)
        if self.first_observation > self.last_observation:
            raise DataContractError("BCCh first_observation cannot follow last_observation.")
        _require_metadata_alignment(
            first_observation=self.first_observation,
            last_observation=self.last_observation,
            frequency_code=self.frequency_code,
        )
        _source_key(
            self.search_series_source_key,
            field="search_series_source_key",
        )
        _sha256(self.search_series_sha256, field="search_series_sha256")


@dataclass(frozen=True, slots=True)
class BcchSeriesMetadata:
    """Strict date-only metadata parsed from a SearchSeries sidecar."""

    source_key: str
    sha256: str
    series_id: str
    frequency_code: str
    spanish_title: str
    english_title: str
    first_observation: date
    last_observation: date
    updated_on: date
    created_on: date

    def __post_init__(self) -> None:
        _source_key(self.source_key, field="source_key")
        _sha256(self.sha256, field="sha256")
        for field in ("series_id", "spanish_title", "english_title"):
            _nonblank(getattr(self, field), field=field)
        _frequency(self.frequency_code)
        for field in (
            "first_observation",
            "last_observation",
            "updated_on",
            "created_on",
        ):
            _canonical_date(getattr(self, field), field=field)
        if self.first_observation > self.last_observation:
            raise DataContractError("BCCh metadata first observation cannot follow the last.")
        _require_metadata_alignment(
            first_observation=self.first_observation,
            last_observation=self.last_observation,
            frequency_code=self.frequency_code,
        )


@dataclass(frozen=True, slots=True)
class BcchReviewedSeriesEntry:
    """One independently authored authorization for two exact BCCh assets."""

    registry_version: str
    series_id: str
    spec: BcchSeriesSpec
    spec_digest: str
    frequency_code: str
    spanish_title: str
    english_title: str
    first_observation: date
    last_observation: date
    updated_on: date
    created_on: date
    get_series_source_key: str
    get_series_sha256: str
    search_series_source_key: str
    search_series_sha256: str
    review_id: str

    def __post_init__(self) -> None:
        for field in (
            "registry_version",
            "series_id",
            "spanish_title",
            "english_title",
            "review_id",
        ):
            _nonblank(getattr(self, field), field=field)
        _frequency(self.frequency_code)
        if not isinstance(self.spec, BcchSeriesSpec):
            raise DataContractError("BCCh reviewed entry requires a typed spec.")
        if self.series_id != self.spec.series_id:
            raise DataContractError("BCCh reviewed entry series_id must equal its semantic spec.")
        _sha256(self.spec_digest, field="spec_digest")
        if self.spec_digest != bcch_spec_digest(self.spec):
            raise DataContractError("BCCh reviewed entry spec digest does not match its spec.")
        for field in (
            "first_observation",
            "last_observation",
            "updated_on",
            "created_on",
        ):
            _canonical_date(getattr(self, field), field=field)
        if self.first_observation > self.last_observation:
            raise DataContractError("BCCh reviewed first observation cannot follow the last.")
        _require_metadata_alignment(
            first_observation=self.first_observation,
            last_observation=self.last_observation,
            frequency_code=self.frequency_code,
        )
        for field in ("get_series_source_key", "search_series_source_key"):
            _source_key(getattr(self, field), field=field)
        if self.get_series_source_key == self.search_series_source_key:
            raise DataContractError("BCCh GetSeries and SearchSeries source keys must differ.")
        for field in ("get_series_sha256", "search_series_sha256"):
            _sha256(getattr(self, field), field=field)

    @property
    def raw_contract(self) -> BcchRawContract:
        """Build a detached strict parser contract from reviewed values."""

        return BcchRawContract(
            series_id=self.series_id,
            frequency_code=self.frequency_code,
            spanish_title=self.spanish_title,
            english_title=self.english_title,
            first_observation=self.first_observation,
            last_observation=self.last_observation,
            updated_on=self.updated_on,
            created_on=self.created_on,
            search_series_source_key=self.search_series_source_key,
            search_series_sha256=self.search_series_sha256,
        )


def _bcch_entry_fingerprint(entry: BcchReviewedSeriesEntry) -> str:
    """Commit to every reviewer-authored authorization field."""

    if not isinstance(entry, BcchReviewedSeriesEntry):
        raise DataContractError("BCCh authorization fingerprint requires a reviewed entry.")
    payload = {
        "schema_version": "bcch-reviewed-series-entry-v1",
        "registry_version": entry.registry_version,
        "series_id": entry.series_id,
        "spec": _bcch_spec_payload(entry.spec),
        "spec_digest": entry.spec_digest,
        "frequency_code": entry.frequency_code,
        "spanish_title": entry.spanish_title,
        "english_title": entry.english_title,
        "first_observation": entry.first_observation,
        "last_observation": entry.last_observation,
        "updated_on": entry.updated_on,
        "created_on": entry.created_on,
        "get_series_source_key": entry.get_series_source_key,
        "get_series_sha256": entry.get_series_sha256,
        "search_series_source_key": entry.search_series_source_key,
        "search_series_sha256": entry.search_series_sha256,
        "review_id": entry.review_id,
    }
    return hashlib.sha256(canonical_identity_bytes(payload)).hexdigest()


def _rebuild_reviewed_entry(
    entry: BcchReviewedSeriesEntry,
) -> BcchReviewedSeriesEntry:
    """Revalidate all nominal entry fields through fresh strict value objects."""

    spec = entry.spec
    rebuilt_spec = BcchSeriesSpec(
        series_id=spec.series_id,
        variable=spec.variable,
        quote_type=spec.quote_type,
        tenor_years=spec.tenor_years,
        currency=spec.currency,
        indexation=spec.indexation,
        instrument_type=spec.instrument_type,
        compounding=spec.compounding,
        day_count=spec.day_count,
        source_unit=spec.source_unit,
        canonical_unit=spec.canonical_unit,
    )
    return BcchReviewedSeriesEntry(
        registry_version=entry.registry_version,
        series_id=entry.series_id,
        spec=rebuilt_spec,
        spec_digest=entry.spec_digest,
        frequency_code=entry.frequency_code,
        spanish_title=entry.spanish_title,
        english_title=entry.english_title,
        first_observation=entry.first_observation,
        last_observation=entry.last_observation,
        updated_on=entry.updated_on,
        created_on=entry.created_on,
        get_series_source_key=entry.get_series_source_key,
        get_series_sha256=entry.get_series_sha256,
        search_series_source_key=entry.search_series_source_key,
        search_series_sha256=entry.search_series_sha256,
        review_id=entry.review_id,
    )


@dataclass(frozen=True, slots=True, init=False)
class BcchReviewedRegistry:
    """Immutable series-id lookup retaining reviewer-authored entry objects."""

    registry_version: str
    entries: Mapping[str, BcchReviewedSeriesEntry]
    _entry_fingerprints: Mapping[str, str]

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("BcchReviewedRegistry must be created by BcchReviewedRegistry.create().")

    @classmethod
    def create(
        cls,
        *,
        registry_version: str,
        entries: Sequence[BcchReviewedSeriesEntry],
    ) -> BcchReviewedRegistry:
        version = _nonblank(registry_version, field="registry_version")
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise DataContractError("BCCh registry entries must be a sequence.")
        try:
            snapshot = tuple(entries)
        except Exception:
            raise DataContractError(
                "BCCh registry entries could not be read as one immutable snapshot."
            ) from None
        by_series: dict[str, BcchReviewedSeriesEntry] = {}
        fingerprints: dict[str, str] = {}
        for entry in snapshot:
            if not isinstance(entry, BcchReviewedSeriesEntry):
                raise DataContractError("BCCh registry values must be reviewed series entries.")
            try:
                rebuilt = _rebuild_reviewed_entry(entry)
            except DataContractError:
                raise
            except Exception:
                raise DataContractError("BCCh reviewed entry could not be revalidated.") from None
            if rebuilt.registry_version != version:
                raise DataContractError("BCCh entry registry version does not match its registry.")
            if rebuilt.spec_digest != bcch_spec_digest(rebuilt.spec):
                raise DataContractError(
                    "BCCh registry entry contains a changed semantic spec digest."
                )
            if rebuilt.series_id in by_series:
                raise DataContractError(
                    "BCCh reviewed series IDs must be unique within a registry."
                )
            by_series[rebuilt.series_id] = entry
            fingerprints[rebuilt.series_id] = _bcch_entry_fingerprint(rebuilt)
        registry = object.__new__(cls)
        object.__setattr__(registry, "registry_version", version)
        object.__setattr__(
            registry,
            "entries",
            MappingProxyType(dict(sorted(by_series.items()))),
        )
        object.__setattr__(
            registry,
            "_entry_fingerprints",
            MappingProxyType(dict(sorted(fingerprints.items()))),
        )
        return registry

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, series_id: object) -> bool:
        return type(series_id) is str and series_id in self.entries

    def __iter__(self) -> Iterator[BcchReviewedSeriesEntry]:
        return iter(self.entries.values())

    def authorization_fingerprint(self, series_id: str) -> str:
        """Return the create-time fingerprint without exposing mutable storage."""

        try:
            return self._entry_fingerprints[series_id]
        except (KeyError, TypeError):
            raise MissingOfficialDataError(
                "The requested BCCh series has no reviewed authorization."
            ) from None

    def entry_matches_snapshot(self, entry: BcchReviewedSeriesEntry) -> bool:
        """Check object authorization and its complete create-time snapshot."""

        try:
            authorized = self.entries.get(entry.series_id)
            expected = self._entry_fingerprints.get(entry.series_id)
            observed = _bcch_entry_fingerprint(_rebuild_reviewed_entry(entry))
        except (AttributeError, DataContractError, TypeError, ValueError, OverflowError):
            return False
        return authorized is entry and expected is not None and observed == expected


PRODUCTION_BCCH_REVIEWED_REGISTRY_V1: Final = BcchReviewedRegistry.create(
    registry_version=_PRODUCTION_REGISTRY_VERSION,
    entries=(),
)


def require_reviewed_bcch_entry(
    series_id: str,
    registry: BcchReviewedRegistry = PRODUCTION_BCCH_REVIEWED_REGISTRY_V1,
) -> BcchReviewedSeriesEntry:
    """Require one exact registry authorization without inspecting raw bytes."""

    if not isinstance(registry, BcchReviewedRegistry):
        raise MissingOfficialDataError("A typed immutable BCCh reviewed registry is required.")
    if type(series_id) is not str:
        raise MissingOfficialDataError("The BCCh series is not reviewed.")
    entry = registry.entries.get(series_id)
    if entry is None:
        raise MissingOfficialDataError(
            "The requested BCCh series is absent from the reviewed registry."
        )
    if not registry.entry_matches_snapshot(entry):
        raise MissingOfficialDataError("The requested BCCh reviewed registry entry has changed.")
    return entry
