"""Strict Banco Central de Chile market-quote parsing and normalization."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from typing import Final, cast

import pandas as pd
from pydantic import HttpUrl

from chile_demographic_pde.core.errors import (
    DataContractError,
    MissingOfficialDataError,
    SourceContractError,
)
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.countries.chile.bcch_registry import (
    PRODUCTION_BCCH_REVIEWED_REGISTRY_V1,
    BcchRawContract,
    BcchReviewedRegistry,
    BcchReviewedSeriesEntry,
    BcchSeriesMetadata,
    BcchSeriesSpec,
    bcch_spec_digest,
    require_reviewed_bcch_entry,
)
from chile_demographic_pde.data.domain_schemas import (
    MARKET_QUOTE_KEY,
    MarketQuoteBatch,
    validate_market_quotes,
)
from chile_demographic_pde.data.envelope import (
    NormalizedDomainBatch,
    assign_observation_ids,
)
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import (
    OfficialFinanceRelease,
    ReleaseManifest,
)
from chile_demographic_pde.data.roles import ObservationDomain

_ROOT_FIELDS: Final = frozenset({"Codigo", "Descripcion", "Series", "SeriesInfos"})
_SERIES_FIELDS: Final = frozenset({"descripEsp", "descripIng", "seriesId", "Obs"})
_OBSERVATION_FIELDS: Final = frozenset({"indexDateString", "value", "statusCode"})
_INFO_FIELDS: Final = frozenset(
    {
        "seriesId",
        "frequencyCode",
        "spanishTitle",
        "englishTitle",
        "firstObservation",
        "lastObservation",
        "updatedAt",
        "createdAt",
    }
)
_EXACT_DATE: Final = re.compile(r"\d{2}-\d{2}-\d{4}")
_EXACT_DECIMAL: Final = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_STATUS_CODE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SAFE_BCCH_DOCUMENTATION_URL: Final = HttpUrl(
    "https://si3.bcentral.cl/estadisticas/Principal1/Web_Services/doc_es.htm"
)
_RATE_TRANSFORMATION_ID: Final = "bcch-published-percent-to-decimal-rate"
_RATE_TRANSFORMATION_VERSION: Final = "v1"
_RATE_QUOTE_TYPES: Final = frozenset({"benchmark_yield", "swap_fixed_rate"})
_FREQUENCIES: Final = frozenset({"DAILY", "MONTHLY", "QUARTERLY", "ANNUAL"})


@dataclass(frozen=True, slots=True)
class BcchObservation:
    """One source observation with explicit publisher missingness."""

    observed_on: date
    published_value: float | None
    status_code: str


@dataclass(frozen=True, slots=True)
class BcchParsedSeries:
    """Strictly parsed GetSeries payload."""

    series_id: str
    spanish_description: str
    english_description: str
    observations: tuple[BcchObservation, ...]


def _duplicate_rejecting_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number {value!r}")


def _json_object(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes:
        raise SourceContractError("BCCh parser input must be immutable bytes.")
    try:
        decoded = raw.decode("utf-8", errors="strict")
        parsed = json.loads(
            decoded,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        raise SourceContractError("BCCh response is not unambiguous strict UTF-8 JSON.") from None
    if type(parsed) is not dict:
        raise SourceContractError("BCCh response root must be a JSON object.")
    return cast(dict[str, object], parsed)


def _exact_object(
    value: object,
    fields: frozenset[str],
    *,
    context: str,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise SourceContractError(f"BCCh {context} must contain exactly the documented fields.")
    return cast(dict[str, object], value)


def _successful_root(raw: bytes) -> dict[str, object]:
    root = _exact_object(_json_object(raw), _ROOT_FIELDS, context="response root")
    code = root["Codigo"]
    if type(code) is not int or code != 0:
        raise SourceContractError("BCCh Codigo must be the integer zero.")
    if type(root["Descripcion"]) is not str or not root["Descripcion"].strip():
        raise SourceContractError("BCCh Descripcion must be a nonblank source string.")
    return root


def _date_dd_mm_yyyy(value: object, *, field: str) -> date:
    if type(value) is not str or _EXACT_DATE.fullmatch(value) is None:
        raise SourceContractError(f"BCCh {field} must be an exact DD-MM-YYYY date.")
    day = int(value[0:2])
    month = int(value[3:5])
    year = int(value[6:10])
    try:
        parsed = date(year, month, day)
    except ValueError:
        raise SourceContractError(f"BCCh {field} contains an invalid calendar date.") from None
    if f"{parsed.day:02d}-{parsed.month:02d}-{parsed.year:04d}" != value:
        raise SourceContractError(f"BCCh {field} is not a canonical DD-MM-YYYY date.")
    return parsed


def _calendar_period_end(observed_on: date, frequency_code: str) -> date:
    """Return the half-open calendar end and enforce period-start alignment."""

    if frequency_code not in _FREQUENCIES:
        raise SourceContractError("BCCh observation frequency is unsupported.")
    try:
        if frequency_code == "DAILY":
            return observed_on + timedelta(days=1)
        if observed_on.day != 1:
            raise SourceContractError(
                "BCCh observation date is not aligned to its calendar frequency."
            )
        step_months = {
            "MONTHLY": 1,
            "QUARTERLY": 3,
            "ANNUAL": 12,
        }[frequency_code]
        if frequency_code == "QUARTERLY" and observed_on.month not in {
            1,
            4,
            7,
            10,
        }:
            raise SourceContractError("BCCh quarterly observation must start a calendar quarter.")
        if frequency_code == "ANNUAL" and observed_on.month != 1:
            raise SourceContractError("BCCh annual observation must start a calendar year.")
        month_index = observed_on.year * 12 + (observed_on.month - 1) + step_months
        return date(month_index // 12, month_index % 12 + 1, 1)
    except (OverflowError, ValueError):
        raise SourceContractError(
            "BCCh calendar period end falls outside the supported date range."
        ) from None


def _validated_raw_contract(
    contract: BcchRawContract,
) -> BcchRawContract:
    """Build a detached parser contract before any source bytes are inspected."""

    if not isinstance(contract, BcchRawContract):
        raise SourceContractError("BCCh parser requires a typed raw contract.")
    try:
        return BcchRawContract(
            series_id=contract.series_id,
            frequency_code=contract.frequency_code,
            spanish_title=contract.spanish_title,
            english_title=contract.english_title,
            first_observation=contract.first_observation,
            last_observation=contract.last_observation,
            updated_on=contract.updated_on,
            created_on=contract.created_on,
            search_series_source_key=contract.search_series_source_key,
            search_series_sha256=contract.search_series_sha256,
        )
    except Exception:
        raise SourceContractError("BCCh raw parser contract could not be revalidated.") from None


def _normalized_description(value: object, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise SourceContractError(f"BCCh {field} must be a nonblank string.")
    return " ".join(unicodedata.normalize("NFC", value).split())


def _require_description(
    actual: object,
    expected: str,
    *,
    field: str,
) -> str:
    normalized = _normalized_description(actual, field=field)
    if normalized != _normalized_description(expected, field=f"reviewed {field}"):
        raise SourceContractError(f"BCCh {field} description does not match the reviewed title.")
    return cast(str, actual)


def parse_bcch_get_series(
    raw: bytes,
    contract: BcchRawContract,
) -> BcchParsedSeries:
    """Parse GetSeries bytes without consulting authorization or acquisition state."""

    contract = _validated_raw_contract(contract)
    root = _successful_root(raw)
    if type(root["SeriesInfos"]) is not list or root["SeriesInfos"]:
        raise SourceContractError("BCCh GetSeries SeriesInfos must be empty.")
    series = _exact_object(root["Series"], _SERIES_FIELDS, context="GetSeries Series")
    if type(series["seriesId"]) is not str or series["seriesId"] != contract.series_id:
        raise SourceContractError("BCCh GetSeries must contain exactly the requested series ID.")
    spanish = _require_description(
        series["descripEsp"],
        contract.spanish_title,
        field="Spanish",
    )
    english = _require_description(
        series["descripIng"],
        contract.english_title,
        field="English",
    )
    raw_observations = series["Obs"]
    if type(raw_observations) is not list or not raw_observations:
        raise SourceContractError("BCCh GetSeries observations must be a nonempty JSON array.")

    observations: list[BcchObservation] = []
    dates: set[date] = set()
    for raw_observation in raw_observations:
        item = _exact_object(
            raw_observation,
            _OBSERVATION_FIELDS,
            context="GetSeries observation",
        )
        observed_on = _date_dd_mm_yyyy(
            item["indexDateString"],
            field="observation date",
        )
        _calendar_period_end(observed_on, contract.frequency_code)
        if observed_on in dates:
            raise SourceContractError("BCCh GetSeries contains a duplicate observation date.")
        if not (contract.first_observation <= observed_on <= contract.last_observation):
            raise SourceContractError(
                "BCCh GetSeries observation date is outside reviewed metadata bounds."
            )
        dates.add(observed_on)
        status = item["statusCode"]
        value = item["value"]
        if type(status) is not str or _STATUS_CODE.fullmatch(status) is None:
            raise SourceContractError("BCCh observation statusCode is not a canonical safe token.")
        if type(value) is not str:
            raise SourceContractError("BCCh observation value must retain its source string form.")
        published: float | None = None
        if status == "OK" and value.strip():
            if _EXACT_DECIMAL.fullmatch(value) is None:
                raise SourceContractError(
                    "BCCh OK observation value must use strict decimal syntax."
                )
            try:
                decimal_value = Decimal(value)
                numeric = float(decimal_value)
            except (InvalidOperation, ValueError, OverflowError):
                raise SourceContractError("BCCh OK observation value must be numeric.") from None
            if not decimal_value.is_finite() or not math.isfinite(numeric):
                raise SourceContractError("BCCh OK observation value must be finite.")
            published = numeric
        observations.append(
            BcchObservation(
                observed_on=observed_on,
                published_value=published,
                status_code=status,
            )
        )
    return BcchParsedSeries(
        series_id=contract.series_id,
        spanish_description=spanish,
        english_description=english,
        observations=tuple(observations),
    )


def parse_bcch_search_series(
    raw: bytes,
    contract: BcchRawContract,
) -> BcchSeriesMetadata:
    """Parse SearchSeries bytes without consulting an authorization registry."""

    contract = _validated_raw_contract(contract)
    root = _successful_root(raw)
    dummy = _exact_object(
        root["Series"],
        _SERIES_FIELDS,
        context="SearchSeries dummy Series",
    )
    if any(value is not None for value in dummy.values()):
        raise SourceContractError("BCCh SearchSeries dummy Series fields must all be null.")
    raw_infos = root["SeriesInfos"]
    if type(raw_infos) is not list:
        raise SourceContractError("BCCh SearchSeries SeriesInfos must be an array.")

    matches: list[dict[str, object]] = []
    for raw_info in raw_infos:
        info = _exact_object(
            raw_info,
            _INFO_FIELDS,
            context="SearchSeries metadata",
        )
        if info["seriesId"] == contract.series_id:
            matches.append(info)
    if len(matches) != 1:
        raise SourceContractError("BCCh SearchSeries requires exactly one matching series.")
    info = matches[0]
    expected_text = {
        "seriesId": contract.series_id,
        "frequencyCode": contract.frequency_code,
        "spanishTitle": contract.spanish_title,
        "englishTitle": contract.english_title,
    }
    if any(type(info[field]) is not str for field in expected_text):
        raise SourceContractError("BCCh SearchSeries metadata contract requires string fields.")
    if any(info[field] != expected for field, expected in expected_text.items()):
        raise SourceContractError("BCCh SearchSeries metadata differs from the reviewed contract.")
    parsed_dates = {
        "firstObservation": _date_dd_mm_yyyy(
            info["firstObservation"],
            field="firstObservation",
        ),
        "lastObservation": _date_dd_mm_yyyy(
            info["lastObservation"],
            field="lastObservation",
        ),
        "updatedAt": _date_dd_mm_yyyy(
            info["updatedAt"],
            field="updatedAt",
        ),
        "createdAt": _date_dd_mm_yyyy(
            info["createdAt"],
            field="createdAt",
        ),
    }
    expected_dates = {
        "firstObservation": contract.first_observation,
        "lastObservation": contract.last_observation,
        "updatedAt": contract.updated_on,
        "createdAt": contract.created_on,
    }
    if parsed_dates != expected_dates:
        raise SourceContractError(
            "BCCh SearchSeries date metadata differs from the reviewed contract."
        )
    if hashlib.sha256(raw).hexdigest() != contract.search_series_sha256:
        raise SourceContractError("BCCh SearchSeries bytes do not match the parser contract.")
    return BcchSeriesMetadata(
        source_key=contract.search_series_source_key,
        sha256=contract.search_series_sha256,
        series_id=contract.series_id,
        frequency_code=contract.frequency_code,
        spanish_title=contract.spanish_title,
        english_title=contract.english_title,
        first_observation=parsed_dates["firstObservation"],
        last_observation=parsed_dates["lastObservation"],
        updated_on=parsed_dates["updatedAt"],
        created_on=parsed_dates["createdAt"],
    )


def _authorized_manifest(
    release: object,
    entry: BcchReviewedSeriesEntry,
) -> ReleaseManifest:
    if type(release) is not OfficialFinanceRelease:
        raise MissingOfficialDataError("BCCh normalization requires an official finance release.")
    manifest = release.assets.manifest
    expected = tuple(
        sorted(
            (
                (entry.get_series_source_key, entry.get_series_sha256),
                (entry.search_series_source_key, entry.search_series_sha256),
            )
        )
    )
    actual = tuple((asset.source_key, asset.sha256) for asset in manifest.assets)
    if actual != expected:
        raise MissingOfficialDataError(
            "BCCh release does not contain exactly the reviewed two-asset set."
        )
    if manifest.primary_fact_source_key != entry.get_series_source_key:
        raise MissingOfficialDataError("BCCh GetSeries must be the release primary fact asset.")
    return manifest


def _quote_semantics(spec: BcchSeriesSpec) -> tuple[str, str, str | None, str | None]:
    if spec.quote_type in _RATE_QUOTE_TYPES:
        if spec.source_unit != "percent" or spec.canonical_unit != "1 / year":
            raise SourceContractError(
                "Reviewed BCCh rate series require percent-to-annual-rate semantics."
            )
        return (
            "rate",
            "market_quote",
            _RATE_TRANSFORMATION_ID,
            _RATE_TRANSFORMATION_VERSION,
        )
    if spec.quote_type == "fixing":
        if spec.source_unit != "CLP" or spec.canonical_unit != "CLP":
            raise SourceContractError("Reviewed BCCh fixing series requires exact CLP units.")
        return ("price", "covariate", None, None)
    if spec.quote_type == "published_index":
        if spec.source_unit != "index_points" or spec.canonical_unit != "dimensionless":
            raise SourceContractError("Reviewed BCCh index series requires index-point semantics.")
        return ("index", "covariate", None, None)
    if spec.quote_type == "published_change":
        if spec.source_unit != "percent" or spec.canonical_unit != "percent":
            raise SourceContractError(
                "Reviewed BCCh change series requires published-percent semantics."
            )
        return ("index", "covariate", None, None)
    raise SourceContractError("Reviewed BCCh quote semantics are unsupported.")


def _source_refs(
    release: OfficialFinanceRelease,
) -> tuple[SourceRef, ...]:
    manifest = release.assets.manifest
    return tuple(
        SourceRef(
            source_key=asset.source_key,
            name=f"Banco Central de Chile BDE {asset.source_key}",
            url=_SAFE_BCCH_DOCUMENTATION_URL,
            release_date=asset.released_at,
            release_missingness_reason=asset.release_missingness_reason,
            retrieved_at=asset.available_at,
            sha256=asset.sha256,
            vintage=release.vintage,
            provisional=False,
        )
        for asset in manifest.assets
    )


class BcchBdeAdapter:
    """Normalize only independently reviewed, exact dual-asset BCCh releases."""

    def __init__(
        self,
        reviewed_registry: BcchReviewedRegistry = (PRODUCTION_BCCH_REVIEWED_REGISTRY_V1),
    ) -> None:
        if not isinstance(reviewed_registry, BcchReviewedRegistry):
            raise DataContractError("BCCh adapter requires an immutable reviewed registry.")
        self._reviewed_registry = reviewed_registry

    def normalize_series(
        self,
        *,
        spec: BcchSeriesSpec,
        registry_entry: BcchReviewedSeriesEntry,
        release: OfficialFinanceRelease,
        identity_registry: IdentityRegistry,
    ) -> MarketQuoteBatch:
        """Authorize before opening bytes, then normalize without imputation."""

        # Authorization steps 1--4 deliberately avoid all release/path access.
        try:
            authorized = require_reviewed_bcch_entry(
                spec.series_id,
                self._reviewed_registry,
            )
        except (AttributeError, TypeError):
            raise MissingOfficialDataError(
                "BCCh caller spec is not authorized by the reviewed registry."
            ) from None
        if registry_entry is not authorized:
            raise MissingOfficialDataError(
                "The supplied BCCh entry is not the authorized registry object."
            )
        if not self._reviewed_registry.entry_matches_snapshot(authorized):
            raise MissingOfficialDataError(
                "The authorized BCCh reviewed registry entry has changed."
            )
        try:
            reviewed_digest = bcch_spec_digest(authorized.spec)
        except (AttributeError, DataContractError, TypeError, ValueError, OverflowError):
            raise MissingOfficialDataError(
                "The authorized BCCh semantic spec digest cannot be verified."
            ) from None
        if (
            authorized.spec_digest != reviewed_digest
            or registry_entry.spec_digest != authorized.spec_digest
        ):
            raise MissingOfficialDataError("The authorized BCCh semantic spec digest has changed.")
        try:
            caller_digest = bcch_spec_digest(spec)
        except (AttributeError, DataContractError, TypeError, ValueError, OverflowError):
            raise MissingOfficialDataError(
                "The caller BCCh spec digest cannot be verified."
            ) from None
        if spec != authorized.spec or caller_digest != authorized.spec_digest:
            raise MissingOfficialDataError(
                "The caller BCCh spec does not match the reviewed spec digest."
            )

        # Step 5 proves the exact asset set before a VerifiedReleaseAssets.open call.
        manifest = _authorized_manifest(release, authorized)
        if not isinstance(identity_registry, IdentityRegistry):
            raise DataContractError("BCCh normalization requires an IdentityRegistry.")

        # Both descriptors are opened/read before either pure parser receives bytes.
        with release.assets.open(authorized.get_series_source_key) as get_stream:
            get_raw = get_stream.read()
        with release.assets.open(authorized.search_series_source_key) as search_stream:
            search_raw = search_stream.read()

        raw_contract = authorized.raw_contract
        parsed = parse_bcch_get_series(get_raw, raw_contract)
        parse_bcch_search_series(search_raw, raw_contract)
        periods = tuple(
            (
                observation,
                _calendar_period_end(
                    observation.observed_on,
                    authorized.frequency_code,
                ),
            )
            for observation in parsed.observations
        )
        if any(
            observation.observed_on < release.reference_start or period_end > release.reference_end
            for observation, period_end in periods
        ):
            raise SourceContractError(
                "BCCh observations fall outside the release reference interval."
            )

        semantic_kind, role, transformation_id, transformation_version = _quote_semantics(spec)
        primary = manifest.primary_fact_asset
        rows: list[dict[str, object]] = []
        for observation, period_end in periods:
            published = observation.published_value
            value = (
                None
                if published is None
                else (
                    published / 100.0 if transformation_id == _RATE_TRANSFORMATION_ID else published
                )
            )
            missingness = (
                "not_applicable_observed"
                if value is not None
                else "bcch_source_observation_missing"
            )
            rows.append(
                {
                    "domain": "market_quote",
                    "variable": spec.variable,
                    "value": value,
                    "semantic_kind": semantic_kind,
                    "unit": spec.canonical_unit,
                    "population_basis": "not_applicable",
                    "observation_role": role,
                    "parity_scope": "not_applicable",
                    "period_start": observation.observed_on,
                    "period_end": period_end,
                    "source_key": primary.source_key,
                    "vintage": release.vintage,
                    "released_at": primary.released_at,
                    "release_missingness_reason": (
                        None
                        if primary.release_missingness_reason is None
                        else primary.release_missingness_reason.value
                    ),
                    "available_at": manifest.available_at,
                    "provisional": False,
                    "transformation_id": transformation_id,
                    "transformation_version": transformation_version,
                    "aggregation_rule": (
                        "one BCCh BDE observation retained at its published date; "
                        "no interpolation or carry-forward"
                    ),
                    "missingness_reason": missingness,
                    "source_status_code": observation.status_code,
                    "source_series_id": spec.series_id,
                    "quote_type": spec.quote_type,
                    "tenor_years": spec.tenor_years,
                    "currency": spec.currency,
                    "indexation": spec.indexation,
                    "instrument_type": spec.instrument_type,
                    "compounding": spec.compounding,
                    "day_count": spec.day_count,
                    "date_basis": "market_observation",
                }
            )
        payload = pd.DataFrame(rows)
        caller_snapshot = identity_registry.snapshot()
        staging_registry = IdentityRegistry()
        for record in caller_snapshot.records:
            staging_registry.record(record)
        assigned = assign_observation_ids(
            payload,
            domain=ObservationDomain.MARKET_QUOTE,
            domain_key=MARKET_QUOTE_KEY,
            source_identity={
                "publisher": "Banco Central de Chile",
                "source_system": "Base de Datos Estadísticos",
                "series_id": authorized.series_id,
            },
            manifest=manifest,
            registry=staging_registry,
        )
        validated = validate_market_quotes(assigned)
        observation_start = min(item.observed_on for item in parsed.observations)
        observation_end = max(period_end for _, period_end in periods)
        provenance = VariableProvenance(
            sources=_source_refs(release),
            release_id=manifest.release_id.value,
            available_at=manifest.available_at,
            observation_start=observation_start,
            observation_end=observation_end,
            dimensions=(
                "market_observation_date",
                "source_series_id",
                "quote_type",
                "tenor_years",
                "currency",
                "indexation",
                "instrument_type",
            ),
            aggregation_rules=(
                "one BCCh BDE observation retained at its published date; "
                "no interpolation or carry-forward",
            ),
            missingness_reason=("non-OK and blank BCCh observations remain explicit missing facts"),
            transformation_ids=(() if transformation_id is None else (transformation_id,)),
            notes=(
                "SearchSeries metadata dates remain date-only source metadata.",
                "Knowledge-time availability is the maximum of both raw assets.",
            ),
        )
        batch = NormalizedDomainBatch.create(
            manifest=manifest,
            observations=validated,
            provenance=provenance,
            identities=staging_registry.snapshot(),
        )
        if identity_registry.snapshot() != caller_snapshot:
            raise DataContractError(
                "BCCh identity registry changed during transactional normalization."
            )
        for record in batch.identities.records:
            identity_registry.record(record)
        return batch
