"""Strict, isolated observation schemas for every normalized data domain."""

from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum
from numbers import Real
from typing import Final

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.data.envelope import (
    NormalizedDomainBatch,
    ObservationEnvelopeSchema,
)

SHARED_FACT_KEY: Final = (
    "variable",
    "semantic_kind",
    "unit",
    "population_basis",
    "observation_role",
    "parity_scope",
    "period_start",
    "period_end",
    "aggregation_rule",
)

DEMOGRAPHIC_OBSERVATION_KEY: Final = (
    *SHARED_FACT_KEY,
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

ENVIRONMENTAL_OBSERVATION_KEY: Final = (
    *SHARED_FACT_KEY,
    "station",
    "region",
    "commune",
    "geography_basis",
    "geography_vintage",
    "pollutant",
    "environmental_measure",
    "averaging_interval",
    "validation_status",
)

HEALTH_SYSTEM_OBSERVATION_KEY: Final = (
    *SHARED_FACT_KEY,
    "facility",
    "health_service",
    "region",
    "commune",
    "geography_basis",
    "geography_vintage",
    "service_measure",
    "capacity_measure",
    "event_measure",
    "reporting_grain",
)

MARKET_QUOTE_KEY: Final = (
    *SHARED_FACT_KEY,
    "source_series_id",
    "quote_type",
    "tenor_years",
    "currency",
    "indexation",
    "instrument_type",
    "compounding",
    "day_count",
    "date_basis",
)

REGULATORY_RATE_KEY: Final = (
    *SHARED_FACT_KEY,
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
    "compounding",
    "day_count",
)

REGULATORY_MORTALITY_KEY: Final = (
    *SHARED_FACT_KEY,
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
)

PENSION_PRODUCT_STATISTIC_KEY: Final = (
    *SHARED_FACT_KEY,
    "financial_entity_id",
    "financial_entity_type",
    "pension_type",
    "modality",
    "offer_type",
    "intermediation_type",
    "contract_state",
    "aggregation_grain",
    "product_term_set_id",
)

type DemographicObservationBatch = NormalizedDomainBatch[pd.DataFrame]
type EnvironmentalObservationBatch = NormalizedDomainBatch[pd.DataFrame]
type HealthSystemObservationBatch = NormalizedDomainBatch[pd.DataFrame]
type MarketQuoteBatch = NormalizedDomainBatch[pd.DataFrame]
type RegulatoryRateBatch = NormalizedDomainBatch[pd.DataFrame]
type RegulatoryMortalityBatch = NormalizedDomainBatch[pd.DataFrame]
type PensionProductStatisticBatch = NormalizedDomainBatch[pd.DataFrame]


class MarketQuoteType(StrEnum):
    """Canonical publisher quote categories supported by the v1 contract."""

    BENCHMARK_YIELD = "benchmark_yield"
    SWAP_FIXED_RATE = "swap_fixed_rate"
    FIXING = "fixing"
    PUBLISHED_INDEX = "published_index"
    PUBLISHED_CHANGE = "published_change"


class RegulatoryQuoteType(StrEnum):
    """Canonical regulatory quote categories supported by the v1 contract."""

    PUBLISHED_ZERO_RATE = "published_zero_rate"
    REGULATORY_DISCOUNT_RATE = "regulatory_discount_rate"
    ASSET_SUFFICIENCY_IRR = "asset_sufficiency_irr"
    TECHNICAL_RATE_INPUT = "technical_rate_input"
    TECHNICAL_RATE = "technical_rate"


class CompoundingConvention(StrEnum):
    """Explicit compounding conventions; unknown conventions remain null."""

    SIMPLE = "simple"
    CONTINUOUS = "continuous"
    ANNUAL_COMPOUNDED = "annual_compounded"
    SEMIANNUAL_COMPOUNDED = "semiannual_compounded"
    QUARTERLY_COMPOUNDED = "quarterly_compounded"
    MONTHLY_COMPOUNDED = "monthly_compounded"


class DayCountConvention(StrEnum):
    """Explicit day-count conventions; unknown conventions remain null."""

    ACTUAL_360 = "actual_360"
    ACTUAL_365_FIXED = "actual_365_fixed"
    ACTUAL_ACTUAL = "actual_actual"
    THIRTY_360 = "thirty_360"
    BUSINESS_252 = "business_252"


class RegulatoryBasis(StrEnum):
    """Reviewed Chilean finance and pension regulatory bases."""

    NCG_209 = "NCG_209"
    VTD_CMF = "VTD_CMF"
    TM_2020 = "TM_2020"
    TITRP = "TITRP"


class FinancialEntityType(StrEnum):
    """Canonical finance publisher entity categories."""

    INSURER = "insurer"
    AFP = "afp"
    MARKET_TOTAL = "market_total"


class PensionType(StrEnum):
    """Conservative v1 pension-type vocabulary."""

    NOT_APPLICABLE = "not_applicable"
    OLD_AGE = "old_age"
    DISABILITY = "disability"
    SURVIVOR = "survivor"


class PensionModality(StrEnum):
    """Conservative v1 pension-payment modality vocabulary."""

    NOT_APPLICABLE = "not_applicable"
    PROGRAMMED_WITHDRAWAL = "programmed_withdrawal"
    TEMPORARY_ANNUITY = "temporary_annuity"
    IMMEDIATE_ANNUITY = "immediate_annuity"
    DEFERRED_ANNUITY = "deferred_annuity"
    COMBINED_ANNUITY = "combined_annuity"
    TEMPORARILY_INCREASED_ANNUITY = "temporarily_increased_annuity"


class OfferType(StrEnum):
    """Reviewed SCOMP offer-type concepts."""

    NOT_APPLICABLE = "not_applicable"
    INTERNAL = "internal"
    EXTERNAL = "external"


class IntermediationType(StrEnum):
    """Intermediation v1 remains closed until a reviewed source contract exists."""

    NOT_APPLICABLE = "not_applicable"


class ContractState(StrEnum):
    """Published annuity contract-state distinctions."""

    NOT_APPLICABLE = "not_applicable"
    CONTRACTED = "contracted"
    ACTIVE = "active"
    RECEIVING = "receiving"


class AggregationGrain(StrEnum):
    """Entity-level aggregation grains supported by the v1 contract."""

    INSURER = "insurer"
    AFP = "afp"
    MARKET_TOTAL = "market_total"


_INVALID_SCALAR: Final = "__invalid_domain_scalar__"
_ISO_DATE_LIKE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ][0-9:.+-]+(?:Z)?)?$")
_DEMOGRAPHIC_NUMERIC_COLUMNS: Final = (
    "value",
    "age_lower",
    "age_upper",
    "age_quantity",
    "birth_order",
)
_OPTIONAL_DEMOGRAPHIC_TEXT: Final = (
    "date_basis",
    "age_role",
    "age_measure_unit",
    "sex_role",
    "commune",
    "geography_basis",
    "geography_vintage",
    "cause_code_system",
    "cause_role",
    "external_cause",
    "nationality",
    "nationality_role",
    "country_of_birth",
    "education_level",
    "residence_five_years_ago",
    "migration_status",
)
_MARKET_QUOTE_TYPES: Final = frozenset(item.value for item in MarketQuoteType)
_REGULATORY_QUOTE_TYPES: Final = frozenset(item.value for item in RegulatoryQuoteType)
_COMPOUNDING_CONVENTIONS: Final = frozenset(item.value for item in CompoundingConvention)
_DAY_COUNT_CONVENTIONS: Final = frozenset(item.value for item in DayCountConvention)
_REGULATORY_BASES: Final = frozenset(item.value for item in RegulatoryBasis)
_REGULATORY_RATE_BASES: Final = frozenset(
    {
        RegulatoryBasis.NCG_209.value,
        RegulatoryBasis.VTD_CMF.value,
        RegulatoryBasis.TITRP.value,
    }
)
_FINANCIAL_ENTITY_TYPES: Final = frozenset(item.value for item in FinancialEntityType)
_PENSION_TYPES: Final = frozenset(item.value for item in PensionType)
_PENSION_MODALITIES: Final = frozenset(item.value for item in PensionModality)
_OFFER_TYPES: Final = frozenset(item.value for item in OfferType)
_INTERMEDIATION_TYPES: Final = frozenset(item.value for item in IntermediationType)
_CONTRACT_STATES: Final = frozenset(item.value for item in ContractState)
_AGGREGATION_GRAINS: Final = frozenset(item.value for item in AggregationGrain)
_FINANCE_OPTIONAL_TEXT: Final = (
    "compounding",
    "day_count",
)
_FINANCE_REQUIRED_ENVELOPE_TEXT: Final = (
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
_FINANCE_OPTIONAL_ENVELOPE_TEXT: Final = (
    "release_missingness_reason",
    "transformation_id",
    "transformation_version",
    "source_status_code",
)
_EFFECTIVE_DATE_COLUMNS: Final = (
    "effective_start",
    "effective_end",
    "maximum_effective_end",
)
_PENSION_STATISTIC_TRIPLES: Final = frozenset(
    {
        ("annuity_new_business_mean_rate", "rate", "1 / year"),
        ("annuity_policy_count", "count", "event"),
        ("annuity_single_premium", "amount", "UF"),
        ("annuity_average_premium", "amount", "UF"),
        ("intermediation_commission_rate", "index", "percent"),
        ("intermediation_commission_amount", "amount", "UF"),
        ("scomp_selected_pension_count", "count", "event"),
        ("scomp_selected_pension_share", "index", "percent"),
    }
)


def _is_missing(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    return False


def _strict_number(value: object) -> object:
    if _is_missing(value):
        return np.nan
    if isinstance(value, bool) or not isinstance(value, Real):
        return _INVALID_SCALAR
    return value


def _strict_integer(value: object) -> object:
    if _is_missing(value):
        return np.nan
    if isinstance(value, bool) or not isinstance(value, Real):
        return _INVALID_SCALAR
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        return _INVALID_SCALAR
    return value


def _effective_date(value: object) -> object:
    if _is_missing(value):
        return None
    if type(value) is date:
        return value
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        return _INVALID_SCALAR
    try:
        return date.fromisoformat(value)
    except ValueError:
        return _INVALID_SCALAR


def _registration_time(value: object) -> object:
    if _is_missing(value):
        return pd.NaT
    if isinstance(value, str) and _ISO_DATE_LIKE.fullmatch(value) is None:
        return _INVALID_SCALAR
    if not isinstance(
        value,
        (str, date, datetime, pd.Timestamp, np.datetime64),
    ):
        return _INVALID_SCALAR
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return _INVALID_SCALAR
    return _INVALID_SCALAR if pd.isna(parsed) else parsed


def _present_nonblank(value: object) -> bool:
    return _is_missing(value) or (isinstance(value, str) and bool(value.strip()))


def _required_nonblank(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    valid = pd.Series(True, index=frame.index, dtype=bool)
    for column in columns:
        valid &= frame[column].map(lambda value: isinstance(value, str) and bool(value.strip()))
    return valid


def _unique_fact_and_key(
    frame: pd.DataFrame,
    key: tuple[str, ...],
) -> pd.Series:
    return pd.Series(
        ~frame.duplicated(["release_id", "fact_id"], keep=False)
        & ~frame.duplicated(list(key), keep=False),
        index=frame.index,
        dtype=bool,
    )


def _optional_nonblank(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    valid = pd.Series(True, index=frame.index, dtype=bool)
    for column in columns:
        valid &= frame[column].map(_present_nonblank)
    return valid


def _finance_text_atoms_are_canonical(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> pd.Series:
    valid = pd.Series(True, index=frame.index, dtype=bool)
    for column in required:
        valid &= frame[column].map(lambda value: type(value) is str and bool(value.strip()))
    for column in optional:
        valid &= frame[column].map(
            lambda value: _is_missing(value) or (type(value) is str and bool(value.strip()))
        )
    return valid


def _finance_envelope_contract(
    frame: pd.DataFrame,
    *,
    role: str,
) -> pd.Series:
    return pd.Series(
        frame["parity_scope"].eq("not_applicable")
        & frame["observation_role"].eq(role)
        & frame["population_basis"].eq("not_applicable"),
        index=frame.index,
        dtype=bool,
    )


def _values_respect_finance_semantics(frame: pd.DataFrame) -> pd.Series:
    value = frame["value"]
    present = value.notna()
    valid = pd.Series(True, index=frame.index, dtype=bool)
    nonnegative = frame["semantic_kind"].isin({"count", "price", "amount", "probability"})
    valid &= ~(present & nonnegative & value.lt(0))
    count = present & frame["semantic_kind"].eq("count")
    valid &= ~count | value.mod(1).eq(0)
    probability = present & frame["semantic_kind"].eq("probability")
    valid &= ~probability | value.between(0, 1)
    discount_factor = present & frame["semantic_kind"].eq("discount_factor")
    valid &= ~discount_factor | value.gt(0)
    return pd.Series(valid, index=frame.index, dtype=bool)


def _effective_interval_is_valid(frame: pd.DataFrame) -> pd.Series:
    def valid_row(start: object, end: object, maximum: object) -> bool:
        if not isinstance(start, date) or isinstance(start, datetime):
            return False
        if end is not None and (
            not isinstance(end, date) or isinstance(end, datetime) or end <= start
        ):
            return False
        if maximum is not None and (
            not isinstance(maximum, date) or isinstance(maximum, datetime) or maximum <= start
        ):
            return False
        return end is None or maximum is None or end <= maximum

    return pd.Series(
        (
            valid_row(start, end, maximum)
            for start, end, maximum in frame.loc[:, list(_EFFECTIVE_DATE_COLUMNS)].itertuples(
                index=False, name=None
            )
        ),
        index=frame.index,
        dtype=bool,
    )


def _finance_envelope_date_atoms_are_canonical(
    frame: pd.DataFrame,
) -> pd.Series:
    return pd.Series(
        (
            type(period_start) is date
            and type(period_end) is date
            and (_is_missing(released_at) or type(released_at) is date)
            and type(available_at) is datetime
            for period_start, period_end, released_at, available_at in frame.loc[
                :,
                [
                    "period_start",
                    "period_end",
                    "released_at",
                    "available_at",
                ],
            ].itertuples(index=False, name=None)
        ),
        index=frame.index,
        dtype=bool,
    )


class DemographicObservationSchema(ObservationEnvelopeSchema):
    """Demographic facts without finance, environment, or health columns."""

    value: Series[float] = pa.Field(nullable=True, coerce=True)
    age_lower: Series[float] = pa.Field(nullable=True, coerce=True)
    age_upper: Series[float] = pa.Field(nullable=True, coerce=True)
    age_open: Series[bool]
    sex: Series[str]
    region: Series[str]
    cause: Series[str]
    date_basis: Series[str] = pa.Field(nullable=True)
    age_role: Series[str] = pa.Field(nullable=True)
    age_measure_unit: Series[str] = pa.Field(nullable=True)
    age_quantity: Series[float] = pa.Field(nullable=True, coerce=True)
    sex_role: Series[str] = pa.Field(nullable=True)
    commune: Series[str] = pa.Field(nullable=True)
    geography_basis: Series[str] = pa.Field(nullable=True)
    geography_vintage: Series[str] = pa.Field(nullable=True)
    cause_code_system: Series[str] = pa.Field(nullable=True)
    cause_role: Series[str] = pa.Field(nullable=True)
    external_cause: Series[str] = pa.Field(nullable=True)
    registration_start: Series[pd.Timestamp] = pa.Field(nullable=True, coerce=True)
    registration_end: Series[pd.Timestamp] = pa.Field(nullable=True, coerce=True)
    birth_order: Series[float] = pa.Field(nullable=True, coerce=True)
    nationality: Series[str] = pa.Field(nullable=True)
    nationality_role: Series[str] = pa.Field(nullable=True)
    country_of_birth: Series[str] = pa.Field(nullable=True)
    education_level: Series[str] = pa.Field(nullable=True)
    residence_five_years_ago: Series[str] = pa.Field(nullable=True)
    migration_status: Series[str] = pa.Field(nullable=True)

    @pa.dataframe_parser
    @classmethod
    def parse_strict_fields(cls, frame: pd.DataFrame) -> pd.DataFrame:
        parsed = frame.copy(deep=True)
        if not parsed.columns.is_unique:
            return parsed
        for column in _DEMOGRAPHIC_NUMERIC_COLUMNS:
            if column in parsed:
                parsed[column] = parsed[column].map(_strict_number)
        for column in ("registration_start", "registration_end"):
            if column in parsed:
                parsed[column] = parsed[column].map(_registration_time)
        return parsed

    @pa.dataframe_check(ignore_na=False)
    def demographic_domain_is_exact(cls, frame: pd.DataFrame) -> pd.Series:
        return frame["domain"].eq("demographic")

    @pa.dataframe_check(ignore_na=False)
    def demographic_age_state_is_valid(cls, frame: pd.DataFrame) -> pd.Series:
        lower = frame["age_lower"]
        upper = frame["age_upper"]
        age_open = frame["age_open"]
        finite_lower = lower.notna() & np.isfinite(lower) & lower.ge(0)
        finite_upper = upper.notna() & np.isfinite(upper) & upper.ge(0)
        not_age_specific = lower.isna() & upper.isna() & ~age_open
        closed = finite_lower & finite_upper & ~age_open & upper.gt(lower)
        open_group = finite_lower & upper.isna() & age_open
        return pd.Series(
            not_age_specific | closed | open_group,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def demographic_age_quantity_is_valid(cls, frame: pd.DataFrame) -> pd.Series:
        quantity = frame["age_quantity"]
        return pd.Series(
            quantity.isna() | (np.isfinite(quantity) & quantity.ge(0)),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def demographic_text_is_nonblank(cls, frame: pd.DataFrame) -> pd.Series:
        valid = _required_nonblank(frame, ("sex", "region", "cause"))
        for column in _OPTIONAL_DEMOGRAPHIC_TEXT:
            valid &= frame[column].map(_present_nonblank)
        return valid

    @pa.dataframe_check(ignore_na=False)
    def demographic_registration_is_paired(cls, frame: pd.DataFrame) -> pd.Series:
        start = frame["registration_start"]
        end = frame["registration_end"]
        absent = start.isna() & end.isna()
        ordered = start.notna() & end.notna() & end.gt(start)
        return pd.Series(absent | ordered, index=frame.index, dtype=bool)

    @pa.dataframe_check(ignore_na=False)
    def parity_contract_is_valid(cls, frame: pd.DataFrame) -> pd.Series:
        births = frame["variable"].eq("births")
        order = frame["birth_order"]
        integral_positive_order = (
            order.notna() & np.isfinite(order) & order.ge(1) & order.mod(1).eq(0)
        )
        nonbirth = ~births & frame["parity_scope"].eq("not_applicable") & order.isna()
        all_orders = births & frame["parity_scope"].eq("all_orders") & order.isna()
        known_order = births & frame["parity_scope"].eq("known_order") & integral_positive_order
        unknown_order = births & frame["parity_scope"].eq("unknown_order") & order.isna()
        return pd.Series(
            nonbirth | all_orders | known_order | unknown_order,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def demographic_keys_are_unique(cls, frame: pd.DataFrame) -> pd.Series:
        return _unique_fact_and_key(frame, DEMOGRAPHIC_OBSERVATION_KEY)

    class Config:
        strict = True
        coerce = False
        unique_column_names = True


class EnvironmentalObservationSchema(ObservationEnvelopeSchema):
    """Environmental observations without demographic or health columns."""

    value: Series[float] = pa.Field(nullable=True, coerce=True)
    station: Series[str] = pa.Field(nullable=True)
    region: Series[str]
    commune: Series[str] = pa.Field(nullable=True)
    geography_basis: Series[str] = pa.Field(nullable=True)
    geography_vintage: Series[str] = pa.Field(nullable=True)
    pollutant: Series[str] = pa.Field(nullable=True)
    environmental_measure: Series[str]
    averaging_interval: Series[str]
    validation_status: Series[str]

    @pa.dataframe_parser
    @classmethod
    def parse_strict_value(cls, frame: pd.DataFrame) -> pd.DataFrame:
        parsed = frame.copy(deep=True)
        if parsed.columns.is_unique and "value" in parsed:
            parsed["value"] = parsed["value"].map(_strict_number)
        return parsed

    @pa.dataframe_check(ignore_na=False)
    def environmental_domain_is_exact(cls, frame: pd.DataFrame) -> pd.Series:
        return frame["domain"].eq("environmental")

    @pa.dataframe_check(ignore_na=False)
    def environmental_fields_are_declared(cls, frame: pd.DataFrame) -> pd.Series:
        valid = _required_nonblank(
            frame,
            (
                "region",
                "environmental_measure",
                "averaging_interval",
                "validation_status",
            ),
        )
        for column in (
            "station",
            "commune",
            "geography_basis",
            "geography_vintage",
            "pollutant",
        ):
            valid &= frame[column].map(_present_nonblank)
        return valid

    @pa.dataframe_check(ignore_na=False)
    def environmental_pollutant_contract_is_compatible(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        pollutant_present = frame["pollutant"].notna()
        station_present = frame["station"].map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        concentration_semantics = (
            frame["semantic_kind"].eq("index")
            & frame["unit"].eq("microgram / meter ** 3")
            & frame["population_basis"].eq("not_applicable")
        )
        explicitly_concentration = frame["environmental_measure"].eq("concentration")
        return pd.Series(
            (~pollutant_present | (station_present & concentration_semantics))
            & (~explicitly_concentration | pollutant_present),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def environmental_keys_are_unique(cls, frame: pd.DataFrame) -> pd.Series:
        return _unique_fact_and_key(frame, ENVIRONMENTAL_OBSERVATION_KEY)

    class Config:
        strict = True
        coerce = False
        unique_column_names = True


class HealthSystemObservationSchema(ObservationEnvelopeSchema):
    """Health-system observations without demographic or environment columns."""

    value: Series[float] = pa.Field(nullable=True, coerce=True)
    facility: Series[str]
    health_service: Series[str]
    region: Series[str]
    commune: Series[str] = pa.Field(nullable=True)
    geography_basis: Series[str] = pa.Field(nullable=True)
    geography_vintage: Series[str] = pa.Field(nullable=True)
    service_measure: Series[str]
    capacity_measure: Series[str]
    event_measure: Series[str]
    reporting_grain: Series[str]

    @pa.dataframe_parser
    @classmethod
    def parse_strict_value(cls, frame: pd.DataFrame) -> pd.DataFrame:
        parsed = frame.copy(deep=True)
        if parsed.columns.is_unique and "value" in parsed:
            parsed["value"] = parsed["value"].map(_strict_number)
        return parsed

    @pa.dataframe_check(ignore_na=False)
    def health_system_domain_is_exact(cls, frame: pd.DataFrame) -> pd.Series:
        return frame["domain"].eq("health_system")

    @pa.dataframe_check(ignore_na=False)
    def health_system_fields_are_declared(cls, frame: pd.DataFrame) -> pd.Series:
        valid = _required_nonblank(
            frame,
            (
                "facility",
                "health_service",
                "region",
                "service_measure",
                "capacity_measure",
                "event_measure",
                "reporting_grain",
            ),
        )
        for column in ("commune", "geography_basis", "geography_vintage"):
            valid &= frame[column].map(_present_nonblank)
        return valid

    @pa.dataframe_check(ignore_na=False)
    def health_system_keys_are_unique(cls, frame: pd.DataFrame) -> pd.Series:
        return _unique_fact_and_key(frame, HEALTH_SYSTEM_OBSERVATION_KEY)

    class Config:
        strict = True
        coerce = False
        unique_column_names = True


class MarketQuoteSchema(ObservationEnvelopeSchema):
    """Observed market quotes at their exact published instrument grain."""

    value: Series[float] = pa.Field(nullable=True, coerce=True)
    source_series_id: Series[str]
    quote_type: Series[str] = pa.Field(isin=sorted(_MARKET_QUOTE_TYPES))
    tenor_years: Series[float] = pa.Field(nullable=True, coerce=True)
    currency: Series[str]
    indexation: Series[str]
    instrument_type: Series[str]
    compounding: Series[str] = pa.Field(
        nullable=True,
        isin=sorted(_COMPOUNDING_CONVENTIONS),
    )
    day_count: Series[str] = pa.Field(
        nullable=True,
        isin=sorted(_DAY_COUNT_CONVENTIONS),
    )
    date_basis: Series[str]

    @pa.dataframe_parser
    @classmethod
    def parse_strict_fields(cls, frame: pd.DataFrame) -> pd.DataFrame:
        parsed = frame.copy(deep=True)
        if not parsed.columns.is_unique:
            return parsed
        if "value" in parsed:
            parsed["value"] = parsed["value"].map(_strict_number)
        if "tenor_years" in parsed:
            parsed["tenor_years"] = parsed["tenor_years"].map(_strict_number)
        return parsed

    @pa.dataframe_check(ignore_na=False)
    def market_quote_domain_is_exact(cls, frame: pd.DataFrame) -> pd.Series:
        return frame["domain"].eq("market_quote")

    @pa.dataframe_check(ignore_na=False)
    def market_quote_canonical_date_atoms_are_owned(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _finance_envelope_date_atoms_are_canonical(frame)

    @pa.dataframe_check(ignore_na=False)
    def market_quote_role_and_parity_are_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        quoted_rate = frame["quote_type"].isin(
            {
                MarketQuoteType.BENCHMARK_YIELD.value,
                MarketQuoteType.SWAP_FIXED_RATE.value,
            }
        ) & frame["observation_role"].eq("market_quote")
        covariate = frame["quote_type"].isin(
            {
                MarketQuoteType.FIXING.value,
                MarketQuoteType.PUBLISHED_INDEX.value,
                MarketQuoteType.PUBLISHED_CHANGE.value,
            }
        ) & frame["observation_role"].eq("covariate")
        return pd.Series(
            frame["parity_scope"].eq("not_applicable")
            & frame["population_basis"].eq("not_applicable")
            & (quoted_rate | covariate),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def market_quote_canonical_fields_are_declared(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _finance_text_atoms_are_canonical(
            frame,
            required=(
                *_FINANCE_REQUIRED_ENVELOPE_TEXT,
                "source_series_id",
                "quote_type",
                "currency",
                "indexation",
                "instrument_type",
                "date_basis",
            ),
            optional=(
                *_FINANCE_OPTIONAL_ENVELOPE_TEXT,
                *_FINANCE_OPTIONAL_TEXT,
            ),
        )

    @pa.dataframe_check(ignore_na=False)
    def market_quote_tenor_is_positive_when_present(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        tenor = frame["tenor_years"]
        return pd.Series(
            tenor.isna() | (np.isfinite(tenor) & tenor.gt(0)),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def market_quote_type_matches_semantics(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        quote_type = frame["quote_type"]
        rate = (
            quote_type.isin(
                {
                    MarketQuoteType.BENCHMARK_YIELD.value,
                    MarketQuoteType.SWAP_FIXED_RATE.value,
                }
            )
            & frame["semantic_kind"].eq("rate")
            & frame["unit"].eq("1 / year")
        )
        fixing = quote_type.eq(MarketQuoteType.FIXING.value) & frame["semantic_kind"].eq("price")
        published_index = (
            quote_type.eq(MarketQuoteType.PUBLISHED_INDEX.value)
            & frame["variable"].eq("consumer_price_index")
            & frame["semantic_kind"].eq("index")
            & frame["unit"].eq("dimensionless")
        )
        published_change = (
            quote_type.eq(MarketQuoteType.PUBLISHED_CHANGE.value)
            & frame["variable"].eq("consumer_price_change")
            & frame["semantic_kind"].eq("index")
            & frame["unit"].eq("percent")
        )
        return pd.Series(
            rate | fixing | published_index | published_change,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def market_quote_instrument_grain_is_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        quoted_curve = frame["quote_type"].isin(
            {
                MarketQuoteType.BENCHMARK_YIELD.value,
                MarketQuoteType.SWAP_FIXED_RATE.value,
            }
        )
        bcp = (
            frame["variable"].eq("official_benchmark_yield")
            & frame["quote_type"].eq(MarketQuoteType.BENCHMARK_YIELD.value)
            & frame["currency"].eq("CLP")
            & frame["indexation"].eq("nominal")
            & frame["instrument_type"].eq("bcp_benchmark_bond")
        )
        bcu_btu = (
            frame["variable"].eq("official_benchmark_yield")
            & frame["quote_type"].eq(MarketQuoteType.BENCHMARK_YIELD.value)
            & frame["currency"].eq("UF")
            & frame["indexation"].eq("UF")
            & frame["instrument_type"].eq("bcu_btu_benchmark_bond")
        )
        nominal_swap = (
            frame["variable"].eq("official_swap_fixed_rate")
            & frame["quote_type"].eq(MarketQuoteType.SWAP_FIXED_RATE.value)
            & frame["currency"].eq("CLP")
            & frame["indexation"].eq("nominal")
            & frame["instrument_type"].eq("spc_nominal_fixed_swap")
        )
        uf_swap = (
            frame["variable"].eq("official_swap_fixed_rate")
            & frame["quote_type"].eq(MarketQuoteType.SWAP_FIXED_RATE.value)
            & frame["currency"].eq("UF")
            & frame["indexation"].eq("UF")
            & frame["instrument_type"].eq("spc_uf_fixed_swap")
        )
        return pd.Series(
            ~quoted_curve | bcp | bcu_btu | nominal_swap | uf_swap,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def market_quote_values_are_valid(cls, frame: pd.DataFrame) -> pd.Series:
        return _values_respect_finance_semantics(frame)

    @pa.dataframe_check(ignore_na=False)
    def uf_fixing_has_exchange_semantics(cls, frame: pd.DataFrame) -> pd.Series:
        is_uf_fixing = frame["variable"].eq("uf_fixing_clp") | frame["instrument_type"].eq(
            "uf_fixing"
        )
        exact = (
            frame["variable"].eq("uf_fixing_clp")
            & frame["quote_type"].eq(MarketQuoteType.FIXING.value)
            & frame["semantic_kind"].eq("price")
            & frame["unit"].eq("CLP")
            & frame["currency"].eq("CLP")
            & frame["indexation"].eq("UF")
            & frame["instrument_type"].eq("uf_fixing")
            & frame["tenor_years"].isna()
            & frame["compounding"].isna()
            & frame["day_count"].isna()
        )
        return pd.Series(
            ~is_uf_fixing | exact,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def market_quote_keys_are_unique(cls, frame: pd.DataFrame) -> pd.Series:
        return _unique_fact_and_key(frame, MARKET_QUOTE_KEY)

    class Config:
        strict = True
        coerce = False
        unique_column_names = True


class RegulatoryRateSchema(ObservationEnvelopeSchema):
    """Published regulatory rates without curve construction or measure labels."""

    value: Series[float] = pa.Field(nullable=True, coerce=True)
    regulatory_basis: Series[str] = pa.Field(isin=sorted(_REGULATORY_RATE_BASES))
    regulatory_use: Series[str]
    quote_type: Series[str] = pa.Field(isin=sorted(_REGULATORY_QUOTE_TYPES))
    effective_start: Series[object]
    effective_end: Series[object] = pa.Field(nullable=True)
    maximum_effective_end: Series[object] = pa.Field(nullable=True)
    tenor_years: Series[float] = pa.Field(nullable=True, coerce=True)
    financial_entity_id: Series[str] = pa.Field(nullable=True)
    financial_entity_type: Series[str] = pa.Field(
        nullable=True,
        isin=sorted(_FINANCIAL_ENTITY_TYPES),
    )
    currency: Series[str]
    indexation: Series[str]
    instrument_type: Series[str]
    compounding: Series[str] = pa.Field(
        nullable=True,
        isin=sorted(_COMPOUNDING_CONVENTIONS),
    )
    day_count: Series[str] = pa.Field(
        nullable=True,
        isin=sorted(_DAY_COUNT_CONVENTIONS),
    )

    @pa.dataframe_parser
    @classmethod
    def parse_strict_fields(cls, frame: pd.DataFrame) -> pd.DataFrame:
        parsed = frame.copy(deep=True)
        if not parsed.columns.is_unique:
            return parsed
        if "value" in parsed:
            parsed["value"] = parsed["value"].map(_strict_number)
        if "tenor_years" in parsed:
            parsed["tenor_years"] = parsed["tenor_years"].map(_strict_number)
        for column in _EFFECTIVE_DATE_COLUMNS:
            if column in parsed:
                parsed[column] = parsed[column].map(_effective_date)
        return parsed

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_domain_is_exact(cls, frame: pd.DataFrame) -> pd.Series:
        return frame["domain"].eq("regulatory_rate")

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_canonical_date_atoms_are_owned(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        effective = frame.loc[:, list(_EFFECTIVE_DATE_COLUMNS)].map(
            lambda value: _is_missing(value) or type(value) is date
        )
        return pd.Series(
            _finance_envelope_date_atoms_are_canonical(frame)
            & effective.all(axis=1)
            & frame["effective_start"].notna(),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_role_and_parity_are_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _finance_envelope_contract(frame, role="regulatory_input")

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_fields_are_declared(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        text = _finance_text_atoms_are_canonical(
            frame,
            required=(
                *_FINANCE_REQUIRED_ENVELOPE_TEXT,
                "regulatory_basis",
                "regulatory_use",
                "quote_type",
                "currency",
                "indexation",
                "instrument_type",
            ),
            optional=(
                *_FINANCE_OPTIONAL_ENVELOPE_TEXT,
                "financial_entity_id",
                "financial_entity_type",
                "compounding",
                "day_count",
            ),
        )
        entity_pair = frame["financial_entity_id"].isna().eq(frame["financial_entity_type"].isna())
        return pd.Series(
            text & entity_pair,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_has_rate_semantics(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return pd.Series(
            frame["semantic_kind"].eq("rate")
            & frame["unit"].eq("1 / year")
            & frame["population_basis"].eq("not_applicable"),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_basis_matches_quote_type(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        vtd = frame["regulatory_basis"].eq(RegulatoryBasis.VTD_CMF.value) & frame[
            "quote_type"
        ].isin(
            {
                RegulatoryQuoteType.PUBLISHED_ZERO_RATE.value,
                RegulatoryQuoteType.REGULATORY_DISCOUNT_RATE.value,
            }
        )
        sufficiency = frame["regulatory_basis"].eq(RegulatoryBasis.NCG_209.value) & frame[
            "quote_type"
        ].eq(RegulatoryQuoteType.ASSET_SUFFICIENCY_IRR.value)
        technical = frame["regulatory_basis"].eq(RegulatoryBasis.TITRP.value) & frame[
            "quote_type"
        ].isin(
            {
                RegulatoryQuoteType.TECHNICAL_RATE_INPUT.value,
                RegulatoryQuoteType.TECHNICAL_RATE.value,
            }
        )
        return pd.Series(
            vtd | sufficiency | technical,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_complete_canonical_grain_is_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        tenor = frame["tenor_years"]
        integral_tenor = tenor.notna() & np.isfinite(tenor) & tenor.gt(0) & tenor.mod(1).eq(0)
        no_entity = frame["financial_entity_id"].isna() & frame["financial_entity_type"].isna()
        common_published_curve = (
            frame["currency"].eq("UF")
            & frame["indexation"].eq("UF")
            & frame["compounding"].eq("annual_compounded")
            & frame["day_count"].isna()
            & no_entity
        )
        vtd_spot = (
            frame["regulatory_basis"].eq(RegulatoryBasis.VTD_CMF.value)
            & frame["quote_type"].eq(RegulatoryQuoteType.REGULATORY_DISCOUNT_RATE.value)
            & frame["variable"].eq("cmf_regulatory_discount_rate")
            & frame["instrument_type"].eq("cmf_vtd_spot_rate_ac")
            & integral_tenor
            & tenor.le(120)
            & common_published_curve
        )
        vtd_zero = (
            frame["regulatory_basis"].eq(RegulatoryBasis.VTD_CMF.value)
            & frame["quote_type"].eq(RegulatoryQuoteType.PUBLISHED_ZERO_RATE.value)
            & frame["variable"].eq("cmf_published_real_zero_rate")
            & frame["instrument_type"].eq("cmf_vtd_real_zero_curve")
            & integral_tenor
            & tenor.le(25)
            & common_published_curve
        )
        insurer_irr = (
            frame["regulatory_basis"].eq(RegulatoryBasis.NCG_209.value)
            & frame["quote_type"].eq(RegulatoryQuoteType.ASSET_SUFFICIENCY_IRR.value)
            & frame["variable"].eq("insurer_asset_sufficiency_irr")
            & frame["currency"].eq("not_applicable")
            & frame["indexation"].eq("not_applicable")
            & frame["instrument_type"].eq("insurer_asset_sufficiency_irr")
            & frame["compounding"].isna()
            & frame["day_count"].isna()
            & tenor.isna()
            & frame["financial_entity_id"].notna()
            & frame["financial_entity_type"].eq("insurer")
        )
        titrp_real_zero = frame["variable"].eq("programmed_withdrawal_real_zero_input") & frame[
            "instrument_type"
        ].eq("titrp_real_zero_input")
        titrp_vector = frame["variable"].eq("programmed_withdrawal_prescribed_vector_rate") & frame[
            "instrument_type"
        ].eq("titrp_prescribed_rate_vector")
        titrp_input = (
            frame["regulatory_basis"].eq(RegulatoryBasis.TITRP.value)
            & frame["quote_type"].eq(RegulatoryQuoteType.TECHNICAL_RATE_INPUT.value)
            & (titrp_real_zero | titrp_vector)
            & integral_tenor
            & tenor.le(20)
            & common_published_curve
        )
        titrp_rate = (
            frame["regulatory_basis"].eq(RegulatoryBasis.TITRP.value)
            & frame["quote_type"].eq(RegulatoryQuoteType.TECHNICAL_RATE.value)
            & frame["variable"].eq("programmed_withdrawal_technical_rate")
            & frame["instrument_type"].eq("programmed_withdrawal_technical_rate")
            & tenor.isna()
            & common_published_curve
        )
        return pd.Series(
            vtd_spot | vtd_zero | insurer_irr | titrp_input | titrp_rate,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_values_are_valid(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _values_respect_finance_semantics(frame)

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_tenor_is_positive_when_present(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        tenor = frame["tenor_years"]
        return pd.Series(
            tenor.isna() | (np.isfinite(tenor) & tenor.gt(0)),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_effective_interval_is_valid(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _effective_interval_is_valid(frame)

    @pa.dataframe_check(ignore_na=False)
    def regulatory_rate_keys_are_unique(cls, frame: pd.DataFrame) -> pd.Series:
        return _unique_fact_and_key(frame, REGULATORY_RATE_KEY)

    class Config:
        strict = True
        coerce = False
        unique_column_names = True


class RegulatoryMortalityTableObservationSchema(ObservationEnvelopeSchema):
    """CMF pensioner mortality-table facts, distinct from population mortality."""

    value: Series[float] = pa.Field(nullable=True, coerce=True)
    regulatory_basis: Series[str] = pa.Field(isin=[RegulatoryBasis.TM_2020.value])
    mortality_table_id: Series[str]
    age: Series[float] = pa.Field(coerce=True)
    sex: Series[str]
    population_segment: Series[str]
    reference_year: Series[float] = pa.Field(nullable=True, coerce=True)
    improvement_year: Series[float] = pa.Field(nullable=True, coerce=True)
    effective_start: Series[object]
    effective_end: Series[object] = pa.Field(nullable=True)
    maximum_effective_end: Series[object] = pa.Field(nullable=True)

    @pa.dataframe_parser
    @classmethod
    def parse_strict_fields(cls, frame: pd.DataFrame) -> pd.DataFrame:
        parsed = frame.copy(deep=True)
        if not parsed.columns.is_unique:
            return parsed
        if "value" in parsed:
            parsed["value"] = parsed["value"].map(_strict_number)
        for column in ("age", "reference_year", "improvement_year"):
            if column in parsed:
                parsed[column] = parsed[column].map(_strict_integer)
        for column in _EFFECTIVE_DATE_COLUMNS:
            if column in parsed:
                parsed[column] = parsed[column].map(_effective_date)
        return parsed

    @pa.dataframe_check(ignore_na=False)
    def regulatory_mortality_domain_is_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return frame["domain"].eq("regulatory_mortality_table")

    @pa.dataframe_check(ignore_na=False)
    def regulatory_mortality_canonical_date_atoms_are_owned(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        effective = frame.loc[:, list(_EFFECTIVE_DATE_COLUMNS)].map(
            lambda value: _is_missing(value) or type(value) is date
        )
        return pd.Series(
            _finance_envelope_date_atoms_are_canonical(frame)
            & effective.all(axis=1)
            & frame["effective_start"].notna(),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_mortality_role_and_parity_are_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return pd.Series(
            frame["parity_scope"].eq("not_applicable")
            & frame["observation_role"].eq("regulatory_input"),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_mortality_fields_are_declared(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _finance_text_atoms_are_canonical(
            frame,
            required=(
                *_FINANCE_REQUIRED_ENVELOPE_TEXT,
                "regulatory_basis",
                "mortality_table_id",
                "sex",
                "population_segment",
            ),
            optional=_FINANCE_OPTIONAL_ENVELOPE_TEXT,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_mortality_age_reference_and_improvement_years_are_valid(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        age = frame["age"]
        reference = frame["reference_year"]
        improvement = frame["improvement_year"]
        return pd.Series(
            np.isfinite(age)
            & age.ge(0)
            & age.mod(1).eq(0)
            & reference.notna()
            & (np.isfinite(reference) & reference.ge(0) & reference.mod(1).eq(0))
            & (
                improvement.isna()
                | (np.isfinite(improvement) & improvement.ge(0) & improvement.mod(1).eq(0))
            ),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_mortality_semantics_are_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        base_probability = (
            frame["variable"].eq("regulatory_mortality_probability")
            & frame["semantic_kind"].eq("probability")
            & frame["unit"].eq("dimensionless")
            & frame["population_basis"].eq("pensioner")
            & frame["improvement_year"].isna()
        )
        improvement_factor = (
            frame["variable"].eq("regulatory_mortality_improvement_factor")
            & frame["semantic_kind"].eq("index")
            & frame["unit"].eq("dimensionless")
            & frame["population_basis"].eq("not_applicable")
            & frame["improvement_year"].notna()
            & (frame["value"].isna() | frame["value"].ge(0))
        )
        return pd.Series(
            (base_probability | improvement_factor) & _values_respect_finance_semantics(frame),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def TM_2020_reference_and_applicability_vintage_are_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return pd.Series(
            frame["regulatory_basis"].eq(RegulatoryBasis.TM_2020.value)
            & frame["reference_year"].eq(2020)
            & frame["population_segment"].eq("pensioner")
            & frame["effective_start"].eq(date(2023, 7, 1))
            & frame["maximum_effective_end"].eq(date(2029, 7, 1)),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def regulatory_mortality_effective_interval_is_valid(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _effective_interval_is_valid(frame)

    @pa.dataframe_check(ignore_na=False)
    def regulatory_mortality_keys_are_unique(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _unique_fact_and_key(frame, REGULATORY_MORTALITY_KEY)

    class Config:
        strict = True
        coerce = False
        unique_column_names = True


class PensionProductStatisticSchema(ObservationEnvelopeSchema):
    """Published pension product statistics, never inferred transaction prices."""

    value: Series[float] = pa.Field(nullable=True, coerce=True)
    financial_entity_id: Series[str]
    financial_entity_type: Series[str] = pa.Field(isin=sorted(_FINANCIAL_ENTITY_TYPES))
    pension_type: Series[str] = pa.Field(isin=sorted(_PENSION_TYPES))
    modality: Series[str] = pa.Field(isin=sorted(_PENSION_MODALITIES))
    offer_type: Series[str] = pa.Field(isin=sorted(_OFFER_TYPES))
    intermediation_type: Series[str] = pa.Field(isin=sorted(_INTERMEDIATION_TYPES))
    contract_state: Series[str] = pa.Field(isin=sorted(_CONTRACT_STATES))
    aggregation_grain: Series[str] = pa.Field(isin=sorted(_AGGREGATION_GRAINS))
    product_term_set_id: Series[str] = pa.Field(nullable=True)

    @pa.dataframe_parser
    @classmethod
    def parse_strict_value(cls, frame: pd.DataFrame) -> pd.DataFrame:
        parsed = frame.copy(deep=True)
        if parsed.columns.is_unique and "value" in parsed:
            parsed["value"] = parsed["value"].map(_strict_number)
        return parsed

    @pa.dataframe_check(ignore_na=False)
    def pension_product_domain_is_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return frame["domain"].eq("pension_product_statistic")

    @pa.dataframe_check(ignore_na=False)
    def pension_product_canonical_date_atoms_are_owned(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _finance_envelope_date_atoms_are_canonical(frame)

    @pa.dataframe_check(ignore_na=False)
    def pension_product_fields_are_declared(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        text = _finance_text_atoms_are_canonical(
            frame,
            required=(
                *_FINANCE_REQUIRED_ENVELOPE_TEXT,
                "financial_entity_id",
                "financial_entity_type",
                "pension_type",
                "modality",
                "offer_type",
                "intermediation_type",
                "contract_state",
                "aggregation_grain",
            ),
            optional=(
                *_FINANCE_OPTIONAL_ENVELOPE_TEXT,
                "product_term_set_id",
            ),
        )
        entity_matches_grain = frame["financial_entity_type"].eq(frame["aggregation_grain"])
        return pd.Series(
            text & entity_matches_grain,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def pension_product_statistic_values_roles_parity_and_semantics_are_exact(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        canonical_statistic = pd.Series(
            (
                triple in _PENSION_STATISTIC_TRIPLES
                for triple in frame.loc[:, ["variable", "semantic_kind", "unit"]].itertuples(
                    index=False, name=None
                )
            ),
            index=frame.index,
            dtype=bool,
        )
        ordinary = frame["observation_role"].eq("product_statistic") & canonical_statistic
        signed = frame["semantic_kind"].eq("signed_amount")
        return pd.Series(
            frame["parity_scope"].eq("not_applicable")
            & frame["population_basis"].eq("not_applicable")
            & (ordinary | signed)
            & _values_respect_finance_semantics(frame),
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def signed_amount_has_exact_regulatory_shape(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        signed = frame["semantic_kind"].eq("signed_amount")
        exact = (
            frame["variable"].eq("asset_sufficiency_balance")
            & frame["unit"].eq("UF")
            & frame["observation_role"].eq("regulatory_input")
            & frame["financial_entity_type"].eq("insurer")
            & frame["aggregation_grain"].eq("insurer")
            & frame["pension_type"].eq("not_applicable")
            & frame["modality"].eq("not_applicable")
            & frame["offer_type"].eq("not_applicable")
            & frame["intermediation_type"].eq("not_applicable")
            & frame["contract_state"].eq("not_applicable")
            & frame["product_term_set_id"].isna()
        )
        return pd.Series(
            ~signed | exact,
            index=frame.index,
            dtype=bool,
        )

    @pa.dataframe_check(ignore_na=False)
    def pension_product_keys_are_unique(
        cls,
        frame: pd.DataFrame,
    ) -> pd.Series:
        return _unique_fact_and_key(frame, PENSION_PRODUCT_STATISTIC_KEY)

    class Config:
        strict = True
        coerce = False
        unique_column_names = True


def _owned_frame(frame: object) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise DataContractError("Domain observation validation requires a pandas DataFrame.")
    return frame.copy(deep=True)


def validate_demographic_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a private copy against the isolated demographic schema."""

    return DemographicObservationSchema.validate(
        _owned_frame(frame),
        lazy=True,
        inplace=False,
    ).copy(deep=True)


def validate_environmental_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a private copy against the isolated environmental schema."""

    return EnvironmentalObservationSchema.validate(
        _owned_frame(frame),
        lazy=True,
        inplace=False,
    ).copy(deep=True)


def validate_health_system_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a private copy against the isolated health-system schema."""

    return HealthSystemObservationSchema.validate(
        _owned_frame(frame),
        lazy=True,
        inplace=False,
    ).copy(deep=True)


def validate_market_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a private copy against the isolated market-quote schema."""

    return MarketQuoteSchema.validate(
        _owned_frame(frame),
        lazy=True,
        inplace=False,
    ).copy(deep=True)


def validate_regulatory_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a private copy against the isolated regulatory-rate schema."""

    return RegulatoryRateSchema.validate(
        _owned_frame(frame),
        lazy=True,
        inplace=False,
    ).copy(deep=True)


def validate_regulatory_mortality(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a private copy against the regulatory-mortality schema."""

    return RegulatoryMortalityTableObservationSchema.validate(
        _owned_frame(frame),
        lazy=True,
        inplace=False,
    ).copy(deep=True)


def validate_pension_product_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a private copy against the pension-product schema."""

    return PensionProductStatisticSchema.validate(
        _owned_frame(frame),
        lazy=True,
        inplace=False,
    ).copy(deep=True)
