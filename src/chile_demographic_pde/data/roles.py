from __future__ import annotations

from enum import StrEnum


class ObservationDomain(StrEnum):
    DEMOGRAPHIC = "demographic"
    ENVIRONMENTAL = "environmental"
    HEALTH_SYSTEM = "health_system"
    MARKET_QUOTE = "market_quote"
    REGULATORY_RATE = "regulatory_rate"
    REGULATORY_MORTALITY_TABLE = "regulatory_mortality_table"
    PENSION_PRODUCT_STATISTIC = "pension_product_statistic"


class ObservationRole(StrEnum):
    OBSERVED_FACT = "observed_fact"
    EXTERNAL_PROJECTION = "external_projection"
    ADMINISTRATIVE_PROXY = "administrative_proxy"
    MARKET_QUOTE = "market_quote"
    REGULATORY_INPUT = "regulatory_input"
    DIAGNOSTIC_CONTROL = "diagnostic_control"
    PRODUCT_STATISTIC = "product_statistic"
    COVARIATE = "covariate"


class ParityScope(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    ALL_ORDERS = "all_orders"
    KNOWN_ORDER = "known_order"
    UNKNOWN_ORDER = "unknown_order"


class ReleaseMissingnessReason(StrEnum):
    PUBLISHER_DATE_NOT_AVAILABLE = "publisher_date_not_available"
