from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd


def _frame(row: dict[str, object]) -> pd.DataFrame:
    frame = pd.DataFrame([row])
    frame["available_at"] = pd.Series(
        [row["available_at"]],
        index=frame.index,
        dtype=object,
    )
    return frame


def _envelope(domain: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "fact_id": "cldemopde:fact:v1:" + "a" * 64,
        "observation_id": "cldemopde:observation:v1:" + "b" * 64,
        "release_id": "cldemopde:release:v1:" + "c" * 64,
        "domain": domain,
        "variable": "deaths",
        "value": 12.0,
        "semantic_kind": "count",
        "unit": "event",
        "population_basis": "resident_estimate",
        "observation_role": "observed_fact",
        "parity_scope": "not_applicable",
        "period_start": date(2024, 1, 1),
        "period_end": date(2025, 1, 1),
        "source_key": "official_fixture",
        "vintage": "2024-final",
        "released_at": date(2025, 6, 30),
        "release_missingness_reason": None,
        "available_at": datetime(2025, 7, 1, tzinfo=UTC),
        "provisional": False,
        "transformation_id": None,
        "transformation_version": None,
        "aggregation_rule": "published total",
        "missingness_reason": "not_applicable_observed",
        "source_status_code": None,
    }
    row.update(overrides)
    return row


def demographic_frame(**overrides: object) -> pd.DataFrame:
    row = _envelope("demographic")
    row.update(
        {
            "age_lower": 40.0,
            "age_upper": 41.0,
            "age_open": False,
            "sex": "female",
            "region": "CL",
            "cause": "all",
            "date_basis": "occurrence_interval",
            "age_role": "decedent",
            "age_measure_unit": "completed_years",
            "age_quantity": 40.0,
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
    )
    row.update(overrides)
    return _frame(row)


def environmental_frame(**overrides: object) -> pd.DataFrame:
    row = _envelope(
        "environmental",
        variable="pm25_concentration",
        value=18.5,
        semantic_kind="index",
        unit="microgram / meter ** 3",
        population_basis="not_applicable",
        observation_role="covariate",
    )
    row.update(
        {
            "station": "STATION-001",
            "region": "RM",
            "commune": "Santiago",
            "geography_basis": "monitor_location",
            "geography_vintage": "dpa-2024",
            "pollutant": "PM2.5",
            "environmental_measure": "concentration",
            "averaging_interval": "24_hour",
            "validation_status": "validated",
        }
    )
    row.update(overrides)
    return _frame(row)


def health_system_frame(**overrides: object) -> pd.DataFrame:
    row = _envelope(
        "health_system",
        variable="installed_beds",
        value=120.0,
        semantic_kind="count",
        unit="event",
        population_basis="not_applicable",
        observation_role="covariate",
    )
    row.update(
        {
            "facility": "HOSP-001",
            "health_service": "SS-METROPOLITANO",
            "region": "RM",
            "commune": "Santiago",
            "geography_basis": "facility_location",
            "geography_vintage": "dpa-2024",
            "service_measure": "not_applicable",
            "capacity_measure": "installed_beds",
            "event_measure": "not_applicable",
            "reporting_grain": "facility",
        }
    )
    row.update(overrides)
    return _frame(row)


def market_quote_frame(**overrides: object) -> pd.DataFrame:
    row = _envelope(
        "market_quote",
        variable="official_benchmark_yield",
        value=0.035,
        semantic_kind="rate",
        unit="1 / year",
        population_basis="not_applicable",
        observation_role="market_quote",
    )
    row.update(
        {
            "source_series_id": "BCCH_SERIES_EXAMPLE",
            "quote_type": "benchmark_yield",
            "tenor_years": 10.0,
            "currency": "CLP",
            "indexation": "nominal",
            "instrument_type": "bcp_benchmark_bond",
            "compounding": "annual_compounded",
            "day_count": None,
            "date_basis": "published_observation_date",
        }
    )
    row.update(overrides)
    return _frame(row)


def regulatory_rate_frame(**overrides: object) -> pd.DataFrame:
    row = _envelope(
        "regulatory_rate",
        variable="cmf_regulatory_discount_rate",
        value=0.025,
        semantic_kind="rate",
        unit="1 / year",
        population_basis="not_applicable",
        observation_role="regulatory_input",
    )
    row.update(
        {
            "regulatory_basis": "VTD_CMF",
            "regulatory_use": "pension_reserve_discounting",
            "quote_type": "regulatory_discount_rate",
            "effective_start": date(2026, 3, 1),
            "effective_end": date(2026, 4, 1),
            "maximum_effective_end": None,
            "tenor_years": 10.0,
            "financial_entity_id": None,
            "financial_entity_type": None,
            "currency": "UF",
            "indexation": "UF",
            "instrument_type": "cmf_vtd_spot_rate_ac",
            "compounding": "annual_compounded",
            "day_count": None,
        }
    )
    row.update(overrides)
    return _frame(row)


def regulatory_mortality_frame(**overrides: object) -> pd.DataFrame:
    row = _envelope(
        "regulatory_mortality_table",
        variable="regulatory_mortality_probability",
        value=0.004,
        semantic_kind="probability",
        unit="dimensionless",
        population_basis="pensioner",
        observation_role="regulatory_input",
    )
    row.update(
        {
            "regulatory_basis": "TM_2020",
            "mortality_table_id": "CB-2020",
            "age": 65.0,
            "sex": "female",
            "population_segment": "pensioner",
            "reference_year": 2020.0,
            "improvement_year": None,
            "effective_start": date(2023, 7, 1),
            "effective_end": None,
            "maximum_effective_end": date(2029, 7, 1),
        }
    )
    row.update(overrides)
    return _frame(row)


def pension_product_frame(**overrides: object) -> pd.DataFrame:
    row = _envelope(
        "pension_product_statistic",
        variable="annuity_single_premium",
        value=1_200.0,
        semantic_kind="amount",
        unit="UF",
        population_basis="not_applicable",
        observation_role="product_statistic",
    )
    row.update(
        {
            "financial_entity_id": "market:total",
            "financial_entity_type": "market_total",
            "pension_type": "old_age",
            "modality": "immediate_annuity",
            "offer_type": "not_applicable",
            "intermediation_type": "not_applicable",
            "contract_state": "contracted",
            "aggregation_grain": "market_total",
            "product_term_set_id": None,
        }
    )
    row.update(overrides)
    return _frame(row)
