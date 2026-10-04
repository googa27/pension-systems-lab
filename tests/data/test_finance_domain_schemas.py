from __future__ import annotations

from datetime import date

import pandas as pd
import pandera.pandas as pa
import pytest
from domain_fixtures import (
    market_quote_frame,
    pension_product_frame,
    regulatory_mortality_frame,
    regulatory_rate_frame,
)

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.data.domain_schemas import (
    MARKET_QUOTE_KEY,
    PENSION_PRODUCT_STATISTIC_KEY,
    REGULATORY_MORTALITY_KEY,
    REGULATORY_RATE_KEY,
    SHARED_FACT_KEY,
    AggregationGrain,
    CompoundingConvention,
    ContractState,
    DayCountConvention,
    FinancialEntityType,
    IntermediationType,
    MarketQuoteBatch,
    MarketQuoteSchema,
    MarketQuoteType,
    OfferType,
    PensionModality,
    PensionProductStatisticBatch,
    PensionProductStatisticSchema,
    PensionType,
    RegulatoryBasis,
    RegulatoryMortalityBatch,
    RegulatoryMortalityTableObservationSchema,
    RegulatoryQuoteType,
    RegulatoryRateBatch,
    RegulatoryRateSchema,
    validate_market_quotes,
    validate_pension_product_statistics,
    validate_regulatory_mortality,
    validate_regulatory_rates,
)

_SCHEMA_FAILURE = (pa.errors.SchemaError, pa.errors.SchemaErrors)


def test_finance_domain_models_are_strict_and_isolated() -> None:
    market = MarketQuoteSchema.validate(market_quote_frame())
    rate = RegulatoryRateSchema.validate(regulatory_rate_frame())
    mortality = RegulatoryMortalityTableObservationSchema.validate(regulatory_mortality_frame())
    pension = PensionProductStatisticSchema.validate(pension_product_frame())

    assert "regulatory_basis" not in market
    assert "mortality_table_id" not in rate
    assert "instrument_type" not in mortality
    assert "tenor_years" not in pension

    for schema, frame, foreign_column in (
        (MarketQuoteSchema, market_quote_frame(), "mortality_table_id"),
        (RegulatoryRateSchema, regulatory_rate_frame(), "pension_type"),
        (
            RegulatoryMortalityTableObservationSchema,
            regulatory_mortality_frame(),
            "instrument_type",
        ),
        (PensionProductStatisticSchema, pension_product_frame(), "tenor_years"),
    ):
        frame[foreign_column] = "foreign"
        with pytest.raises(_SCHEMA_FAILURE):
            schema.validate(frame)


def test_finance_keys_and_batch_aliases_are_exact() -> None:
    assert (
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
    ) == MARKET_QUOTE_KEY
    assert (
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
    ) == REGULATORY_RATE_KEY
    assert (
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
    ) == REGULATORY_MORTALITY_KEY
    assert (
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
    ) == PENSION_PRODUCT_STATISTIC_KEY
    assert MarketQuoteBatch is not None
    assert RegulatoryRateBatch is not None
    assert RegulatoryMortalityBatch is not None
    assert PensionProductStatisticBatch is not None


def test_closed_finance_v1_vocabularies_are_explicit() -> None:
    assert {member.value for member in MarketQuoteType} == {
        "benchmark_yield",
        "swap_fixed_rate",
        "fixing",
        "published_index",
        "published_change",
    }
    assert {member.value for member in RegulatoryQuoteType} == {
        "published_zero_rate",
        "regulatory_discount_rate",
        "asset_sufficiency_irr",
        "technical_rate_input",
        "technical_rate",
    }
    assert {member.value for member in CompoundingConvention} == {
        "simple",
        "continuous",
        "annual_compounded",
        "semiannual_compounded",
        "quarterly_compounded",
        "monthly_compounded",
    }
    assert {member.value for member in DayCountConvention} == {
        "actual_360",
        "actual_365_fixed",
        "actual_actual",
        "thirty_360",
        "business_252",
    }
    assert {member.value for member in RegulatoryBasis} == {
        "NCG_209",
        "VTD_CMF",
        "TM_2020",
        "TITRP",
    }
    assert {member.value for member in FinancialEntityType} == {
        "insurer",
        "afp",
        "market_total",
    }
    assert {member.value for member in PensionType} == {
        "not_applicable",
        "old_age",
        "disability",
        "survivor",
    }
    assert {member.value for member in PensionModality} == {
        "not_applicable",
        "programmed_withdrawal",
        "temporary_annuity",
        "immediate_annuity",
        "deferred_annuity",
        "combined_annuity",
        "temporarily_increased_annuity",
    }
    assert {member.value for member in OfferType} == {
        "not_applicable",
        "internal",
        "external",
    }
    assert {member.value for member in IntermediationType} == {"not_applicable"}
    assert {member.value for member in ContractState} == {
        "not_applicable",
        "contracted",
        "active",
        "receiving",
    }
    assert {member.value for member in AggregationGrain} == {
        "insurer",
        "afp",
        "market_total",
    }


@pytest.mark.parametrize(
    ("schema", "frame"),
    [
        (MarketQuoteSchema, market_quote_frame(parity_scope="unknown_order")),
        (RegulatoryRateSchema, regulatory_rate_frame(parity_scope="all_orders")),
        (
            RegulatoryMortalityTableObservationSchema,
            regulatory_mortality_frame(parity_scope="known_order"),
        ),
        (
            PensionProductStatisticSchema,
            pension_product_frame(parity_scope="unknown_order"),
        ),
    ],
)
def test_finance_domains_always_have_not_applicable_parity(
    schema: object,
    frame: pd.DataFrame,
) -> None:
    with pytest.raises(_SCHEMA_FAILURE, match="parity"):
        schema.validate(frame)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("schema", "frame"),
    [
        (MarketQuoteSchema, market_quote_frame(observation_role="observed_fact")),
        (RegulatoryRateSchema, regulatory_rate_frame(observation_role="market_quote")),
        (
            RegulatoryMortalityTableObservationSchema,
            regulatory_mortality_frame(observation_role="product_statistic"),
        ),
        (
            PensionProductStatisticSchema,
            pension_product_frame(observation_role="market_quote"),
        ),
    ],
)
def test_finance_roles_are_domain_specific(
    schema: object,
    frame: pd.DataFrame,
) -> None:
    with pytest.raises(_SCHEMA_FAILURE, match="role"):
        schema.validate(frame)  # type: ignore[attr-defined]


def test_negative_regulatory_rates_are_valid_but_amounts_are_not() -> None:
    RegulatoryRateSchema.validate(regulatory_rate_frame(value=-0.002))
    with pytest.raises(_SCHEMA_FAILURE, match=r"amount|value"):
        PensionProductStatisticSchema.validate(
            pension_product_frame(
                semantic_kind="amount",
                unit="UF",
                value=-1.0,
            )
        )


def test_signed_sufficiency_balance_has_one_exact_legal_shape() -> None:
    exact = pension_product_frame(
        variable="asset_sufficiency_balance",
        semantic_kind="signed_amount",
        unit="UF",
        value=-303.0,
        observation_role="regulatory_input",
        financial_entity_id="cl-rut:12345678-5",
        financial_entity_type="insurer",
        pension_type="not_applicable",
        modality="not_applicable",
        offer_type="not_applicable",
        intermediation_type="not_applicable",
        contract_state="not_applicable",
        aggregation_grain="insurer",
        product_term_set_id=None,
    )
    PensionProductStatisticSchema.validate(exact)

    for mutation in (
        {"variable": "other_balance"},
        {"observation_role": "product_statistic"},
        {"financial_entity_type": "market_total"},
        {"aggregation_grain": "market_total"},
        {"pension_type": "old_age"},
        {"modality": "immediate_annuity"},
        {"offer_type": "internal"},
        {"contract_state": "active"},
        {"product_term_set_id": "scomp:v1:terms"},
    ):
        with pytest.raises(_SCHEMA_FAILURE, match="signed_amount"):
            PensionProductStatisticSchema.validate(exact.assign(**mutation))

    with pytest.raises(_SCHEMA_FAILURE, match=r"domain|signed_amount"):
        RegulatoryRateSchema.validate(regulatory_rate_frame(semantic_kind="signed_amount"))


@pytest.mark.parametrize("value", [-0.01, -1.0])
def test_prices_counts_and_amounts_are_nonnegative(value: float) -> None:
    with pytest.raises(_SCHEMA_FAILURE, match="value"):
        MarketQuoteSchema.validate(
            market_quote_frame(
                variable="uf_fixing_clp",
                quote_type="fixing",
                semantic_kind="price",
                unit="CLP",
                value=value,
                observation_role="covariate",
                tenor_years=None,
                currency="CLP",
                indexation="UF",
                instrument_type="uf_fixing",
                compounding=None,
                day_count=None,
            )
        )
    with pytest.raises(_SCHEMA_FAILURE, match="value"):
        PensionProductStatisticSchema.validate(pension_product_frame(value=value))


def test_uf_fixing_has_exact_exchange_semantics() -> None:
    fixing = market_quote_frame(
        variable="uf_fixing_clp",
        quote_type="fixing",
        semantic_kind="price",
        unit="CLP",
        value=39_000.0,
        observation_role="covariate",
        tenor_years=None,
        currency="CLP",
        indexation="UF",
        instrument_type="uf_fixing",
        compounding=None,
        day_count=None,
    )
    MarketQuoteSchema.validate(fixing)
    for mutation in (
        {"currency": "UF"},
        {"indexation": "nominal"},
        {"instrument_type": "other_fixing"},
        {"semantic_kind": "index", "unit": "dimensionless"},
    ):
        with pytest.raises(_SCHEMA_FAILURE, match=r"UF|fixing"):
            MarketQuoteSchema.validate(fixing.assign(**mutation))


@pytest.mark.parametrize(
    ("instrument_type", "currency", "indexation", "quote_type", "variable"),
    [
        (
            "bcp_benchmark_bond",
            "CLP",
            "nominal",
            "benchmark_yield",
            "official_benchmark_yield",
        ),
        (
            "bcu_btu_benchmark_bond",
            "UF",
            "UF",
            "benchmark_yield",
            "official_benchmark_yield",
        ),
        (
            "spc_nominal_fixed_swap",
            "CLP",
            "nominal",
            "swap_fixed_rate",
            "official_swap_fixed_rate",
        ),
        (
            "spc_uf_fixed_swap",
            "UF",
            "UF",
            "swap_fixed_rate",
            "official_swap_fixed_rate",
        ),
    ],
)
def test_benchmark_and_swap_instrument_grains_are_exact(
    instrument_type: str,
    currency: str,
    indexation: str,
    quote_type: str,
    variable: str,
) -> None:
    frame = market_quote_frame(
        instrument_type=instrument_type,
        currency=currency,
        indexation=indexation,
        quote_type=quote_type,
        variable=variable,
    )
    MarketQuoteSchema.validate(frame)
    with pytest.raises(_SCHEMA_FAILURE, match="instrument_grain"):
        MarketQuoteSchema.validate(frame.assign(currency="foreign"))
    with pytest.raises(_SCHEMA_FAILURE, match="instrument_grain"):
        MarketQuoteSchema.validate(frame.assign(indexation="foreign"))
    with pytest.raises(_SCHEMA_FAILURE, match="instrument_grain"):
        MarketQuoteSchema.validate(frame.assign(instrument_type="foreign"))


def test_unknown_compounding_and_day_count_remain_null() -> None:
    MarketQuoteSchema.validate(market_quote_frame(compounding=None, day_count=None))
    RegulatoryRateSchema.validate(
        regulatory_rate_frame(
            variable="insurer_asset_sufficiency_irr",
            regulatory_basis="NCG_209",
            quote_type="asset_sufficiency_irr",
            tenor_years=None,
            financial_entity_id="cl-rut:12345678-5",
            financial_entity_type="insurer",
            currency="not_applicable",
            indexation="not_applicable",
            instrument_type="insurer_asset_sufficiency_irr",
            compounding=None,
            day_count=None,
        )
    )
    with pytest.raises(_SCHEMA_FAILURE):
        MarketQuoteSchema.validate(market_quote_frame(compounding="unknown"))
    with pytest.raises(_SCHEMA_FAILURE):
        RegulatoryRateSchema.validate(regulatory_rate_frame(day_count="unknown"))


@pytest.mark.parametrize(
    ("quote_type", "semantic_kind", "unit", "valid"),
    [
        ("published_index", "index", "dimensionless", True),
        ("published_index", "index", "percent", False),
        ("published_change", "index", "percent", True),
        ("published_change", "index", "dimensionless", False),
    ],
)
def test_published_index_and_change_units_remain_distinct(
    quote_type: str,
    semantic_kind: str,
    unit: str,
    valid: bool,
) -> None:
    frame = market_quote_frame(
        variable=(
            "consumer_price_index" if quote_type == "published_index" else "consumer_price_change"
        ),
        quote_type=quote_type,
        semantic_kind=semantic_kind,
        unit=unit,
        observation_role="covariate",
    )
    if valid:
        MarketQuoteSchema.validate(frame)
    else:
        with pytest.raises(_SCHEMA_FAILURE, match="semantics"):
            MarketQuoteSchema.validate(frame)
    with pytest.raises(_SCHEMA_FAILURE, match="semantics"):
        MarketQuoteSchema.validate(frame.assign(variable="interchangeable_index_variable"))


@pytest.mark.parametrize(
    ("quote_type", "role", "valid"),
    [
        ("benchmark_yield", "market_quote", True),
        ("swap_fixed_rate", "market_quote", True),
        ("fixing", "covariate", True),
        ("published_index", "covariate", True),
        ("published_change", "covariate", True),
        ("benchmark_yield", "covariate", False),
        ("fixing", "market_quote", False),
        ("published_index", "market_quote", False),
    ],
)
def test_market_quote_role_depends_on_published_quote_type(
    quote_type: str,
    role: str,
    valid: bool,
) -> None:
    overrides: dict[str, object] = {
        "quote_type": quote_type,
        "observation_role": role,
    }
    if quote_type == "fixing":
        overrides.update(
            {
                "variable": "uf_fixing_clp",
                "semantic_kind": "price",
                "unit": "CLP",
                "tenor_years": None,
                "currency": "CLP",
                "indexation": "UF",
                "instrument_type": "uf_fixing",
                "compounding": None,
                "day_count": None,
            }
        )
    elif quote_type == "swap_fixed_rate":
        overrides.update(
            {
                "variable": "official_swap_fixed_rate",
                "instrument_type": "spc_nominal_fixed_swap",
            }
        )
    elif quote_type == "published_index":
        overrides.update(
            {
                "variable": "consumer_price_index",
                "semantic_kind": "index",
                "unit": "dimensionless",
            }
        )
    elif quote_type == "published_change":
        overrides.update(
            {
                "variable": "consumer_price_change",
                "semantic_kind": "index",
                "unit": "percent",
            }
        )
    frame = market_quote_frame(**overrides)
    if valid:
        MarketQuoteSchema.validate(frame)
    else:
        with pytest.raises(_SCHEMA_FAILURE, match="role"):
            MarketQuoteSchema.validate(frame)


@pytest.mark.parametrize(
    ("basis", "quote_type", "valid"),
    [
        ("VTD_CMF", "asset_sufficiency_irr", False),
        ("NCG_209", "technical_rate", False),
        ("TITRP", "regulatory_discount_rate", False),
    ],
)
def test_regulatory_basis_and_quote_type_mapping_is_exact(
    basis: str,
    quote_type: str,
    valid: bool,
) -> None:
    frame = regulatory_rate_frame(
        regulatory_basis=basis,
        quote_type=quote_type,
    )
    if valid:
        RegulatoryRateSchema.validate(frame)
    else:
        with pytest.raises(_SCHEMA_FAILURE, match="basis"):
            RegulatoryRateSchema.validate(frame)


def test_regulatory_rate_rows_bind_the_complete_canonical_grain() -> None:
    vtd_spot = regulatory_rate_frame()
    vtd_zero = regulatory_rate_frame(
        variable="cmf_published_real_zero_rate",
        quote_type="published_zero_rate",
        tenor_years=25,
        instrument_type="cmf_vtd_real_zero_curve",
    )
    insurer_irr = regulatory_rate_frame(
        variable="insurer_asset_sufficiency_irr",
        regulatory_basis="NCG_209",
        quote_type="asset_sufficiency_irr",
        tenor_years=None,
        financial_entity_id="cl-rut:12345678-5",
        financial_entity_type="insurer",
        currency="not_applicable",
        indexation="not_applicable",
        instrument_type="insurer_asset_sufficiency_irr",
        compounding=None,
    )
    titrp_input = regulatory_rate_frame(
        variable="programmed_withdrawal_real_zero_input",
        regulatory_basis="TITRP",
        quote_type="technical_rate_input",
        tenor_years=20,
        financial_entity_id=None,
        financial_entity_type=None,
        instrument_type="titrp_real_zero_input",
    )
    titrp_rate = regulatory_rate_frame(
        variable="programmed_withdrawal_technical_rate",
        regulatory_basis="TITRP",
        quote_type="technical_rate",
        tenor_years=None,
        financial_entity_id=None,
        financial_entity_type=None,
        instrument_type="programmed_withdrawal_technical_rate",
    )
    for frame in (vtd_spot, vtd_zero, insurer_irr, titrp_input, titrp_rate):
        RegulatoryRateSchema.validate(frame)

    invalid = (
        vtd_spot.assign(financial_entity_id="market:total", financial_entity_type="market_total"),
        vtd_spot.assign(currency="CLP"),
        vtd_spot.assign(indexation="nominal"),
        vtd_spot.assign(instrument_type="cmf_vtd_real_zero_curve"),
        vtd_spot.assign(compounding=None),
        vtd_spot.assign(tenor_years=121),
        vtd_zero.assign(tenor_years=26),
        insurer_irr.assign(financial_entity_id=None, financial_entity_type=None),
        insurer_irr.assign(tenor_years=1),
        titrp_input.assign(currency="CLP"),
        titrp_input.assign(tenor_years=21),
        titrp_rate.assign(tenor_years=1),
        titrp_rate.assign(instrument_type="cmf_vtd_spot_rate_ac"),
    )
    for frame in invalid:
        with pytest.raises(_SCHEMA_FAILURE, match="grain"):
            RegulatoryRateSchema.validate(frame)


@pytest.mark.parametrize(
    ("factory", "schema"),
    [
        (market_quote_frame, MarketQuoteSchema),
        (regulatory_rate_frame, RegulatoryRateSchema),
        (
            regulatory_mortality_frame,
            RegulatoryMortalityTableObservationSchema,
        ),
        (pension_product_frame, PensionProductStatisticSchema),
    ],
)
def test_finance_domain_discriminator_is_exact(
    factory: object,
    schema: object,
) -> None:
    with pytest.raises(_SCHEMA_FAILURE, match="domain"):
        schema.validate(factory(domain="demographic"))  # type: ignore[operator, attr-defined]


def test_market_quote_requires_exact_instrument_grain() -> None:
    valid = market_quote_frame()
    MarketQuoteSchema.validate(valid)
    with pytest.raises(_SCHEMA_FAILURE):
        MarketQuoteSchema.validate(valid.assign(source_series_id=None))


@pytest.mark.parametrize(
    ("start", "end", "maximum", "valid"),
    [
        (date(2026, 1, 1), None, None, True),
        (date(2026, 1, 1), None, date(2029, 1, 1), True),
        (date(2026, 1, 1), date(2027, 1, 1), None, True),
        (date(2026, 1, 1), date(2027, 1, 1), date(2029, 1, 1), True),
        (None, date(2027, 1, 1), None, False),
        (date(2027, 1, 1), date(2026, 1, 1), None, False),
        (date(2027, 1, 1), None, date(2026, 1, 1), False),
        (date(2026, 1, 1), date(2029, 1, 1), date(2027, 1, 1), False),
    ],
)
def test_effective_intervals_are_half_open_and_bounds_are_distinct(
    start: object,
    end: object,
    maximum: object,
    valid: bool,
) -> None:
    frame = regulatory_rate_frame(
        effective_start=start,
        effective_end=end,
        maximum_effective_end=maximum,
    )
    if valid:
        RegulatoryRateSchema.validate(frame)
    else:
        with pytest.raises(_SCHEMA_FAILURE, match="effective"):
            RegulatoryRateSchema.validate(frame)


def test_mortality_table_probability_and_age_are_bounded() -> None:
    RegulatoryMortalityTableObservationSchema.validate(regulatory_mortality_frame(value=0.0, age=0))
    RegulatoryMortalityTableObservationSchema.validate(
        regulatory_mortality_frame(value=1.0, age=120)
    )
    for mutation in (
        {"value": -0.001},
        {"value": 1.001},
        {"age": -1},
        {"age": 65.5},
        {"reference_year": 2020.5},
        {"improvement_year": True},
    ):
        with pytest.raises(_SCHEMA_FAILURE):
            RegulatoryMortalityTableObservationSchema.validate(
                regulatory_mortality_frame(**mutation)
            )
    with pytest.raises(_SCHEMA_FAILURE, match="reference"):
        RegulatoryMortalityTableObservationSchema.validate(
            regulatory_mortality_frame(reference_year=None)
        )


def test_mortality_improvement_factor_is_nonnegative_and_year_specific() -> None:
    improvement = regulatory_mortality_frame(
        variable="regulatory_mortality_improvement_factor",
        semantic_kind="index",
        population_basis="not_applicable",
        improvement_year=2026,
        value=0.012,
    )
    RegulatoryMortalityTableObservationSchema.validate(improvement)
    with pytest.raises(_SCHEMA_FAILURE, match="semantics"):
        RegulatoryMortalityTableObservationSchema.validate(improvement.assign(value=-0.001))


def test_tm2020_reference_and_applicability_vintage_are_exact() -> None:
    valid = regulatory_mortality_frame()
    RegulatoryMortalityTableObservationSchema.validate(valid)
    for mutation in (
        {"reference_year": 2021},
        {"effective_start": date(2024, 1, 1)},
        {"maximum_effective_end": None},
        {"maximum_effective_end": date(2030, 7, 1)},
    ):
        with pytest.raises(_SCHEMA_FAILURE, match="TM_2020"):
            RegulatoryMortalityTableObservationSchema.validate(valid.assign(**mutation))


def test_pension_product_counts_are_nonnegative_integers() -> None:
    PensionProductStatisticSchema.validate(
        pension_product_frame(
            variable="annuity_policy_count",
            semantic_kind="count",
            unit="event",
            value=12,
        )
    )
    for value in (-1, 1.5):
        with pytest.raises(_SCHEMA_FAILURE, match="values"):
            PensionProductStatisticSchema.validate(
                pension_product_frame(
                    variable="annuity_policy_count",
                    semantic_kind="count",
                    unit="event",
                    value=value,
                )
            )


@pytest.mark.parametrize(
    ("variable", "semantic_kind", "unit"),
    [
        ("annuity_new_business_mean_rate", "rate", "1 / year"),
        ("annuity_policy_count", "count", "event"),
        ("annuity_single_premium", "amount", "UF"),
        ("annuity_average_premium", "amount", "UF"),
        ("intermediation_commission_rate", "index", "percent"),
        ("intermediation_commission_amount", "amount", "UF"),
        ("scomp_selected_pension_count", "count", "event"),
        ("scomp_selected_pension_share", "index", "percent"),
    ],
)
def test_pension_product_variables_have_exact_semantic_triples(
    variable: str,
    semantic_kind: str,
    unit: str,
) -> None:
    PensionProductStatisticSchema.validate(
        pension_product_frame(
            variable=variable,
            semantic_kind=semantic_kind,
            unit=unit,
        )
    )


@pytest.mark.parametrize(
    ("variable", "semantic_kind", "unit"),
    [
        ("annuity_single_premium", "count", "event"),
        ("annuity_policy_count", "amount", "UF"),
        ("intermediation_commission_rate", "index", "millimeter"),
        ("annuity_new_business_mean_rate", "rate", "1 / month"),
        ("unreviewed_product_variable", "amount", "UF"),
    ],
)
def test_pension_product_variables_cannot_be_interchanged(
    variable: str,
    semantic_kind: str,
    unit: str,
) -> None:
    with pytest.raises(_SCHEMA_FAILURE, match="statistic"):
        PensionProductStatisticSchema.validate(
            pension_product_frame(
                variable=variable,
                semantic_kind=semantic_kind,
                unit=unit,
            )
        )


@pytest.mark.parametrize(
    ("factory", "schema"),
    [
        (market_quote_frame, MarketQuoteSchema),
        (regulatory_rate_frame, RegulatoryRateSchema),
        (
            regulatory_mortality_frame,
            RegulatoryMortalityTableObservationSchema,
        ),
        (pension_product_frame, PensionProductStatisticSchema),
    ],
)
def test_finance_keys_reject_duplicate_facts_and_domain_keys(
    factory: object,
    schema: object,
) -> None:
    frame = factory()  # type: ignore[operator]
    with pytest.raises(_SCHEMA_FAILURE, match="unique"):
        schema.validate(pd.concat([frame, frame], ignore_index=True))  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "validator",
    [
        validate_market_quotes,
        validate_regulatory_rates,
        validate_regulatory_mortality,
        validate_pension_product_statistics,
    ],
)
@pytest.mark.parametrize("value", [None, "frame", [], {}])
def test_finance_validation_wrappers_reject_nonframes(
    validator: object,
    value: object,
) -> None:
    with pytest.raises(DataContractError, match="DataFrame"):
        validator(value)  # type: ignore[operator]


def test_finance_validation_wrappers_own_input_and_output() -> None:
    original = market_quote_frame()
    validated = validate_market_quotes(original)
    validated.loc[0, "value"] = 99.0
    assert original.loc[0, "value"] == 0.035


@pytest.mark.parametrize(
    "column",
    [
        "source_series_id",
        "variable",
        "aggregation_rule",
        "source_key",
        "semantic_kind",
    ],
)
def test_finance_wrapper_rejects_mutable_string_subclass_scalars(
    column: str,
) -> None:
    class MutableString(str):
        payload: list[str]

        def __new__(cls, value: str) -> MutableString:
            instance = super().__new__(cls, value)
            instance.payload = []
            return instance

    frame = market_quote_frame()
    mutable_value = MutableString(str(frame.loc[0, column]))
    with pytest.raises(_SCHEMA_FAILURE, match="canonical"):
        validate_market_quotes(frame.assign(**{column: mutable_value}))


def test_finance_wrappers_reject_mutable_date_subclass_scalars() -> None:
    class MutableDate(date):
        payload: list[str]

        def __new__(cls, year: int, month: int, day: int) -> MutableDate:
            instance = super().__new__(cls, year, month, day)
            instance.payload = []
            return instance

    mutable = MutableDate(2024, 1, 1)
    for validator, frame, column in (
        (validate_market_quotes, market_quote_frame(), "period_start"),
        (validate_market_quotes, market_quote_frame(), "released_at"),
        (
            validate_regulatory_rates,
            regulatory_rate_frame(),
            "effective_start",
        ),
    ):
        with pytest.raises(_SCHEMA_FAILURE, match="canonical_date"):
            validator(frame.assign(**{column: mutable}))
