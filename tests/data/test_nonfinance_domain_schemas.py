from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
import pytest
from domain_fixtures import (
    demographic_frame,
    environmental_frame,
    health_system_frame,
)

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.data.domain_schemas import (
    DEMOGRAPHIC_OBSERVATION_KEY,
    ENVIRONMENTAL_OBSERVATION_KEY,
    HEALTH_SYSTEM_OBSERVATION_KEY,
    SHARED_FACT_KEY,
    DemographicObservationBatch,
    DemographicObservationSchema,
    EnvironmentalObservationBatch,
    EnvironmentalObservationSchema,
    HealthSystemObservationBatch,
    HealthSystemObservationSchema,
    validate_demographic_observations,
    validate_environmental_observations,
    validate_health_system_observations,
)

_SCHEMA_FAILURE = (pa.errors.SchemaError, pa.errors.SchemaErrors)


def test_domain_models_have_no_foreign_null_columns() -> None:
    demographic = DemographicObservationSchema.validate(demographic_frame())
    environmental = EnvironmentalObservationSchema.validate(environmental_frame())
    health = HealthSystemObservationSchema.validate(health_system_frame())
    assert "tenor_years" not in demographic
    assert "facility" not in demographic
    assert "age_lower" not in environmental
    assert "pollutant" not in health


@pytest.mark.parametrize(
    ("factory", "schema", "foreign_column"),
    [
        (demographic_frame, DemographicObservationSchema, "facility"),
        (environmental_frame, EnvironmentalObservationSchema, "age_lower"),
        (health_system_frame, HealthSystemObservationSchema, "pollutant"),
    ],
)
def test_strict_domain_schema_rejects_foreign_columns(
    factory: object, schema: object, foreign_column: str
) -> None:
    frame = factory()  # type: ignore[operator]
    frame[foreign_column] = "foreign"
    with pytest.raises(_SCHEMA_FAILURE):
        schema.validate(frame)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("variable", "scope", "order", "valid"),
    [
        ("deaths", "not_applicable", None, True),
        ("deaths", "all_orders", None, False),
        ("deaths", "known_order", 1, False),
        ("births", "all_orders", None, True),
        ("births", "known_order", 1, True),
        ("births", "known_order", None, False),
        ("births", "known_order", 0, False),
        ("births", "known_order", 1.5, False),
        ("births", "unknown_order", None, True),
        ("births", "unknown_order", 1, False),
        ("births", "not_applicable", None, False),
    ],
)
def test_parity_truth_table(variable: str, scope: str, order: object, valid: bool) -> None:
    frame = demographic_frame(
        variable=variable,
        parity_scope=scope,
        birth_order=order,
    )
    if valid:
        DemographicObservationSchema.validate(frame)
    else:
        with pytest.raises(_SCHEMA_FAILURE, match="parity"):
            DemographicObservationSchema.validate(frame)


def test_domain_discriminator_mismatch_fails_before_normalization() -> None:
    frame = environmental_frame(domain="demographic")
    with pytest.raises(_SCHEMA_FAILURE, match="domain"):
        EnvironmentalObservationSchema.validate(frame)


def test_each_schema_owns_a_minimal_key() -> None:
    assert "source_key" not in DEMOGRAPHIC_OBSERVATION_KEY
    assert "release_id" not in DEMOGRAPHIC_OBSERVATION_KEY
    assert "value" not in DEMOGRAPHIC_OBSERVATION_KEY


def test_domain_key_order_is_exact_and_batch_aliases_are_exported() -> None:
    assert SHARED_FACT_KEY == (
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
    assert DEMOGRAPHIC_OBSERVATION_KEY[: len(SHARED_FACT_KEY)] == SHARED_FACT_KEY
    assert (
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
    ) == ENVIRONMENTAL_OBSERVATION_KEY
    assert (
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
    ) == HEALTH_SYSTEM_OBSERVATION_KEY
    assert DemographicObservationBatch is not None
    assert EnvironmentalObservationBatch is not None
    assert HealthSystemObservationBatch is not None


@pytest.mark.parametrize(
    ("factory", "schema", "column"),
    [
        (demographic_frame, DemographicObservationSchema, "sex"),
        (environmental_frame, EnvironmentalObservationSchema, "station"),
        (health_system_frame, HealthSystemObservationSchema, "facility"),
    ],
)
def test_missing_and_duplicate_columns_are_rejected(
    factory: object,
    schema: object,
    column: str,
) -> None:
    frame = factory()  # type: ignore[operator]
    with pytest.raises(_SCHEMA_FAILURE):
        schema.validate(frame.drop(columns=column))  # type: ignore[attr-defined]

    duplicate = pd.concat([frame, frame[[column]]], axis=1)
    with pytest.raises(_SCHEMA_FAILURE):
        schema.validate(duplicate)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("factory", "schema", "column"),
    [
        (demographic_frame, DemographicObservationSchema, "sex"),
        (environmental_frame, EnvironmentalObservationSchema, "pollutant"),
        (health_system_frame, HealthSystemObservationSchema, "facility"),
    ],
)
def test_unhashable_categorical_cells_are_schema_failures(
    factory: object,
    schema: object,
    column: str,
) -> None:
    frame = factory()  # type: ignore[operator]
    frame.at[0, column] = ["not", "a", "scalar"]
    with pytest.raises(_SCHEMA_FAILURE):
        schema.validate(frame)  # type: ignore[attr-defined]


@pytest.mark.parametrize("value", [True, "40"])
def test_strict_numeric_precoercion_rejects_booleans_and_strings(value: object) -> None:
    with pytest.raises(_SCHEMA_FAILURE):
        DemographicObservationSchema.validate(demographic_frame(age_lower=value))


@pytest.mark.parametrize(
    ("lower", "upper", "open_group", "valid"),
    [
        (None, None, False, True),
        (40.0, 41.0, False, True),
        (85.0, None, True, True),
        (None, 41.0, False, False),
        (40.0, None, False, False),
        (40.0, 41.0, True, False),
        (41.0, 40.0, False, False),
        (-1.0, 0.0, False, False),
    ],
)
def test_demographic_age_state_is_exact(
    lower: object,
    upper: object,
    open_group: bool,
    valid: bool,
) -> None:
    frame = demographic_frame(
        age_lower=lower,
        age_upper=upper,
        age_open=open_group,
    )
    if valid:
        DemographicObservationSchema.validate(frame)
    else:
        with pytest.raises(_SCHEMA_FAILURE, match="age"):
            DemographicObservationSchema.validate(frame)


@pytest.mark.parametrize(
    ("start", "end", "valid"),
    [
        (None, None, True),
        ("2024-01-01", "2024-02-01", True),
        ("2024-01-01", None, False),
        (None, "2024-02-01", False),
        ("2024-02-01", "2024-01-01", False),
        ("not-a-date", "2024-02-01", False),
        (True, "2024-02-01", False),
    ],
)
def test_registration_interval_is_paired_strict_and_ordered(
    start: object,
    end: object,
    valid: bool,
) -> None:
    frame = demographic_frame(
        registration_start=start,
        registration_end=end,
    )
    if valid:
        DemographicObservationSchema.validate(frame)
    else:
        with pytest.raises(_SCHEMA_FAILURE, match="registration"):
            DemographicObservationSchema.validate(frame)


def test_values_are_finite_or_explicitly_missing() -> None:
    with pytest.raises(_SCHEMA_FAILURE, match="missing"):
        DemographicObservationSchema.validate(demographic_frame(value=float("nan")))
    DemographicObservationSchema.validate(
        demographic_frame(
            value=float("nan"),
            missingness_reason="publisher_cell_blank",
        )
    )
    with pytest.raises(_SCHEMA_FAILURE):
        DemographicObservationSchema.validate(demographic_frame(value=float("inf")))


@pytest.mark.parametrize(
    "column",
    [
        "station",
        "region",
        "pollutant",
        "environmental_measure",
        "averaging_interval",
        "validation_status",
    ],
)
def test_environmental_measure_fields_are_declared(column: str) -> None:
    with pytest.raises(_SCHEMA_FAILURE, match="environmental"):
        EnvironmentalObservationSchema.validate(environmental_frame(**{column: "   "}))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("semantic_kind", "rate"),
        ("unit", "dimensionless"),
        ("population_basis", "resident_estimate"),
    ],
)
def test_environmental_concentration_semantics_are_compatible(
    column: str,
    value: str,
) -> None:
    with pytest.raises(_SCHEMA_FAILURE):
        EnvironmentalObservationSchema.validate(environmental_frame(**{column: value}))


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "environmental_measure": "daily_mean",
            "semantic_kind": "count",
            "unit": "event",
            "population_basis": "resident_estimate",
        },
        {
            "environmental_measure": "temperature",
            "semantic_kind": "index",
            "unit": "dimensionless",
            "population_basis": "not_applicable",
        },
        {"station": None},
    ],
)
def test_pollutant_rows_require_station_and_concentration_semantics(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(_SCHEMA_FAILURE, match="environmental"):
        EnvironmentalObservationSchema.validate(environmental_frame(**overrides))


def test_nonpollutant_environmental_measures_remain_extensible() -> None:
    EnvironmentalObservationSchema.validate(
        environmental_frame(
            variable="mean_temperature",
            unit="degree_Celsius",
            environmental_measure="temperature",
            station=None,
            pollutant=None,
        )
    )
    EnvironmentalObservationSchema.validate(
        environmental_frame(
            variable="precipitation",
            unit="millimeter",
            environmental_measure="precipitation",
            station=None,
            pollutant=None,
        )
    )


@pytest.mark.parametrize(
    ("factory", "schema"),
    [
        (demographic_frame, DemographicObservationSchema),
        (environmental_frame, EnvironmentalObservationSchema),
        (health_system_frame, HealthSystemObservationSchema),
    ],
)
def test_nullable_values_accept_pd_na_with_explicit_missingness(
    factory: object,
    schema: object,
) -> None:
    frame = factory(  # type: ignore[operator]
        value=pd.NA,
        missingness_reason="publisher_cell_blank",
    )
    schema.validate(frame)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "column",
    ["age_lower", "age_upper", "age_quantity", "birth_order"],
)
def test_nullable_demographic_numerics_accept_pd_na(column: str) -> None:
    overrides: dict[str, object] = {column: pd.NA}
    if column in {"age_lower", "age_upper"}:
        overrides.update(
            {
                "age_lower": pd.NA,
                "age_upper": pd.NA,
                "age_open": False,
            }
        )
    DemographicObservationSchema.validate(demographic_frame(**overrides))


@pytest.mark.parametrize(
    "validator",
    [
        validate_demographic_observations,
        validate_environmental_observations,
        validate_health_system_observations,
    ],
)
@pytest.mark.parametrize("value", [None, "frame", [], {}])
def test_validation_wrappers_reject_nonframes_with_domain_error(
    validator: object,
    value: object,
) -> None:
    with pytest.raises(DataContractError, match="DataFrame"):
        validator(value)  # type: ignore[operator]


@pytest.mark.parametrize(
    "column",
    [
        "facility",
        "health_service",
        "region",
        "service_measure",
        "capacity_measure",
        "event_measure",
        "reporting_grain",
    ],
)
def test_health_system_measure_fields_are_declared(column: str) -> None:
    with pytest.raises(_SCHEMA_FAILURE, match="health_system"):
        HealthSystemObservationSchema.validate(health_system_frame(**{column: ""}))


@pytest.mark.parametrize(
    ("factory", "key_column"),
    [
        (demographic_frame, "sex"),
        (environmental_frame, "station"),
        (health_system_frame, "facility"),
    ],
)
def test_duplicate_fact_and_domain_keys_are_rejected(factory: object, key_column: str) -> None:
    first = factory()  # type: ignore[operator]
    duplicate_fact = pd.concat([first, first], ignore_index=True)
    with pytest.raises(_SCHEMA_FAILURE):
        if key_column == "sex":
            DemographicObservationSchema.validate(duplicate_fact)
        elif key_column == "station":
            EnvironmentalObservationSchema.validate(duplicate_fact)
        else:
            HealthSystemObservationSchema.validate(duplicate_fact)

    duplicate_key = pd.concat([first, first], ignore_index=True)
    duplicate_key.loc[1, "fact_id"] = "cldemopde:fact:v1:" + "d" * 64
    duplicate_key.loc[1, "observation_id"] = "cldemopde:observation:v1:" + "e" * 64
    with pytest.raises(_SCHEMA_FAILURE):
        if key_column == "sex":
            DemographicObservationSchema.validate(duplicate_key)
        elif key_column == "station":
            EnvironmentalObservationSchema.validate(duplicate_key)
        else:
            HealthSystemObservationSchema.validate(duplicate_key)


def test_validation_wrapper_owns_a_copy() -> None:
    cases = (
        (demographic_frame(), validate_demographic_observations, "sex"),
        (
            environmental_frame(),
            validate_environmental_observations,
            "station",
        ),
        (
            health_system_frame(),
            validate_health_system_observations,
            "facility",
        ),
    )
    for frame, validator, mutable_column in cases:
        original = frame.copy(deep=True)
        validated = validator(frame)
        validated.loc[0, mutable_column] = "changed"
        pd.testing.assert_frame_equal(frame, original)
