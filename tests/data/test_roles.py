from chile_demographic_pde.core.errors import (
    AmbiguousRevisionError,
    DataContractError,
    IdentityCollisionError,
    ReportingProjectionInputError,
    SourceContractError,
)
from chile_demographic_pde.data.roles import (
    ObservationDomain,
    ObservationRole,
    ParityScope,
    ReleaseMissingnessReason,
)


def test_shared_domains_roles_and_parity_are_exact_closed_sets() -> None:
    assert {item.value for item in ObservationDomain} == {
        "demographic",
        "environmental",
        "health_system",
        "market_quote",
        "regulatory_rate",
        "regulatory_mortality_table",
        "pension_product_statistic",
    }
    assert {item.value for item in ObservationRole} == {
        "observed_fact",
        "external_projection",
        "administrative_proxy",
        "market_quote",
        "regulatory_input",
        "diagnostic_control",
        "product_statistic",
        "covariate",
    }
    assert {item.value for item in ParityScope} == {
        "not_applicable",
        "all_orders",
        "known_order",
        "unknown_order",
    }
    assert {item.value for item in ReleaseMissingnessReason} == {"publisher_date_not_available"}


def test_new_data_contract_errors_share_the_expected_base_class() -> None:
    for error_type in (
        IdentityCollisionError,
        AmbiguousRevisionError,
        ReportingProjectionInputError,
        SourceContractError,
    ):
        assert issubclass(error_type, DataContractError)
