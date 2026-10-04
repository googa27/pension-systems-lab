import pytest

from chile_demographic_pde.core.errors import UnitMismatchError
from chile_demographic_pde.core.semantics import (
    PopulationBasis,
    SemanticKind,
    Unit,
    assert_semantic_compatible,
    assert_unit_compatible,
)


def test_semantic_kinds_and_population_bases_are_explicit() -> None:
    assert SemanticKind.RATE.value == "rate"
    assert SemanticKind.EXPOSURE.value == "exposure"
    assert SemanticKind.COUNT.value == "count"
    assert PopulationBasis.CENSUS_ENUMERATED.value == "census_enumerated"
    assert PopulationBasis.RESIDENT_ESTIMATE.value == "resident_estimate"


def test_unit_compatibility_uses_physical_dimensions() -> None:
    assert_unit_compatible(Unit.PER_YEAR, Unit.PER_YEAR)
    with pytest.raises(UnitMismatchError, match="1 / year"):
        assert_unit_compatible(Unit.PER_YEAR, Unit.PERSON_YEAR)


def test_amount_and_signed_amount_have_distinct_value_domains() -> None:
    assert_semantic_compatible(
        SemanticKind.AMOUNT,
        Unit.UF,
        PopulationBasis.NOT_APPLICABLE,
    )
    assert_semantic_compatible(
        SemanticKind.SIGNED_AMOUNT,
        Unit.CLP,
        PopulationBasis.NOT_APPLICABLE,
    )
