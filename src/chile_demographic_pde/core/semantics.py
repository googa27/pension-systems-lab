from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

import pint

from chile_demographic_pde.core.errors import PopulationBasisError, UnitMismatchError

UNIT_REGISTRY: pint.UnitRegistry[Any] = pint.UnitRegistry()


class SemanticKind(StrEnum):
    COUNT = "count"
    EXPOSURE = "exposure"
    POPULATION = "population"
    RATE = "rate"
    PROBABILITY = "probability"
    INDEX = "index"
    PRICE = "price"
    AMOUNT = "amount"
    SIGNED_AMOUNT = "signed_amount"
    DISCOUNT_FACTOR = "discount_factor"


class PopulationBasis(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    CENSUS_ENUMERATED = "census_enumerated"
    RESIDENT_ESTIMATE = "resident_estimate"
    PERSON_TIME = "person_time"
    PENSIONER = "pensioner"


class Unit(StrEnum):
    DIMENSIONLESS = "dimensionless"
    PERSON = "person"
    PERSON_YEAR = "person * year"
    PER_YEAR = "1 / year"
    PER_MONTH = "1 / month"
    PERCENT = "percent"
    CLP = "CLP"
    UF = "UF"
    EVENT = "event"
    DAY = "day"
    DEGREE_CELSIUS = "degree_Celsius"
    MILLIMETER = "millimeter"
    MICROGRAM_PER_CUBIC_METER = "microgram / meter ** 3"


SEMANTIC_COMPATIBILITY: Final[Mapping[SemanticKind, Mapping[Unit, frozenset[PopulationBasis]]]] = (
    MappingProxyType(
        {
            SemanticKind.COUNT: MappingProxyType(
                {
                    Unit.PERSON: frozenset(
                        {
                            PopulationBasis.CENSUS_ENUMERATED,
                            PopulationBasis.RESIDENT_ESTIMATE,
                            PopulationBasis.PENSIONER,
                        }
                    ),
                    Unit.EVENT: frozenset(
                        {
                            PopulationBasis.NOT_APPLICABLE,
                            PopulationBasis.CENSUS_ENUMERATED,
                            PopulationBasis.RESIDENT_ESTIMATE,
                            PopulationBasis.PENSIONER,
                        }
                    ),
                }
            ),
            SemanticKind.EXPOSURE: MappingProxyType(
                {
                    Unit.PERSON_YEAR: frozenset(
                        {
                            PopulationBasis.RESIDENT_ESTIMATE,
                            PopulationBasis.PERSON_TIME,
                            PopulationBasis.PENSIONER,
                        }
                    )
                }
            ),
            SemanticKind.POPULATION: MappingProxyType(
                {
                    Unit.PERSON: frozenset(
                        {
                            PopulationBasis.CENSUS_ENUMERATED,
                            PopulationBasis.RESIDENT_ESTIMATE,
                            PopulationBasis.PENSIONER,
                        }
                    )
                }
            ),
            SemanticKind.RATE: MappingProxyType(
                {
                    Unit.PER_YEAR: frozenset(
                        {
                            PopulationBasis.NOT_APPLICABLE,
                            PopulationBasis.CENSUS_ENUMERATED,
                            PopulationBasis.RESIDENT_ESTIMATE,
                            PopulationBasis.PERSON_TIME,
                            PopulationBasis.PENSIONER,
                        }
                    ),
                    Unit.PER_MONTH: frozenset(
                        {
                            PopulationBasis.NOT_APPLICABLE,
                            PopulationBasis.CENSUS_ENUMERATED,
                            PopulationBasis.RESIDENT_ESTIMATE,
                            PopulationBasis.PERSON_TIME,
                            PopulationBasis.PENSIONER,
                        }
                    ),
                }
            ),
            SemanticKind.PROBABILITY: MappingProxyType(
                {
                    Unit.DIMENSIONLESS: frozenset(
                        {
                            PopulationBasis.NOT_APPLICABLE,
                            PopulationBasis.RESIDENT_ESTIMATE,
                            PopulationBasis.PENSIONER,
                        }
                    ),
                    Unit.PERCENT: frozenset(
                        {
                            PopulationBasis.NOT_APPLICABLE,
                            PopulationBasis.RESIDENT_ESTIMATE,
                            PopulationBasis.PENSIONER,
                        }
                    ),
                }
            ),
            SemanticKind.INDEX: MappingProxyType(
                {
                    Unit.DIMENSIONLESS: frozenset({PopulationBasis.NOT_APPLICABLE}),
                    Unit.PERCENT: frozenset({PopulationBasis.NOT_APPLICABLE}),
                    Unit.DAY: frozenset({PopulationBasis.NOT_APPLICABLE}),
                    Unit.DEGREE_CELSIUS: frozenset({PopulationBasis.NOT_APPLICABLE}),
                    Unit.MILLIMETER: frozenset({PopulationBasis.NOT_APPLICABLE}),
                    Unit.MICROGRAM_PER_CUBIC_METER: frozenset({PopulationBasis.NOT_APPLICABLE}),
                }
            ),
            SemanticKind.PRICE: MappingProxyType(
                {
                    Unit.CLP: frozenset({PopulationBasis.NOT_APPLICABLE}),
                    Unit.UF: frozenset({PopulationBasis.NOT_APPLICABLE}),
                }
            ),
            SemanticKind.AMOUNT: MappingProxyType(
                {
                    Unit.CLP: frozenset({PopulationBasis.NOT_APPLICABLE}),
                    Unit.UF: frozenset({PopulationBasis.NOT_APPLICABLE}),
                }
            ),
            SemanticKind.SIGNED_AMOUNT: MappingProxyType(
                {
                    Unit.CLP: frozenset({PopulationBasis.NOT_APPLICABLE}),
                    Unit.UF: frozenset({PopulationBasis.NOT_APPLICABLE}),
                }
            ),
            SemanticKind.DISCOUNT_FACTOR: MappingProxyType(
                {Unit.DIMENSIONLESS: frozenset({PopulationBasis.NOT_APPLICABLE})}
            ),
        }
    )
)


for definition in (
    "person = [population]",
    "event = [event]",
    "CLP = [currency]",
    "UF = [indexed_currency]",
):
    try:
        UNIT_REGISTRY.define(definition)
    except pint.errors.DefinitionSyntaxError:
        raise
    except pint.errors.RedefinitionError:
        pass


def assert_unit_compatible(left: Unit, right: Unit) -> None:
    left_unit = UNIT_REGISTRY.parse_units(left.value)
    right_unit = UNIT_REGISTRY.parse_units(right.value)
    if left_unit.dimensionality != right_unit.dimensionality:
        raise UnitMismatchError(f"Incompatible units: {left.value!r} and {right.value!r}.")


def assert_semantic_compatible(
    kind: SemanticKind,
    unit: Unit,
    population_basis: PopulationBasis,
) -> None:
    """Require a supported semantic-kind, unit, and population-basis contract."""

    unit_contracts = SEMANTIC_COMPATIBILITY[kind]
    if unit not in unit_contracts:
        expected = sorted(candidate.value for candidate in unit_contracts)
        raise UnitMismatchError(
            f"Semantic kind {kind.value!r} is incompatible with unit "
            f"{unit.value!r}; expected one of {expected!r}."
        )

    compatible_bases = unit_contracts[unit]
    if population_basis not in compatible_bases:
        expected = sorted(candidate.value for candidate in compatible_bases)
        raise PopulationBasisError(
            f"Semantic kind {kind.value!r} is incompatible with population basis "
            f"{population_basis.value!r}; expected one of {expected!r}."
        )
