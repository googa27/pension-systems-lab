"""Versioned child rows for Chilean pension-product terms."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Final, cast

from chile_demographic_pde.core.errors import DataContractError

_SET_ID: Final = re.compile(
    r"(?P<source>[a-z0-9][a-z0-9_]*):v"
    r"(?P<version>[1-9][0-9]*):(?P<id>[a-z0-9][a-z0-9_]*)"
)
_ATOM: Final = re.compile(r"[a-z0-9][a-z0-9_]*")
_ROLES: Final = frozenset({"contract_term", "published_aggregate_grouping"})
_MAPPING_PROXY_TYPE: Final[type[object]] = type(MappingProxyType({}))


def _text(value: object, name: str, pattern: re.Pattern[str] | None = None) -> str:
    if type(value) is not str:
        raise DataContractError(f"Product term {name} must be an exact built-in string.")
    has_control = any(unicodedata.category(character).startswith("C") for character in value)
    if (
        not value
        or value.strip() != value
        or has_control
        or (pattern is not None and pattern.fullmatch(value) is None)
    ):
        raise DataContractError(f"Product term {name} is not a canonical text atom.")
    return value


def _number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise DataContractError(f"Product term {name} must be an exact built-in number.")
    try:
        owned = float(cast("int | float", value))
    except OverflowError:
        raise DataContractError(f"Product term {name} must be finite.") from None
    if not math.isfinite(owned):
        raise DataContractError(f"Product term {name} must be finite.")
    return owned


@dataclass(frozen=True, slots=True)
class PensionProductTerm:
    """One immutable scalar clause or published aggregate interval."""

    product_term_set_id: str
    term_ordinal: int
    term_code: str
    term_role: str
    numeric_value: float | None
    lower_bound: float | None
    upper_bound: float | None
    unit: str
    source_label: str

    def __post_init__(self) -> None:
        for value, name, pattern in (
            (self.product_term_set_id, "product_term_set_id", _SET_ID),
            (self.term_code, "term_code", _ATOM),
            (self.term_role, "term_role", None),
            (self.unit, "unit", _ATOM),
            (self.source_label, "source_label", None),
        ):
            _text(value, name, pattern)
        if self.term_role not in _ROLES:
            raise DataContractError("Product term role is outside its closed vocabulary.")
        if type(self.term_ordinal) is not int or self.term_ordinal < 0:
            raise DataContractError(
                "Product term ordinal atom must be an exact nonnegative built-in integer."
            )

        scalar = (
            self.numeric_value is not None and self.lower_bound is None and self.upper_bound is None
        )
        interval = (
            self.numeric_value is None
            and self.lower_bound is not None
            and self.upper_bound is not None
        )
        if scalar == interval:
            raise DataContractError(
                "Product term requires exactly one scalar or complete interval shape."
            )
        if scalar:
            object.__setattr__(self, "numeric_value", _number(self.numeric_value, "numeric_value"))
            return
        lower = _number(self.lower_bound, "lower_bound")
        upper = _number(self.upper_bound, "upper_bound")
        if lower >= upper:
            raise DataContractError("Product term interval requires lower < upper.")
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)


@dataclass(frozen=True, slots=True)
class _Rule:
    code: str
    role: str
    shape: str
    unit: str
    integral: bool
    maximum: float | None
    cardinality: int | None


type _RuleRow = tuple[str, str, str, str, bool, float | None, int | None]
_C = "contract_term"
_A = "published_aggregate_grouping"
_RULE_ROWS: Final[tuple[_RuleRow, ...]] = (
    ("guarantee_years", _C, "scalar", "year", True, None, None),
    ("guarantee_years_band", _A, "interval", "year", True, None, 1),
    ("survivor_percentage", _C, "scalar", "dimensionless", False, 1.0, None),
    ("survivor_percentage_band", _A, "interval", "dimensionless", False, 1.0, 1),
    ("temporary_increase_months", _C, "scalar", "month", True, None, None),
    ("temporary_increase_months_band", _A, "interval", "month", True, None, 1),
    (
        "temporary_increase_percentage",
        _C,
        "scalar",
        "dimensionless",
        False,
        1.0,
        None,
    ),
    (
        "temporary_increase_percentage_band",
        _A,
        "interval",
        "dimensionless",
        False,
        1.0,
        1,
    ),
)


def _fresh_rules() -> tuple[_Rule, ...]:
    return tuple(_Rule(*row) for row in _RULE_ROWS)


@dataclass(frozen=True, slots=True, init=False)
class ProductTermVocabulary:
    """Factory-created immutable rules for one source vocabulary version."""

    _source: str
    _version: int
    _rules: tuple[_Rule, ...]
    _lookup: Mapping[str, _Rule] = field(compare=False, hash=False, repr=False)

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("ProductTermVocabulary requires a versioned factory.")

    @classmethod
    def _from_rules(
        cls,
        source: str,
        version: int,
        rules: tuple[_Rule, ...],
    ) -> ProductTermVocabulary:
        value = object.__new__(cls)
        object.__setattr__(value, "_source", source)
        object.__setattr__(value, "_version", version)
        object.__setattr__(value, "_rules", rules)
        object.__setattr__(
            value,
            "_lookup",
            MappingProxyType({rule.code: rule for rule in rules}),
        )
        return value

    @classmethod
    def scomp_v1(cls) -> ProductTermVocabulary:
        return cls._from_rules("scomp", 1, _fresh_rules())

    def __len__(self) -> int:
        return len(self._rules)

    def __contains__(self, code: object) -> bool:
        return type(code) is str and code in self._lookup

    def __iter__(self) -> Iterator[str]:
        return (rule.code for rule in self._rules)


def _vocabulary(value: object) -> ProductTermVocabulary:
    if type(value) is not ProductTermVocabulary:
        raise DataContractError("An exact product-term vocabulary is required.")
    expected = ProductTermVocabulary._from_rules("scomp", 1, _fresh_rules())
    try:
        lookup = value._lookup
        exact_representation = type(lookup) is _MAPPING_PROXY_TYPE
        exact_contents = dict(lookup) == dict(expected._lookup)
        exact = value == expected and exact_representation and exact_contents
    except Exception:
        exact = False
    if not exact:
        raise DataContractError("The product-term vocabulary is malformed.")
    return expected


def _term(value: object) -> PensionProductTerm:
    if type(value) is not PensionProductTerm:
        raise DataContractError("An exact PensionProductTerm value is required.")
    try:
        return replace(value)
    except DataContractError:
        raise
    except (AttributeError, TypeError, ValueError):
        raise DataContractError("The exact product term is malformed.") from None


def _check_rule(term: PensionProductTerm, rule: _Rule) -> None:
    shape = "scalar" if term.numeric_value is not None else "interval"
    if (term.term_role, shape, term.unit) != (rule.role, rule.shape, rule.unit):
        raise DataContractError("Product term violates its role, shape, or unit rule.")
    optional = (term.numeric_value,) if shape == "scalar" else (term.lower_bound, term.upper_bound)
    numbers = tuple(value for value in optional if value is not None)
    if rule.integral and any(not value.is_integer() for value in numbers):
        raise DataContractError("Product term violates its integral-value rule.")
    if shape == "scalar":
        if numbers[0] <= 0 or (rule.maximum is not None and numbers[0] > rule.maximum):
            raise DataContractError("Product term scalar is outside its positive range.")
    elif numbers[0] < 0 or (rule.maximum is not None and numbers[1] > rule.maximum):
        raise DataContractError("Product term interval is outside its range.")


def validate_product_term_set(
    terms: Sequence[PensionProductTerm],
    vocabulary: ProductTermVocabulary,
) -> tuple[PensionProductTerm, ...]:
    """Return one detached, validated term set in ordinal order."""

    rules = _vocabulary(vocabulary)
    if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence):
        raise DataContractError("Product terms must be a non-string Sequence.")
    try:
        snapshot = tuple(terms)
    except Exception as error:
        raise DataContractError("Product term sequence snapshot failed.") from error
    if not snapshot:
        return ()
    owned = tuple(_term(value) for value in snapshot)
    set_ids = {term.product_term_set_id for term in owned}
    if len(set_ids) != 1:
        raise DataContractError("Terms must share exactly one term-set identity.")
    parsed = _SET_ID.fullmatch(next(iter(set_ids)))
    if (
        parsed is None
        or parsed.group("source") != rules._source
        or parsed.group("version") != str(rules._version)
    ):
        raise DataContractError("Term-set source/version does not match its vocabulary.")

    ordered = tuple(sorted(owned, key=lambda term: term.term_ordinal))
    if tuple(term.term_ordinal for term in ordered) != tuple(range(len(ordered))):
        raise DataContractError("Term-set ordinals must be contiguous from zero.")
    counts: dict[str, int] = {}
    for term in ordered:
        rule = rules._lookup.get(term.term_code)
        if rule is None:
            raise DataContractError("Term code is absent from its vocabulary.")
        _check_rule(term, rule)
        counts[term.term_code] = counts.get(term.term_code, 0) + 1
        if rule.cardinality is not None and counts[term.term_code] > rule.cardinality:
            raise DataContractError("Aggregate band exceeds its code cardinality.")
    return ordered
