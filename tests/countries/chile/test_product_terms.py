from __future__ import annotations

from asyncio import CancelledError
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import FrozenInstanceError, fields
from decimal import Decimal

import numpy as np
import pytest

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.countries.chile.product_terms import (
    PensionProductTerm,
    ProductTermVocabulary,
    validate_product_term_set,
)

_FIELD_NAMES = (
    "product_term_set_id",
    "term_ordinal",
    "term_code",
    "term_role",
    "numeric_value",
    "lower_bound",
    "upper_bound",
    "unit",
    "source_label",
)
_CODES = (
    "guarantee_years",
    "guarantee_years_band",
    "survivor_percentage",
    "survivor_percentage_band",
    "temporary_increase_months",
    "temporary_increase_months_band",
    "temporary_increase_percentage",
    "temporary_increase_percentage_band",
)


class _StringSubclass(str):
    pass


def _term(
    *,
    product_term_set_id: str = "scomp:v1:fixture",
    term_ordinal: int = 0,
    term_code: str = "guarantee_years",
    term_role: str = "contract_term",
    numeric_value: int | float | None = 10,
    lower_bound: int | float | None = None,
    upper_bound: int | float | None = None,
    unit: str = "year",
    source_label: str = "10 años garantizados",
) -> PensionProductTerm:
    return PensionProductTerm(
        product_term_set_id=product_term_set_id,
        term_ordinal=term_ordinal,
        term_code=term_code,
        term_role=term_role,
        numeric_value=numeric_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unit=unit,
        source_label=source_label,
    )


def _band(
    *,
    product_term_set_id: str = "scomp:v1:fixture",
    term_ordinal: int = 0,
    term_code: str = "guarantee_years_band",
    term_role: str = "published_aggregate_grouping",
    lower_bound: int | float = 0,
    upper_bound: int | float = 10,
    unit: str = "year",
    source_label: str = "[0, 10) años",
) -> PensionProductTerm:
    return _term(
        product_term_set_id=product_term_set_id,
        term_ordinal=term_ordinal,
        term_code=term_code,
        term_role=term_role,
        numeric_value=None,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unit=unit,
        source_label=source_label,
    )


@pytest.fixture
def vocabulary() -> ProductTermVocabulary:
    return ProductTermVocabulary.scomp_v1()


def test_term_has_exact_frozen_slotted_value_fields() -> None:
    term = _term()
    equal = _term()

    assert tuple(field.name for field in fields(PensionProductTerm)) == _FIELD_NAMES
    assert not hasattr(term, "__dict__")
    assert term == equal
    assert hash(term) == hash(equal)
    assert repr(term) == repr(equal)
    assert term.__eq__(object()) is NotImplemented
    with pytest.raises(FrozenInstanceError):
        term.source_label = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        _ = term < equal  # type: ignore[operator]


def test_term_owns_builtin_numbers_as_floats_and_preserves_source_label() -> None:
    scalar = _term(numeric_value=10, source_label="Aumento temporal: 100% — 12 meses")
    interval = _band(lower_bound=0, upper_bound=10)

    assert type(scalar.numeric_value) is float
    assert scalar.numeric_value == 10.0
    assert type(interval.lower_bound) is float
    assert type(interval.upper_bound) is float
    assert interval.lower_bound == 0.0
    assert interval.upper_bound == 10.0
    assert scalar.source_label == "Aumento temporal: 100% — 12 meses"


@pytest.mark.parametrize(
    ("numeric_value", "lower_bound", "upper_bound"),
    [
        (None, None, None),
        (1, 0, 2),
        (None, 0, None),
        (None, None, 1),
    ],
)
def test_term_requires_exactly_one_complete_value_shape(
    numeric_value: int | None,
    lower_bound: int | None,
    upper_bound: int | None,
) -> None:
    with pytest.raises(DataContractError, match=r"shape|scalar|interval|exactly"):
        _term(
            numeric_value=numeric_value,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound"),
    [(1, 1), (2, 1)],
)
def test_interval_requires_strictly_ordered_bounds(
    lower_bound: int,
    upper_bound: int,
) -> None:
    with pytest.raises(DataContractError, match=r"lower|upper|interval"):
        _band(lower_bound=lower_bound, upper_bound=upper_bound)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("numeric_value", float("nan")),
        ("numeric_value", float("inf")),
        ("numeric_value", float("-inf")),
        ("lower_bound", float("nan")),
        ("upper_bound", float("inf")),
    ],
)
def test_term_rejects_nonfinite_numbers(field: str, value: float) -> None:
    arguments: dict[str, object]
    if field == "numeric_value":
        arguments = {"numeric_value": value}
    else:
        arguments = {
            "numeric_value": None,
            "lower_bound": 0,
            "upper_bound": 1,
            field: value,
        }
    with pytest.raises(DataContractError, match="finite"):
        _term(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [True, np.int64(10), np.float64(10), Decimal("10"), "10"],
    ids=["bool", "numpy-int", "numpy-float", "decimal", "string"],
)
def test_numeric_atoms_require_exact_builtin_numbers(value: object) -> None:
    with pytest.raises(DataContractError, match=r"numeric|built-in|number"):
        _term(numeric_value=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "ordinal",
    [-1, True, 0.0, np.int64(0)],
    ids=["negative", "bool", "float", "numpy-int"],
)
def test_ordinal_is_an_exact_nonnegative_builtin_integer(ordinal: object) -> None:
    with pytest.raises(DataContractError, match="ordinal"):
        _term(term_ordinal=ordinal)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("product_term_set_id", ""),
        ("product_term_set_id", " scomp:v1:fixture"),
        ("product_term_set_id", "scomp:v0:fixture"),
        ("product_term_set_id", "scomp:v1:"),
        ("term_code", "GuaranteeYears"),
        ("term_code", "guarantee-years"),
        ("term_role", "contract"),
        ("unit", "Year"),
        ("source_label", ""),
        ("source_label", " padded "),
        ("source_label", "line\nbreak"),
        ("source_label", "zero\u200bwidth"),
    ],
)
def test_text_atoms_are_canonical_or_preserved_metadata(
    field: str,
    value: str,
) -> None:
    arguments = {field: value}
    with pytest.raises(DataContractError):
        _term(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", _FIELD_NAMES[0:4] + _FIELD_NAMES[7:9])
def test_text_atoms_require_exact_builtin_strings(field: str) -> None:
    with pytest.raises(DataContractError, match=r"string|text|atom"):
        _term(**{field: _StringSubclass("valid")})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    (
        "term_code",
        "term_role",
        "numeric_value",
        "lower_bound",
        "upper_bound",
        "unit",
    ),
    [
        ("guarantee_years", "contract_term", 10, None, None, "year"),
        (
            "survivor_percentage",
            "contract_term",
            0.6,
            None,
            None,
            "dimensionless",
        ),
        (
            "temporary_increase_percentage",
            "contract_term",
            1.0,
            None,
            None,
            "dimensionless",
        ),
        (
            "temporary_increase_months",
            "contract_term",
            12,
            None,
            None,
            "month",
        ),
        (
            "guarantee_years_band",
            "published_aggregate_grouping",
            None,
            0,
            10,
            "year",
        ),
        (
            "survivor_percentage_band",
            "published_aggregate_grouping",
            None,
            0,
            1,
            "dimensionless",
        ),
        (
            "temporary_increase_percentage_band",
            "published_aggregate_grouping",
            None,
            0.5,
            1,
            "dimensionless",
        ),
        (
            "temporary_increase_months_band",
            "published_aggregate_grouping",
            None,
            0,
            12,
            "month",
        ),
    ],
)
def test_scomp_v1_accepts_each_exact_rule(
    vocabulary: ProductTermVocabulary,
    term_code: str,
    term_role: str,
    numeric_value: int | float | None,
    lower_bound: int | float | None,
    upper_bound: int | float | None,
    unit: str,
) -> None:
    term = _term(
        term_code=term_code,
        term_role=term_role,
        numeric_value=numeric_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unit=unit,
    )

    validated = validate_product_term_set((term,), vocabulary)

    assert validated == (term,)
    assert validated[0] is not term


@pytest.mark.parametrize(
    ("term_code", "changes"),
    [
        ("guarantee_years", {"numeric_value": 0}),
        ("survivor_percentage", {"numeric_value": 1.1, "unit": "dimensionless"}),
        (
            "temporary_increase_percentage",
            {"numeric_value": 0, "unit": "dimensionless"},
        ),
        (
            "temporary_increase_months",
            {"numeric_value": 1.5, "unit": "month"},
        ),
        (
            "guarantee_years_band",
            {
                "numeric_value": None,
                "lower_bound": -1,
                "upper_bound": 10,
                "term_role": "published_aggregate_grouping",
            },
        ),
        (
            "survivor_percentage_band",
            {
                "numeric_value": None,
                "lower_bound": 0,
                "upper_bound": 1.1,
                "term_role": "published_aggregate_grouping",
                "unit": "dimensionless",
            },
        ),
        (
            "temporary_increase_percentage_band",
            {
                "numeric_value": None,
                "lower_bound": -0.1,
                "upper_bound": 1,
                "term_role": "published_aggregate_grouping",
                "unit": "dimensionless",
            },
        ),
        (
            "temporary_increase_months_band",
            {
                "numeric_value": None,
                "lower_bound": 0,
                "upper_bound": 2.5,
                "term_role": "published_aggregate_grouping",
                "unit": "month",
            },
        ),
    ],
)
def test_scomp_v1_rejects_each_rule_outside_its_range_or_integrality(
    vocabulary: ProductTermVocabulary,
    term_code: str,
    changes: dict[str, object],
) -> None:
    term = _term(term_code=term_code, **changes)  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match=r"range|integral|positive|unit|rule"):
        validate_product_term_set((term,), vocabulary)


@pytest.mark.parametrize(
    "changes",
    [
        {"term_role": "published_aggregate_grouping"},
        {"unit": "month"},
        {
            "numeric_value": None,
            "lower_bound": 1,
            "upper_bound": 2,
        },
        {"term_code": "unknown_term"},
    ],
)
def test_validation_checks_code_role_shape_and_unit_jointly(
    vocabulary: ProductTermVocabulary,
    changes: dict[str, object],
) -> None:
    term = _term(**changes)  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match=r"code|role|shape|unit|rule"):
        validate_product_term_set((term,), vocabulary)


def test_scalar_codes_are_repeatable_but_each_band_code_is_unique(
    vocabulary: ProductTermVocabulary,
) -> None:
    repeated = (
        _term(term_ordinal=0, numeric_value=10),
        _term(term_ordinal=1, numeric_value=15),
    )
    assert validate_product_term_set(repeated, vocabulary) == repeated

    duplicate_band = (
        _band(term_ordinal=0, lower_bound=0, upper_bound=10),
        _band(term_ordinal=1, lower_bound=10, upper_bound=20),
    )
    with pytest.raises(DataContractError, match=r"cardinality|band|duplicate"):
        validate_product_term_set(duplicate_band, vocabulary)


@pytest.mark.parametrize(
    "terms",
    [
        (
            _term(product_term_set_id="scomp:v1:a", term_ordinal=0),
            _term(product_term_set_id="scomp:v1:b", term_ordinal=1),
        ),
        (_term(term_ordinal=0), _term(term_ordinal=0)),
        (_term(term_ordinal=1),),
        (_term(term_ordinal=0), _term(term_ordinal=2)),
        (_term(product_term_set_id="other:v1:a"),),
        (_term(product_term_set_id="scomp:v2:a"),),
    ],
    ids=[
        "mixed-set-id",
        "duplicate-ordinal",
        "does-not-start-zero",
        "ordinal-gap",
        "wrong-source",
        "wrong-version",
    ],
)
def test_set_identity_and_ordinals_are_exact(
    vocabulary: ProductTermVocabulary,
    terms: tuple[PensionProductTerm, ...],
) -> None:
    with pytest.raises(DataContractError, match=r"set|ordinal|source|version"):
        validate_product_term_set(terms, vocabulary)


class _OnePassTerms(Sequence[PensionProductTerm]):
    def __init__(self, values: tuple[PensionProductTerm, ...]) -> None:
        self._values = values
        self.iterations = 0

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> PensionProductTerm:
        return self._values[index]

    def __iter__(self) -> Iterator[PensionProductTerm]:
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("input sequence was iterated more than once")
        return iter(self._values)


def test_validation_snapshots_once_returns_copies_and_sorts_by_ordinal(
    vocabulary: ProductTermVocabulary,
) -> None:
    later = _term(term_ordinal=1, term_code="guarantee_years", numeric_value=15)
    earlier = _term(
        term_ordinal=0,
        term_code="survivor_percentage",
        numeric_value=0.6,
        unit="dimensionless",
    )
    sequence = _OnePassTerms((later, earlier))

    result = validate_product_term_set(sequence, vocabulary)

    assert sequence.iterations == 1
    assert tuple(term.term_ordinal for term in result) == (0, 1)
    assert result == (earlier, later)
    assert result[0] is not earlier
    assert result[1] is not later


@pytest.mark.parametrize("bad_input", ["terms", object(), {1, 2}, iter(())])
def test_validation_requires_a_nonstring_sequence(
    vocabulary: ProductTermVocabulary,
    bad_input: object,
) -> None:
    with pytest.raises(DataContractError, match=r"Sequence|sequence"):
        validate_product_term_set(bad_input, vocabulary)  # type: ignore[arg-type]


def test_empty_term_set_is_valid(vocabulary: ProductTermVocabulary) -> None:
    assert validate_product_term_set((), vocabulary) == ()
    assert validate_product_term_set([], vocabulary) == ()


def test_vocabulary_is_factory_only_frozen_hashable_and_deterministic() -> None:
    with pytest.raises(TypeError):
        ProductTermVocabulary()  # type: ignore[call-arg]

    left = ProductTermVocabulary.scomp_v1()
    right = ProductTermVocabulary.scomp_v1()

    assert left == right
    assert hash(left) == hash(right)
    assert not hasattr(left, "__dict__")
    assert len(left) == 8
    assert tuple(left) == _CODES
    assert all(code in left for code in _CODES)
    assert "unknown_term" not in left
    assert 1 not in left  # type: ignore[operator]
    assert not hasattr(left, "rules")
    assert not hasattr(left, "lookup")
    with pytest.raises(TypeError):
        _ = left["guarantee_years"]  # type: ignore[index]
    with pytest.raises((FrozenInstanceError, AttributeError)):
        left._rules = ()  # type: ignore[attr-defined]


def test_validation_rejects_forged_vocabulary_and_term_subclasses() -> None:
    forged_vocabulary = object.__new__(ProductTermVocabulary)
    with pytest.raises(DataContractError, match="vocabulary"):
        validate_product_term_set((), forged_vocabulary)

    class TermSubclass(PensionProductTerm):
        pass

    subclass = object.__new__(TermSubclass)
    for field in fields(PensionProductTerm):
        object.__setattr__(subclass, field.name, getattr(_term(), field.name))

    with pytest.raises(DataContractError, match=r"exact|term"):
        validate_product_term_set(
            (subclass,),
            ProductTermVocabulary.scomp_v1(),
        )


def test_validation_reconstructs_terms_and_rejects_forged_control_atoms(
    vocabulary: ProductTermVocabulary,
) -> None:
    forged = object.__new__(PensionProductTerm)
    valid = _term()
    for field in fields(PensionProductTerm):
        object.__setattr__(forged, field.name, getattr(valid, field.name))
    object.__setattr__(forged, "source_label", "hidden\u202econtrol")

    with pytest.raises(DataContractError, match=r"source_label|control|term"):
        validate_product_term_set((forged,), vocabulary)


def test_oversized_positive_set_version_fails_with_typed_contract_error(
    vocabulary: ProductTermVocabulary,
) -> None:
    oversized_version = "9" * 5000
    term = _term(
        product_term_set_id=f"scomp:v{oversized_version}:fixture",
    )

    with pytest.raises(DataContractError, match=r"set|source|version"):
        validate_product_term_set((term,), vocabulary)


def test_corrupting_one_vocabulary_rule_cannot_affect_a_fresh_factory() -> None:
    corrupted = ProductTermVocabulary.scomp_v1()
    rule = corrupted._rules[0]
    original_maximum = rule.maximum
    object.__setattr__(rule, "maximum", 5.0)
    try:
        fresh = ProductTermVocabulary.scomp_v1()

        assert fresh._rules[0] is not rule
        assert fresh._rules[0].maximum is original_maximum
    finally:
        object.__setattr__(rule, "maximum", original_maximum)


def test_validation_rejects_a_vocabulary_with_a_corrupted_private_rule() -> None:
    corrupted = ProductTermVocabulary.scomp_v1()
    rule = corrupted._rules[0]
    original_maximum = rule.maximum
    object.__setattr__(rule, "maximum", 5.0)
    try:
        with pytest.raises(DataContractError, match=r"vocabulary|malformed"):
            validate_product_term_set((), corrupted)
    finally:
        object.__setattr__(rule, "maximum", original_maximum)


class _FailingSnapshotTerms(Sequence[PensionProductTerm]):
    def __init__(self, value: PensionProductTerm) -> None:
        self._value = value
        self.iterations = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> PensionProductTerm:
        if index == 0:
            return self._value
        raise IndexError(index)

    def __iter__(self) -> Iterator[PensionProductTerm]:
        self.iterations += 1
        yield self._value
        raise RuntimeError("hostile sequence payload")


def test_snapshot_failure_is_typed_and_does_not_retry(
    vocabulary: ProductTermVocabulary,
) -> None:
    sequence = _FailingSnapshotTerms(_term())

    with pytest.raises(DataContractError, match=r"sequence|snapshot"):
        validate_product_term_set(sequence, vocabulary)

    assert sequence.iterations == 1


class _RaisingSnapshotTerms(Sequence[PensionProductTerm]):
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.iterations = 0

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> PensionProductTerm:
        if index == 0:
            return _term()
        raise IndexError(index)

    def __iter__(self) -> Iterator[PensionProductTerm]:
        self.iterations += 1
        raise self._error


def test_snapshot_cancellation_propagates_unchanged(
    vocabulary: ProductTermVocabulary,
) -> None:
    cancellation = CancelledError("cancel snapshot")
    sequence = _RaisingSnapshotTerms(cancellation)

    with pytest.raises(CancelledError) as captured:
        validate_product_term_set(sequence, vocabulary)

    assert captured.value is cancellation
    assert sequence.iterations == 1


def test_snapshot_interrupt_group_propagates_unchanged(
    vocabulary: ProductTermVocabulary,
) -> None:
    interrupt_group = BaseExceptionGroup(
        "interrupt snapshot",
        [KeyboardInterrupt()],
    )
    sequence = _RaisingSnapshotTerms(interrupt_group)

    with pytest.raises(BaseExceptionGroup) as captured:
        validate_product_term_set(sequence, vocabulary)

    assert captured.value is interrupt_group
    assert sequence.iterations == 1


class _RaisingLookup(Mapping[str, object]):
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.iterations = 0

    def __len__(self) -> int:
        return 8

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        self.iterations += 1
        raise self._error


def test_validation_rejects_equivalent_mutable_vocabulary_lookup() -> None:
    vocabulary = ProductTermVocabulary.scomp_v1()
    object.__setattr__(vocabulary, "_lookup", dict(vocabulary._lookup))

    with pytest.raises(DataContractError, match=r"vocabulary|malformed|lookup"):
        validate_product_term_set((), vocabulary)


def test_hostile_vocabulary_lookup_failure_is_typed() -> None:
    vocabulary = ProductTermVocabulary.scomp_v1()
    lookup = _RaisingLookup(RuntimeError("hostile lookup payload"))
    object.__setattr__(vocabulary, "_lookup", lookup)

    with pytest.raises(DataContractError, match=r"vocabulary|malformed|lookup"):
        validate_product_term_set((), vocabulary)

    assert lookup.iterations == 1


def test_vocabulary_lookup_cancellation_propagates_unchanged() -> None:
    vocabulary = ProductTermVocabulary.scomp_v1()
    cancellation = CancelledError("cancel lookup")
    lookup = _RaisingLookup(cancellation)
    object.__setattr__(vocabulary, "_lookup", lookup)

    with pytest.raises(CancelledError) as captured:
        validate_product_term_set((), vocabulary)

    assert captured.value is cancellation
    assert lookup.iterations == 1
