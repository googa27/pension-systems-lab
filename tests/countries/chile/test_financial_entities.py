from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime

import pytest

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.countries.chile.financial_entities import (
    FinancialEntityAlias,
    normalize_chilean_rut,
    validate_financial_entity,
)
from chile_demographic_pde.data.domain_schemas import FinancialEntityType

_ENTITY_ID = "cl-rut:12345678-5"
_SOURCE_KEY = "cmf_entity_registry"


class _StringSubclass(str):
    pass


class _DateSubclass(date):
    pass


def _alias(**overrides: object) -> FinancialEntityAlias:
    values: dict[str, object] = {
        "entity_id": _ENTITY_ID,
        "label": "Compañía de Seguros",
        "valid_start": date(2020, 1, 1),
        "valid_end": date(2024, 1, 1),
        "source_key": _SOURCE_KEY,
    }
    values.update(overrides)
    return FinancialEntityAlias(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 12.345.678-5 ", _ENTITY_ID),
        ("12345678-5", _ENTITY_ID),
        ("123456785", _ENTITY_ID),
        (_ENTITY_ID, _ENTITY_ID),
        ("96.573.600-k", "cl-rut:96573600-K"),
        ("cl-rut:96573600-k", "cl-rut:96573600-K"),
        ("123456-0", "cl-rut:123456-0"),
        ("6-k", "cl-rut:6-K"),
        ("00000006-k", "cl-rut:6-K"),
        ("00123456-0", "cl-rut:123456-0"),
    ],
)
def test_rut_normalization_is_canonical_and_idempotent(
    raw: str,
    expected: str,
) -> None:
    normalized = normalize_chilean_rut(raw)

    assert normalized == expected
    assert normalize_chilean_rut(normalized) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "12.345.678-9",
        "",
        "   ",
        ".",
        "12345678",
        "-5",
        "0-0",
        "123456789-2",
        "000.000.006-K",
        "12..345.678-5",
        ".12.345.678-5",
        "12.345.678.-5",
        "12-345-678-5",
        "12 345 678-5",
        "12_345_678-5",
        "12/345/678-5",
        "+12345678-5",
        "\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668-\u0665",
        "12345678-\uff2b",
        "12345678\u20135",
        "12345678-\u200b5",
        "CL-RUT:12345678-5",
        "cl-rut:cl-rut:12345678-5",
        "market:total",
        "K2345678-5",
    ],
)
def test_rut_normalization_rejects_malformed_or_wrong_check_digits(
    raw: str,
) -> None:
    with pytest.raises(DataContractError, match=r"RUT|check digit"):
        normalize_chilean_rut(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "\n12.345.678-5",
        "12.345.678-5\t",
        "\x1c12.345.678-5",
    ],
)
def test_rut_normalization_rejects_outer_control_characters(
    raw: str,
) -> None:
    with pytest.raises(DataContractError, match="RUT"):
        normalize_chilean_rut(raw)


@pytest.mark.parametrize(
    "raw",
    [None, 12_345_678, b"12345678-5", True, _StringSubclass("12345678-5")],
)
def test_rut_normalization_requires_an_exact_builtin_string(raw: object) -> None:
    with pytest.raises(DataContractError, match="RUT"):
        normalize_chilean_rut(raw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("entity_id", "entity_type"),
    [
        (_ENTITY_ID, "insurer"),
        (_ENTITY_ID, FinancialEntityType.INSURER),
        (_ENTITY_ID, "afp"),
        (_ENTITY_ID, FinancialEntityType.AFP),
        ("market:total", "market_total"),
        ("market:total", FinancialEntityType.MARKET_TOTAL),
    ],
)
def test_entity_type_and_canonical_identity_agree(
    entity_id: str,
    entity_type: FinancialEntityType | str,
) -> None:
    assert validate_financial_entity(entity_id, entity_type) == entity_id


@pytest.mark.parametrize(
    ("entity_id", "entity_type"),
    [
        ("market:total", "insurer"),
        ("market:total", "afp"),
        (_ENTITY_ID, "market_total"),
        ("12.345.678-5", "insurer"),
        ("cl-rut:12345678-9", "insurer"),
        ("entity:unknown", "insurer"),
        ("", "insurer"),
        (_ENTITY_ID, "bank"),
        (_ENTITY_ID, ""),
        (_StringSubclass(_ENTITY_ID), "insurer"),
        (_ENTITY_ID, _StringSubclass("insurer")),
        (None, "insurer"),
        (_ENTITY_ID, None),
    ],
)
def test_entity_validation_rejects_cross_pairs_and_noncanonical_atoms(
    entity_id: object,
    entity_type: object,
) -> None:
    with pytest.raises(DataContractError, match="entity"):
        validate_financial_entity(  # type: ignore[arg-type]
            entity_id,
            entity_type,
        )


def test_alias_changes_preserve_entity_identity_and_are_half_open() -> None:
    old = _alias()
    new = _alias(
        label="Nombre actual",
        valid_start=old.valid_end,
        valid_end=None,
    )

    assert old.entity_id == new.entity_id
    assert date(2020, 1, 1) in old
    assert date(2023, 12, 31) in old
    assert date(2024, 1, 1) not in old
    assert date(2019, 12, 31) not in old
    assert date(2124, 1, 1) in new


def test_alias_is_a_frozen_hashable_value_object() -> None:
    first = _alias()
    second = _alias()
    changed = _alias(label="Otra compañía")

    assert first == second
    assert hash(first) == hash(second)
    assert first != changed
    assert first.__eq__(object()) is NotImplemented
    with pytest.raises(FrozenInstanceError):
        first.label = "mutated"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"entity_id": "market:total"}, "entity"),
        ({"entity_id": "12.345.678-5"}, "entity"),
        ({"entity_id": "cl-rut:12345678-9"}, "entity"),
        ({"entity_id": _StringSubclass(_ENTITY_ID)}, "entity"),
        ({"label": ""}, "label"),
        ({"label": "   "}, "label"),
        ({"label": " Nombre"}, "label"),
        ({"label": "Nombre "}, "label"),
        ({"label": _StringSubclass("Nombre")}, "label"),
        ({"label": "Nombre\ninyectado"}, "label"),
        ({"label": "Nombre\tinyectado"}, "label"),
        ({"label": "Nombre\x00inyectado"}, "label"),
        ({"label": "Nombre\x1finyectado"}, "label"),
        ({"source_key": ""}, "source_key"),
        ({"source_key": "CMF"}, "source_key"),
        ({"source_key": "cmf-registry"}, "source_key"),
        ({"source_key": " cmf_registry"}, "source_key"),
        ({"source_key": _StringSubclass(_SOURCE_KEY)}, "source_key"),
        ({"valid_start": datetime(2020, 1, 1)}, "valid_start"),
        ({"valid_start": _DateSubclass(2020, 1, 1)}, "valid_start"),
        ({"valid_start": "2020-01-01"}, "valid_start"),
        ({"valid_start": None}, "valid_start"),
        ({"valid_end": datetime(2024, 1, 1)}, "valid_end"),
        ({"valid_end": _DateSubclass(2024, 1, 1)}, "valid_end"),
        ({"valid_end": "2024-01-01"}, "valid_end"),
        ({"valid_end": date(2020, 1, 1)}, "half-open"),
        ({"valid_end": date(2019, 12, 31)}, "half-open"),
    ],
)
def test_alias_rejects_invalid_identity_text_and_intervals(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(DataContractError, match=match):
        _alias(**overrides)


@pytest.mark.parametrize(
    "candidate",
    [
        datetime(2022, 1, 1),
        _DateSubclass(2022, 1, 1),
        "2022-01-01",
        None,
        object(),
    ],
)
def test_alias_membership_rejects_non_exact_dates(candidate: object) -> None:
    assert candidate not in _alias()
