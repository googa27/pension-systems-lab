"""Canonical Chilean financial-entity identities and temporal legal aliases."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Final

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.data.domain_schemas import FinancialEntityType

_CANONICAL_RUT: Final = re.compile(r"cl-rut:(?P<body>[1-9][0-9]{0,7})-(?P<check_digit>[0-9Kk])")
_DOTTED_RUT: Final = re.compile(
    r"(?P<body>[0-9]{1,3}(?:\.[0-9]{3}){1,2})-"
    r"(?P<check_digit>[0-9Kk])"
)
_HYPHENATED_RUT: Final = re.compile(r"(?P<body>[0-9]{1,8})-(?P<check_digit>[0-9Kk])")
_COMPACT_RUT: Final = re.compile(r"(?P<body>[0-9]{1,8})(?P<check_digit>[0-9Kk])")
_SOURCE_KEY: Final = re.compile(r"[a-z0-9][a-z0-9_]*")


def _expected_check_digit(body: str) -> str:
    total = 0
    weight = 2
    for digit in reversed(body):
        total += int(digit) * weight
        weight = 2 if weight == 7 else weight + 1

    candidate = 11 - total % 11
    if candidate == 11:
        return "0"
    if candidate == 10:
        return "K"
    return str(candidate)


def _parsed_rut(raw: str) -> tuple[str, str]:
    canonical = _CANONICAL_RUT.fullmatch(raw)
    if canonical is not None:
        return canonical.group("body"), canonical.group("check_digit").upper()

    parsed = (
        _DOTTED_RUT.fullmatch(raw) or _HYPHENATED_RUT.fullmatch(raw) or _COMPACT_RUT.fullmatch(raw)
    )
    if parsed is None:
        raise DataContractError("Invalid Chilean RUT spelling.")

    body = parsed.group("body").replace(".", "")
    check_digit = parsed.group("check_digit").upper()
    return body, check_digit


def normalize_chilean_rut(raw: str) -> str:
    """Normalize and validate one raw Chilean RUT spelling."""

    if type(raw) is not str:
        raise DataContractError("Chilean RUT input must be an exact built-in string.")
    if any(unicodedata.category(character).startswith("C") for character in raw):
        raise DataContractError("Chilean RUT input cannot contain control characters.")

    body, check_digit = _parsed_rut(raw.strip())
    if not 1 <= len(body) <= 8:
        raise DataContractError("Chilean RUT body must contain one to eight digits.")
    canonical_body = body.lstrip("0")
    if not canonical_body:
        raise DataContractError("Chilean RUT body cannot be all zeroes.")
    if _expected_check_digit(canonical_body) != check_digit:
        raise DataContractError("Chilean RUT check digit is invalid.")
    return f"cl-rut:{canonical_body}-{check_digit}"


def _require_canonical_rut(entity_id: object) -> str:
    if type(entity_id) is not str:
        raise DataContractError("Financial entity ID must be an exact built-in canonical string.")
    try:
        normalized = normalize_chilean_rut(entity_id)
    except DataContractError:
        raise DataContractError(
            "Financial entity ID must be a valid canonical Chilean RUT."
        ) from None
    if normalized != entity_id:
        raise DataContractError("Financial entity ID must use canonical cl-rut spelling.")
    return entity_id


def _require_entity_type(
    entity_type: FinancialEntityType | str,
) -> FinancialEntityType:
    if isinstance(entity_type, FinancialEntityType):
        return entity_type
    if type(entity_type) is not str:
        raise DataContractError(
            "Financial entity type must use the closed FinancialEntityType vocabulary."
        )
    try:
        return FinancialEntityType(entity_type)
    except ValueError:
        raise DataContractError(
            "Financial entity type must use the closed FinancialEntityType vocabulary."
        ) from None


def validate_financial_entity(
    entity_id: str,
    entity_type: FinancialEntityType | str,
) -> str:
    """Return a strict canonical entity ID after checking its closed type."""

    canonical_type = _require_entity_type(entity_type)
    if canonical_type is FinancialEntityType.MARKET_TOTAL:
        if type(entity_id) is not str or entity_id != "market:total":
            raise DataContractError(
                "Financial entity type market_total requires entity ID 'market:total'."
            )
        return entity_id

    if entity_id == "market:total":
        raise DataContractError(
            "Financial entity types insurer and afp require a canonical Chilean RUT."
        )
    return _require_canonical_rut(entity_id)


@dataclass(frozen=True, slots=True)
class FinancialEntityAlias:
    """One official legal label over a half-open validity interval."""

    entity_id: str
    label: str
    valid_start: date
    valid_end: date | None
    source_key: str

    def __post_init__(self) -> None:
        try:
            _require_canonical_rut(self.entity_id)
        except DataContractError:
            raise DataContractError(
                "Financial entity alias entity_id must be a canonical Chilean RUT."
            ) from None
        if (
            type(self.label) is not str
            or not self.label.strip()
            or self.label.strip() != self.label
            or any(unicodedata.category(character).startswith("C") for character in self.label)
        ):
            raise DataContractError(
                "Financial entity alias label must be a nonblank, outer-trimmed string."
            )
        if type(self.source_key) is not str or _SOURCE_KEY.fullmatch(self.source_key) is None:
            raise DataContractError(
                "Financial entity alias source_key must use lowercase snake-case."
            )
        if type(self.valid_start) is not date:
            raise DataContractError("Financial entity alias valid_start must be an exact date.")
        if self.valid_end is not None and type(self.valid_end) is not date:
            raise DataContractError(
                "Financial entity alias valid_end must be an exact date or None."
            )
        if self.valid_end is not None and self.valid_start >= self.valid_end:
            raise DataContractError(
                "Financial entity alias requires an ordered half-open interval."
            )

    def __contains__(self, candidate: object) -> bool:
        if type(candidate) is not date:
            return False
        return self.valid_start <= candidate and (
            self.valid_end is None or candidate < self.valid_end
        )
