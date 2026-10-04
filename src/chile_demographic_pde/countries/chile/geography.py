"""Shared canonicalization of documented Chilean region labels."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

_REGION_CODES: Final = {
    "de tarapaca": "01",
    "de antofagasta": "02",
    "de atacama": "03",
    "de coquimbo": "04",
    "de valparaiso": "05",
    "del libertador general bernardo ohiggins": "06",
    "del libertador general bernardo o higgins": "06",
    "del libertador b ohiggins": "06",
    "del libertador b o higgins": "06",
    "del maule": "07",
    "del biobio": "08",
    "de la araucania": "09",
    "de los lagos": "10",
    "aysen del general carlos ibanez del campo": "11",
    "de aysen del general carlos ibanez del campo": "11",
    "de aisen del gral c ibanez del campo": "11",
    "de magallanes y de la antartica chilena": "12",
    "metropolitana de santiago": "13",
    "de los rios": "14",
    "de arica y parinacota": "15",
    "de nuble": "16",
}


def normalize_region_label(value: str) -> str:
    """Normalize a documented region label without guessing a region code."""

    decomposed = unicodedata.normalize("NFKD", value.strip().lower())
    ascii_text = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text))


def region_code_for_label(value: str) -> str | None:
    """Return the two-digit code for a documented label, if it is known."""

    return _REGION_CODES.get(normalize_region_label(value))
