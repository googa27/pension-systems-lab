"""Strict normalization for documented Spanish categorical labels."""

from __future__ import annotations

import unicodedata

_SPANISH_TO_ASCII = str.maketrans(
    {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }
)
_DOCUMENTED_SPANISH_CHARACTERS = frozenset("áéíóúüñ")


def normalize_spanish_label(value: str, *, field: str) -> str:
    """Normalize documented accents/case/spacing without discarding characters."""

    normalized = unicodedata.normalize("NFC", value).lower().strip()
    unsupported = [
        character
        for character in normalized
        if not (character.isascii() and (character.isalnum() or character == "."))
        and not character.isspace()
        and character not in _DOCUMENTED_SPANISH_CHARACTERS
    ]
    if unsupported:
        raise ValueError(f"{field} contains unsupported Unicode characters {unsupported!r}.")
    ascii_text = normalized.translate(_SPANISH_TO_ASCII)
    return " ".join(ascii_text.split())
