from __future__ import annotations

from datetime import UTC, date, datetime

from chile_demographic_pde.data.roles import ObservationDomain


def identity_fixture_value(name: str) -> object:
    fixtures: dict[str, object] = {
        "null": None,
        "bool_true": True,
        "int_one": 1,
        "float_one": 1.0,
        "float_negative_zero": -0.0,
        "nfc_string": "e\u0301",
        "date": date(2026, 7, 1),
        "timestamp": datetime(2026, 7, 1, 12, tzinfo=UTC),
        "enum": ObservationDomain.REGULATORY_RATE,
        "mapping": {"b": 2, "a": 1},
        "list": [1, "x"],
        "tuple": (1, "x"),
        "set": {2, 1},
    }
    try:
        return fixtures[name]
    except KeyError as error:
        raise ValueError(f"Unknown independent identity fixture: {name}") from error
