from __future__ import annotations

import importlib.util


def test_legacy_global_schema_module_is_removed() -> None:
    assert importlib.util.find_spec("chile_demographic_pde.data.schemas") is None
