from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.data.identity import CanonicalIdentityRecord, IdentityRegistrySnapshot


@pytest.mark.parametrize("attribute", ["records", "_by_value", "unexpected_attribute"])
def test_snapshot_attribute_mutation_has_stable_frozen_exception(attribute: str) -> None:
    snapshot = IdentityRegistrySnapshot((CanonicalIdentityRecord("synthetic-id", b"{}"),))
    with pytest.raises(FrozenInstanceError):
        setattr(snapshot, attribute, ())
    assert snapshot.require("synthetic-id").canonical_payload == b"{}"


def test_snapshot_dict_tampering_is_detected_before_lookup() -> None:
    snapshot = IdentityRegistrySnapshot((CanonicalIdentityRecord("synthetic-id", b"{}"),))
    snapshot.__dict__["_by_value"] = MappingProxyType({})
    with pytest.raises(DataContractError, match="changed after creation"):
        snapshot.get("synthetic-id")
