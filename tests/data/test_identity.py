from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any

import pytest
from identity_fixtures import identity_fixture_value

from chile_demographic_pde.core.errors import (
    DataContractError,
    IdentityCollisionError,
)
from chile_demographic_pde.data.identity import (
    CanonicalIdentityRecord,
    FactId,
    FactIdFactory,
    IdentityRegistry,
    IdentityRegistrySnapshot,
    ObservationId,
    ObservationIdFactory,
    ReleaseId,
    ReleaseIdFactory,
    TransformationIdentity,
    canonical_identity_bytes,
)
from chile_demographic_pde.data.roles import ObservationDomain


class Example(StrEnum):
    VALUE = "value"


@dataclass(frozen=True, slots=True)
class CollisionFixtureIdentity:
    value: str
    canonical_payload: bytes


def _golden() -> dict[str, Any]:
    return json.loads(
        Path("tests/data/fixtures/identity_v1_golden.json").read_text(encoding="utf-8")
    )


def _golden_fact() -> FactId:
    return FactIdFactory.v1(
        ObservationDomain.REGULATORY_RATE,
        {"publisher": "SP", "rule_id": "2417"},
        {"variable": "programmed_withdrawal_technical_rate"},
    )


def _golden_release() -> ReleaseId:
    return ReleaseIdFactory.v1(
        (("pdf", "a" * 64), ("sidecar", "b" * 64)),
    )


def _golden_observation() -> ObservationId:
    return ObservationIdFactory.v1(
        _golden_fact(),
        _golden_release(),
        "circular-2417",
        TransformationIdentity("percent-to-decimal", "v1"),
        datetime(2026, 7, 7, tzinfo=UTC),
    )


def test_golden_fixture_has_exact_reviewable_shape() -> None:
    golden = _golden()
    expected_names = [
        "null",
        "bool_true",
        "int_one",
        "float_one",
        "float_negative_zero",
        "nfc_string",
        "date",
        "timestamp",
        "enum",
        "mapping",
        "list",
        "tuple",
        "set",
    ]
    assert set(golden) == {"schema_version", "canonical_vectors", "identity_vectors"}
    assert golden["schema_version"] == "v1"
    assert [vector["name"] for vector in golden["canonical_vectors"]] == expected_names
    assert all(
        set(vector) == {"name", "canonical_hex", "sha256"} for vector in golden["canonical_vectors"]
    )
    assert set(golden["identity_vectors"]) == {"fact", "release", "observation"}
    assert all(
        set(vector) == {"canonical_hex", "id"} for vector in golden["identity_vectors"].values()
    )


def test_all_thirteen_v1_golden_vectors_are_literal_and_exact() -> None:
    for vector in _golden()["canonical_vectors"]:
        actual = canonical_identity_bytes(identity_fixture_value(vector["name"]))
        assert actual.hex() == vector["canonical_hex"]
        assert hashlib.sha256(actual).hexdigest() == vector["sha256"]


def test_atoms_are_typed_and_float_zero_is_canonical() -> None:
    assert canonical_identity_bytes(1) != canonical_identity_bytes(1.0)
    assert canonical_identity_bytes(True) != canonical_identity_bytes(1)
    assert canonical_identity_bytes(-0.0) == canonical_identity_bytes(0.0)
    assert b'"t":"enum"' in canonical_identity_bytes(Example.VALUE)
    assert b"test_identity.Example" in canonical_identity_bytes(Example.VALUE)


def test_container_types_and_order_have_distinct_canonical_meaning() -> None:
    assert canonical_identity_bytes([1, "x"]) != canonical_identity_bytes((1, "x"))
    assert canonical_identity_bytes([1, 2]) != canonical_identity_bytes([2, 1])
    assert canonical_identity_bytes({1, 2}) == canonical_identity_bytes({2, 1})


def test_mappings_sort_normalized_keys_and_reject_nfc_collision() -> None:
    assert canonical_identity_bytes({"b": 2, "a": 1}) == canonical_identity_bytes({"a": 1, "b": 2})
    with pytest.raises(DataContractError, match="NFC"):
        canonical_identity_bytes({"é": 1, "e\u0301": 2})


def test_mapping_rejects_non_string_keys() -> None:
    with pytest.raises(DataContractError, match="string"):
        canonical_identity_bytes({1: "one"})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_identity_numbers_fail(value: float) -> None:
    with pytest.raises(DataContractError, match="finite"):
        canonical_identity_bytes(value)


@pytest.mark.parametrize("value", [b"bytes", Decimal("1.0"), frozenset({1})])
def test_unsupported_identity_atoms_fail_explicitly(value: object) -> None:
    with pytest.raises(DataContractError, match="Unsupported"):
        canonical_identity_bytes(value)


def test_naive_timestamp_fails_and_aware_time_normalizes_to_utc_ns() -> None:
    with pytest.raises(DataContractError, match="timezone"):
        canonical_identity_bytes(datetime(2026, 7, 1, 12))
    utc = canonical_identity_bytes(datetime(2026, 7, 1, 16, tzinfo=UTC))
    chile = canonical_identity_bytes(datetime(2026, 7, 1, 12, tzinfo=timezone(timedelta(hours=-4))))
    assert utc == chile
    assert b"2026-07-01T16:00:00.000000000Z" in utc
    assert b'"t":"date"' in canonical_identity_bytes(date(2026, 7, 1))


def test_timestamp_year_is_fixed_width_at_datetime_lower_bound() -> None:
    utc = canonical_identity_bytes(datetime(1, 1, 1, 0, 0, 0, 7, tzinfo=UTC))
    offset = canonical_identity_bytes(
        datetime(1, 1, 1, 4, 0, 0, 7, tzinfo=timezone(timedelta(hours=4)))
    )
    assert utc == offset
    assert b"0001-01-01T00:00:00.000007000Z" in utc


def test_fact_factory_is_pure_versioned_and_excludes_non_key_state() -> None:
    arguments = (
        ObservationDomain.REGULATORY_RATE,
        {"publisher": "SP", "rule_id": "2417"},
        {"variable": "programmed_withdrawal_technical_rate"},
    )
    first = FactIdFactory.v1(*arguments)
    second = FactIdFactory.v1(*arguments)
    assert first == second
    assert first is not second
    assert first.value.startswith("cldemopde:fact:v1:")
    assert first.value.count(":") == 3


def test_release_asset_order_is_irrelevant_but_membership_is_not() -> None:
    first = ReleaseIdFactory.v1(
        assets=(("pdf", "a" * 64), ("sidecar", "b" * 64)),
    )
    reverse = ReleaseIdFactory.v1(
        assets=(("sidecar", "b" * 64), ("pdf", "a" * 64)),
    )
    changed = ReleaseIdFactory.v1(
        assets=(("pdf", "a" * 64),),
    )
    assert first == reverse
    assert first != changed
    assert first.value.startswith("cldemopde:release:v1:")


@pytest.mark.parametrize(
    "assets, match",
    [
        ((), "nonempty"),
        ((("pdf", "a" * 64), ("pdf", "a" * 64)), "unique"),
        ((("pdf", "a" * 64), ("pdf", "b" * 64)), "unique"),
        ((("", "a" * 64),), "source key"),
        ((("PDF", "a" * 64),), "source key"),
        ((("pdf-with-dash", "a" * 64),), "source key"),
        ((("pdf", "A" * 64),), "lowercase SHA-256"),
        ((("pdf", "a" * 63),), "lowercase SHA-256"),
        ((("pdf", "g" * 64),), "lowercase SHA-256"),
    ],
)
def test_release_factory_rejects_invalid_assets(
    assets: tuple[tuple[str, str], ...],
    match: str,
) -> None:
    with pytest.raises(DataContractError, match=match):
        ReleaseIdFactory.v1(assets)


def test_observation_id_commits_to_release_availability() -> None:
    fact = _golden_fact()
    release = ReleaseIdFactory.v1(
        (("sp_circular_2417_pdf", "a" * 64),),
    )
    early = ObservationIdFactory.v1(
        fact,
        release,
        "circular-2417",
        TransformationIdentity("percent-to-decimal", "v1"),
        datetime(2026, 7, 7, tzinfo=UTC),
    )
    late = ObservationIdFactory.v1(
        fact,
        release,
        "circular-2417",
        TransformationIdentity("percent-to-decimal", "v1"),
        datetime(2026, 7, 8, tzinfo=UTC),
    )
    assert early != late
    assert early.value.startswith("cldemopde:observation:v1:")


def test_revision_keeps_fact_id_but_changes_release_and_observation_id() -> None:
    fact = FactIdFactory.v1(
        ObservationDomain.DEMOGRAPHIC,
        {"publisher": "DEIS", "series": "deaths"},
        {"variable": "deaths", "period_start": date(2023, 1, 1)},
    )
    same_fact_after_value_revision = FactIdFactory.v1(
        ObservationDomain.DEMOGRAPHIC,
        {"publisher": "DEIS", "series": "deaths"},
        {"variable": "deaths", "period_start": date(2023, 1, 1)},
    )
    first_release = ReleaseIdFactory.v1((("deis_deaths", "a" * 64),))
    second_release = ReleaseIdFactory.v1((("deis_deaths", "b" * 64),))
    first_observation = ObservationIdFactory.v1(
        fact,
        first_release,
        "preliminary",
        TransformationIdentity("event-count", "v1"),
        datetime(2024, 1, 1, tzinfo=UTC),
    )
    second_observation = ObservationIdFactory.v1(
        same_fact_after_value_revision,
        second_release,
        "final",
        TransformationIdentity("event-count", "v1"),
        datetime(2024, 7, 1, tzinfo=UTC),
    )
    assert fact == same_fact_after_value_revision
    assert first_release != second_release
    assert first_observation != second_observation


def test_complete_wrapper_payloads_and_ids_match_independent_literals() -> None:
    actual = {
        "fact": _golden_fact(),
        "release": _golden_release(),
        "observation": _golden_observation(),
    }
    for name, identity in actual.items():
        expected = _golden()["identity_vectors"][name]
        assert identity.canonical_payload.hex() == expected["canonical_hex"]
        assert identity.value == expected["id"]
        assert identity.value.endswith(hashlib.sha256(identity.canonical_payload).hexdigest())


def test_transformation_identity_requires_a_complete_nonempty_pair() -> None:
    assert TransformationIdentity(None, None) == TransformationIdentity(None, None)
    for identity, version in ((None, "v1"), ("event-count", None), ("", ""), ("x", "")):
        with pytest.raises(DataContractError, match="transformation"):
            TransformationIdentity(identity, version)


def test_observation_factory_rejects_empty_vintage_and_naive_availability() -> None:
    with pytest.raises(DataContractError, match="vintage"):
        ObservationIdFactory.v1(
            _golden_fact(),
            _golden_release(),
            "",
            TransformationIdentity(None, None),
            datetime(2026, 7, 7, tzinfo=UTC),
        )
    with pytest.raises(DataContractError, match="timezone"):
        ObservationIdFactory.v1(
            _golden_fact(),
            _golden_release(),
            "vintage",
            TransformationIdentity(None, None),
            datetime(2026, 7, 7),
        )


def test_identity_value_constructors_are_not_public() -> None:
    for identity_type in (FactId, ReleaseId, ObservationId):
        with pytest.raises(TypeError, match="factory"):
            identity_type("cldemopde:forged:v1:" + "0" * 64, b"forged")


def test_registry_snapshot_is_sorted_auditable_and_deeply_immutable() -> None:
    registry = IdentityRegistry()
    release = _golden_release()
    fact = _golden_fact()
    registry.record(release)
    registry.record(fact)
    snapshot = registry.snapshot()

    assert tuple(record.value for record in snapshot.records) == tuple(
        sorted((release.value, fact.value))
    )
    assert snapshot.get(fact.value) is not None
    assert snapshot.require(fact.value).canonical_payload == fact.canonical_payload
    assert snapshot.get("missing") is None
    with pytest.raises(DataContractError, match="missing"):
        snapshot.require("missing")
    with pytest.raises(FrozenInstanceError):
        snapshot.records = ()
    with pytest.raises(FrozenInstanceError):
        snapshot.records[0].value = "forged"

    registry.record(_golden_observation())
    assert len(snapshot.records) == 2


def test_direct_snapshot_construction_sorts_and_collapses_identical_duplicates() -> None:
    first = CanonicalIdentityRecord("cldemopde:fact:v1:" + "b" * 64, b"second")
    second = CanonicalIdentityRecord("cldemopde:fact:v1:" + "a" * 64, b"first")
    duplicate = CanonicalIdentityRecord(second.value, b"first")

    snapshot = IdentityRegistrySnapshot((first, second, duplicate))

    assert snapshot.records == (second, first)
    assert snapshot.require(second.value) == second
    with pytest.raises(FrozenInstanceError):
        snapshot.records[0].canonical_payload = b"forged"


def test_direct_snapshot_construction_rejects_conflicting_duplicate_ids() -> None:
    value = "cldemopde:fact:v1:" + "a" * 64
    with pytest.raises(IdentityCollisionError, match="collision"):
        IdentityRegistrySnapshot(
            (
                CanonicalIdentityRecord(value, b"first"),
                CanonicalIdentityRecord(value, b"second"),
            )
        )


def test_direct_snapshot_construction_rejects_invalid_record_container() -> None:
    with pytest.raises(DataContractError, match="tuple"):
        IdentityRegistrySnapshot([])  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="record"):
        IdentityRegistrySnapshot(("not-a-record",))  # type: ignore[arg-type]


def test_large_identity_snapshot_lookup_is_indexed() -> None:
    records = tuple(
        CanonicalIdentityRecord(f"benchmark:{index:08d}", b"payload") for index in range(8_000)
    )
    snapshot = IdentityRegistrySnapshot(records)

    started = perf_counter()
    for record in snapshot.records:
        assert snapshot.require(record.value) is not None
    elapsed = perf_counter() - started

    assert elapsed < 1.0, (
        f"8,000 identity lookups must use the snapshot index; observed {elapsed:.3f}s"
    )


def test_identity_snapshot_records_cannot_be_hostilely_rebound() -> None:
    first = CanonicalIdentityRecord("cldemopde:fact:v1:" + "a" * 64, b"first")
    second = CanonicalIdentityRecord("cldemopde:fact:v1:" + "b" * 64, b"second")
    snapshot = IdentityRegistrySnapshot((first, second))

    with pytest.raises((AttributeError, FrozenInstanceError)):
        object.__setattr__(snapshot, "records", (snapshot.records[0],))

    assert snapshot.require(second.value).canonical_payload == b"second"


@pytest.mark.parametrize("value_equal", [False, True])
def test_identity_snapshot_rejects_hostile_lookup_index_rebinding(
    value_equal: bool,
) -> None:
    first = CanonicalIdentityRecord("cldemopde:fact:v1:" + "a" * 64, b"first")
    second = CanonicalIdentityRecord("cldemopde:fact:v1:" + "b" * 64, b"second")
    snapshot = IdentityRegistrySnapshot((first, second))
    replacement = (
        MappingProxyType({record.value: record for record in snapshot.records})
        if value_equal
        else MappingProxyType({first.value: first})
    )
    object.__setattr__(snapshot, "_by_value", replacement)

    with pytest.raises(DataContractError, match=r"snapshot.*changed"):
        snapshot.require(first.value)


def test_identity_snapshot_revalidates_hostile_record_mutation() -> None:
    record = CanonicalIdentityRecord(
        "cldemopde:fact:v1:" + "a" * 64,
        b"first",
    )
    snapshot = IdentityRegistrySnapshot((record,))
    stored = snapshot.records[0]
    object.__setattr__(stored, "canonical_payload", b"changed")

    with pytest.raises(DataContractError, match=r"snapshot.*changed"):
        snapshot.require(stored.value)


def test_collision_guard_is_idempotent_but_never_overwrites_existing_bytes() -> None:
    registry = IdentityRegistry()
    first = CollisionFixtureIdentity(
        value="cldemopde:fact:v1:" + "a" * 64,
        canonical_payload=b"first",
    )
    same = CollisionFixtureIdentity(
        value=first.value,
        canonical_payload=b"first",
    )
    collision = CollisionFixtureIdentity(
        value=first.value,
        canonical_payload=b"second",
    )
    registry.record(first)
    registry.record(same)
    with pytest.raises(IdentityCollisionError):
        registry.record(collision)
    assert registry.snapshot().require(first.value).canonical_payload == b"first"
