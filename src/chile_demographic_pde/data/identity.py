from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from types import MappingProxyType
from typing import Final, Protocol
from weakref import ReferenceType, ref

from chile_demographic_pde.core.errors import (
    DataContractError,
    IdentityCollisionError,
)
from chile_demographic_pde.data.roles import ObservationDomain

type CanonicalJson = None | bool | str | list[CanonicalJson] | dict[str, CanonicalJson]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")


def _json_bytes(value: CanonicalJson) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _normalized_string(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _canonical_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataContractError("Canonical identity timestamps require a timezone.")
    utc_value = value.astimezone(UTC)
    return (
        f"{utc_value.year:04d}-{utc_value.month:02d}-{utc_value.day:02d}"
        f"T{utc_value.hour:02d}:{utc_value.minute:02d}:{utc_value.second:02d}"
        f".{utc_value.microsecond:06d}000Z"
    )


def _canonical_mapping(value: Mapping[object, object]) -> CanonicalJson:
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise DataContractError("Canonical identity mapping keys must be strings.")
        normalized_key = _normalized_string(key)
        if normalized_key in normalized:
            raise DataContractError(
                "Canonical identity mapping keys collide after NFC normalization."
            )
        normalized[normalized_key] = item

    encoded_pairs: list[tuple[bytes, list[CanonicalJson]]] = []
    for key, item in normalized.items():
        encoded_key = _canonical_value(key)
        encoded_pairs.append(
            (
                _json_bytes(encoded_key),
                [encoded_key, _canonical_value(item)],
            )
        )
    encoded_pairs.sort(key=lambda pair: pair[0])
    return {"t": "mapping", "v": [pair for _, pair in encoded_pairs]}


def _canonical_value(value: object) -> CanonicalJson:
    if value is None:
        return {"t": "null"}
    if isinstance(value, bool):
        return {"t": "bool", "v": value}
    if isinstance(value, Enum):
        if not isinstance(value.value, str):
            raise DataContractError("Canonical identity enum values must be strings.")
        enum_name = f"{type(value).__module__}.{type(value).__qualname__}"
        return {
            "n": enum_name,
            "t": "enum",
            "v": _normalized_string(value.value),
        }
    if isinstance(value, int):
        return {"t": "int", "v": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataContractError("Canonical identity floats must be finite.")
        normalized = 0.0 if value == 0.0 else value
        return {"t": "float64", "v": struct.pack(">d", normalized).hex()}
    if isinstance(value, datetime):
        return {"t": "timestamp_utc_ns", "v": _canonical_timestamp(value)}
    if isinstance(value, date):
        return {"t": "date", "v": value.isoformat()}
    if isinstance(value, str):
        return {"t": "str", "v": _normalized_string(value)}
    if isinstance(value, Mapping):
        return _canonical_mapping(value)
    if isinstance(value, list):
        return {"t": "list", "v": [_canonical_value(item) for item in value]}
    if isinstance(value, tuple):
        return {"t": "tuple", "v": [_canonical_value(item) for item in value]}
    if isinstance(value, set):
        encoded = [_canonical_value(item) for item in value]
        encoded.sort(key=_json_bytes)
        return {"t": "set", "v": encoded}
    raise DataContractError(
        f"Unsupported canonical identity value type: {type(value).__qualname__}."
    )


def canonical_identity_bytes(value: object) -> bytes:
    return _json_bytes(_canonical_value(value))


def _identity_wrapper(namespace: str, payload: object) -> bytes:
    return _json_bytes(
        {
            "namespace": namespace,
            "payload": _canonical_value(payload),
            "schema_version": "v1",
        }
    )


class CanonicalIdentity(Protocol):
    @property
    def value(self) -> str: ...

    @property
    def canonical_payload(self) -> bytes: ...


@dataclass(frozen=True, slots=True)
class CanonicalIdentityRecord:
    value: str
    canonical_payload: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise DataContractError("Canonical identity records require a nonempty ID.")
        if not isinstance(self.canonical_payload, bytes):
            raise DataContractError("Canonical identity payloads must be immutable bytes.")


@dataclass(frozen=True, init=False)
class IdentityRegistrySnapshot:
    _by_value: Mapping[str, CanonicalIdentityRecord] = field(
        repr=False,
    )

    def __init__(self, records: tuple[CanonicalIdentityRecord, ...]) -> None:
        if type(records) is not tuple:
            raise DataContractError("Identity snapshot records must be a tuple.")
        by_value: dict[str, CanonicalIdentityRecord] = {}
        for record in records:
            if not isinstance(record, CanonicalIdentityRecord):
                raise DataContractError(
                    "Identity snapshot entries must be canonical identity records."
                )
            immutable_record = CanonicalIdentityRecord(
                record.value,
                bytes(record.canonical_payload),
            )
            existing = by_value.get(immutable_record.value)
            if (
                existing is not None
                and existing.canonical_payload != immutable_record.canonical_payload
            ):
                raise IdentityCollisionError(
                    f"Canonical identity collision in snapshot for {immutable_record.value}."
                )
            if existing is None:
                by_value[immutable_record.value] = immutable_record
        object.__setattr__(
            self,
            "_by_value",
            MappingProxyType({value: by_value[value] for value in sorted(by_value)}),
        )
        _seal_identity_snapshot(self)

    @property
    def records(self) -> tuple[CanonicalIdentityRecord, ...]:
        """Return the sorted, immutable records from the sealed lookup source."""

        index, sealed_records = self._require_unchanged()
        return tuple(
            _revalidate_snapshot_record(
                value,
                record,
                sealed_records[value],
            )
            for value, record in index.items()
        )

    def get(self, value: str) -> CanonicalIdentityRecord | None:
        index, sealed_records = self._require_unchanged()
        record = index.get(value)
        if record is None:
            return None
        return _revalidate_snapshot_record(
            value,
            record,
            sealed_records[value],
        )

    def require(self, value: str) -> CanonicalIdentityRecord:
        record = self.get(value)
        if record is None:
            raise DataContractError(f"Canonical identity is missing from snapshot: {value}")
        return record

    def _require_unchanged(
        self,
    ) -> tuple[
        Mapping[str, CanonicalIdentityRecord],
        Mapping[str, tuple[int, str, bytes]],
    ]:
        try:
            (
                reference,
                sealed_index_id,
                sealed_records,
            ) = _IDENTITY_SNAPSHOT_SEALS[id(self)]
            index = self._by_value
            if (
                reference() is not self
                or id(index) != sealed_index_id
                or type(index) is not MappingProxyType
            ):
                raise KeyError("changed snapshot")
        except Exception:
            raise DataContractError("Canonical identity snapshot changed after creation.") from None
        return index, sealed_records


_IDENTITY_SNAPSHOT_SEALS: Final[
    dict[
        int,
        tuple[
            ReferenceType[IdentityRegistrySnapshot],
            int,
            Mapping[str, tuple[int, str, bytes]],
        ],
    ]
] = {}


def _seal_identity_snapshot(snapshot: IdentityRegistrySnapshot) -> None:
    snapshot_id = id(snapshot)

    def remove_seal(
        reference: ReferenceType[IdentityRegistrySnapshot],
        *,
        sealed_snapshot_id: int = snapshot_id,
    ) -> None:
        current = _IDENTITY_SNAPSHOT_SEALS.get(sealed_snapshot_id)
        if current is not None and current[0] is reference:
            _IDENTITY_SNAPSHOT_SEALS.pop(sealed_snapshot_id, None)

    reference = ref(snapshot, remove_seal)
    _IDENTITY_SNAPSHOT_SEALS[snapshot_id] = (
        reference,
        id(snapshot._by_value),
        MappingProxyType(
            {
                key: (
                    id(record),
                    record.value,
                    record.canonical_payload,
                )
                for key, record in snapshot._by_value.items()
            }
        ),
    )


def _revalidate_snapshot_record(
    key: str,
    record: CanonicalIdentityRecord,
    sealed: tuple[int, str, bytes],
) -> CanonicalIdentityRecord:
    try:
        sealed_id, sealed_value, sealed_payload = sealed
        if type(record) is not CanonicalIdentityRecord:
            raise DataContractError("invalid identity record type")
        rebuilt = CanonicalIdentityRecord(
            value=record.value,
            canonical_payload=bytes(record.canonical_payload),
        )
        if (
            id(record) != sealed_id
            or key != sealed_value
            or rebuilt.value != sealed_value
            or rebuilt.canonical_payload != sealed_payload
        ):
            raise DataContractError("changed identity record")
    except Exception:
        raise DataContractError("Canonical identity snapshot changed after creation.") from None
    return record


class IdentityRegistry:
    def __init__(self) -> None:
        self._records: dict[str, CanonicalIdentityRecord] = {}

    def record(self, identity: CanonicalIdentity) -> None:
        record = CanonicalIdentityRecord(
            value=identity.value,
            canonical_payload=identity.canonical_payload,
        )
        existing = self._records.get(record.value)
        if existing is not None and existing.canonical_payload != record.canonical_payload:
            raise IdentityCollisionError(
                f"Canonical identity collision for {record.value}; existing bytes retained."
            )
        if existing is None:
            self._records[record.value] = record

    def snapshot(self) -> IdentityRegistrySnapshot:
        return IdentityRegistrySnapshot(
            tuple(self._records[value] for value in sorted(self._records))
        )


@dataclass(frozen=True, slots=True)
class TransformationIdentity:
    transformation_id: str | None
    transformation_version: str | None

    def __post_init__(self) -> None:
        fields = (self.transformation_id, self.transformation_version)
        if fields == (None, None):
            return
        if not all(isinstance(field, str) and bool(field.strip()) for field in fields):
            raise DataContractError(
                "A transformation requires both a nonempty transformation_id "
                "and transformation_version."
            )


@dataclass(frozen=True, slots=True, init=False)
class FactId:
    value: str
    canonical_payload: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("FactId must be created by FactIdFactory.v1 factory.")


@dataclass(frozen=True, slots=True, init=False)
class ReleaseId:
    value: str
    canonical_payload: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("ReleaseId must be created by ReleaseIdFactory.v1 factory.")


@dataclass(frozen=True, slots=True, init=False)
class ObservationId:
    value: str
    canonical_payload: bytes

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("ObservationId must be created by ObservationIdFactory.v1 factory.")


def _new_identity[IdentityValue: (FactId, ReleaseId, ObservationId)](
    identity_type: type[IdentityValue],
    prefix: str,
    canonical_payload: bytes,
) -> IdentityValue:
    identity = object.__new__(identity_type)
    object.__setattr__(
        identity,
        "value",
        prefix + hashlib.sha256(canonical_payload).hexdigest(),
    )
    object.__setattr__(identity, "canonical_payload", canonical_payload)
    return identity


class FactIdFactory:
    @classmethod
    def v1(
        cls,
        domain: ObservationDomain,
        source_identity: Mapping[str, object],
        key_fields: Mapping[str, object],
    ) -> FactId:
        if not isinstance(domain, ObservationDomain):
            raise DataContractError("Fact identity domain must be an ObservationDomain.")
        if not isinstance(source_identity, Mapping) or not source_identity:
            raise DataContractError("Fact source_identity must be a nonempty mapping.")
        if not isinstance(key_fields, Mapping) or not key_fields:
            raise DataContractError("Fact key_fields must be a nonempty mapping.")
        payload = _identity_wrapper(
            "cldemopde:fact",
            {
                "domain": domain,
                "key_fields": key_fields,
                "source_identity": source_identity,
            },
        )
        return _new_identity(FactId, "cldemopde:fact:v1:", payload)


class ReleaseIdFactory:
    @classmethod
    def v1(
        cls,
        assets: Sequence[tuple[str, str]],
    ) -> ReleaseId:
        if isinstance(assets, (str, bytes)) or not isinstance(assets, Sequence):
            raise DataContractError("Release assets must be a nonempty sequence of pairs.")
        if not assets:
            raise DataContractError("Release assets must be nonempty.")

        normalized: list[tuple[str, str]] = []
        source_keys: set[str] = set()
        for asset in assets:
            if not isinstance(asset, tuple) or len(asset) != 2:
                raise DataContractError("Each release asset must be a (source key, SHA-256) pair.")
            source_key, sha256 = asset
            if not isinstance(source_key, str) or _SOURCE_KEY_PATTERN.fullmatch(source_key) is None:
                raise DataContractError(
                    "Release asset source key must use lowercase snake-case characters."
                )
            if source_key in source_keys:
                raise DataContractError("Release asset source keys must be unique.")
            if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
                raise DataContractError(
                    "Release asset hashes must be lowercase SHA-256 hex strings."
                )
            source_keys.add(source_key)
            normalized.append((source_key, sha256))

        normalized.sort(key=lambda item: item[0])
        payload = _identity_wrapper(
            "cldemopde:release",
            {
                "assets": [
                    {"sha256": sha256, "source_key": source_key}
                    for source_key, sha256 in normalized
                ]
            },
        )
        return _new_identity(ReleaseId, "cldemopde:release:v1:", payload)


class ObservationIdFactory:
    @classmethod
    def v1(
        cls,
        fact_id: FactId,
        release_id: ReleaseId,
        vintage: str,
        transformation: TransformationIdentity,
        available_at: datetime,
    ) -> ObservationId:
        if not isinstance(fact_id, FactId):
            raise DataContractError("Observation identity requires a FactId.")
        if not isinstance(release_id, ReleaseId):
            raise DataContractError("Observation identity requires a ReleaseId.")
        if not isinstance(vintage, str) or not vintage.strip():
            raise DataContractError("Observation identity vintage must be nonempty.")
        if not isinstance(transformation, TransformationIdentity):
            raise DataContractError("Observation identity requires a TransformationIdentity.")
        if not isinstance(available_at, datetime):
            raise DataContractError("Observation availability must be a datetime.")

        payload = _identity_wrapper(
            "cldemopde:observation",
            {
                "available_at": available_at,
                "fact_id": fact_id.value,
                "release_id": release_id.value,
                "transformation": {
                    "transformation_id": transformation.transformation_id,
                    "transformation_version": transformation.transformation_version,
                },
                "vintage": vintage,
            },
        )
        return _new_identity(
            ObservationId,
            "cldemopde:observation:v1:",
            payload,
        )
