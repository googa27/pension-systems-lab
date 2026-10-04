"""Immutable authorizations for reviewed Chilean official-finance releases.

An official URL or caller-created dataclass is not an authorization.  The
registry commits to the complete asset set, publisher-date metadata, primary
fact source, release schema, and parser contract selected during independent
review.  Authorization happens before any verified descriptor is opened.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Final
from weakref import WeakKeyDictionary

from chile_demographic_pde.core.errors import (
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.data.identity import (
    IdentityRegistry,
    ReleaseId,
    canonical_identity_bytes,
)
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
    VerifiedReleaseAssets,
)
from chile_demographic_pde.data.roles import ReleaseMissingnessReason

_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_CANONICAL_KEY: Final = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_REGISTRY_VERSION: Final = "chile-reviewed-official-finance-v1"
_ENTRY_SCHEMA_VERSION: Final = "reviewed-official-release-entry-v1"
_REGISTRY_SEALS: Final[
    WeakKeyDictionary[
        ReviewedOfficialReleaseRegistry,
        tuple[int, int, tuple[tuple[str, int], ...], str],
    ]
] = WeakKeyDictionary()


def _canonical_text(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or unicodedata.normalize("NFC", value) != value
        or any(unicodedata.category(character).startswith("C") for character in value)
    ):
        raise DataContractError(
            f"Reviewed official release {field} must be canonical nonblank text."
        )
    return value


def _canonical_key(value: object, *, field: str) -> str:
    text = _canonical_text(value, field=field)
    if _CANONICAL_KEY.fullmatch(text) is None:
        raise DataContractError(f"Reviewed official release {field} has unsupported characters.")
    return text


def _sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise DataContractError(f"Reviewed official release {field} must be a lowercase SHA-256.")
    return value


@dataclass(frozen=True, slots=True)
class ReviewedOfficialAsset:
    """One exact reviewed member and its publisher-date interpretation."""

    source_key: str
    sha256: str
    released_at: date | None
    release_missingness_reason: ReleaseMissingnessReason | None

    def __post_init__(self) -> None:
        _canonical_key(self.source_key, field="asset source_key")
        _sha256(self.sha256, field="asset sha256")
        if self.released_at is not None and type(self.released_at) is not date:
            raise DataContractError(
                "Reviewed official asset released_at must be a date without a time."
            )
        if (
            self.release_missingness_reason is not None
            and type(self.release_missingness_reason) is not ReleaseMissingnessReason
        ):
            raise DataContractError("Reviewed official asset missingness reason must be typed.")
        if (self.released_at is None) == (self.release_missingness_reason is None):
            raise DataContractError(
                "Reviewed official asset requires exactly one publisher date "
                "or typed missingness reason."
            )


@dataclass(frozen=True, slots=True)
class ReviewedOfficialReleaseEntry:
    """Reviewer-authored authorization for one exact composite release."""

    registry_version: str
    registry_key: str
    release_schema_version: str
    primary_fact_source_key: str
    assets: tuple[ReviewedOfficialAsset, ...]
    contract_id: str
    review_id: str

    def __post_init__(self) -> None:
        _canonical_key(self.registry_version, field="registry_version")
        _canonical_key(self.registry_key, field="registry_key")
        _canonical_key(
            self.release_schema_version,
            field="release_schema_version",
        )
        _canonical_key(
            self.primary_fact_source_key,
            field="primary_fact_source_key",
        )
        _canonical_key(self.contract_id, field="contract_id")
        _canonical_text(self.review_id, field="review_id")
        if type(self.assets) is not tuple or not self.assets:
            raise DataContractError("Reviewed official release assets must be a nonempty tuple.")
        if any(type(asset) is not ReviewedOfficialAsset for asset in self.assets):
            raise DataContractError("Reviewed official release contains an invalid asset entry.")
        source_keys = tuple(asset.source_key for asset in self.assets)
        if len(source_keys) != len(set(source_keys)):
            raise DataContractError("Reviewed official release asset source keys must be unique.")
        if source_keys.count(self.primary_fact_source_key) != 1:
            raise DataContractError("Reviewed official release primary source must name one asset.")
        if source_keys != tuple(sorted(source_keys)):
            raise DataContractError(
                "Reviewed official release assets must use canonical key order."
            )


def _rebuild_asset(asset: ReviewedOfficialAsset) -> ReviewedOfficialAsset:
    if type(asset) is not ReviewedOfficialAsset:
        raise DataContractError("Reviewed official release asset type is invalid.")
    return ReviewedOfficialAsset(
        source_key=asset.source_key,
        sha256=asset.sha256,
        released_at=asset.released_at,
        release_missingness_reason=asset.release_missingness_reason,
    )


def _rebuild_entry(
    entry: ReviewedOfficialReleaseEntry,
) -> ReviewedOfficialReleaseEntry:
    if type(entry) is not ReviewedOfficialReleaseEntry:
        raise DataContractError("Reviewed official release entry type is invalid.")
    try:
        assets = tuple(_rebuild_asset(asset) for asset in entry.assets)
    except DataContractError:
        raise
    except Exception:
        raise DataContractError(
            "Reviewed official release assets could not be revalidated."
        ) from None
    return ReviewedOfficialReleaseEntry(
        registry_version=entry.registry_version,
        registry_key=entry.registry_key,
        release_schema_version=entry.release_schema_version,
        primary_fact_source_key=entry.primary_fact_source_key,
        assets=assets,
        contract_id=entry.contract_id,
        review_id=entry.review_id,
    )


def _entry_fingerprint(entry: ReviewedOfficialReleaseEntry) -> str:
    rebuilt = _rebuild_entry(entry)
    payload = {
        "schema_version": _ENTRY_SCHEMA_VERSION,
        "registry_version": rebuilt.registry_version,
        "registry_key": rebuilt.registry_key,
        "release_schema_version": rebuilt.release_schema_version,
        "primary_fact_source_key": rebuilt.primary_fact_source_key,
        "assets": tuple(
            {
                "source_key": asset.source_key,
                "sha256": asset.sha256,
                "released_at": asset.released_at,
                "release_missingness_reason": asset.release_missingness_reason,
            }
            for asset in rebuilt.assets
        ),
        "contract_id": rebuilt.contract_id,
        "review_id": rebuilt.review_id,
    }
    return hashlib.sha256(canonical_identity_bytes(payload)).hexdigest()


def _registry_fingerprint(
    registry_version: str,
    fingerprints: Mapping[str, str],
) -> str:
    return hashlib.sha256(
        canonical_identity_bytes(
            {
                "schema_version": "reviewed-official-release-registry-v1",
                "registry_version": registry_version,
                "entries": tuple(sorted(fingerprints.items())),
            }
        )
    ).hexdigest()


@dataclass(
    frozen=True,
    slots=True,
    init=False,
    eq=False,
    weakref_slot=True,
)
class ReviewedOfficialReleaseRegistry:
    """Immutable exact-release lookup with create-time tamper fingerprints."""

    registry_version: str
    entries: Mapping[str, ReviewedOfficialReleaseEntry]
    _fingerprints: Mapping[str, str]

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("ReviewedOfficialReleaseRegistry must be created by create().")

    @classmethod
    def create(
        cls,
        *,
        registry_version: str,
        entries: Sequence[ReviewedOfficialReleaseEntry],
    ) -> ReviewedOfficialReleaseRegistry:
        version = _canonical_key(registry_version, field="registry_version")
        if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
            raise DataContractError("Reviewed official registry entries must be a sequence.")
        try:
            snapshot = tuple(entries)
        except Exception:
            raise DataContractError(
                "Reviewed official registry entries could not be snapshotted."
            ) from None
        by_key: dict[str, ReviewedOfficialReleaseEntry] = {}
        fingerprints: dict[str, str] = {}
        for entry in snapshot:
            rebuilt = _rebuild_entry(entry)
            if rebuilt.registry_version != version:
                raise DataContractError("Reviewed official entry registry version does not match.")
            if rebuilt.registry_key in by_key:
                raise DataContractError("Reviewed official registry keys must be unique.")
            by_key[rebuilt.registry_key] = entry
            fingerprints[rebuilt.registry_key] = _entry_fingerprint(rebuilt)
        registry = object.__new__(cls)
        object.__setattr__(registry, "registry_version", version)
        object.__setattr__(
            registry,
            "entries",
            MappingProxyType(dict(sorted(by_key.items()))),
        )
        object.__setattr__(
            registry,
            "_fingerprints",
            MappingProxyType(dict(sorted(fingerprints.items()))),
        )
        _REGISTRY_SEALS[registry] = (
            id(registry.entries),
            id(registry._fingerprints),
            tuple((key, id(entry)) for key, entry in registry.entries.items()),
            _registry_fingerprint(version, fingerprints),
        )
        return registry

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, registry_key: object) -> bool:
        return type(registry_key) is str and registry_key in self.entries

    def __iter__(self) -> Iterator[ReviewedOfficialReleaseEntry]:
        return iter(self.entries.values())

    def _require_unchanged(self, registry_key: str) -> ReviewedOfficialReleaseEntry:
        try:
            (
                sealed_entries_id,
                sealed_fingerprints_id,
                sealed_entry_ids,
                sealed_fingerprint,
            ) = _REGISTRY_SEALS[self]
            registry_version = self.registry_version
            entries = self.entries
            fingerprints = self._fingerprints
            if (
                type(registry_version) is not str
                or id(entries) != sealed_entries_id
                or id(fingerprints) != sealed_fingerprints_id
            ):
                raise KeyError("changed registry")
            current_fingerprint = _registry_fingerprint(
                registry_version,
                fingerprints,
            )
            current_entry_ids = tuple((key, id(entry)) for key, entry in entries.items())
            if current_entry_ids != sealed_entry_ids or current_fingerprint != sealed_fingerprint:
                raise KeyError("changed registry")
            entry = _rebuild_entry(entries[registry_key])
            expected = fingerprints[registry_key]
            observed = _entry_fingerprint(entry)
        except Exception:
            raise MissingOfficialDataError(
                "The requested official finance release is not reviewed."
            ) from None
        if observed != expected:
            raise MissingOfficialDataError(
                "The reviewed official finance authorization has changed."
            )
        return entry


def _release_manifest(value: object) -> ReleaseManifest:
    if type(value) is ReleaseManifest:
        manifest = value
    else:
        assets = getattr(value, "assets", None)
        if type(assets) is not VerifiedReleaseAssets:
            raise MissingOfficialDataError("A typed verified official finance release is required.")
        manifest = assets.manifest
    if type(manifest) is not ReleaseManifest:
        raise MissingOfficialDataError("A typed verified official finance release is required.")

    try:
        if (
            type(manifest.schema_version) is not str
            or type(manifest.primary_fact_source_key) is not str
            or type(manifest.available_at) is not datetime
            or type(manifest.release_id) is not ReleaseId
            or type(manifest.release_id.value) is not str
            or type(manifest.release_id.canonical_payload) is not bytes
            or type(manifest.assets) is not tuple
            or any(
                type(asset) is not RequiredReleaseAsset
                or type(asset.source_key) is not str
                or type(asset.sha256) is not str
                or type(asset.available_at) is not datetime
                or (asset.released_at is not None and type(asset.released_at) is not date)
                or (
                    asset.release_missingness_reason is not None
                    and type(asset.release_missingness_reason) is not ReleaseMissingnessReason
                )
                for asset in manifest.assets
            )
        ):
            raise DataContractError("invalid manifest asset types")
        rebuilt_assets = tuple(
            RequiredReleaseAsset(
                source_key=asset.source_key,
                sha256=asset.sha256,
                available_at=asset.available_at,
                released_at=asset.released_at,
                release_missingness_reason=asset.release_missingness_reason,
            )
            for asset in manifest.assets
        )
        rebuilt = ReleaseManifest.create(
            schema_version=manifest.schema_version,
            assets=rebuilt_assets,
            primary_fact_source_key=manifest.primary_fact_source_key,
            identity_registry=IdentityRegistry(),
        )
        unchanged = (
            manifest.assets == rebuilt.assets
            and manifest.release_id == rebuilt.release_id
            and manifest.release_id.canonical_payload == rebuilt.release_id.canonical_payload
            and manifest.available_at == rebuilt.available_at
        )
    except (AttributeError, DataContractError, TypeError, ValueError, OverflowError):
        unchanged = False
    if not unchanged:
        raise MissingOfficialDataError(
            "The verified official finance release manifest has changed."
        )
    return rebuilt


def require_reviewed_official_release(
    release: object,
    registry_key: str,
    registry: ReviewedOfficialReleaseRegistry | None = None,
) -> ReviewedOfficialReleaseEntry:
    """Authorize an exact manifest without opening any release member."""

    selected = PRODUCTION_OFFICIAL_RELEASE_REGISTRY_V1 if registry is None else registry
    if type(selected) is not ReviewedOfficialReleaseRegistry:
        raise MissingOfficialDataError(
            "A typed immutable reviewed official-finance registry is required."
        )
    if type(registry_key) is not str:
        raise MissingOfficialDataError("The requested official finance release is not reviewed.")
    entry = selected._require_unchanged(registry_key)
    manifest = _release_manifest(release)
    actual = tuple(
        (
            asset.source_key,
            asset.sha256,
            asset.released_at,
            asset.release_missingness_reason,
        )
        for asset in manifest.assets
    )
    expected = tuple(
        (
            asset.source_key,
            asset.sha256,
            asset.released_at,
            asset.release_missingness_reason,
        )
        for asset in entry.assets
    )
    if (
        manifest.schema_version != entry.release_schema_version
        or manifest.primary_fact_source_key != entry.primary_fact_source_key
        or actual != expected
    ):
        raise MissingOfficialDataError(
            "The supplied official finance release does not match its reviewed authorization."
        )
    return entry


_TM2020_HISTORICAL_ENTRY: Final = ReviewedOfficialReleaseEntry(
    registry_version=_REGISTRY_VERSION,
    registry_key="cmf_tm2020_historical",
    release_schema_version="cmf-tm2020-historical-xlsx-v1",
    primary_fact_source_key="cmf_tm2020_historical_xlsx",
    assets=(
        ReviewedOfficialAsset(
            source_key="cmf_tm2020_historical_xlsx",
            sha256=("5c8a01aa2dd8b356038d33894b77bc765e1fa2281d53c8e336c750a74feab747"),
            released_at=None,
            release_missingness_reason=(ReleaseMissingnessReason.PUBLISHER_DATE_NOT_AVAILABLE),
        ),
    ),
    contract_id="cmf-tm2020-historical-workbook-v1",
    review_id="cmf-tm2020-historical-independent-review-2026-07-19",
)

PRODUCTION_OFFICIAL_RELEASE_REGISTRY_V1: Final = ReviewedOfficialReleaseRegistry.create(
    registry_version=_REGISTRY_VERSION,
    entries=(_TM2020_HISTORICAL_ENTRY,),
)
