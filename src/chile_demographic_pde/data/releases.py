"""Immutable composite releases and descriptor-safe access to official bytes.

The verified-byte boundary assumes a content-addressed cache whose files are
immutable and have a single writer. Descriptor verification prevents pathname
replacement races, but cannot freeze an inode against an in-place write that
starts after its checksum has been computed.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import BinaryIO

from chile_demographic_pde.core.errors import (
    ChecksumMismatchError,
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.data.acquisition import AcquiredAsset, sha256_stream
from chile_demographic_pde.data.identity import (
    IdentityRegistry,
    ReleaseId,
    ReleaseIdFactory,
)
from chile_demographic_pde.data.roles import ReleaseMissingnessReason

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_KEY_PATTERN = re.compile(r"[a-z0-9][a-z0-9_]*")
_O_NOFOLLOW: int | None = getattr(os, "O_NOFOLLOW", None)


@dataclass(frozen=True, slots=True)
class RequiredReleaseAsset:
    """One independently dated, checksum-pinned member of an official release."""

    source_key: str
    sha256: str
    available_at: datetime
    released_at: date | None
    release_missingness_reason: ReleaseMissingnessReason | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_key, str)
            or _SOURCE_KEY_PATTERN.fullmatch(self.source_key) is None
        ):
            raise DataContractError(
                "Required release asset source key must use lowercase snake-case characters."
            )
        if not isinstance(self.sha256, str) or _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise DataContractError(
                "Required release asset SHA-256 must be a lowercase 64-character digest."
            )
        if (
            not isinstance(self.available_at, datetime)
            or self.available_at.tzinfo is None
            or self.available_at.utcoffset() is None
        ):
            raise DataContractError(
                "Required release asset available_at must be a timezone-aware datetime."
            )
        if self.released_at is not None and (
            not isinstance(self.released_at, date) or isinstance(self.released_at, datetime)
        ):
            raise DataContractError("Required release asset released_at must be a date or None.")
        if self.release_missingness_reason is not None and not isinstance(
            self.release_missingness_reason,
            ReleaseMissingnessReason,
        ):
            raise DataContractError("Required release asset missingness reason must be typed.")
        if (self.released_at is None) == (self.release_missingness_reason is None):
            raise DataContractError(
                "Required release asset requires exactly one of released_at "
                "or release_missingness_reason."
            )
        object.__setattr__(self, "available_at", self.available_at.astimezone(UTC))


@dataclass(frozen=True, slots=True, init=False)
class ReleaseManifest:
    """A factory-created, content-addressed composite official release."""

    schema_version: str
    assets: tuple[RequiredReleaseAsset, ...]
    primary_fact_source_key: str
    release_id: ReleaseId
    available_at: datetime

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("ReleaseManifest must be created by ReleaseManifest.create().")

    @classmethod
    def create(
        cls,
        *,
        schema_version: str,
        assets: tuple[RequiredReleaseAsset, ...],
        primary_fact_source_key: str,
        identity_registry: IdentityRegistry,
    ) -> ReleaseManifest:
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise DataContractError("Release manifest schema_version must be a nonempty string.")
        if not isinstance(assets, tuple):
            raise DataContractError("Release manifest assets must be an immutable tuple.")
        if not assets:
            raise DataContractError("Release manifest assets must be nonempty.")
        if not all(isinstance(asset, RequiredReleaseAsset) for asset in assets):
            raise DataContractError("Release manifest entries must be RequiredReleaseAsset values.")
        source_keys = tuple(asset.source_key for asset in assets)
        if len(set(source_keys)) != len(source_keys):
            raise DataContractError(
                "Release manifest source keys must be unique; duplicate source key found."
            )
        if (
            not isinstance(primary_fact_source_key, str)
            or source_keys.count(primary_fact_source_key) != 1
        ):
            raise DataContractError(
                "Release manifest primary_fact_source_key must name exactly one asset."
            )
        if not isinstance(identity_registry, IdentityRegistry):
            raise DataContractError(
                "Release manifest identity_registry must be an IdentityRegistry."
            )

        normalized_assets = tuple(sorted(assets, key=lambda asset: asset.source_key))
        release_id = ReleaseIdFactory.v1(
            tuple((asset.source_key, asset.sha256) for asset in normalized_assets)
        )
        identity_registry.record(release_id)

        manifest = object.__new__(cls)
        object.__setattr__(manifest, "schema_version", schema_version)
        object.__setattr__(manifest, "assets", normalized_assets)
        object.__setattr__(
            manifest,
            "primary_fact_source_key",
            primary_fact_source_key,
        )
        object.__setattr__(manifest, "release_id", release_id)
        object.__setattr__(
            manifest,
            "available_at",
            max(asset.available_at for asset in normalized_assets),
        )
        return manifest

    @property
    def primary_fact_asset(self) -> RequiredReleaseAsset:
        return self.asset(self.primary_fact_source_key)

    def asset(self, source_key: str) -> RequiredReleaseAsset:
        for asset in self.assets:
            if asset.source_key == source_key:
                return asset
        raise DataContractError("unknown release asset source key.")


class VerifiedReleaseAssets:
    """Opaque verified paths whose bytes are rechecked at each parser boundary.

    Backing cache entries must remain immutable and single-writer while an
    ``open()`` context is active. The parser receives the same descriptor that
    was hashed; this prevents path swaps, not later writes to that same inode.
    """

    __slots__ = ("_manifest", "_paths")

    _manifest: ReleaseManifest
    _paths: Mapping[str, Path]

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("VerifiedReleaseAssets must be created by verify_release_assets().")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("VerifiedReleaseAssets is immutable.")

    @property
    def manifest(self) -> ReleaseManifest:
        return self._manifest

    @property
    def release_id(self) -> ReleaseId:
        return self._manifest.release_id

    @property
    def available_at(self) -> datetime:
        return self._manifest.available_at

    def open(self, source_key: str) -> AbstractContextManager[BinaryIO]:
        """Recheck and expose one descriptor under the immutable-cache contract."""

        if not isinstance(source_key, str) or source_key not in self._paths:
            raise DataContractError("unknown verified release asset source key.")
        asset = self._manifest.asset(source_key)
        return _open_verified_asset(
            source_key=source_key,
            path=self._paths[source_key],
            expected_sha256=asset.sha256,
        )


def verify_release_assets(
    manifest: ReleaseManifest,
    acquired_assets: Sequence[AcquiredAsset],
) -> VerifiedReleaseAssets:
    """Bind a manifest to an exact acquired set without exposing raw paths."""

    if not isinstance(manifest, ReleaseManifest):
        raise DataContractError("Release verification requires a ReleaseManifest.")
    if isinstance(acquired_assets, (str, bytes)) or not isinstance(
        acquired_assets,
        Sequence,
    ):
        raise DataContractError("Acquired release assets must be supplied as a sequence.")
    if not all(isinstance(asset, AcquiredAsset) for asset in acquired_assets):
        raise DataContractError("Acquired release entries must be AcquiredAsset values.")
    validated_assets = tuple(_validated_acquired_asset(asset) for asset in acquired_assets)

    acquired_by_key: dict[str, AcquiredAsset] = {}
    for acquired in validated_assets:
        if acquired.key in acquired_by_key:
            raise DataContractError(
                "Acquired release asset source keys must be unique; duplicate found."
            )
        acquired_by_key[acquired.key] = acquired

    required_keys = {asset.source_key for asset in manifest.assets}
    if set(acquired_by_key) != required_keys:
        raise DataContractError("Manifest and acquired release asset sets must match exactly.")
    for required in manifest.assets:
        acquired = acquired_by_key[required.source_key]
        if acquired.sha256 != required.sha256:
            raise ChecksumMismatchError(
                f"Acquired asset {required.source_key!r} does not match its manifest SHA-256."
            )

    verified = object.__new__(VerifiedReleaseAssets)
    object.__setattr__(verified, "_manifest", manifest)
    object.__setattr__(
        verified,
        "_paths",
        MappingProxyType(
            {
                source_key: Path(acquired_by_key[source_key].path)
                for source_key in sorted(acquired_by_key)
            }
        ),
    )
    return verified


def _validated_acquired_asset(asset: AcquiredAsset) -> AcquiredAsset:
    missing = object()
    key = getattr(asset, "key", missing)
    if not isinstance(key, str) or not key.strip():
        raise DataContractError("Acquired release asset key must be a nonempty string.")

    path = getattr(asset, "path", missing)
    if not isinstance(path, Path):
        raise DataContractError("Acquired release asset path must be a pathlib.Path.")

    sha256 = getattr(asset, "sha256", missing)
    if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
        raise DataContractError(
            "Acquired release asset sha256 must be a lowercase 64-character digest."
        )

    retrieved_at = getattr(asset, "retrieved_at", missing)
    if not isinstance(retrieved_at, datetime):
        raise DataContractError("Acquired release asset retrieved_at must be a datetime.")
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise DataContractError("Acquired release asset retrieved_at must be timezone-aware.")

    return AcquiredAsset(
        key=key,
        path=path,
        sha256=sha256,
        retrieved_at=retrieved_at,
    )


@contextmanager
def _open_verified_asset(
    *,
    source_key: str,
    path: Path,
    expected_sha256: str,
) -> Iterator[BinaryIO]:
    pre_open_stat: os.stat_result | None = None
    if _O_NOFOLLOW is None:
        pre_open_stat = _sanitized_lstat(path, source_key)
        if stat.S_ISLNK(pre_open_stat.st_mode) or not stat.S_ISREG(pre_open_stat.st_mode):
            raise MissingOfficialDataError(
                f"Required official asset {source_key!r} is not a regular file."
            )

    flags = os.O_RDONLY
    if _O_NOFOLLOW is not None:
        flags |= _O_NOFOLLOW
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)

    descriptor: int | None = None
    parser_stream: BinaryIO | None = None
    try:
        descriptor = _sanitized_open(path, flags, source_key)
        descriptor_stat = _sanitized_fstat(descriptor, source_key)
        if not stat.S_ISREG(descriptor_stat.st_mode):
            raise MissingOfficialDataError(
                f"Required official asset {source_key!r} is not a regular file."
            )
        if pre_open_stat is not None and (
            pre_open_stat.st_dev != descriptor_stat.st_dev
            or pre_open_stat.st_ino != descriptor_stat.st_ino
        ):
            raise MissingOfficialDataError(
                f"Required official asset {source_key!r} changed while being opened."
            )

        observed_sha256 = _sanitized_descriptor_hash(descriptor, source_key)
        if observed_sha256 != expected_sha256:
            raise ChecksumMismatchError(
                f"Official asset {source_key!r} has SHA-256 {observed_sha256}; "
                f"expected {expected_sha256}."
            )

        parser_stream = _sanitized_prepare_stream(descriptor, source_key)
        descriptor = None
        try:
            yield parser_stream
        finally:
            parser_stream.close()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sanitized_lstat(path: Path, source_key: str) -> os.stat_result:
    try:
        result = os.lstat(path)
    except OSError:
        pass
    else:
        return result
    raise MissingOfficialDataError(
        f"Required official asset {source_key!r} is missing or inaccessible."
    )


def _sanitized_open(path: Path, flags: int, source_key: str) -> int:
    try:
        descriptor = os.open(path, flags)
    except OSError:
        pass
    else:
        return descriptor
    raise MissingOfficialDataError(f"Required official asset {source_key!r} is missing or unsafe.")


def _sanitized_fstat(descriptor: int, source_key: str) -> os.stat_result:
    try:
        result = os.fstat(descriptor)
    except OSError:
        pass
    else:
        return result
    raise MissingOfficialDataError(
        f"Required official asset {source_key!r} could not be inspected."
    )


def _sanitized_descriptor_hash(descriptor: int, source_key: str) -> str:
    try:
        with os.fdopen(
            descriptor,
            "rb",
            buffering=0,
            closefd=False,
        ) as hash_stream:
            digest = sha256_stream(hash_stream)
    except OSError:
        pass
    else:
        return digest
    raise MissingOfficialDataError(f"Required official asset {source_key!r} could not be read.")


def _sanitized_prepare_stream(descriptor: int, source_key: str) -> BinaryIO:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        stream = os.fdopen(descriptor, "rb", closefd=True)
    except OSError:
        pass
    else:
        return stream
    raise MissingOfficialDataError(f"Required official asset {source_key!r} could not be prepared.")


@dataclass(frozen=True, slots=True)
class OfficialFinanceRelease:
    """A verified finance release over an ordered half-open reference interval."""

    assets: VerifiedReleaseAssets
    vintage: str
    reference_start: date
    reference_end: date

    def __post_init__(self) -> None:
        if not isinstance(self.assets, VerifiedReleaseAssets):
            raise DataContractError(
                "Official finance release assets must be VerifiedReleaseAssets."
            )
        if not isinstance(self.vintage, str) or not self.vintage.strip():
            raise DataContractError("Official finance release vintage must be a nonempty string.")
        if (
            not isinstance(self.reference_start, date)
            or isinstance(self.reference_start, datetime)
            or not isinstance(self.reference_end, date)
            or isinstance(self.reference_end, datetime)
        ):
            raise DataContractError("Official finance reference bounds must be date values.")
        if self.reference_start >= self.reference_end:
            raise DataContractError(
                "Official finance release requires reference_start < reference_end "
                "for its half-open interval."
            )

    @property
    def released_at(self) -> date | None:
        return self.assets.manifest.primary_fact_asset.released_at

    @property
    def release_missingness_reason(self) -> ReleaseMissingnessReason | None:
        return self.assets.manifest.primary_fact_asset.release_missingness_reason
