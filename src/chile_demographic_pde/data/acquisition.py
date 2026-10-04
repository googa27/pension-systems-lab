"""Checksum-pinned acquisition policy for official source assets."""

from __future__ import annotations

import hashlib
import string
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import pooch  # type: ignore[import-untyped]

from chile_demographic_pde.core.errors import (
    ChecksumMismatchError,
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.data.catalog import AcquisitionMode, OfficialAsset

_HASH_BLOCK_SIZE = 1 << 20


@dataclass(frozen=True, slots=True)
class AcquiredAsset:
    """An immutable, checksum-identified official file acquired for a snapshot."""

    key: str
    path: Path
    sha256: str
    retrieved_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise DataContractError("Acquired asset key must be a nonempty string.")
        if not isinstance(self.path, Path):
            raise DataContractError("Acquired asset path must be a pathlib.Path.")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or self.sha256 != self.sha256.lower()
            or any(character not in string.hexdigits for character in self.sha256)
        ):
            raise ValueError("Acquired asset SHA-256 must be a lowercase 64-character digest.")
        if not isinstance(self.retrieved_at, datetime):
            raise DataContractError("Acquired asset retrieved_at must be a datetime.")
        if self.retrieved_at.tzinfo is None or self.retrieved_at.utcoffset() is None:
            raise ValueError("Acquired asset retrieved_at must be timezone-aware.")
        object.__setattr__(
            self,
            "retrieved_at",
            self.retrieved_at.astimezone(UTC),
        )


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of ``path`` without loading it all at once."""

    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream: BinaryIO) -> str:
    """Hash bytes read from the caller-owned binary stream at its current position."""

    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(_HASH_BLOCK_SIZE), b""):
        digest.update(block)
    return digest.hexdigest()


def acquire_asset(
    asset: OfficialAsset,
    cache_dir: Path,
    *,
    progressbar: bool = False,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> AcquiredAsset:
    """Acquire one declared asset, never substituting absent official data."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / asset.expected_filename
    if destination.is_symlink() or destination.exists():
        return _verify(asset, destination, clock)

    if asset.acquisition is AcquisitionMode.MANUAL:
        raise MissingOfficialDataError(
            f"Expected official file {asset.expected_filename!r} in {cache_dir}; "
            f"obtain it from {asset.landing_url}."
        )

    if asset.download_url is None or asset.sha256 is None:
        raise MissingOfficialDataError(
            f"Direct official asset {asset.key!r} requires a declared download URL and SHA-256."
        )

    try:
        downloaded = Path(
            pooch.retrieve(
                url=str(asset.download_url),
                known_hash=f"sha256:{asset.sha256}",
                fname=asset.expected_filename,
                path=cache_dir,
                progressbar=progressbar,
            )
        )
    except ValueError as error:
        if _is_pooch_hash_error(error):
            raise ChecksumMismatchError(
                f"Downloaded {asset.expected_filename!r} did not match expected SHA-256 "
                f"{asset.sha256}: {error}"
            ) from error
        raise
    return _verify(asset, downloaded, clock)


def _verify(asset: OfficialAsset, path: Path, clock: Callable[[], datetime]) -> AcquiredAsset:
    if path.is_symlink() or not path.is_file():
        raise MissingOfficialDataError(
            f"Expected regular file {asset.expected_filename!r} at {path}; found a non-file entry."
        )
    observed = sha256_file(path)
    if asset.sha256 is not None and observed != asset.sha256:
        raise ChecksumMismatchError(f"{path.name} has SHA-256 {observed}; expected {asset.sha256}.")
    retrieved_at = clock()
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise ValueError("Acquisition clock must return a timezone-aware datetime.")
    return AcquiredAsset(
        key=asset.key,
        path=path,
        sha256=observed,
        retrieved_at=retrieved_at,
    )


def _is_pooch_hash_error(error: ValueError) -> bool:
    """Identify Pooch's documented downloaded-hash failure without masking other errors."""

    message = str(error).lower()
    return (
        "hash of downloaded file" in message
        and "does not match" in message
        and "known hash" in message
    )
