"""Checksum-pinned identities and acquisition policies for official data."""

from __future__ import annotations

import unicodedata
from datetime import date
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{number}" for number in range(1, 10)}
    | {f"LPT{number}" for number in range(1, 10)}
)
WINDOWS_RESERVED_FILENAME_CHARACTERS = frozenset('<>:"|?*')


class AcquisitionMode(StrEnum):
    """How an official asset is obtained."""

    DIRECT = "direct"
    MANUAL = "manual"


class OfficialAsset(BaseModel):
    """Identity and acquisition policy for one official source asset."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    landing_url: HttpUrl
    acquisition: AcquisitionMode
    expected_filename: str = Field(min_length=1)
    vintage: str = Field(min_length=1)
    provisional: bool
    download_url: HttpUrl | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    required: bool = True
    notes: tuple[str, ...] = ()

    @field_validator("title", "publisher", "vintage")
    @classmethod
    def validate_nonblank_metadata(cls, value: str) -> str:
        """Reject whitespace-only descriptive metadata."""

        if not value.strip():
            raise ValueError("Asset metadata must not be blank.")
        return value

    @field_validator("expected_filename")
    @classmethod
    def validate_expected_filename(cls, value: str) -> str:
        """Require one safe filename rather than a path supplied by a source."""

        posix_path = PurePosixPath(value)
        windows_path = PureWindowsPath(value)
        windows_device_stem = (
            unicodedata.normalize("NFKC", value.split(".", maxsplit=1)[0]).rstrip(" .").upper()
        )
        if (
            not value.strip()
            or value != posix_path.name
            or value != windows_path.name
            or value.endswith((".", " "))
            or any(unicodedata.category(character) == "Cc" for character in value)
            or any(character in WINDOWS_RESERVED_FILENAME_CHARACTERS for character in value)
            or windows_device_stem in WINDOWS_RESERVED_BASENAMES
        ):
            raise ValueError("Expected filename must be a safe basename.")
        return value

    @model_validator(mode="after")
    def validate_acquisition(self) -> Self:
        """Require reproducible identity details for direct downloads."""

        if self.acquisition is AcquisitionMode.DIRECT and (
            self.download_url is None or self.sha256 is None
        ):
            raise ValueError("Direct assets require download_url and sha256.")
        return self


class SourceManifest(BaseModel):
    """The official source inventory for one ISO-3 country."""

    model_config = ConfigDict(frozen=True)

    country: str = Field(pattern=r"^[A-Z]{3}$")
    assets: tuple[OfficialAsset, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_keys(self) -> Self:
        """Reject ambiguous references to official assets."""

        keys = tuple(asset.key for asset in self.assets)
        if len(keys) != len(set(keys)):
            raise ValueError("Asset keys must be unique.")
        return self

    def asset(self, key: str) -> OfficialAsset:
        """Return the uniquely declared official asset for ``key``."""

        for asset in self.assets:
            if asset.key == key:
                return asset
        raise KeyError(f"Unknown asset key {key!r} for country {self.country!r}.")


class DataSnapshot(BaseModel):
    """Immutable identity and explicit availability for a materialized snapshot."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str = Field(min_length=1)
    country: str = Field(pattern=r"^[A-Z]{3}$")
    as_of: date
    root: Path
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    acquired_assets: tuple[str, ...]
    available_assets: tuple[str, ...]
    unavailable_optional_assets: tuple[str, ...] = ()

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        """Reject a whitespace-only snapshot identity."""

        if not value.strip():
            raise ValueError("Snapshot ID must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_asset_coverage(self) -> Self:
        """Keep explicit asset availability internally consistent."""

        fields = (
            ("acquired_assets", self.acquired_assets),
            ("available_assets", self.available_assets),
            ("unavailable_optional_assets", self.unavailable_optional_assets),
        )
        for name, keys in fields:
            if len(keys) != len(set(keys)):
                raise ValueError(f"{name} must not contain duplicate asset keys.")
        available = set(self.available_assets)
        if not set(self.acquired_assets).issubset(available):
            raise ValueError("Acquired assets must be a subset of available assets.")
        if set(self.unavailable_optional_assets).intersection(available):
            raise ValueError("Unavailable optional assets must be disjoint from available assets.")
        return self
