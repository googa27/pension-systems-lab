from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from chile_demographic_pde.data.roles import ReleaseMissingnessReason

_SOURCE_KEY = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_RELEASE_ID = re.compile(r"^cldemopde:release:v1:[0-9a-f]{64}$")


class SourceRef(BaseModel):
    """Immutable reference to one retrieved source artifact."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    source_key: str
    name: str = Field(min_length=1)
    url: HttpUrl
    release_date: date | None
    release_missingness_reason: ReleaseMissingnessReason | None
    retrieved_at: datetime
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vintage: str = Field(min_length=1)
    provisional: bool

    @field_validator("source_key")
    @classmethod
    def _require_canonical_source_key(cls, value: str) -> str:
        if _SOURCE_KEY.fullmatch(value) is None:
            raise ValueError("source_key must use lowercase snake-case characters")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def _require_timezone_aware_retrieval(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _require_release_date_xor_reason(self) -> Self:
        if (self.release_date is None) == (self.release_missingness_reason is None):
            raise ValueError(
                "release metadata requires exactly one of release_date "
                "or release_missingness_reason"
            )
        return self


class VariableProvenance(BaseModel):
    """Immutable provenance attached to one demographic variable."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    sources: tuple[SourceRef, ...] = Field(min_length=1)
    release_id: str
    available_at: datetime
    observation_start: date
    observation_end: date
    dimensions: tuple[str, ...]
    aggregation_rules: tuple[str, ...] = Field(min_length=1)
    missingness_reason: str
    transformation_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @field_validator("release_id")
    @classmethod
    def _require_versioned_release_id(cls, value: str) -> str:
        if _RELEASE_ID.fullmatch(value) is None:
            raise ValueError("release_id must use the cldemopde:release:v1 SHA-256 form")
        return value

    @field_validator("available_at")
    @classmethod
    def _require_utc_availability(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("available_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("dimensions")
    @classmethod
    def _require_unique_nonblank_dimensions(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not dimension.strip() for dimension in value):
            raise ValueError("dimensions must contain only nonblank names")
        if len(set(value)) != len(value):
            raise ValueError("dimensions must contain unique names")
        return value

    @field_validator("aggregation_rules")
    @classmethod
    def _require_nonblank_aggregation_rules(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not rule.strip() for rule in value):
            raise ValueError("aggregation_rules must contain only nonblank rules")
        return value

    @field_validator("missingness_reason")
    @classmethod
    def _require_nonblank_missingness_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("missingness_reason must be nonblank")
        return value

    @model_validator(mode="after")
    def _require_ordered_observation_dates(self) -> Self:
        if self.observation_start >= self.observation_end:
            raise ValueError(
                "observation_start must precede observation_end for a half-open interval"
            )
        return self
