"""Checksum-attested boundaries for reviewed legacy curated tables."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, cast

import numpy as np
import pandas as pd
from pydantic import ValidationError

from chile_demographic_pde.core.errors import (
    ChecksumMismatchError,
    DataContractError,
)
from chile_demographic_pde.core.provenance import VariableProvenance
from chile_demographic_pde.data.domain_schemas import (
    DEMOGRAPHIC_OBSERVATION_KEY,
    DemographicObservationBatch,
    validate_demographic_observations,
)
from chile_demographic_pde.data.envelope import (
    NormalizedDomainBatch,
    assign_observation_ids,
)
from chile_demographic_pde.data.identity import (
    IdentityRegistry,
    ReleaseIdFactory,
    canonical_identity_bytes,
)
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
)
from chile_demographic_pde.data.roles import ObservationDomain

_CONTRACT_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*-v[1-9][0-9]*$")
_LOGICAL_TYPES: Final = frozenset({"integer", "float64", "string", "date", "timestamp_utc"})
_SCHEMA_VERSION: Final = "curated-domain-table-v1"

type _CanonicalJson = None | bool | str | list[_CanonicalJson] | dict[str, _CanonicalJson]


@dataclass(frozen=True, slots=True)
class CuratedTableContract:
    """Exact physical columns and reviewed scalar types for a curated table."""

    contract_id: str
    columns: tuple[str, ...]
    logical_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.contract_id) is not str
            or self.contract_id.strip() != self.contract_id
            or _CONTRACT_ID.fullmatch(self.contract_id) is None
        ):
            raise DataContractError(
                "Curated table contract_id must be a nonblank versioned identifier."
            )
        if type(self.columns) is not tuple or not self.columns:
            raise DataContractError("Curated table columns must be a nonempty immutable tuple.")
        if any(
            type(column) is not str or not column.strip() or column.strip() != column
            for column in self.columns
        ):
            raise DataContractError("Curated table columns must be nonblank built-in strings.")
        if len(set(self.columns)) != len(self.columns):
            raise DataContractError("Curated table column names must be unique.")
        if type(self.logical_types) is not tuple or len(self.logical_types) != len(self.columns):
            raise DataContractError(
                "Curated table logical types must exactly align with its columns."
            )
        if any(
            type(logical_type) is not str or logical_type not in _LOGICAL_TYPES
            for logical_type in self.logical_types
        ):
            raise DataContractError("Curated table contains an unsupported logical scalar type.")


@dataclass(frozen=True, slots=True)
class CuratedDomainRelease:
    """Manifest, provenance, and table contract for one attested curated input."""

    manifest: ReleaseManifest
    provenance: VariableProvenance
    vintage: str
    curated_source_key: str
    table_contract: CuratedTableContract

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ReleaseManifest):
            raise DataContractError("Curated domain release requires a ReleaseManifest.")
        if not isinstance(self.provenance, VariableProvenance):
            raise DataContractError("Curated domain release requires VariableProvenance.")
        if (
            type(self.vintage) is not str
            or not self.vintage.strip()
            or self.vintage.strip() != self.vintage
        ):
            raise DataContractError("Curated domain release vintage must be a nonblank string.")
        if type(self.curated_source_key) is not str or not self.curated_source_key:
            raise DataContractError("Curated domain release source key must be a nonblank string.")
        if not isinstance(self.table_contract, CuratedTableContract):
            raise DataContractError("Curated domain release requires a CuratedTableContract.")


def _validated_table_contract(
    contract: CuratedTableContract,
) -> CuratedTableContract:
    """Reconstruct a nominal contract instead of trusting forged slots."""

    if type(contract) is not CuratedTableContract:
        raise DataContractError("Canonical curated input requires a CuratedTableContract.")
    try:
        return CuratedTableContract(
            contract_id=contract.contract_id,
            columns=contract.columns,
            logical_types=contract.logical_types,
        )
    except (
        AttributeError,
        DataContractError,
        TypeError,
        ValueError,
    ):
        raise DataContractError("Curated table contract is malformed.") from None


def _is_explicit_null(value: object) -> bool:
    return value is None or value is pd.NA or value is pd.NaT


def _contains_explicit_null(series: pd.Series) -> bool:
    return any(_is_explicit_null(series.iloc[position]) for position in range(len(series)))


def _canonical_atom(value: object) -> _CanonicalJson:
    if _is_explicit_null(value):
        return {"t": "null"}
    if type(value) not in {int, float, str, date, datetime, pd.Timestamp}:
        raise DataContractError("Curated table cells must use reviewed immutable scalar atoms.")
    if type(value) is float and not math.isfinite(value):
        raise DataContractError("Curated table float64 values must be finite.")
    if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() is None):
        raise DataContractError("Curated table timestamps must be timezone-aware.")
    try:
        encoded = json.loads(canonical_identity_bytes(value))
    except (DataContractError, TypeError, ValueError, OverflowError):
        raise DataContractError(
            "Curated table cell cannot be encoded as a canonical scalar."
        ) from None
    return cast(_CanonicalJson, encoded)


def _dtype_matches(series: pd.Series, logical_type: str) -> bool:
    dtype = series.dtype
    if logical_type == "integer":
        return bool(
            dtype == np.dtype("int64")
            or (isinstance(dtype, pd.Int64Dtype) and _contains_explicit_null(series))
        )
    if logical_type == "float64":
        return bool(
            dtype == np.dtype("float64")
            or (isinstance(dtype, pd.Float64Dtype) and _contains_explicit_null(series))
        )
    if logical_type == "string":
        return pd.api.types.is_object_dtype(dtype)
    if logical_type == "date":
        return pd.api.types.is_object_dtype(dtype)
    if logical_type == "timestamp_utc":
        return isinstance(dtype, pd.DatetimeTZDtype)
    return False


def _cell_matches_logical_type(value: object, logical_type: str) -> bool:
    if _is_explicit_null(value):
        return True
    if logical_type == "integer":
        return type(value) is int
    if logical_type == "float64":
        return type(value) is float and math.isfinite(value)
    if logical_type == "string":
        return type(value) is str
    if logical_type == "date":
        return type(value) is date
    if logical_type == "timestamp_utc":
        return (
            type(value) in {datetime, pd.Timestamp}
            and isinstance(value, datetime)
            and value.tzinfo is not None
            and value.utcoffset() is not None
        )
    return False


def _python_cell(series: pd.Series, position: int) -> object:
    """Return pandas numeric scalars as their exact built-in reviewed atoms."""

    value = series.iloc[position]
    if isinstance(value, np.integer) and not isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def canonical_curated_table_bytes(
    frame: pd.DataFrame,
    contract: CuratedTableContract,
) -> bytes:
    """Serialize one reviewed table to stable typed canonical JSON bytes."""

    if type(frame) is not pd.DataFrame or frame.empty:
        raise DataContractError("Canonical curated input must be a nonempty pandas DataFrame.")
    validated_contract = _validated_table_contract(contract)
    if (
        not isinstance(frame.index, pd.RangeIndex)
        or frame.index.start != 0
        or frame.index.stop != len(frame)
        or frame.index.step != 1
        or frame.index.name is not None
    ):
        raise DataContractError(
            "Canonical curated input requires the default zero-based RangeIndex."
        )
    if not frame.columns.is_unique or tuple(frame.columns) != validated_contract.columns:
        raise DataContractError(
            "Canonical curated input columns must exactly match contract order."
        )

    for column, logical_type in zip(
        validated_contract.columns,
        validated_contract.logical_types,
        strict=True,
    ):
        if not _dtype_matches(frame[column], logical_type):
            raise DataContractError(
                "Canonical curated input physical dtype does not match its "
                f"declared logical type for column {column!r}."
            )

    typed_rows: list[list[_CanonicalJson]] = []
    for row_position in range(len(frame)):
        typed_row: list[_CanonicalJson] = []
        for column, logical_type in zip(
            validated_contract.columns,
            validated_contract.logical_types,
            strict=True,
        ):
            value = _python_cell(frame[column], row_position)
            if not _cell_matches_logical_type(value, logical_type):
                raise DataContractError(
                    "Canonical curated input cell does not match its declared "
                    f"logical type for column {column!r}."
                )
            typed_row.append(_canonical_atom(value))
        typed_rows.append(typed_row)

    payload = {
        "columns": list(validated_contract.columns),
        "contract_id": validated_contract.contract_id,
        "logical_types": list(validated_contract.logical_types),
        "schema_version": _SCHEMA_VERSION,
        "typed_rows": typed_rows,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validated_manifest(manifest: ReleaseManifest) -> dict[str, RequiredReleaseAsset]:
    if type(manifest) is not ReleaseManifest:
        raise DataContractError("Curated release manifest must be a ReleaseManifest.")
    try:
        schema_version = manifest.schema_version
        assets = manifest.assets
        primary_source_key = manifest.primary_fact_source_key
        available_at = manifest.available_at
        release_id = manifest.release_id
    except AttributeError:
        raise DataContractError("Curated release manifest is malformed.") from None
    if type(schema_version) is not str or not schema_version.strip():
        raise DataContractError("Curated release manifest schema version is malformed.")
    if type(assets) is not tuple or not assets:
        raise DataContractError("Curated release manifest assets are malformed.")
    validated_assets: list[RequiredReleaseAsset] = []
    for asset in assets:
        if type(asset) is not RequiredReleaseAsset:
            raise DataContractError("Curated release manifest assets are malformed.")
        try:
            validated_assets.append(
                RequiredReleaseAsset(
                    source_key=asset.source_key,
                    sha256=asset.sha256,
                    available_at=asset.available_at,
                    released_at=asset.released_at,
                    release_missingness_reason=asset.release_missingness_reason,
                )
            )
        except (
            AttributeError,
            DataContractError,
            TypeError,
            ValueError,
        ):
            raise DataContractError("Curated release manifest assets are malformed.") from None
    if tuple(asset.source_key for asset in validated_assets) != tuple(
        sorted(asset.source_key for asset in validated_assets)
    ):
        raise DataContractError("Curated release manifest assets must use canonical source order.")
    by_key = {asset.source_key: asset for asset in validated_assets}
    if len(by_key) != len(assets):
        raise DataContractError("Curated release manifest asset source keys must be unique.")
    if (
        type(primary_source_key) is not str
        or primary_source_key not in by_key
        or available_at != max(asset.available_at for asset in validated_assets)
    ):
        raise DataContractError(
            "Curated release manifest primary asset or availability is invalid."
        )
    try:
        expected_release_id = ReleaseIdFactory.v1(
            tuple((asset.source_key, asset.sha256) for asset in validated_assets)
        )
        release_identity_matches = (
            release_id.value == expected_release_id.value
            and release_id.canonical_payload == expected_release_id.canonical_payload
        )
    except (AttributeError, DataContractError, TypeError, ValueError):
        release_identity_matches = False
    if not release_identity_matches:
        raise DataContractError("Curated release manifest release identity is invalid.")
    return by_key


def _validated_provenance(
    provenance: VariableProvenance,
) -> VariableProvenance:
    if not isinstance(provenance, VariableProvenance):
        raise DataContractError("Curated release provenance must be VariableProvenance.")
    try:
        payload = provenance.model_dump(mode="python", warnings="none")
        return VariableProvenance.model_validate(payload)
    except (ValidationError, TypeError, ValueError, AttributeError):
        raise DataContractError("Curated release provenance violates its typed contract.") from None


def _validate_curated_release(
    release: CuratedDomainRelease,
) -> tuple[
    ReleaseManifest,
    RequiredReleaseAsset,
    VariableProvenance,
    CuratedTableContract,
]:
    if type(release) is not CuratedDomainRelease:
        raise DataContractError("Curated normalization requires a CuratedDomainRelease.")
    try:
        manifest = release.manifest
        raw_provenance = release.provenance
        curated_source_key = release.curated_source_key
        vintage = release.vintage
        raw_contract = release.table_contract
    except AttributeError:
        raise DataContractError("Curated domain release is malformed.") from None
    assets = _validated_manifest(manifest)
    provenance = _validated_provenance(raw_provenance)
    contract = _validated_table_contract(raw_contract)
    if provenance.release_id != manifest.release_id.value:
        raise DataContractError(
            "Curated release provenance release identity does not match manifest."
        )
    if provenance.available_at != manifest.available_at:
        raise DataContractError("Curated release provenance availability does not match manifest.")
    sources = {source.source_key: source for source in provenance.sources}
    if len(sources) != len(provenance.sources) or set(sources) != set(assets):
        raise DataContractError("Curated release provenance assets do not exactly match manifest.")
    for source_key, asset in assets.items():
        source = sources[source_key]
        if (
            source.sha256 != asset.sha256
            or source.release_date != asset.released_at
            or source.release_missingness_reason != asset.release_missingness_reason
            or source.retrieved_at != asset.available_at
        ):
            raise DataContractError(
                "Curated release provenance publisher metadata or asset "
                "availability does not match manifest."
            )
    if type(curated_source_key) is not str or list(assets).count(curated_source_key) != 1:
        raise DataContractError("Curated source key must name exactly one manifest asset.")
    if (
        type(vintage) is not str
        or not vintage.strip()
        or sources[curated_source_key].vintage != vintage
    ):
        raise DataContractError("Curated release vintage must match the attested source.")
    return (
        manifest,
        assets[curated_source_key],
        provenance,
        contract,
    )


def _attest_curated_release(
    frame: pd.DataFrame,
    release: CuratedDomainRelease,
) -> None:
    """Validate release metadata, then require exact curated-table bytes."""

    _, curated_asset, _, contract = _validate_curated_release(release)
    digest = hashlib.sha256(canonical_curated_table_bytes(frame, contract)).hexdigest()
    if digest != curated_asset.sha256:
        raise ChecksumMismatchError(
            "Curated table bytes do not match the attested manifest checksum."
        )


def _finalize_demographic_batch(
    payload: pd.DataFrame,
    *,
    release: CuratedDomainRelease,
    identity_registry: IdentityRegistry,
    source_identity: Mapping[str, object],
) -> DemographicObservationBatch:
    """Assign IDs, validate the strict domain, and create an immutable batch."""

    manifest, _, provenance, _ = _validate_curated_release(release)
    if not isinstance(identity_registry, IdentityRegistry):
        raise DataContractError("Curated demographic normalization requires an IdentityRegistry.")
    if not isinstance(payload, pd.DataFrame):
        raise DataContractError("Curated demographic payload must be a pandas DataFrame.")
    assigned = assign_observation_ids(
        payload.copy(deep=True),
        domain=ObservationDomain.DEMOGRAPHIC,
        domain_key=DEMOGRAPHIC_OBSERVATION_KEY,
        source_identity=source_identity,
        manifest=manifest,
        registry=identity_registry,
    )
    validated = validate_demographic_observations(assigned)
    return NormalizedDomainBatch.create(
        manifest=manifest,
        observations=validated,
        provenance=provenance,
        identities=identity_registry.snapshot(),
    )
