from __future__ import annotations

import copy
import hashlib
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime, timedelta, timezone
from typing import cast

import numpy as np
import pandas as pd
import pytest

import chile_demographic_pde.countries.chile as chile
from chile_demographic_pde.core.errors import (
    ChecksumMismatchError,
    DataContractError,
)
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.countries.chile import census, population
from chile_demographic_pde.countries.chile.census import (
    CuratedDomainRelease,
    CuratedTableContract,
    canonical_curated_table_bytes,
    normalize_censo_grouped_curated,
)
from chile_demographic_pde.countries.chile.population import (
    normalize_ine_population_curated,
)
from chile_demographic_pde.data.domain_schemas import (
    DemographicObservationSchema,
    validate_demographic_observations,
)
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
)

_CENSO_CONTRACT = CuratedTableContract(
    contract_id="censo2024-grouped-curated-v1",
    columns=("age_group", "population_male", "population_female"),
    logical_types=("string", "integer", "integer"),
)
_CENSO_CANONICAL_BYTES = (
    b'{"columns":["age_group","population_male","population_female"],'
    b'"contract_id":"censo2024-grouped-curated-v1",'
    b'"logical_types":["string","integer","integer"],'
    b'"schema_version":"curated-domain-table-v1","typed_rows":'
    b'[[{"t":"str","v":"0 a 4"},{"t":"int","v":"100"},'
    b'{"t":"int","v":"105"}],[{"t":"str","v":"85 o m\xc3\xa1s"},'
    b'{"t":"int","v":"10"},{"t":"int","v":"15"}]]}'
)
_CENSO_SHA256 = "68af86657cc959c58d0e2051d127edf530b036f01d67729e8f8f27fc14b9ec7b"

_INE_CONTRACT = CuratedTableContract(
    contract_id="ine-base2024-population-curated-v1",
    columns=("year", "age", "sex", "population"),
    logical_types=("integer", "integer", "string", "float64"),
)
_INE_CANONICAL_BYTES = (
    b'{"columns":["year","age","sex","population"],'
    b'"contract_id":"ine-base2024-population-curated-v1",'
    b'"logical_types":["integer","integer","string","float64"],'
    b'"schema_version":"curated-domain-table-v1","typed_rows":'
    b'[[{"t":"int","v":"2024"},{"t":"int","v":"0"},'
    b'{"t":"str","v":"Hombres"},'
    b'{"t":"float64","v":"4059000000000000"}],'
    b'[{"t":"int","v":"2025"},{"t":"int","v":"0"},'
    b'{"t":"str","v":"Hombres"},'
    b'{"t":"float64","v":"4059400000000000"}]]}'
)
_INE_SHA256 = "fca8a8b9adc01b93e9c0048dc757ef9c7d51ccc3cf9f3d7f7f1b18a4cbc996f0"

_SCALAR_CONTRACT = CuratedTableContract(
    contract_id="curated-scalar-grammar-v1",
    columns=("label", "count", "rate", "as_of", "retrieved_at", "optional"),
    logical_types=(
        "string",
        "integer",
        "float64",
        "date",
        "timestamp_utc",
        "string",
    ),
)
_SCALAR_CANONICAL_BYTES = (
    b'{"columns":["label","count","rate","as_of","retrieved_at","optional"],'
    b'"contract_id":"curated-scalar-grammar-v1",'
    b'"logical_types":["string","integer","float64","date","timestamp_utc","string"],'
    b'"schema_version":"curated-domain-table-v1","typed_rows":'
    b'[[{"t":"str","v":"Ni\xc3\xb1o"},{"t":"int","v":"3"},'
    b'{"t":"float64","v":"3ff8000000000000"},'
    b'{"t":"date","v":"2024-04-16"},'
    b'{"t":"timestamp_utc_ns","v":"2025-03-27T15:00:00.000000000Z"},'
    b'{"t":"null"}]]}'
)
_SCALAR_SHA256 = "24413a8566a3a081d5f76c8e3516d69ef51bfb79e968b65824051644aabd38fa"

_DEMOGRAPHIC_COLUMNS = tuple(DemographicObservationSchema.to_schema().columns)
_FOREIGN_DOMAIN_COLUMNS = {
    "tenor_years",
    "currency",
    "facility",
    "station",
    "pollutant",
    "pension_type",
    "mortality_table_id",
}


class _MutableString(str):
    marker: list[str]

    def __new__(cls, value: str) -> _MutableString:
        instance = cast("_MutableString", super().__new__(cls, value))
        instance.marker = []
        return instance


def _censo_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age_group": ["0 a 4", "85 o más"],
            "population_male": [100, 10],
            "population_female": [105, 15],
        }
    )


def _ine_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2024, 2025],
            "age": [0, 0],
            "sex": ["Hombres", "Hombres"],
            "population": [100.0, 101.0],
        }
    )


def _scalar_frame(*, label: str = "Niño") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [label],
            "count": [3],
            "rate": [1.5],
            "as_of": [date(2024, 4, 16)],
            "retrieved_at": [
                datetime(
                    2025,
                    3,
                    27,
                    12,
                    tzinfo=timezone(timedelta(hours=-3)),
                )
            ],
            "optional": [None],
        }
    )


def _release(
    *,
    source_key: str,
    expected_sha256: str,
    contract: CuratedTableContract,
    vintage: str,
    released_at: date,
    observation_start: date,
    observation_end: date,
) -> CuratedDomainRelease:
    registry = IdentityRegistry()
    available_at = datetime(2026, 7, 19, 15, tzinfo=UTC)
    manifest = ReleaseManifest.create(
        schema_version="curated-domain-release-v1",
        assets=(
            RequiredReleaseAsset(
                source_key=source_key,
                sha256=expected_sha256,
                available_at=available_at,
                released_at=released_at,
                release_missingness_reason=None,
            ),
        ),
        primary_fact_source_key=source_key,
        identity_registry=registry,
    )
    provenance = VariableProvenance(
        sources=(
            SourceRef(
                source_key=source_key,
                name=source_key,
                url=f"https://example.invalid/{source_key}",
                release_date=released_at,
                release_missingness_reason=None,
                retrieved_at=available_at,
                sha256=expected_sha256,
                vintage=vintage,
                provisional=False,
            ),
        ),
        release_id=manifest.release_id.value,
        available_at=manifest.available_at,
        observation_start=observation_start,
        observation_end=observation_end,
        dimensions=("period", "age", "sex", "region"),
        aggregation_rules=("published population stock",),
        missingness_reason="not_applicable_observed",
    )
    return CuratedDomainRelease(
        manifest=manifest,
        provenance=provenance,
        vintage=vintage,
        curated_source_key=source_key,
        table_contract=contract,
    )


def _censo_release() -> CuratedDomainRelease:
    return _release(
        source_key="censo2024_grouped_curated",
        expected_sha256=_CENSO_SHA256,
        contract=_CENSO_CONTRACT,
        vintage="2024-final",
        released_at=date(2025, 3, 27),
        observation_start=date(2024, 4, 16),
        observation_end=date(2024, 4, 17),
    )


def _ine_release() -> CuratedDomainRelease:
    return _release(
        source_key="ine_population_base2024_curated",
        expected_sha256=_INE_SHA256,
        contract=_INE_CONTRACT,
        vintage="base2024-reviewed",
        released_at=date(2026, 1, 28),
        observation_start=date(2024, 6, 30),
        observation_end=date(2025, 7, 1),
    )


def _normalize_censo(
    frame: pd.DataFrame | None = None,
    *,
    release: CuratedDomainRelease | None = None,
    registry: IdentityRegistry | None = None,
):
    return normalize_censo_grouped_curated(
        _censo_frame() if frame is None else frame,
        release=_censo_release() if release is None else release,
        reference_date=date(2024, 4, 16),
        identity_registry=IdentityRegistry() if registry is None else registry,
    )


def _normalize_ine(
    frame: pd.DataFrame | None = None,
    *,
    release: CuratedDomainRelease | None = None,
    registry: IdentityRegistry | None = None,
):
    return normalize_ine_population_curated(
        _ine_frame() if frame is None else frame,
        release=_ine_release() if release is None else release,
        historical_end_year=2024,
        identity_registry=IdentityRegistry() if registry is None else registry,
    )


def test_curated_boundary_names_replace_legacy_exports() -> None:
    assert census.normalize_censo_grouped_curated is normalize_censo_grouped_curated
    assert population.normalize_ine_population_curated is normalize_ine_population_curated
    assert chile.normalize_censo_grouped_curated is normalize_censo_grouped_curated
    assert chile.normalize_ine_population_curated is normalize_ine_population_curated
    assert not hasattr(census, "normalize_censo_grouped")
    assert not hasattr(population, "normalize_ine_population")
    assert "normalize_censo_grouped" not in chile.__all__
    assert "normalize_ine_population" not in chile.__all__


def test_canonical_curated_bytes_match_independent_literal_fixture() -> None:
    assert (
        canonical_curated_table_bytes(
            _scalar_frame(),
            _SCALAR_CONTRACT,
        )
        == _SCALAR_CANONICAL_BYTES
    )
    assert hashlib.sha256(_SCALAR_CANONICAL_BYTES).hexdigest() == _SCALAR_SHA256


def test_curated_fixture_attestations_are_independent_literals() -> None:
    assert (
        canonical_curated_table_bytes(
            _censo_frame(),
            _CENSO_CONTRACT,
        )
        == _CENSO_CANONICAL_BYTES
    )
    assert hashlib.sha256(_CENSO_CANONICAL_BYTES).hexdigest() == _CENSO_SHA256
    assert (
        canonical_curated_table_bytes(
            _ine_frame(),
            _INE_CONTRACT,
        )
        == _INE_CANONICAL_BYTES
    )
    assert hashlib.sha256(_INE_CANONICAL_BYTES).hexdigest() == _INE_SHA256


def test_string_atoms_are_normalized_to_nfc() -> None:
    assert (
        canonical_curated_table_bytes(
            _scalar_frame(label="Nin\u0303o"),
            _SCALAR_CONTRACT,
        )
        == _SCALAR_CANONICAL_BYTES
    )


def test_nullable_float64_uses_an_explicit_typed_null_atom() -> None:
    frame = pd.DataFrame({"rate": pd.Series([1.5, pd.NA], dtype="Float64")})
    contract = CuratedTableContract(
        contract_id="nullable-float64-v1",
        columns=("rate",),
        logical_types=("float64",),
    )
    expected = (
        b'{"columns":["rate"],"contract_id":"nullable-float64-v1",'
        b'"logical_types":["float64"],'
        b'"schema_version":"curated-domain-table-v1","typed_rows":'
        b'[[{"t":"float64","v":"3ff8000000000000"}],[{"t":"null"}]]}'
    )

    assert canonical_curated_table_bytes(frame, contract) == expected


@pytest.mark.parametrize("dtype", ["int32", "uint16", "Int64"])
def test_integer_physical_dtype_mutations_fail_closed(dtype: str) -> None:
    frame = pd.DataFrame({"value": pd.Series([1, 2], dtype=dtype)})
    contract = CuratedTableContract(
        contract_id="exact-int64-v1",
        columns=("value",),
        logical_types=("integer",),
    )

    with pytest.raises(DataContractError, match=r"dtype|physical"):
        canonical_curated_table_bytes(frame, contract)


def test_nullable_integer_requires_an_actual_explicit_null() -> None:
    frame = pd.DataFrame({"value": pd.Series([1, pd.NA], dtype="Int64")})
    contract = CuratedTableContract(
        contract_id="nullable-integer-v1",
        columns=("value",),
        logical_types=("integer",),
    )

    assert b'{"t":"null"}' in canonical_curated_table_bytes(frame, contract)


def test_forged_nominal_contract_and_release_fail_as_data_contract_errors() -> None:
    forged_contract = object.__new__(CuratedTableContract)
    with pytest.raises(DataContractError, match=r"contract|malformed"):
        canonical_curated_table_bytes(
            pd.DataFrame({"value": [1]}),
            forged_contract,
        )

    forged_release = object.__new__(CuratedDomainRelease)
    with pytest.raises(DataContractError, match=r"release|malformed"):
        normalize_censo_grouped_curated(
            _censo_frame(),
            release=forged_release,
            reference_date=date(2024, 4, 16),
            identity_registry=IdentityRegistry(),
        )


def test_forged_nested_manifest_asset_fails_as_a_data_contract_error() -> None:
    release = copy.copy(_censo_release())
    manifest = copy.copy(release.manifest)
    forged_asset = object.__new__(RequiredReleaseAsset)
    object.__setattr__(manifest, "assets", (forged_asset,))
    object.__setattr__(release, "manifest", manifest)

    with pytest.raises(DataContractError, match=r"asset|manifest|malformed"):
        _normalize_censo(release=release)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("value", ChecksumMismatchError),
        ("row_order", ChecksumMismatchError),
        ("dtype", DataContractError),
        ("column_order", DataContractError),
        ("index", DataContractError),
        ("missingness", ChecksumMismatchError),
    ],
)
def test_censo_frame_must_match_attested_deterministic_bytes(
    mutation: str,
    expected_error: type[Exception],
) -> None:
    changed = _censo_frame()
    if mutation == "value":
        changed.loc[0, "population_male"] = 101
    elif mutation == "row_order":
        changed = changed.iloc[::-1].reset_index(drop=True)
    elif mutation == "dtype":
        changed = changed.astype({"population_male": "float64"})
    elif mutation == "column_order":
        changed = changed.loc[:, list(reversed(changed.columns))]
    elif mutation == "index":
        changed.index = pd.Index([1, 2])
    else:
        changed.loc[0, "age_group"] = None

    with pytest.raises(expected_error):
        _normalize_censo(changed)


@pytest.mark.parametrize(
    ("value", "logical_type"),
    [
        (True, "integer"),
        (np.inf, "float64"),
        ("3", "integer"),
        (_MutableString("reviewed"), "string"),
        (object(), "string"),
        (datetime(2025, 1, 1), "timestamp_utc"),
    ],
)
def test_canonical_curated_table_rejects_pathological_scalars(
    value: object,
    logical_type: str,
) -> None:
    frame = pd.DataFrame({"value": [value]}, dtype=object)
    contract = CuratedTableContract(
        contract_id="pathological-scalars-v1",
        columns=("value",),
        logical_types=(logical_type,),
    )

    with pytest.raises(DataContractError):
        canonical_curated_table_bytes(frame, contract)


@pytest.mark.parametrize(
    "frame",
    [
        None,
        [],
        pd.DataFrame(),
        pd.DataFrame({"value": [1]}, index=pd.Index([5])),
    ],
)
def test_canonical_curated_table_rejects_nonframes_empty_or_nondefault_index(
    frame: object,
) -> None:
    contract = CuratedTableContract(
        contract_id="integer-table-v1",
        columns=("value",),
        logical_types=("integer",),
    )

    with pytest.raises(DataContractError):
        canonical_curated_table_bytes(cast(pd.DataFrame, frame), contract)


@pytest.mark.parametrize(
    ("contract_id", "columns", "logical_types"),
    [
        ("", ("value",), ("integer",)),
        ("unversioned", ("value",), ("integer",)),
        ("table-v1", (), ()),
        ("table-v1", ("value", "value"), ("integer", "integer")),
        ("table-v1", ("value",), ()),
        ("table-v1", ("value",), ("unsupported",)),
    ],
)
def test_malformed_curated_contracts_are_rejected(
    contract_id: str,
    columns: tuple[str, ...],
    logical_types: tuple[str, ...],
) -> None:
    with pytest.raises(DataContractError):
        contract = CuratedTableContract(
            contract_id=contract_id,
            columns=columns,
            logical_types=logical_types,
        )
        canonical_curated_table_bytes(pd.DataFrame({"value": [1]}), contract)


@pytest.mark.parametrize(
    "mismatch",
    ["release_id", "availability", "source_asset", "publisher_metadata"],
)
def test_normalizer_revalidates_manifest_and_provenance_before_mapping(
    mismatch: str,
) -> None:
    release = copy.copy(_censo_release())
    provenance_payload = release.provenance.model_dump(mode="python")
    if mismatch == "release_id":
        provenance_payload["release_id"] = "cldemopde:release:v1:" + "f" * 64
    elif mismatch == "availability":
        provenance_payload["available_at"] += timedelta(seconds=1)
    elif mismatch == "source_asset":
        provenance_payload["sources"][0]["sha256"] = "f" * 64
    else:
        provenance_payload["sources"][0]["release_date"] = date(2025, 3, 28)
    object.__setattr__(
        release,
        "provenance",
        VariableProvenance.model_validate(provenance_payload),
    )

    with pytest.raises(DataContractError, match=r"release|provenance|availability|asset"):
        _normalize_censo(release=release)


def test_normalizer_rejects_unknown_curated_source_key_before_mapping() -> None:
    release = copy.copy(_censo_release())
    object.__setattr__(release, "curated_source_key", "unknown_curated_source")

    with pytest.raises(DataContractError, match=r"source|asset"):
        _normalize_censo(release=release)


def test_censo_batch_has_strict_demographic_identity_and_copy_boundaries() -> None:
    frame = _censo_frame()
    original = frame.copy(deep=True)
    release = _censo_release()
    registry = IdentityRegistry()

    batch = _normalize_censo(frame, release=release, registry=registry)
    observations = batch.observations

    pd.testing.assert_frame_equal(frame, original)
    assert tuple(observations.columns) == _DEMOGRAPHIC_COLUMNS
    assert _FOREIGN_DOMAIN_COLUMNS.isdisjoint(observations.columns)
    pd.testing.assert_frame_equal(
        validate_demographic_observations(observations),
        observations,
    )
    assert set(observations["domain"]) == {"demographic"}
    assert set(observations["observation_role"]) == {"observed_fact"}
    assert set(observations["population_basis"]) == {"census_enumerated"}
    assert set(observations["parity_scope"]) == {"not_applicable"}
    assert set(observations["period_start"]) == {date(2024, 4, 16)}
    assert set(observations["period_end"]) == {date(2024, 4, 17)}
    assert set(observations["release_id"]) == {release.manifest.release_id.value}
    assert len(set(observations["fact_id"])) == len(observations)
    assert len(set(observations["observation_id"])) == len(observations)
    assert all(value.startswith("cldemopde:fact:v1:") for value in observations["fact_id"])
    assert all(
        value.startswith("cldemopde:observation:v1:") for value in observations["observation_id"]
    )
    assert batch.manifest is release.manifest
    assert batch.provenance == release.provenance
    assert batch.identities.require(release.manifest.release_id.value)

    observations.loc[:, "value"] = -1.0
    frame.loc[:, "population_male"] = 0
    assert (batch.observations["value"] >= 0).all()
    with pytest.raises((AttributeError, FrozenInstanceError)):
        batch.manifest = release.manifest


def test_ine_roles_split_at_reviewed_historical_end_year() -> None:
    batch = _normalize_ine()
    observations = batch.observations
    years = observations["period_start"].map(lambda value: value.year)

    assert tuple(observations.columns) == _DEMOGRAPHIC_COLUMNS
    assert _FOREIGN_DOMAIN_COLUMNS.isdisjoint(observations.columns)
    assert set(observations["domain"]) == {"demographic"}
    assert set(observations["population_basis"]) == {"resident_estimate"}
    assert set(observations["parity_scope"]) == {"not_applicable"}
    assert set(observations.loc[years.le(2024), "observation_role"]) == {"observed_fact"}
    assert set(observations.loc[years.gt(2024), "observation_role"]) == {"external_projection"}
    assert set(observations["period_start"]) == {
        date(2024, 6, 30),
        date(2025, 6, 30),
    }
    assert set(observations["period_end"]) == {
        date(2024, 7, 1),
        date(2025, 7, 1),
    }
    assert len(set(observations["fact_id"])) == len(observations)
    assert len(set(observations["observation_id"])) == len(observations)


def test_normalizer_rejects_wrong_reviewed_controls() -> None:
    with pytest.raises(DataContractError, match=r"reference|Censo|2024"):
        normalize_censo_grouped_curated(
            _censo_frame(),
            release=_censo_release(),
            reference_date=date(2024, 4, 15),
            identity_registry=IdentityRegistry(),
        )
    with pytest.raises(DataContractError, match=r"historical|year"):
        normalize_ine_population_curated(
            _ine_frame(),
            release=_ine_release(),
            historical_end_year=True,
            identity_registry=IdentityRegistry(),
        )
    with pytest.raises(DataContractError, match=r"registry|Identity"):
        normalize_censo_grouped_curated(
            _censo_frame(),
            release=_censo_release(),
            reference_date=date(2024, 4, 16),
            identity_registry=cast(IdentityRegistry, object()),
        )
