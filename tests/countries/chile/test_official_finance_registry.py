from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from chile_demographic_pde.core.errors import (
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.countries.chile.official_finance_registry import (
    PRODUCTION_OFFICIAL_RELEASE_REGISTRY_V1,
    ReviewedOfficialAsset,
    ReviewedOfficialReleaseEntry,
    ReviewedOfficialReleaseRegistry,
    require_reviewed_official_release,
)
from chile_demographic_pde.data.acquisition import AcquiredAsset
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.releases import (
    ReleaseManifest,
    RequiredReleaseAsset,
    VerifiedReleaseAssets,
    verify_release_assets,
)
from chile_demographic_pde.data.roles import ReleaseMissingnessReason

_REGISTRY_VERSION = "chile-reviewed-official-finance-v1"
_KEY = "cmf_tm2020_historical"
_SOURCE_KEY = "cmf_tm2020_historical_xlsx"
_DIGEST = hashlib.sha256(b"historical fixture").hexdigest()
_RELEASE_REASON = ReleaseMissingnessReason.PUBLISHER_DATE_NOT_AVAILABLE


def _asset(*, sha256: str = _DIGEST) -> ReviewedOfficialAsset:
    return ReviewedOfficialAsset(
        source_key=_SOURCE_KEY,
        sha256=sha256,
        released_at=None,
        release_missingness_reason=_RELEASE_REASON,
    )


def _entry(
    *,
    registry_key: str = _KEY,
    asset: ReviewedOfficialAsset | None = None,
) -> ReviewedOfficialReleaseEntry:
    return ReviewedOfficialReleaseEntry(
        registry_version=_REGISTRY_VERSION,
        registry_key=registry_key,
        release_schema_version="cmf-tm2020-historical-xlsx-v1",
        primary_fact_source_key=_SOURCE_KEY,
        assets=(_asset() if asset is None else asset,),
        contract_id="cmf-tm2020-historical-workbook-v1",
        review_id="independent-review-fixture",
    )


def _registry(
    entry: ReviewedOfficialReleaseEntry | None = None,
) -> ReviewedOfficialReleaseRegistry:
    return ReviewedOfficialReleaseRegistry.create(
        registry_version=_REGISTRY_VERSION,
        entries=(_entry() if entry is None else entry,),
    )


def _manifest(
    *,
    sha256: str = _DIGEST,
    source_key: str = _SOURCE_KEY,
    primary: str = _SOURCE_KEY,
) -> ReleaseManifest:
    return ReleaseManifest.create(
        schema_version="cmf-tm2020-historical-xlsx-v1",
        assets=(
            RequiredReleaseAsset(
                source_key=source_key,
                sha256=sha256,
                available_at=datetime(2026, 7, 19, tzinfo=UTC),
                released_at=None,
                release_missingness_reason=_RELEASE_REASON,
            ),
        ),
        primary_fact_source_key=primary,
        identity_registry=IdentityRegistry(),
    )


def test_production_registry_is_immutable_and_diagnostic_stays_locked() -> None:
    assert isinstance(
        PRODUCTION_OFFICIAL_RELEASE_REGISTRY_V1.entries,
        type(MappingProxyType({})),
    )
    assert "cmf_tm2020_construction_diagnostic" not in (PRODUCTION_OFFICIAL_RELEASE_REGISTRY_V1)
    assert not hasattr(PRODUCTION_OFFICIAL_RELEASE_REGISTRY_V1, "register")
    assert _DIGEST not in {
        asset.sha256 for entry in PRODUCTION_OFFICIAL_RELEASE_REGISTRY_V1 for asset in entry.assets
    }
    historical = PRODUCTION_OFFICIAL_RELEASE_REGISTRY_V1.entries["cmf_tm2020_historical"]
    assert historical.contract_id == "cmf-tm2020-historical-workbook-v1"
    assert historical.assets[0].sha256 == (
        "5c8a01aa2dd8b356038d33894b77bc765e1fa2281d53c8e336c750a74feab747"
    )
    with pytest.raises(TypeError):
        PRODUCTION_OFFICIAL_RELEASE_REGISTRY_V1.entries["caller"] = _entry()  # type: ignore[index]


def test_reviewed_registry_has_only_meaningful_collection_dunders() -> None:
    entry = _entry()
    registry = _registry(entry)

    assert len(registry) == 1
    assert _KEY in registry
    assert tuple(registry) == (entry,)
    assert "not-reviewed" not in registry


def test_exact_manifest_is_authorized_by_injected_registry() -> None:
    entry = require_reviewed_official_release(
        _manifest(),
        _KEY,
        _registry(),
    )

    assert entry.registry_key == _KEY
    assert entry.primary_fact_source_key == _SOURCE_KEY


@pytest.mark.parametrize(
    ("manifest", "registry_key"),
    [
        (_manifest(sha256="0" * 64), _KEY),
        (
            _manifest(
                source_key="unreviewed_fixture",
                primary="unreviewed_fixture",
            ),
            _KEY,
        ),
        (_manifest(), "cmf_tm2020_construction_diagnostic"),
    ],
)
def test_changed_asset_or_contract_key_fails_authorization(
    manifest: ReleaseManifest,
    registry_key: str,
) -> None:
    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(manifest, registry_key, _registry())


def test_caller_created_entry_is_not_authorization() -> None:
    caller_entry = _entry(registry_key="caller_created")

    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(
            _manifest(),
            caller_entry.registry_key,
            _registry(),
        )


def test_registry_revalidates_entries_and_detects_postcreate_tampering() -> None:
    entry = _entry()
    object.__setattr__(entry, "primary_fact_source_key", "changed")
    with pytest.raises(DataContractError):
        _registry(entry)

    registry = _registry()
    stored = next(iter(registry))
    object.__setattr__(stored, "review_id", "changed")
    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(_manifest(), _KEY, registry)


def test_registry_rejects_coherent_mapping_and_fingerprint_rebinding() -> None:
    original = _registry()
    replacement_entry = _entry(registry_key="replacement")
    replacement = _registry(replacement_entry)
    object.__setattr__(original, "entries", replacement.entries)
    object.__setattr__(original, "_fingerprints", replacement._fingerprints)

    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(
            _manifest(),
            replacement_entry.registry_key,
            original,
        )


def test_registry_rejects_value_equal_caller_entry_mapping_rebinding() -> None:
    registry = _registry()
    original = next(iter(registry))
    caller_clone = replace(original)
    object.__setattr__(
        registry,
        "entries",
        MappingProxyType({_KEY: caller_clone}),
    )

    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(_manifest(), _KEY, registry)


def test_release_hook_cannot_mutate_exact_two_asset_authorization(
    tmp_path: Path,
) -> None:
    reviewed_payloads = {
        "reviewed_primary": b"reviewed primary",
        "reviewed_support": b"reviewed support",
    }
    unreviewed_payloads = {
        "reviewed_primary": b"unreviewed primary",
        "reviewed_support": b"unreviewed support",
    }
    reviewed_assets = tuple(
        ReviewedOfficialAsset(
            source_key=source_key,
            sha256=hashlib.sha256(payload).hexdigest(),
            released_at=None,
            release_missingness_reason=_RELEASE_REASON,
        )
        for source_key, payload in sorted(reviewed_payloads.items())
    )
    entry = ReviewedOfficialReleaseEntry(
        registry_version=_REGISTRY_VERSION,
        registry_key="two_asset_release",
        release_schema_version="two-asset-release-v1",
        primary_fact_source_key="reviewed_primary",
        assets=reviewed_assets,
        contract_id="two-asset-contract-v1",
        review_id="independent-review-fixture",
    )
    registry = ReviewedOfficialReleaseRegistry.create(
        registry_version=_REGISTRY_VERSION,
        entries=(entry,),
    )

    required: list[RequiredReleaseAsset] = []
    acquired: list[AcquiredAsset] = []
    for source_key, payload in sorted(unreviewed_payloads.items()):
        path = tmp_path / source_key
        path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        required.append(
            RequiredReleaseAsset(
                source_key=source_key,
                sha256=digest,
                available_at=datetime(2026, 7, 19, tzinfo=UTC),
                released_at=None,
                release_missingness_reason=_RELEASE_REASON,
            )
        )
        acquired.append(
            AcquiredAsset(
                key=source_key,
                path=path,
                sha256=digest,
                retrieved_at=datetime(2026, 7, 19, tzinfo=UTC),
            )
        )
    manifest = ReleaseManifest.create(
        schema_version="two-asset-release-v1",
        assets=tuple(required),
        primary_fact_source_key="reviewed_primary",
        identity_registry=IdentityRegistry(),
    )
    verified = verify_release_assets(manifest, acquired)

    class MutatingRelease:
        @property
        def assets(self) -> VerifiedReleaseAssets:
            for reviewed, actual in zip(
                entry.assets,
                manifest.assets,
                strict=True,
            ):
                object.__setattr__(reviewed, "sha256", actual.sha256)
            return verified

    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(
            MutatingRelease(),
            entry.registry_key,
            registry,
        )


def test_registry_rejects_registry_version_str_subclass_rebinding() -> None:
    class Text(str):
        pass

    registry = _registry()
    object.__setattr__(registry, "registry_version", Text(_REGISTRY_VERSION))

    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(_manifest(), _KEY, registry)


def test_registry_rejects_hostile_fingerprint_mapping_before_traversal() -> None:
    class HostileMapping(Mapping[str, str]):
        def __init__(self) -> None:
            self.traversed = False

        def __getitem__(self, key: str) -> str:
            self.traversed = True
            raise AssertionError(key)

        def __iter__(self) -> Iterator[str]:
            self.traversed = True
            raise AssertionError("iteration")

        def __len__(self) -> int:
            self.traversed = True
            raise AssertionError("length")

        def items(self) -> object:
            self.traversed = True
            raise AssertionError("items")

    registry = _registry()
    hostile = HostileMapping()
    object.__setattr__(registry, "_fingerprints", hostile)

    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(_manifest(), _KEY, registry)
    assert not hostile.traversed


def test_manifest_identity_and_availability_are_revalidated() -> None:
    manifest = _manifest()
    object.__setattr__(manifest, "available_at", datetime(2030, 1, 1, tzinfo=UTC))
    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(manifest, _KEY, _registry())

    manifest = _manifest()
    object.__setattr__(manifest, "release_id", _manifest(sha256="1" * 64).release_id)
    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(manifest, _KEY, _registry())


def test_manifest_rejects_canonically_equal_primitive_subclass_rebinding() -> None:
    class Text(str):
        pass

    manifest = _manifest()
    object.__setattr__(
        manifest,
        "schema_version",
        Text("cmf-tm2020-historical-xlsx-v1"),
    )
    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(manifest, _KEY, _registry())

    manifest = _manifest()
    object.__setattr__(
        manifest.assets[0],
        "source_key",
        Text(_SOURCE_KEY),
    )
    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(manifest, _KEY, _registry())


def test_unknown_key_is_rejected_before_release_is_inspected() -> None:
    class SentinelRelease:
        @property
        def assets(self) -> object:
            raise AssertionError("release was inspected before authorization")

    with pytest.raises(MissingOfficialDataError):
        require_reviewed_official_release(
            SentinelRelease(),
            "not_reviewed",
            _registry(),
        )


def test_registry_rejects_uninitialized_or_hostile_entry_sequences() -> None:
    with pytest.raises(DataContractError):
        _registry(object.__new__(ReviewedOfficialReleaseEntry))

    class HostileSequence(Sequence[ReviewedOfficialReleaseEntry]):
        def __getitem__(self, index: int) -> ReviewedOfficialReleaseEntry:
            raise RuntimeError(index)

        def __len__(self) -> int:
            raise RuntimeError("length")

        def __iter__(self) -> Iterator[ReviewedOfficialReleaseEntry]:
            raise RuntimeError("iteration")

    with pytest.raises(DataContractError):
        ReviewedOfficialReleaseRegistry.create(
            registry_version=_REGISTRY_VERSION,
            entries=HostileSequence(),
        )


def test_entry_validation_is_closed_and_dates_are_not_datetimes() -> None:
    with pytest.raises(DataContractError):
        replace(_asset(), released_at=datetime(2023, 2, 24, tzinfo=UTC))
    with pytest.raises(DataContractError):
        replace(_entry(), assets=(_asset(), _asset()))
    with pytest.raises(DataContractError):
        replace(_entry(), primary_fact_source_key="not_present")
    with pytest.raises(DataContractError):
        replace(_entry(), registry_key="not canonical")
    with pytest.raises(DataContractError):
        replace(_asset(), sha256="A" * 64)


def test_entry_declares_exact_publisher_date_or_reason_per_asset() -> None:
    dated = ReviewedOfficialAsset(
        source_key=_SOURCE_KEY,
        sha256=_DIGEST,
        released_at=date(2023, 2, 24),
        release_missingness_reason=None,
    )
    assert dated.released_at == date(2023, 2, 24)
    with pytest.raises(DataContractError):
        replace(dated, release_missingness_reason=_RELEASE_REASON)
    with pytest.raises(DataContractError):
        replace(dated, released_at=None)
