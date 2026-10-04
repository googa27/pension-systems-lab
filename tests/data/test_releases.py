from __future__ import annotations

import errno
import hashlib
import os
import traceback
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from zoneinfo import ZoneInfo

import pytest

import chile_demographic_pde.data.releases as releases_module
from chile_demographic_pde.core.errors import (
    ChecksumMismatchError,
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.data.acquisition import AcquiredAsset, sha256_file
from chile_demographic_pde.data.identity import IdentityRegistry, ReleaseIdFactory
from chile_demographic_pde.data.releases import (
    OfficialFinanceRelease,
    ReleaseManifest,
    RequiredReleaseAsset,
    VerifiedReleaseAssets,
    verify_release_assets,
)
from chile_demographic_pde.data.roles import ReleaseMissingnessReason

_MISSING_DATE = ReleaseMissingnessReason.PUBLISHER_DATE_NOT_AVAILABLE
_NOW = datetime(2026, 7, 19, 12, tzinfo=UTC)


def _required(
    source_key: str = "asset",
    sha256: str = "a" * 64,
    *,
    available_at: datetime = _NOW,
    released_at: date | None = None,
    reason: ReleaseMissingnessReason | None = _MISSING_DATE,
) -> RequiredReleaseAsset:
    return RequiredReleaseAsset(
        source_key=source_key,
        sha256=sha256,
        available_at=available_at,
        released_at=released_at,
        release_missingness_reason=reason,
    )


def _manifest(
    assets: tuple[RequiredReleaseAsset, ...] | None = None,
    *,
    primary: str = "asset",
    registry: IdentityRegistry | None = None,
) -> ReleaseManifest:
    return ReleaseManifest.create(
        schema_version="official-release-v1",
        assets=assets if assets is not None else (_required(),),
        primary_fact_source_key=primary,
        identity_registry=registry if registry is not None else IdentityRegistry(),
    )


def _acquired(path: Path, key: str = "asset") -> AcquiredAsset:
    return AcquiredAsset(
        key=key,
        path=path,
        sha256=sha256_file(path),
        retrieved_at=_NOW,
    )


def _forged_acquired(
    *,
    key: object = "asset",
    path: object = Path("asset.bin"),
    sha256: object = "a" * 64,
    retrieved_at: object = _NOW,
    missing_field: str | None = None,
) -> AcquiredAsset:
    forged = object.__new__(AcquiredAsset)
    values = {
        "key": key,
        "path": path,
        "sha256": sha256,
        "retrieved_at": retrieved_at,
    }
    for field, value in values.items():
        if field != missing_field:
            object.__setattr__(forged, field, value)
    return forged


def _verified(path: Path, key: str = "asset") -> VerifiedReleaseAssets:
    acquired = _acquired(path, key)
    manifest = _manifest(
        (
            _required(
                key,
                acquired.sha256,
                available_at=acquired.retrieved_at,
            ),
        ),
        primary=key,
    )
    return verify_release_assets(manifest, (acquired,))


def _assert_private_path_is_sanitized(
    error: BaseException,
    private_path: Path,
) -> None:
    private_text = str(private_path)
    rendered = "".join(traceback.TracebackException.from_exception(error).format(chain=True))
    assert private_text not in rendered
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        assert private_text not in str(current)
        assert private_text not in repr(current)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)


@pytest.fixture
def composite_manifest_with_distinct_asset_dates() -> ReleaseManifest:
    return _manifest(
        (
            _required(
                "workbook",
                "a" * 64,
                available_at=datetime(2026, 7, 7, tzinfo=UTC),
                released_at=date(2026, 6, 30),
                reason=None,
            ),
            _required(
                "entity_registry",
                "b" * 64,
                available_at=datetime(2026, 7, 9, tzinfo=UTC),
            ),
        ),
        primary="workbook",
    )


@pytest.fixture
def verified_assets(tmp_path: Path) -> VerifiedReleaseAssets:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    return _verified(path)


def test_required_asset_normalizes_available_at_to_utc() -> None:
    asset = _required(
        available_at=datetime(
            2026,
            7,
            19,
            9,
            tzinfo=ZoneInfo("America/Santiago"),
        )
    )

    assert asset.available_at == datetime(2026, 7, 19, 13, tzinfo=UTC)


@pytest.mark.parametrize(
    ("source_key", "sha256", "message"),
    [
        ("", "a" * 64, "source key"),
        ("UpperCase", "a" * 64, "source key"),
        ("has-dash", "a" * 64, "source key"),
        ("asset", "a" * 63, "SHA-256"),
        ("asset", "A" * 64, "SHA-256"),
        ("asset", "z" * 64, "SHA-256"),
    ],
)
def test_required_asset_rejects_invalid_identity_fields(
    source_key: str,
    sha256: str,
    message: str,
) -> None:
    with pytest.raises(DataContractError, match=message):
        _required(source_key, sha256)


def test_required_asset_rejects_naive_available_at() -> None:
    with pytest.raises(DataContractError, match="timezone"):
        _required(available_at=datetime(2026, 7, 19))


@pytest.mark.parametrize(
    ("released_at", "reason"),
    [
        (None, None),
        (date(2026, 7, 18), _MISSING_DATE),
    ],
)
def test_required_asset_enforces_publisher_date_reason_xor(
    released_at: date | None,
    reason: ReleaseMissingnessReason | None,
) -> None:
    with pytest.raises(DataContractError, match=r"exactly one|exclusive"):
        _required(released_at=released_at, reason=reason)


def test_required_asset_rejects_untyped_release_metadata() -> None:
    with pytest.raises(DataContractError, match="released_at"):
        _required(released_at="2026-07-18", reason=None)  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="missingness"):
        _required(reason="publisher_date_not_available")  # type: ignore[arg-type]


def test_manifest_available_at_is_latest_required_asset_time() -> None:
    manifest = _manifest(
        (
            _required(
                "pdf",
                "a" * 64,
                available_at=datetime(2026, 7, 7, tzinfo=UTC),
                released_at=date(2026, 7, 6),
                reason=None,
            ),
            _required(
                "sidecar",
                "b" * 64,
                available_at=datetime(2026, 7, 9, tzinfo=UTC),
            ),
        ),
        primary="pdf",
    )

    assert manifest.available_at == datetime(2026, 7, 9, tzinfo=UTC)
    assert manifest.primary_fact_asset.released_at == date(2026, 7, 6)


def test_composite_assets_keep_distinct_publisher_dates(
    composite_manifest_with_distinct_asset_dates: ReleaseManifest,
) -> None:
    manifest = composite_manifest_with_distinct_asset_dates

    assert manifest.primary_fact_asset.source_key == "workbook"
    assert manifest.primary_fact_asset.released_at == date(2026, 6, 30)
    sidecar = manifest.asset("entity_registry")
    assert sidecar.released_at is None
    assert sidecar.release_missingness_reason is _MISSING_DATE


def test_manifest_normalizes_declared_local_knowledge_time_to_utc() -> None:
    manifest = _manifest(
        (
            _required(
                available_at=datetime(
                    2026,
                    7,
                    19,
                    9,
                    tzinfo=ZoneInfo("America/Santiago"),
                ),
                released_at=date(2026, 7, 18),
                reason=None,
            ),
        )
    )

    assert manifest.available_at == datetime(2026, 7, 19, 13, tzinfo=UTC)


def test_manifest_sorts_assets_and_records_release_identity_immediately() -> None:
    registry = IdentityRegistry()
    manifest = _manifest(
        (
            _required("sidecar", "b" * 64),
            _required("asset", "a" * 64),
        ),
        registry=registry,
    )

    assert tuple(asset.source_key for asset in manifest.assets) == ("asset", "sidecar")
    assert registry.snapshot().require(manifest.release_id.value).canonical_payload == (
        manifest.release_id.canonical_payload
    )


def test_manifest_identity_uses_only_sorted_key_hash_pairs() -> None:
    first = _manifest(
        (
            _required(
                "asset",
                "a" * 64,
                available_at=datetime(2026, 7, 8, tzinfo=UTC),
                released_at=date(2026, 7, 7),
                reason=None,
            ),
            _required("sidecar", "b" * 64),
        )
    )
    second = _manifest(
        (
            _required(
                "sidecar",
                "b" * 64,
                available_at=datetime(2026, 8, 1, tzinfo=UTC),
            ),
            _required(
                "asset",
                "a" * 64,
                available_at=datetime(2026, 8, 2, tzinfo=UTC),
            ),
        )
    )
    expected = ReleaseIdFactory.v1((("asset", "a" * 64), ("sidecar", "b" * 64)))

    assert first.release_id == second.release_id == expected


@pytest.mark.parametrize("schema_version", ["", "   ", 1])
def test_manifest_rejects_invalid_schema_version(schema_version: object) -> None:
    with pytest.raises(DataContractError, match="schema_version"):
        ReleaseManifest.create(
            schema_version=schema_version,  # type: ignore[arg-type]
            assets=(_required(),),
            primary_fact_source_key="asset",
            identity_registry=IdentityRegistry(),
        )


def test_manifest_rejects_empty_or_mutable_asset_container() -> None:
    with pytest.raises(DataContractError, match="nonempty"):
        _manifest(())
    with pytest.raises(DataContractError, match="tuple"):
        ReleaseManifest.create(
            schema_version="official-release-v1",
            assets=[_required()],  # type: ignore[arg-type]
            primary_fact_source_key="asset",
            identity_registry=IdentityRegistry(),
        )


def test_manifest_rejects_non_asset_entry() -> None:
    with pytest.raises(DataContractError, match="RequiredReleaseAsset"):
        _manifest(("not-an-asset",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "assets",
    [
        (_required("asset", "a" * 64), _required("asset", "a" * 64)),
        (_required("asset", "a" * 64), _required("asset", "b" * 64)),
    ],
)
def test_manifest_rejects_duplicate_source_keys(
    assets: tuple[RequiredReleaseAsset, ...],
) -> None:
    with pytest.raises(DataContractError, match=r"duplicate|source key"):
        _manifest(assets)


@pytest.mark.parametrize("primary", ["", "missing", "UpperCase"])
def test_manifest_requires_exact_primary_fact_member(primary: str) -> None:
    with pytest.raises(DataContractError, match="primary"):
        _manifest(primary=primary)


def test_manifest_rejects_invalid_identity_registry() -> None:
    with pytest.raises(DataContractError, match="IdentityRegistry"):
        ReleaseManifest.create(
            schema_version="official-release-v1",
            assets=(_required(),),
            primary_fact_source_key="asset",
            identity_registry=object(),  # type: ignore[arg-type]
        )


def test_manifest_direct_constructor_cannot_bypass_factory() -> None:
    release_id = ReleaseIdFactory.v1((("asset", "a" * 64),))

    with pytest.raises(TypeError, match="create"):
        ReleaseManifest(  # type: ignore[call-arg]
            "official-release-v1",
            (_required(),),
            "asset",
            release_id,
            _NOW,
        )


def test_manifest_asset_rejects_unknown_key_without_leaking_members() -> None:
    with pytest.raises(DataContractError, match="unknown"):
        _manifest().asset("missing")


def test_verified_release_requires_factory() -> None:
    with pytest.raises(TypeError, match="verify_release_assets"):
        VerifiedReleaseAssets()  # type: ignore[call-arg]


def test_manifest_and_acquired_sets_must_match_exactly(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"x")
    acquired = _acquired(path, key="extra")

    with pytest.raises(DataContractError, match="exact"):
        verify_release_assets(_manifest(), (acquired,))


def test_verified_release_rejects_missing_acquired_asset(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"x")
    acquired = _acquired(path)
    manifest = _manifest(
        (
            _required("asset", acquired.sha256),
            _required("sidecar", "b" * 64),
        )
    )

    with pytest.raises(DataContractError, match="exact"):
        verify_release_assets(manifest, (acquired,))


def test_verified_release_rejects_extra_acquired_asset(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    acquired = _acquired(first)
    extra = _acquired(second, key="extra")
    manifest = _manifest((_required("asset", acquired.sha256),))

    with pytest.raises(DataContractError, match="exact"):
        verify_release_assets(manifest, (acquired, extra))


@pytest.mark.parametrize("conflicting_content", [False, True])
def test_verified_release_rejects_duplicate_acquired_keys(
    tmp_path: Path,
    conflicting_content: bool,
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"approved")
    second.write_bytes(b"changed" if conflicting_content else b"approved")
    acquired = _acquired(first)
    duplicate = _acquired(second)
    manifest = _manifest((_required("asset", acquired.sha256),))

    with pytest.raises(DataContractError, match=r"duplicate|exact"):
        verify_release_assets(manifest, (acquired, duplicate))


def test_verified_release_rejects_manifest_acquired_hash_conflict(
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"changed")
    acquired = _acquired(path)

    with pytest.raises(ChecksumMismatchError, match="asset"):
        verify_release_assets(
            _manifest((_required("asset", hashlib.sha256(b"approved").hexdigest()),)),
            (acquired,),
        )


def test_verified_release_rejects_invalid_acquired_sequence() -> None:
    with pytest.raises(DataContractError, match="sequence"):
        verify_release_assets(_manifest(), "asset")  # type: ignore[arg-type]
    with pytest.raises(DataContractError, match="AcquiredAsset"):
        verify_release_assets(_manifest(), ("asset",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key", []),
        ("key", None),
        ("key", ""),
        ("key", "   "),
        ("path", None),
        ("path", "asset.bin"),
        ("path", []),
        ("sha256", None),
        ("sha256", "a" * 63),
        ("sha256", "A" * 64),
        ("sha256", "z" * 64),
        ("retrieved_at", None),
        ("retrieved_at", "2026-07-19T00:00:00Z"),
        ("retrieved_at", datetime(2026, 7, 19)),
    ],
)
def test_verification_defensively_rejects_forged_acquired_fields(
    field: str,
    value: object,
) -> None:
    forged = _forged_acquired(**{field: value})

    with pytest.raises(DataContractError, match=field):
        verify_release_assets(_manifest(), (forged,))


@pytest.mark.parametrize("missing_field", ["key", "path", "sha256", "retrieved_at"])
def test_verification_defensively_rejects_deleted_acquired_slots(
    missing_field: str,
) -> None:
    forged = _forged_acquired(missing_field=missing_field)

    with pytest.raises(DataContractError, match=missing_field):
        verify_release_assets(_manifest(), (forged,))


def test_verification_rejects_forged_none_path_before_open() -> None:
    forged = _forged_acquired(path=None)

    with pytest.raises(DataContractError, match="path"):
        verify_release_assets(_manifest(), (forged,))


def test_verified_release_copies_private_paths_and_exposes_only_safe_api(
    verified_assets: VerifiedReleaseAssets,
) -> None:
    assert not hasattr(verified_assets, "__dict__")
    assert not hasattr(verified_assets, "acquired_assets")
    assert not hasattr(verified_assets, "assets")
    assert not hasattr(verified_assets, "paths")
    assert not isinstance(verified_assets._paths, dict)
    assert isinstance(verified_assets._paths, MappingProxyType)
    assert verified_assets.release_id == verified_assets.manifest.release_id
    assert verified_assets.available_at == verified_assets.manifest.available_at
    assert "asset.bin" not in repr(verified_assets)
    with pytest.raises(AttributeError):
        verified_assets._manifest = _manifest()  # type: ignore[misc]


def test_verified_release_recomputes_hash_before_parser_access(
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    path.write_bytes(b"tampered")

    with pytest.raises(ChecksumMismatchError), verified.open("asset"):
        pytest.fail("tampered bytes must never reach the parser")


def test_verified_release_rejects_replacement_after_verification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"tampered")
    replacement.replace(path)

    with pytest.raises(ChecksumMismatchError), verified.open("asset"):
        pytest.fail("replacement bytes must never reach the parser")


def test_verified_release_hashes_and_reads_the_same_open_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"tampered")
    real_lseek = os.lseek
    replaced = False

    def replace_at_rewind(fd: int, position: int, how: int) -> int:
        nonlocal replaced
        if position == 0 and how == os.SEEK_SET and not replaced:
            replacement.replace(path)
            replaced = True
        return real_lseek(fd, position, how)

    monkeypatch.setattr(releases_module.os, "lseek", replace_at_rewind)

    with verified.open("asset") as stream:
        assert stream.read() == b"approved"
    assert path.read_bytes() == b"tampered"


def test_verified_release_open_returns_read_only_descriptor_backed_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    real_open = os.open
    opened: list[int] = []

    def tracking_open(target: os.PathLike[str], flags: int) -> int:
        fd = real_open(target, flags)
        opened.append(fd)
        return fd

    monkeypatch.setattr(releases_module.os, "open", tracking_open)

    with verified.open("asset") as stream:
        assert stream.fileno() == opened[-1]
        assert stream.readable()
        assert not stream.writable()
        assert stream.read() == b"approved"
    with pytest.raises(OSError) as error:
        os.fstat(opened[-1])
    assert error.value.errno == errno.EBADF


def test_verified_release_closes_descriptor_on_checksum_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    path.write_bytes(b"tampered")
    real_open = os.open
    opened: list[int] = []

    def tracking_open(target: os.PathLike[str], flags: int) -> int:
        fd = real_open(target, flags)
        opened.append(fd)
        return fd

    monkeypatch.setattr(releases_module.os, "open", tracking_open)

    with pytest.raises(ChecksumMismatchError), verified.open("asset"):
        pass
    with pytest.raises(OSError) as error:
        os.fstat(opened[-1])
    assert error.value.errno == errno.EBADF


def test_verified_release_rejects_unknown_source_key(
    verified_assets: VerifiedReleaseAssets,
) -> None:
    with pytest.raises(DataContractError, match="unknown"), verified_assets.open("missing"):
        pass


def test_verified_release_rejects_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    path.unlink()

    with pytest.raises(MissingOfficialDataError, match="asset") as captured, verified.open("asset"):
        pass
    _assert_private_path_is_sanitized(captured.value, path)


@pytest.mark.parametrize("dangling", [False, True])
def test_verified_release_rejects_symlinks(
    tmp_path: Path,
    dangling: bool,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    path.unlink()
    target = tmp_path / "missing.bin" if dangling else tmp_path / "target.bin"
    if not dangling:
        target.write_bytes(b"approved")
    path.symlink_to(target)

    with pytest.raises(MissingOfficialDataError, match="asset") as captured, verified.open("asset"):
        pass
    _assert_private_path_is_sanitized(captured.value, path)


def test_verified_release_rejects_directory(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    path.unlink()
    path.mkdir()

    with (
        pytest.raises(MissingOfficialDataError, match="regular") as captured,
        verified.open("asset"),
    ):
        pass
    _assert_private_path_is_sanitized(captured.value, path)


def test_verified_release_sanitizes_open_error_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)

    def failed_open(_target: os.PathLike[str], _flags: int) -> int:
        raise OSError(f"private open failure: {path}")

    monkeypatch.setattr(releases_module.os, "open", failed_open)

    with pytest.raises(MissingOfficialDataError) as captured, verified.open("asset"):
        pass
    _assert_private_path_is_sanitized(captured.value, path)


def test_verified_release_sanitizes_fstat_error_chain_and_closes_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    real_open = os.open
    real_fstat = os.fstat
    opened: list[int] = []

    def tracking_open(target: os.PathLike[str], flags: int) -> int:
        descriptor = real_open(target, flags)
        opened.append(descriptor)
        return descriptor

    def failed_fstat(_descriptor: int) -> os.stat_result:
        raise OSError(f"private fstat failure: {path}")

    monkeypatch.setattr(releases_module.os, "open", tracking_open)
    monkeypatch.setattr(releases_module.os, "fstat", failed_fstat)

    with pytest.raises(MissingOfficialDataError) as captured, verified.open("asset"):
        pass
    _assert_private_path_is_sanitized(captured.value, path)
    with pytest.raises(OSError) as closed:
        real_fstat(opened[-1])
    assert closed.value.errno == errno.EBADF


def test_verified_release_sanitizes_read_error_chain_and_closes_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    real_open = os.open
    opened: list[int] = []

    def tracking_open(target: os.PathLike[str], flags: int) -> int:
        descriptor = real_open(target, flags)
        opened.append(descriptor)
        return descriptor

    def failed_hash(_stream: object) -> str:
        raise OSError(f"private read failure: {path}")

    monkeypatch.setattr(releases_module.os, "open", tracking_open)
    monkeypatch.setattr(releases_module, "sha256_stream", failed_hash)

    with pytest.raises(MissingOfficialDataError) as captured, verified.open("asset"):
        pass
    _assert_private_path_is_sanitized(captured.value, path)
    with pytest.raises(OSError) as closed:
        os.fstat(opened[-1])
    assert closed.value.errno == errno.EBADF


def test_verified_release_sanitizes_seek_error_chain_and_closes_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    real_open = os.open
    opened: list[int] = []

    def tracking_open(target: os.PathLike[str], flags: int) -> int:
        descriptor = real_open(target, flags)
        opened.append(descriptor)
        return descriptor

    def failed_seek(_descriptor: int, _position: int, _how: int) -> int:
        raise OSError(f"private seek failure: {path}")

    monkeypatch.setattr(releases_module.os, "open", tracking_open)
    monkeypatch.setattr(releases_module.os, "lseek", failed_seek)

    with pytest.raises(MissingOfficialDataError) as captured, verified.open("asset"):
        pass
    _assert_private_path_is_sanitized(captured.value, path)
    with pytest.raises(OSError) as closed:
        os.fstat(opened[-1])
    assert closed.value.errno == errno.EBADF


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="platform has no FIFO support")
def test_verified_release_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    path.unlink()
    os.mkfifo(path)

    with pytest.raises(MissingOfficialDataError, match="regular"), verified.open("asset"):
        pass


@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"),
    reason="platform has no O_NOFOLLOW",
)
def test_verified_release_uses_read_only_no_follow_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    real_open = os.open
    observed_flags: list[int] = []

    def tracking_open(target: os.PathLike[str], flags: int) -> int:
        observed_flags.append(flags)
        return real_open(target, flags)

    monkeypatch.setattr(releases_module.os, "open", tracking_open)

    with verified.open("asset") as stream:
        assert stream.read() == b"approved"
    assert observed_flags[-1] & os.O_NOFOLLOW
    assert observed_flags[-1] & os.O_ACCMODE == os.O_RDONLY


def test_verified_release_portable_fallback_compares_lstat_and_fstat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    real_lstat = os.lstat
    real_fstat = os.fstat
    lstats: list[Path] = []
    fstats: list[int] = []

    def tracking_lstat(target: os.PathLike[str]) -> os.stat_result:
        lstats.append(Path(target))
        return real_lstat(target)

    def tracking_fstat(fd: int) -> os.stat_result:
        fstats.append(fd)
        return real_fstat(fd)

    monkeypatch.setattr(releases_module, "_O_NOFOLLOW", None)
    monkeypatch.setattr(releases_module.os, "lstat", tracking_lstat)
    monkeypatch.setattr(releases_module.os, "fstat", tracking_fstat)

    with verified.open("asset") as stream:
        assert stream.read() == b"approved"
    assert lstats == [path]
    assert fstats


def test_verified_release_portable_fallback_rejects_replace_between_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"approved")
    real_open = os.open

    def replace_then_open(target: os.PathLike[str], flags: int) -> int:
        replacement.replace(path)
        return real_open(target, flags)

    monkeypatch.setattr(releases_module, "_O_NOFOLLOW", None)
    monkeypatch.setattr(releases_module.os, "open", replace_then_open)

    with pytest.raises(MissingOfficialDataError, match="changed"), verified.open("asset"):
        pass


def test_verified_release_portable_fallback_rejects_symlink_before_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"approved")
    path = tmp_path / "asset.bin"
    path.write_bytes(b"approved")
    verified = _verified(path)
    path.unlink()
    path.symlink_to(target)

    def forbidden_open(_target: os.PathLike[str], _flags: int) -> int:
        pytest.fail("portable fallback must reject a symlink before os.open")

    monkeypatch.setattr(releases_module, "_O_NOFOLLOW", None)
    monkeypatch.setattr(releases_module.os, "open", forbidden_open)

    with pytest.raises(MissingOfficialDataError, match="asset"), verified.open("asset"):
        pass


def test_finance_release_uses_primary_asset_publisher_metadata(
    verified_assets: VerifiedReleaseAssets,
) -> None:
    release = OfficialFinanceRelease(
        assets=verified_assets,
        vintage="2026",
        reference_start=date(2026, 1, 1),
        reference_end=date(2027, 1, 1),
    )

    assert release.released_at == verified_assets.manifest.primary_fact_asset.released_at
    assert release.release_missingness_reason is _MISSING_DATE


@pytest.mark.parametrize("vintage", ["", "   ", 2026])
def test_finance_release_rejects_invalid_vintage(
    verified_assets: VerifiedReleaseAssets,
    vintage: object,
) -> None:
    with pytest.raises(DataContractError, match="vintage"):
        OfficialFinanceRelease(
            assets=verified_assets,
            vintage=vintage,  # type: ignore[arg-type]
            reference_start=date(2026, 1, 1),
            reference_end=date(2027, 1, 1),
        )


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (date(2026, 1, 1), date(2026, 1, 1)),
        (date(2027, 1, 1), date(2026, 1, 1)),
    ],
)
def test_finance_release_requires_ordered_half_open_interval(
    verified_assets: VerifiedReleaseAssets,
    start: date,
    end: date,
) -> None:
    with pytest.raises(DataContractError, match=r"reference_start|half-open"):
        OfficialFinanceRelease(
            assets=verified_assets,
            vintage="2026",
            reference_start=start,
            reference_end=end,
        )


def test_finance_release_rejects_untyped_fields(
    verified_assets: VerifiedReleaseAssets,
) -> None:
    with pytest.raises(DataContractError, match="VerifiedReleaseAssets"):
        OfficialFinanceRelease(
            assets=object(),  # type: ignore[arg-type]
            vintage="2026",
            reference_start=date(2026, 1, 1),
            reference_end=date(2027, 1, 1),
        )
    with pytest.raises(DataContractError, match="date"):
        OfficialFinanceRelease(
            assets=verified_assets,
            vintage="2026",
            reference_start="2026-01-01",  # type: ignore[arg-type]
            reference_end=date(2027, 1, 1),
        )
