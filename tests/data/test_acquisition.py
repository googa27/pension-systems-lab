from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pooch
import pytest

from chile_demographic_pde.core.errors import (
    ChecksumMismatchError,
    DataContractError,
    MissingOfficialDataError,
)
from chile_demographic_pde.data.acquisition import (
    AcquiredAsset,
    acquire_asset,
    sha256_file,
    sha256_stream,
)
from chile_demographic_pde.data.catalog import AcquisitionMode, OfficialAsset


def _manual_asset(checksum: str | None = None) -> OfficialAsset:
    return OfficialAsset(
        key="deis",
        title="DEIS deaths",
        publisher="DEIS",
        landing_url="https://deis.minsal.cl/",
        acquisition=AcquisitionMode.MANUAL,
        expected_filename="deis.csv",
        vintage="2024-preliminary",
        sha256=checksum,
        provisional=True,
    )


def _direct_asset(checksum: str) -> OfficialAsset:
    return OfficialAsset(
        key="ine_population",
        title="INE population",
        publisher="INE",
        landing_url="https://www.ine.gob.cl/",
        acquisition=AcquisitionMode.DIRECT,
        expected_filename="population.csv",
        vintage="2024-final",
        download_url="https://www.ine.gob.cl/population.csv",
        sha256=checksum,
        provisional=False,
    )


def test_sha256_file_reads_fixed_size_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"a" * ((1 << 20) + 5)
    reads: list[int] = []

    class TrackingStream(BytesIO):
        def read(self, size: int = -1) -> bytes:
            reads.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda _path, _mode: TrackingStream(content))

    assert sha256_file(tmp_path / "official.csv") == hashlib.sha256(content).hexdigest()
    assert reads == [1 << 20, 1 << 20, 1 << 20]


def test_sha256_stream_hashes_the_supplied_descriptor_without_reopening() -> None:
    content = b"descriptor-backed bytes"
    stream = BytesIO(content)

    assert sha256_stream(stream) == hashlib.sha256(content).hexdigest()
    assert stream.tell() == len(content)


def test_missing_manual_asset_raises_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(MissingOfficialDataError) as error:
        acquire_asset(_manual_asset(), tmp_path)

    message = str(error.value)
    assert "deis.csv" in message
    assert str(tmp_path) in message
    assert "https://deis.minsal.cl/" in message


def test_manual_file_without_declared_checksum_returns_observed_checksum(tmp_path: Path) -> None:
    content = b"official manual excerpt"
    path = tmp_path / "deis.csv"
    path.write_bytes(content)

    acquired = acquire_asset(_manual_asset(), tmp_path)

    assert acquired.path == path
    assert acquired.sha256 == hashlib.sha256(content).hexdigest()


def test_manual_file_with_declared_checksum_is_verified(tmp_path: Path) -> None:
    content = b"official manual excerpt"
    path = tmp_path / "deis.csv"
    path.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()

    acquired = acquire_asset(_manual_asset(checksum), tmp_path)

    assert acquired.sha256 == checksum


def test_declared_local_checksum_mismatch_includes_observed_and_expected(tmp_path: Path) -> None:
    path = tmp_path / "deis.csv"
    path.write_bytes(b"modified")
    expected = hashlib.sha256(b"official").hexdigest()
    observed = hashlib.sha256(b"modified").hexdigest()

    with pytest.raises(ChecksumMismatchError) as error:
        acquire_asset(_manual_asset(expected), tmp_path)

    assert observed in str(error.value)
    assert expected in str(error.value)


def test_non_file_at_expected_destination_raises_official_data_error(tmp_path: Path) -> None:
    (tmp_path / "deis.csv").mkdir()

    with pytest.raises(MissingOfficialDataError, match="regular file"):
        acquire_asset(_manual_asset(), tmp_path)


def test_dangling_direct_destination_symlink_is_rejected_before_pooch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checksum = hashlib.sha256(b"official").hexdigest()
    (tmp_path / "population.csv").symlink_to(tmp_path / "missing.csv")

    def retrieve(**_kwargs: object) -> str:
        pytest.fail("a dangling cache symlink must not trigger Pooch")

    monkeypatch.setattr(pooch, "retrieve", retrieve)

    with pytest.raises(MissingOfficialDataError, match="regular file"):
        acquire_asset(_direct_asset(checksum), tmp_path)


def test_symlink_to_regular_manual_file_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "official.csv"
    target.write_bytes(b"official")
    (tmp_path / "deis.csv").symlink_to(target)

    with pytest.raises(MissingOfficialDataError, match="regular file"):
        acquire_asset(_manual_asset(), tmp_path)


def test_direct_acquisition_passes_pinned_pooch_arguments_and_rechecks_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"pinned official download"
    checksum = hashlib.sha256(content).hexdigest()
    calls: list[dict[str, object]] = []

    def retrieve(**kwargs: object) -> str:
        calls.append(kwargs)
        destination = Path(str(kwargs["path"])) / str(kwargs["fname"])
        destination.write_bytes(content)
        return str(destination)

    monkeypatch.setattr(pooch, "retrieve", retrieve)

    acquired = acquire_asset(_direct_asset(checksum), tmp_path, progressbar=True)

    assert calls == [
        {
            "url": "https://www.ine.gob.cl/population.csv",
            "known_hash": f"sha256:{checksum}",
            "fname": "population.csv",
            "path": tmp_path,
            "progressbar": True,
        }
    ]
    assert acquired.path == tmp_path / "population.csv"
    assert acquired.sha256 == checksum


def test_direct_acquisition_rejects_returned_file_that_fails_independent_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checksum = hashlib.sha256(b"official").hexdigest()

    def retrieve(**kwargs: object) -> str:
        destination = Path(str(kwargs["path"])) / str(kwargs["fname"])
        destination.write_bytes(b"changed")
        return str(destination)

    monkeypatch.setattr(pooch, "retrieve", retrieve)

    with pytest.raises(ChecksumMismatchError, match=checksum):
        acquire_asset(_direct_asset(checksum), tmp_path)


def test_downloaded_hash_value_error_is_translated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checksum = hashlib.sha256(b"official").hexdigest()

    def retrieve(**_kwargs: object) -> str:
        raise ValueError("Hash of downloaded file does not match known hash")

    monkeypatch.setattr(pooch, "retrieve", retrieve)

    with pytest.raises(ChecksumMismatchError, match=checksum):
        acquire_asset(_direct_asset(checksum), tmp_path)


def test_unrelated_retrieval_errors_are_not_mislabeled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checksum = hashlib.sha256(b"official").hexdigest()

    def retrieve(**_kwargs: object) -> str:
        raise RuntimeError("transport unavailable")

    monkeypatch.setattr(pooch, "retrieve", retrieve)

    with pytest.raises(RuntimeError, match="transport unavailable"):
        acquire_asset(_direct_asset(checksum), tmp_path)


def test_unrelated_value_error_with_hash_words_is_not_mislabeled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checksum = hashlib.sha256(b"official").hexdigest()

    def retrieve(**_kwargs: object) -> str:
        raise ValueError("cache hash match metadata is malformed")

    monkeypatch.setattr(pooch, "retrieve", retrieve)

    with pytest.raises(ValueError, match="metadata is malformed"):
        acquire_asset(_direct_asset(checksum), tmp_path)


def test_direct_cached_checksum_mismatch_does_not_call_pooch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = hashlib.sha256(b"official").hexdigest()
    (tmp_path / "population.csv").write_bytes(b"changed")

    def retrieve(**_kwargs: object) -> str:
        pytest.fail("a mismatching cache entry must not be replaced")

    monkeypatch.setattr(pooch, "retrieve", retrieve)

    with pytest.raises(ChecksumMismatchError):
        acquire_asset(_direct_asset(expected), tmp_path)


def test_acquisition_uses_injected_aware_clock(tmp_path: Path) -> None:
    path = tmp_path / "deis.csv"
    path.write_bytes(b"official")
    timestamp = datetime(2026, 7, 19, 12, 30, tzinfo=UTC)

    acquired = acquire_asset(_manual_asset(), tmp_path, clock=lambda: timestamp)

    assert acquired.retrieved_at == timestamp


def test_acquisition_rejects_naive_clock(tmp_path: Path) -> None:
    path = tmp_path / "deis.csv"
    path.write_bytes(b"official")

    with pytest.raises(ValueError, match="timezone-aware"):
        acquire_asset(_manual_asset(), tmp_path, clock=lambda: datetime(2026, 7, 19))


def test_acquired_asset_is_frozen_value_object() -> None:
    timestamp = datetime(2026, 7, 19, tzinfo=UTC)
    first = AcquiredAsset("deis", Path("deis.csv"), "a" * 64, timestamp)
    second = AcquiredAsset("deis", Path("deis.csv"), "a" * 64, timestamp)

    assert first == second
    assert hash(first) == hash(second)
    with pytest.raises(AttributeError):
        first.key = "other"  # type: ignore[misc]


def test_acquired_asset_rejects_non_hexadecimal_digest() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        AcquiredAsset(
            "deis",
            Path("deis.csv"),
            "z" * 64,
            datetime(2026, 7, 19, tzinfo=UTC),
        )


@pytest.mark.parametrize("key", ["", "   ", [], None])
def test_acquired_asset_rejects_invalid_key_type_or_blank_value(
    key: object,
) -> None:
    with pytest.raises(DataContractError, match="key"):
        AcquiredAsset(
            key=key,  # type: ignore[arg-type]
            path=Path("deis.csv"),
            sha256="a" * 64,
            retrieved_at=datetime(2026, 7, 19, tzinfo=UTC),
        )


@pytest.mark.parametrize("path", [None, "deis.csv", [], 1])
def test_acquired_asset_rejects_non_path_values(path: object) -> None:
    with pytest.raises(DataContractError, match="path"):
        AcquiredAsset(
            key="deis",
            path=path,  # type: ignore[arg-type]
            sha256="a" * 64,
            retrieved_at=datetime(2026, 7, 19, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    "retrieved_at",
    [None, "2026-07-19T00:00:00Z", date(2026, 7, 19)],
)
def test_acquired_asset_rejects_invalid_retrieved_at_type(
    retrieved_at: object,
) -> None:
    with pytest.raises(DataContractError, match="retrieved_at"):
        AcquiredAsset(
            key="deis",
            path=Path("deis.csv"),
            sha256="a" * 64,
            retrieved_at=retrieved_at,  # type: ignore[arg-type]
        )


def test_acquired_asset_accepts_path_value_and_normalizes_timestamp_to_utc() -> None:
    local_time = datetime(
        2026,
        7,
        19,
        9,
        tzinfo=ZoneInfo("America/Santiago"),
    )

    acquired = AcquiredAsset(
        key="deis",
        path=Path("deis.csv"),
        sha256="a" * 64,
        retrieved_at=local_time,
    )

    assert acquired.path == Path("deis.csv")
    assert acquired.retrieved_at == datetime(2026, 7, 19, 13, tzinfo=UTC)
