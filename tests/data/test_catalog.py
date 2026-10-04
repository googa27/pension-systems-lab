from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from chile_demographic_pde.core.errors import (
    ChecksumMismatchError,
    DemographicPDEError,
    MissingOfficialDataError,
)
from chile_demographic_pde.data.catalog import (
    AcquisitionMode,
    DataSnapshot,
    OfficialAsset,
    SourceManifest,
)


def manual_asset(**changes: object) -> OfficialAsset:
    values: dict[str, object] = {
        "key": "censo_2024",
        "title": "Censo 2024 results",
        "publisher": "Instituto Nacional de Estadisticas",
        "landing_url": "https://censo2024.ine.gob.cl/resultados/",
        "acquisition": AcquisitionMode.MANUAL,
        "expected_filename": "censo_2024.xlsx",
        "vintage": "2024-final",
        "provisional": False,
    }
    values.update(changes)
    return OfficialAsset(**values)


def direct_asset(**changes: object) -> OfficialAsset:
    values: dict[str, object] = {
        "key": "ine_population",
        "title": "INE population estimates",
        "publisher": "Instituto Nacional de Estadisticas",
        "landing_url": "https://www.ine.gob.cl/estadisticas/",
        "acquisition": AcquisitionMode.DIRECT,
        "expected_filename": "population.xlsx",
        "vintage": "2024-final",
        "download_url": "https://www.ine.gob.cl/files/population.xlsx",
        "sha256": "a" * 64,
        "provisional": False,
    }
    values.update(changes)
    return OfficialAsset(**values)


def snapshot(**changes: object) -> DataSnapshot:
    values: dict[str, object] = {
        "snapshot_id": "chile-2024-final",
        "country": "CHL",
        "as_of": date(2026, 7, 18),
        "root": Path("data/snapshots/chile-2024-final"),
        "manifest_sha256": "b" * 64,
        "content_sha256": "c" * 64,
        "acquired_assets": ("censo_2024",),
        "available_assets": ("censo_2024", "ine_population"),
    }
    values.update(changes)
    return DataSnapshot(**values)


def test_domain_data_errors_inherit_base_error() -> None:
    assert issubclass(MissingOfficialDataError, DemographicPDEError)
    assert issubclass(ChecksumMismatchError, DemographicPDEError)


def test_direct_asset_requires_download_url_and_lowercase_sha256() -> None:
    with pytest.raises(ValidationError, match="download_url and sha256"):
        direct_asset(download_url=None, sha256=None)
    with pytest.raises(ValidationError):
        direct_asset(sha256="A" * 64)


@pytest.mark.parametrize(
    "filename",
    [
        "../censo.xlsx",
        "folder/censo.xlsx",
        r"folder\\censo.xlsx",
        r"C:\\data\\censo.xlsx",
        r"\\\\server\\share\\censo.xlsx",
        "C:censo.xlsx",
        ".",
        "censo.xlsx.",
        "censo.xlsx ",
        "censo\x00.xlsx",
        "censo\n.xlsx",
        "censo\t.xlsx",
        "CON",
        "PRN.csv",
        "AUX.txt",
        "NUL",
        "COM1.csv",
        "COM9",
        "LPT1.txt",
        "LPT9",
        "CON .txt",
        "COM1 .csv",
        "COM¹.txt",
        "LPT².csv",
        "CONIN$",
        "CONOUT$.txt",
        "censo<2024.xlsx",
        "censo>2024.xlsx",
        "censo:2024.xlsx",
        'censo"2024.xlsx',
        "censo|2024.xlsx",
        "censo?2024.xlsx",
        "censo*2024.xlsx",
        "censo\x7f.xlsx",
        "censo\x85.xlsx",
        "censo\x9f.xlsx",
    ],
)
def test_asset_filename_must_be_a_safe_basename(filename: str) -> None:
    with pytest.raises(ValidationError, match="safe basename"):
        manual_asset(expected_filename=filename)


@pytest.mark.parametrize("filename", ["Censo Ñandú 2024.xlsx", "tabla población.xlsx"])
def test_asset_filename_allows_safe_unicode_and_internal_spaces(filename: str) -> None:
    assert manual_asset(expected_filename=filename).expected_filename == filename


@pytest.mark.parametrize("field", ["title", "publisher", "vintage"])
def test_asset_text_metadata_cannot_be_blank(field: str) -> None:
    with pytest.raises(ValidationError):
        manual_asset(**{field: "   "})


def test_asset_requires_explicit_provisional_status() -> None:
    with pytest.raises(ValidationError, match="provisional"):
        OfficialAsset(
            key="censo_2024",
            title="Censo 2024 results",
            publisher="Instituto Nacional de Estadisticas",
            landing_url="https://censo2024.ine.gob.cl/resultados/",
            acquisition=AcquisitionMode.MANUAL,
            expected_filename="censo_2024.xlsx",
            vintage="2024-final",
        )


def test_manual_asset_can_omit_download_details_and_notes_are_immutable() -> None:
    asset = manual_asset(notes=["verified", "official"])
    assert asset.download_url is None
    assert asset.sha256 is None
    assert asset.notes == ("verified", "official")
    with pytest.raises(ValidationError):
        asset.title = "Changed"  # type: ignore[misc]


def test_asset_json_round_trip() -> None:
    asset = direct_asset(notes=("source release",))
    assert OfficialAsset.model_validate_json(asset.model_dump_json()) == asset


def test_manifest_requires_assets_unique_keys_and_uppercase_country() -> None:
    asset = manual_asset()
    with pytest.raises(ValidationError):
        SourceManifest(country="CHL", assets=())
    with pytest.raises(ValidationError, match="unique"):
        SourceManifest(country="CHL", assets=(asset, asset))
    with pytest.raises(ValidationError):
        SourceManifest(country="chl", assets=(asset,))


def test_manifest_returns_asset_or_contextual_key_error() -> None:
    manifest = SourceManifest(country="CHL", assets=(manual_asset(),))
    assert manifest.asset("censo_2024") == manual_asset()
    with pytest.raises(KeyError, match="censo_missing"):
        manifest.asset("censo_missing")


def test_manifest_json_round_trip() -> None:
    manifest = SourceManifest(country="CHL", assets=(manual_asset(), direct_asset()))
    assert SourceManifest.model_validate_json(manifest.model_dump_json()) == manifest


def test_snapshot_requires_lowercase_hashes_and_nonblank_identity() -> None:
    with pytest.raises(ValidationError):
        snapshot(manifest_sha256="B" * 64)
    with pytest.raises(ValidationError):
        snapshot(content_sha256="short")
    with pytest.raises(ValidationError):
        snapshot(snapshot_id="   ")
    with pytest.raises(ValidationError):
        snapshot(country="chl")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("acquired_assets", ("censo_2024", "censo_2024")),
        ("available_assets", ("censo_2024", "censo_2024")),
        ("unavailable_optional_assets", ("ine_population", "ine_population")),
    ],
)
def test_snapshot_asset_tuple_fields_cannot_contain_duplicates(
    field: str, value: tuple[str, str]
) -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        snapshot(**{field: value})


def test_snapshot_requires_acquired_subset_and_optional_unavailable_disjointness() -> None:
    with pytest.raises(ValidationError, match="subset"):
        snapshot(acquired_assets=("missing",))
    with pytest.raises(ValidationError, match="disjoint"):
        snapshot(unavailable_optional_assets=("ine_population",))


def test_snapshot_json_round_trip() -> None:
    result = snapshot(unavailable_optional_assets=("future_projection",))
    assert DataSnapshot.model_validate_json(result.model_dump_json()) == result
