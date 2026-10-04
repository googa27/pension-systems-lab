from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

import chile_demographic_pde.countries.chile.market as market_module
from chile_demographic_pde.core.errors import (
    DataContractError,
    MissingOfficialDataError,
    SourceContractError,
)
from chile_demographic_pde.countries.chile.bcch_registry import (
    PRODUCTION_BCCH_REVIEWED_REGISTRY_V1,
    BcchRawContract,
    BcchReviewedRegistry,
    BcchReviewedSeriesEntry,
    BcchSeriesMetadata,
    BcchSeriesSpec,
    bcch_spec_digest,
    require_reviewed_bcch_entry,
)
from chile_demographic_pde.countries.chile.market import (
    BcchBdeAdapter,
    parse_bcch_get_series,
    parse_bcch_search_series,
)
from chile_demographic_pde.data.acquisition import AcquiredAsset, sha256_file
from chile_demographic_pde.data.domain_schemas import (
    MARKET_QUOTE_KEY,
    validate_market_quotes,
)
from chile_demographic_pde.data.identity import (
    IdentityRegistry,
    canonical_identity_bytes,
)
from chile_demographic_pde.data.releases import (
    OfficialFinanceRelease,
    ReleaseManifest,
    RequiredReleaseAsset,
    verify_release_assets,
)
from chile_demographic_pde.data.roles import ReleaseMissingnessReason

_FIXTURES = Path(__file__).with_name("fixtures")
_GET_KEY = "bcch_get_series"
_SEARCH_KEY = "bcch_search_series"
_GET_SHA256 = "c7f4255d68cfda8238ed17950a3783cfc4f5883264480c3537086789f1db81a0"
_SEARCH_SHA256 = "b3bc7e23e0e461fb53a13962802bb88b091582de3ae9d0a9a22dd2a5f68a2d19"
_SPEC_DIGEST = "cdbf110861ece7c24e726ee216359b5676b8449435061a7277d19d9f4d9d3012"
_NOW_GET = datetime(2026, 7, 18, 14, tzinfo=UTC)
_NOW_SEARCH = datetime(2026, 7, 18, 16, tzinfo=UTC)
_MISSING_RELEASE = ReleaseMissingnessReason.PUBLISHER_DATE_NOT_AVAILABLE


def _get_raw() -> bytes:
    return (_FIXTURES / "bcch_get_series_schema.json").read_bytes()


def _search_raw() -> bytes:
    return (_FIXTURES / "bcch_search_series_schema.json").read_bytes()


def _spec() -> BcchSeriesSpec:
    return BcchSeriesSpec(
        series_id="F022.BUF.TIS.AN10.UF.Z.D",
        variable="official_benchmark_yield",
        quote_type="benchmark_yield",
        tenor_years=10.0,
        currency="UF",
        indexation="UF",
        instrument_type="bcu_btu_benchmark_bond",
        compounding=None,
        day_count=None,
        source_unit="percent",
        canonical_unit="1 / year",
    )


def _raw_contract() -> BcchRawContract:
    return BcchRawContract(
        series_id="F022.BUF.TIS.AN10.UF.Z.D",
        frequency_code="DAILY",
        spanish_title="Tasa de interés de los bonos en UF a 10 años (porcentaje)",
        english_title="10-year UF-indexed bond interest rate (percentage)",
        first_observation=date(2002, 1, 2),
        last_observation=date(2026, 1, 3),
        updated_on=date(2026, 1, 4),
        created_on=date(2012, 3, 15),
        search_series_source_key=_SEARCH_KEY,
        search_series_sha256=_SEARCH_SHA256,
    )


def _reviewed_entry(
    *,
    spec: BcchSeriesSpec | None = None,
    get_raw: bytes | None = None,
    search_raw: bytes | None = None,
    spanish_title: str | None = None,
) -> BcchReviewedSeriesEntry:
    reviewed_spec = _spec() if spec is None else spec
    get_payload = _get_raw() if get_raw is None else get_raw
    search_payload = _search_raw() if search_raw is None else search_raw
    return BcchReviewedSeriesEntry(
        registry_version="bcch-reviewed-series-v1",
        series_id=reviewed_spec.series_id,
        spec=reviewed_spec,
        spec_digest=(_SPEC_DIGEST if spec is None else bcch_spec_digest(reviewed_spec)),
        frequency_code="DAILY",
        spanish_title=(
            "Tasa de interés de los bonos en UF a 10 años (porcentaje)"
            if spanish_title is None
            else spanish_title
        ),
        english_title="10-year UF-indexed bond interest rate (percentage)",
        first_observation=date(2002, 1, 2),
        last_observation=date(2026, 1, 3),
        updated_on=date(2026, 1, 4),
        created_on=date(2012, 3, 15),
        get_series_source_key=_GET_KEY,
        get_series_sha256=(
            _GET_SHA256 if get_raw is None else hashlib.sha256(get_payload).hexdigest()
        ),
        search_series_source_key=_SEARCH_KEY,
        search_series_sha256=(
            _SEARCH_SHA256 if search_raw is None else hashlib.sha256(search_payload).hexdigest()
        ),
        review_id="independent-review-2026-07-18",
    )


def _release(
    tmp_path: Path,
    identity_registry: IdentityRegistry,
    *,
    asset_set: str = "reviewed",
    get_raw: bytes | None = None,
    search_raw: bytes | None = None,
    primary_fact_source_key: str | None = None,
) -> OfficialFinanceRelease:
    tmp_path.mkdir(parents=True, exist_ok=True)
    get_payload = _get_raw() if get_raw is None else get_raw
    search_payload = _search_raw() if search_raw is None else search_raw
    payloads = {
        _GET_KEY: (get_payload, _NOW_GET),
        _SEARCH_KEY: (search_payload, _NOW_SEARCH),
    }
    if asset_set == "get_only":
        payloads.pop(_SEARCH_KEY)
    elif asset_set == "search_only":
        payloads.pop(_GET_KEY)
    elif asset_set == "extra_asset":
        payloads["unreviewed_sidecar"] = (b"{}", _NOW_SEARCH)
    elif asset_set != "reviewed":
        raise AssertionError(f"unknown test asset set {asset_set!r}")

    acquired: list[AcquiredAsset] = []
    required: list[RequiredReleaseAsset] = []
    for key, (payload, available_at) in payloads.items():
        path = tmp_path / f"{key}.json"
        path.write_bytes(payload)
        digest = sha256_file(path)
        acquired.append(
            AcquiredAsset(
                key=key,
                path=path,
                sha256=digest,
                retrieved_at=available_at,
            )
        )
        required.append(
            RequiredReleaseAsset(
                source_key=key,
                sha256=digest,
                available_at=available_at,
                released_at=None,
                release_missingness_reason=_MISSING_RELEASE,
            )
        )
    primary = (
        (_GET_KEY if _GET_KEY in payloads else _SEARCH_KEY)
        if primary_fact_source_key is None
        else primary_fact_source_key
    )
    manifest = ReleaseManifest.create(
        schema_version="bcch-bde-dual-json-v1",
        assets=tuple(required),
        primary_fact_source_key=primary,
        identity_registry=identity_registry,
    )
    return OfficialFinanceRelease(
        assets=verify_release_assets(manifest, tuple(acquired)),
        vintage="test-snapshot-2026-07-18",
        reference_start=date(2026, 1, 1),
        reference_end=date(2026, 1, 4),
    )


def _adapter_entry_registry() -> tuple[BcchReviewedSeriesEntry, BcchReviewedRegistry]:
    entry = _reviewed_entry()
    return entry, BcchReviewedRegistry.create(
        registry_version="bcch-reviewed-series-v1",
        entries=(entry,),
    )


def _json_mutation(raw: bytes, mutate: object) -> bytes:
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    assert callable(mutate)
    mutate(payload)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _monthly_index_get_raw() -> bytes:
    def mutate(payload: dict[str, object]) -> None:
        series = payload["Series"]
        assert isinstance(series, dict)
        series["seriesId"] = "F074.IPC.IND.TEST.M"
        series["descripEsp"] = "Índice de precios al consumidor"
        series["descripIng"] = "Consumer price index"
        observations = series["Obs"]
        assert isinstance(observations, list)
        for item, observed_on, value in zip(
            observations,
            ("01-01-2026", "01-02-2026", "01-03-2026"),
            ("100.0", "101.0", "102.0"),
            strict=True,
        ):
            assert isinstance(item, dict)
            item["indexDateString"] = observed_on
            item["value"] = value
            item["statusCode"] = "OK"

    return _json_mutation(_get_raw(), mutate)


def _monthly_index_search_raw() -> bytes:
    def mutate(payload: dict[str, object]) -> None:
        infos = payload["SeriesInfos"]
        assert isinstance(infos, list)
        matching = infos[0]
        assert isinstance(matching, dict)
        matching.update(
            {
                "seriesId": "F074.IPC.IND.TEST.M",
                "frequencyCode": "MONTHLY",
                "spanishTitle": "Índice de precios al consumidor",
                "englishTitle": "Consumer price index",
                "firstObservation": "01-01-2026",
                "lastObservation": "01-03-2026",
                "updatedAt": "02-04-2026",
                "createdAt": "01-01-2026",
            }
        )

    return _json_mutation(_search_raw(), mutate)


def test_candidate_series_ids_do_not_unlock_production() -> None:
    assert not PRODUCTION_BCCH_REVIEWED_REGISTRY_V1.entries
    with pytest.raises(MissingOfficialDataError):
        require_reviewed_bcch_entry("F022.BUF.TIS.AN10.UF.Z.D")


def test_unreviewed_lookup_never_echoes_credential_like_caller_input() -> None:
    canary = "user=alice&pass=TOPSECRET"

    with pytest.raises(MissingOfficialDataError) as captured:
        require_reviewed_bcch_entry(canary)

    assert canary not in str(captured.value)
    assert "TOPSECRET" not in repr(captured.value)


def test_recorded_parser_fixtures_have_independently_pinned_hashes() -> None:
    assert hashlib.sha256(_get_raw()).hexdigest() == _GET_SHA256
    assert hashlib.sha256(_search_raw()).hexdigest() == _SEARCH_SHA256


def test_production_registry_is_immutable_and_empty() -> None:
    assert isinstance(
        PRODUCTION_BCCH_REVIEWED_REGISTRY_V1.entries,
        type(MappingProxyType({})),
    )
    with pytest.raises(TypeError):
        PRODUCTION_BCCH_REVIEWED_REGISTRY_V1.entries["series"] = _reviewed_entry()  # type: ignore[index]


def test_spec_digest_commits_to_v1_typed_bytes_of_every_field() -> None:
    spec = _spec()
    expected_payload = {
        "schema_version": "bcch-series-spec-v1",
        "series_id": spec.series_id,
        "variable": spec.variable,
        "quote_type": spec.quote_type,
        "tenor_years": spec.tenor_years,
        "currency": spec.currency,
        "indexation": spec.indexation,
        "instrument_type": spec.instrument_type,
        "compounding": spec.compounding,
        "day_count": spec.day_count,
        "source_unit": spec.source_unit,
        "canonical_unit": spec.canonical_unit,
    }
    expected = hashlib.sha256(canonical_identity_bytes(expected_payload)).hexdigest()

    assert expected == _SPEC_DIGEST
    assert bcch_spec_digest(spec) == _SPEC_DIGEST
    assert bcch_spec_digest(replace(spec, compounding="annual_compounded")) != expected


def test_series_spec_rejects_cross_semantic_relabeling_and_control_text() -> None:
    with pytest.raises(DataContractError, match=r"variable|semantic|benchmark"):
        replace(_spec(), variable="official_swap_fixed_rate")
    with pytest.raises(DataContractError, match=r"canonical|control|variable"):
        replace(_spec(), variable="official_benchmark\nyield")


def test_registry_rejects_bad_spec_digest() -> None:
    with pytest.raises(DataContractError, match="digest"):
        BcchReviewedRegistry.create(
            registry_version="bcch-reviewed-series-v1",
            entries=(replace(_reviewed_entry(), spec_digest="0" * 64),),
        )


def test_registry_revalidates_entry_invariants_after_precreate_mutation() -> None:
    entry = _reviewed_entry()
    object.__setattr__(entry, "frequency_code", "WEEKLY")

    with pytest.raises(DataContractError, match=r"frequency|entry|reviewed"):
        BcchReviewedRegistry.create(
            registry_version="bcch-reviewed-series-v1",
            entries=(entry,),
        )


def test_registry_rejects_uninitialized_nominal_entry_with_typed_error() -> None:
    forged = object.__new__(BcchReviewedSeriesEntry)

    with pytest.raises(DataContractError):
        BcchReviewedRegistry.create(
            registry_version="bcch-reviewed-series-v1",
            entries=(forged,),
        )


def test_registry_translates_hostile_sequence_failure_without_swallowing_interrupts() -> None:
    class HostileSequence(Sequence[BcchReviewedSeriesEntry]):
        def __getitem__(self, index: int) -> BcchReviewedSeriesEntry:
            raise RuntimeError(f"hostile index {index}")

        def __len__(self) -> int:
            raise RuntimeError("hostile length")

        def __iter__(self) -> Iterator[BcchReviewedSeriesEntry]:
            raise RuntimeError("hostile iteration")

    with pytest.raises(DataContractError, match=r"snapshot|read|sequence"):
        BcchReviewedRegistry.create(
            registry_version="bcch-reviewed-series-v1",
            entries=HostileSequence(),
        )


def test_registry_supports_only_meaningful_collection_dunders() -> None:
    entry, registry = _adapter_entry_registry()

    assert len(registry) == 1
    assert entry.series_id in registry
    assert tuple(registry) == (entry,)
    assert "unknown" not in registry


def test_valid_get_series_parser_preserves_missing_observations() -> None:
    parsed = parse_bcch_get_series(_get_raw(), _raw_contract())

    assert parsed.series_id == _spec().series_id
    assert [item.observed_on for item in parsed.observations] == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
    ]
    assert parsed.observations[0].published_value == 4.5
    assert parsed.observations[1].published_value is None
    assert parsed.observations[2].published_value is None
    assert [item.status_code for item in parsed.observations] == ["OK", "ND", "ND"]


def test_valid_search_parser_keeps_all_metadata_dates_as_dates() -> None:
    metadata = parse_bcch_search_series(_search_raw(), _raw_contract())

    assert type(metadata.first_observation) is date
    assert type(metadata.last_observation) is date
    assert type(metadata.updated_on) is date
    assert type(metadata.created_on) is date
    assert metadata.updated_on == date(2026, 1, 4)
    assert metadata.source_key == _SEARCH_KEY
    assert metadata.sha256 == _SEARCH_SHA256


def test_public_bcch_metadata_value_validates_its_date_and_hash_atoms() -> None:
    with pytest.raises(DataContractError, match=r"updated|date"):
        BcchSeriesMetadata(
            source_key=_SEARCH_KEY,
            sha256=_SEARCH_SHA256,
            series_id=_spec().series_id,
            frequency_code="DAILY",
            spanish_title="Título",
            english_title="Title",
            first_observation=date(2026, 1, 1),
            last_observation=date(2026, 1, 2),
            updated_on=datetime(2026, 1, 3, tzinfo=UTC),  # type: ignore[arg-type]
            created_on=date(2026, 1, 1),
        )


def test_raw_contract_rejects_non_official_frequency() -> None:
    with pytest.raises(DataContractError, match="frequency"):
        replace(_raw_contract(), frequency_code="WEEKLY")


@pytest.mark.parametrize(
    ("raw_factory", "parser"),
    [
        (
            lambda: _json_mutation(
                _get_raw(),
                lambda payload: payload.__setitem__("Codigo", 12),
            ),
            parse_bcch_get_series,
        ),
        (
            lambda: _json_mutation(
                _search_raw(),
                lambda payload: payload.__setitem__("Codigo", 12),
            ),
            parse_bcch_search_series,
        ),
    ],
)
def test_nonzero_source_codes_emit_no_parsed_facts(
    raw_factory: object,
    parser: object,
) -> None:
    assert callable(raw_factory)
    assert callable(parser)
    with pytest.raises(SourceContractError, match="Codigo"):
        parser(raw_factory(), _raw_contract())


@pytest.mark.parametrize(
    "raw",
    [
        b'{"Codigo":0,"Codigo":1,"Descripcion":"x","Series":{},"SeriesInfos":[]}',
        b"\xff\xfe",
        b"{not-json}",
    ],
)
def test_parser_rejects_ambiguous_or_malformed_json(raw: bytes) -> None:
    with pytest.raises(SourceContractError):
        parse_bcch_get_series(raw, _raw_contract())


@pytest.mark.parametrize(
    "parser",
    [parse_bcch_get_series, parse_bcch_search_series],
)
def test_parsers_reject_uninitialized_nominal_contract_with_typed_error(
    parser: object,
) -> None:
    assert callable(parser)
    forged = object.__new__(BcchRawContract)

    with pytest.raises(SourceContractError):
        parser(_get_raw(), forged)


def test_parser_revalidates_raw_contract_after_mutation() -> None:
    contract = _raw_contract()
    object.__setattr__(contract, "frequency_code", "WEEKLY")

    with pytest.raises(SourceContractError, match="contract"):
        parse_bcch_get_series(_get_raw(), contract)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("unexpected", "field"),
        lambda payload: payload["Series"].__setitem__("unexpected", "field"),
        lambda payload: payload["Series"]["Obs"][0].__setitem__(
            "unexpected",
            "field",
        ),
        lambda payload: payload.pop("SeriesInfos"),
        lambda payload: payload.__setitem__("Codigo", False),
    ],
)
def test_get_parser_requires_the_exact_official_field_contract(
    mutate: object,
) -> None:
    raw = _json_mutation(_get_raw(), mutate)

    with pytest.raises(SourceContractError):
        parse_bcch_get_series(raw, _raw_contract())


@pytest.mark.parametrize(
    "bad_value",
    ["4,5", "NaN", "Infinity", "-Infinity", "1_000", "+4.5", "4.5 ", 4.5],
)
def test_get_parser_rejects_malformed_ok_numeric_values(
    bad_value: object,
) -> None:
    raw = _json_mutation(
        _get_raw(),
        lambda payload: payload["Series"]["Obs"][0].__setitem__(
            "value",
            bad_value,
        ),
    )

    with pytest.raises(SourceContractError, match=r"value|numeric|observation"):
        parse_bcch_get_series(raw, _raw_contract())


@pytest.mark.parametrize(
    "status_code",
    [" OK", "OK\n", "ND\t", "ND\nuser=alice&pass=TOPSECRET"],
)
def test_get_parser_rejects_noncanonical_status_without_echo(
    status_code: str,
) -> None:
    raw = _json_mutation(
        _get_raw(),
        lambda payload: payload["Series"]["Obs"][0].__setitem__(
            "statusCode",
            status_code,
        ),
    )

    with pytest.raises(SourceContractError) as captured:
        parse_bcch_get_series(raw, _raw_contract())

    assert status_code not in str(captured.value)
    assert "TOPSECRET" not in repr(captured.value)


def test_period_arithmetic_overflow_is_a_sanitized_source_contract_error() -> None:
    raw = _json_mutation(
        _get_raw(),
        lambda payload: payload["Series"]["Obs"][0].__setitem__(
            "indexDateString",
            "31-12-9999",
        ),
    )

    with pytest.raises(SourceContractError, match=r"period|date|calendar"):
        parse_bcch_get_series(raw, _raw_contract())


def test_get_parser_rejects_duplicate_dates() -> None:
    raw = _json_mutation(
        _get_raw(),
        lambda payload: payload["Series"]["Obs"].append(  # type: ignore[index]
            {
                "indexDateString": "01-01-2026",
                "value": "4.6",
                "statusCode": "OK",
            }
        ),
    )

    with pytest.raises(SourceContractError, match="duplicate"):
        parse_bcch_get_series(raw, _raw_contract())


@pytest.mark.parametrize(
    "invalid_date",
    ["1-01-2026", "01-1-2026", "2026-01-01", "31-02-2026"],
)
def test_get_parser_requires_exact_unambiguous_dates(invalid_date: str) -> None:
    raw = _json_mutation(
        _get_raw(),
        lambda payload: payload["Series"]["Obs"][0].__setitem__(  # type: ignore[index]
            "indexDateString",
            invalid_date,
        ),
    )

    with pytest.raises(SourceContractError, match="date"):
        parse_bcch_get_series(raw, _raw_contract())


def test_get_parser_accepts_only_documented_description_whitespace_normalization() -> None:
    equivalent = _json_mutation(
        _get_raw(),
        lambda payload: payload["Series"].__setitem__(  # type: ignore[index]
            "descripEsp",
            "  Tasa de interés de los bonos en UF a 10 años\n(porcentaje)  ",
        ),
    )
    changed = _json_mutation(
        _get_raw(),
        lambda payload: payload["Series"].__setitem__(  # type: ignore[index]
            "descripEsp",
            "Tasa de política monetaria",
        ),
    )

    parsed = parse_bcch_get_series(equivalent, _raw_contract())
    assert parsed.series_id == _spec().series_id
    with pytest.raises(SourceContractError, match=r"description|title"):
        parse_bcch_get_series(changed, _raw_contract())


def test_search_parser_requires_exactly_one_matching_series() -> None:
    duplicate = _json_mutation(
        _search_raw(),
        lambda payload: payload["SeriesInfos"].append(  # type: ignore[index]
            dict(payload["SeriesInfos"][0])  # type: ignore[index]
        ),
    )

    with pytest.raises(SourceContractError, match="exactly one"):
        parse_bcch_search_series(duplicate, _raw_contract())


@pytest.mark.parametrize(
    "invalid_date",
    ["2-01-2002", "02-1-2002", "2002-01-02", "31-02-2002"],
)
def test_search_parser_requires_exact_metadata_date_spelling(
    invalid_date: str,
) -> None:
    raw = _json_mutation(
        _search_raw(),
        lambda payload: payload["SeriesInfos"][0].__setitem__(
            "firstObservation",
            invalid_date,
        ),
    )

    with pytest.raises(SourceContractError, match=r"date|metadata|contract"):
        parse_bcch_search_series(raw, _raw_contract())


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("frequencyCode", "MONTHLY"),
        ("spanishTitle", "Título cambiado"),
        ("englishTitle", "Changed title"),
        ("firstObservation", "03-01-2002"),
        ("lastObservation", "02-01-2026"),
        ("updatedAt", "05-01-2026"),
        ("createdAt", "16-03-2012"),
    ],
)
def test_search_parser_checks_every_reviewed_metadata_field(
    field: str,
    changed: str,
) -> None:
    raw = _json_mutation(
        _search_raw(),
        lambda payload: payload["SeriesInfos"][0].__setitem__(field, changed),  # type: ignore[index]
    )

    with pytest.raises(SourceContractError, match=r"metadata|contract"):
        parse_bcch_search_series(raw, _raw_contract())


@pytest.mark.parametrize("asset_set", ["get_only", "search_only", "extra_asset"])
def test_bcch_requires_exact_reviewed_two_asset_release(
    asset_set: str,
    tmp_path: Path,
) -> None:
    entry, registry = _adapter_entry_registry()
    release_registry = IdentityRegistry()
    release = _release(
        tmp_path,
        release_registry,
        asset_set=asset_set,
    )

    with pytest.raises(MissingOfficialDataError):
        BcchBdeAdapter(registry).normalize_series(
            spec=_spec(),
            registry_entry=entry,
            release=release,
            identity_registry=release_registry,
        )


def test_bcch_requires_get_series_as_primary_fact_asset(tmp_path: Path) -> None:
    entry, registry = _adapter_entry_registry()
    identity_registry = IdentityRegistry()
    release = _release(
        tmp_path,
        identity_registry,
        primary_fact_source_key=_SEARCH_KEY,
    )

    with pytest.raises(MissingOfficialDataError, match="primary"):
        BcchBdeAdapter(registry).normalize_series(
            spec=_spec(),
            registry_entry=entry,
            release=release,
            identity_registry=identity_registry,
        )


def test_caller_created_equal_entry_is_not_authorization(tmp_path: Path) -> None:
    entry, registry = _adapter_entry_registry()
    copied_entry = replace(entry)
    identity_registry = IdentityRegistry()
    release = _release(tmp_path, identity_registry)

    assert copied_entry == entry
    assert copied_entry is not entry
    with pytest.raises(MissingOfficialDataError, match=r"authorized|registry"):
        BcchBdeAdapter(registry).normalize_series(
            spec=_spec(),
            registry_entry=copied_entry,
            release=release,
            identity_registry=identity_registry,
        )


def test_authorization_failure_occurs_before_verified_bytes_are_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, registry = _adapter_entry_registry()
    identity_registry = IdentityRegistry()
    release = _release(tmp_path, identity_registry)

    def forbidden_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("authorization failure attempted byte access")

    monkeypatch.setattr(type(release.assets), "open", forbidden_open)

    with pytest.raises(MissingOfficialDataError, match=r"authorized|registry"):
        BcchBdeAdapter(registry).normalize_series(
            spec=_spec(),
            registry_entry=replace(entry),
            release=release,
            identity_registry=identity_registry,
        )


def test_registry_snapshot_rejects_valid_post_creation_entry_rebinding(
    tmp_path: Path,
) -> None:
    entry, registry = _adapter_entry_registry()
    changed_title = "Título alterado después de la revisión"
    changed_get = _json_mutation(
        _get_raw(),
        lambda payload: payload["Series"].__setitem__(
            "descripEsp",
            changed_title,
        ),
    )
    changed_search = _json_mutation(
        _search_raw(),
        lambda payload: payload["SeriesInfos"][0].__setitem__(
            "spanishTitle",
            changed_title,
        ),
    )
    object.__setattr__(entry, "spanish_title", changed_title)
    object.__setattr__(
        entry,
        "get_series_sha256",
        hashlib.sha256(changed_get).hexdigest(),
    )
    object.__setattr__(
        entry,
        "search_series_sha256",
        hashlib.sha256(changed_search).hexdigest(),
    )
    identity_registry = IdentityRegistry()
    release = _release(
        tmp_path,
        identity_registry,
        get_raw=changed_get,
        search_raw=changed_search,
    )

    with pytest.raises(MissingOfficialDataError, match=r"changed|registry|reviewed"):
        BcchBdeAdapter(registry).normalize_series(
            spec=_spec(),
            registry_entry=entry,
            release=release,
            identity_registry=identity_registry,
        )


def test_registry_snapshot_revalidates_canonically_equal_subclass_rebinding_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TextSubclass(str):
        pass

    class DateSubclass(date):
        pass

    def forbidden_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("exact-type authorization attempted byte access")

    for ordinal, (field, changed) in enumerate(
        (
            (
                "spanish_title",
                TextSubclass("Tasa de interés de los bonos en UF a 10 años (porcentaje)"),
            ),
            ("first_observation", DateSubclass(2002, 1, 2)),
        )
    ):
        entry, registry = _adapter_entry_registry()
        object.__setattr__(entry, field, changed)
        identity_registry = IdentityRegistry()
        release = _release(tmp_path / str(ordinal), identity_registry)
        monkeypatch.setattr(type(release.assets), "open", forbidden_open)

        with pytest.raises(
            MissingOfficialDataError,
            match=r"changed|registry|reviewed",
        ):
            BcchBdeAdapter(registry).normalize_series(
                spec=_spec(),
                registry_entry=entry,
                release=release,
                identity_registry=identity_registry,
            )


def test_bcch_contract_rejects_metadata_copied_from_changed_sidecar(
    tmp_path: Path,
) -> None:
    entry, registry = _adapter_entry_registry()
    changed_title = "Título cambiado sin revisión independiente"
    changed_raw = _json_mutation(
        _search_raw(),
        lambda payload: payload["SeriesInfos"][0].__setitem__(  # type: ignore[index]
            "spanishTitle",
            changed_title,
        ),
    )
    tautological_entry = _reviewed_entry(
        search_raw=changed_raw,
        spanish_title=changed_title,
    )
    identity_registry = IdentityRegistry()
    release = _release(
        tmp_path,
        identity_registry,
        search_raw=changed_raw,
    )

    assert tautological_entry is not entry
    with pytest.raises(MissingOfficialDataError):
        BcchBdeAdapter(registry).normalize_series(
            spec=_spec(),
            registry_entry=tautological_entry,
            release=release,
            identity_registry=identity_registry,
        )


def test_caller_cannot_relabel_reviewed_bcch_series(tmp_path: Path) -> None:
    entry, registry = _adapter_entry_registry()
    relabeled = replace(
        _spec(),
        variable="official_swap_fixed_rate",
        quote_type="swap_fixed_rate",
        instrument_type="spc_uf_fixed_swap",
    )
    identity_registry = IdentityRegistry()
    release = _release(tmp_path, identity_registry)

    with pytest.raises(MissingOfficialDataError, match=r"spec|digest"):
        BcchBdeAdapter(registry).normalize_series(
            spec=relabeled,
            registry_entry=entry,
            release=release,
            identity_registry=identity_registry,
        )


def test_independently_reviewed_two_asset_test_release_normalizes(
    tmp_path: Path,
) -> None:
    entry, registry = _adapter_entry_registry()
    identity_registry = IdentityRegistry()
    release = _release(tmp_path, identity_registry)

    batch = BcchBdeAdapter(registry).normalize_series(
        spec=_spec(),
        registry_entry=entry,
        release=release,
        identity_registry=identity_registry,
    )
    observations = batch.observations

    validate_market_quotes(observations)
    assert tuple(column for column in MARKET_QUOTE_KEY if column in observations)
    assert set(observations["domain"]) == {"market_quote"}
    assert set(observations["observation_role"]) == {"market_quote"}
    assert list(observations["value"].iloc[:1]) == [0.045]
    assert observations["value"].iloc[1:].isna().all()
    assert list(observations["source_status_code"]) == ["OK", "ND", "ND"]
    assert set(observations["transformation_id"]) == {"bcch-published-percent-to-decimal-rate"}
    assert set(observations["transformation_version"]) == {"v1"}
    assert set(observations["unit"]) == {"1 / year"}
    assert set(observations["source_key"]) == {_GET_KEY}
    assert set(observations["date_basis"]) == {"market_observation"}
    assert list(observations["period_start"]) == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
    ]
    assert observations["available_at"].map(lambda value: value == _NOW_SEARCH).all()
    assert observations["released_at"].isna().all()
    assert set(observations["release_missingness_reason"]) == {_MISSING_RELEASE.value}
    assert len(batch.provenance.sources) == 2
    assert {source.source_key for source in batch.provenance.sources} == {
        _GET_KEY,
        _SEARCH_KEY,
    }
    rendered_sources = " ".join(str(source.url) for source in batch.provenance.sources)
    assert "user=" not in rendered_sources
    assert "pass=" not in rendered_sources
    assert "timeseries=" not in rendered_sources
    assert batch.provenance.available_at == _NOW_SEARCH
    assert np.isnan(observations["value"].iloc[2])


def test_bcch_fact_identity_is_stable_across_value_and_hash_revisions(
    tmp_path: Path,
) -> None:
    identity_registry = IdentityRegistry()
    first_entry, first_reviewed_registry = _adapter_entry_registry()
    first_release = _release(tmp_path / "first", identity_registry)
    first = (
        BcchBdeAdapter(first_reviewed_registry)
        .normalize_series(
            spec=_spec(),
            registry_entry=first_entry,
            release=first_release,
            identity_registry=identity_registry,
        )
        .observations
    )

    changed_get = _json_mutation(
        _get_raw(),
        lambda payload: payload["Series"]["Obs"][0].__setitem__("value", "4.6"),
    )
    second_entry = replace(
        _reviewed_entry(get_raw=changed_get),
        review_id="independent-review-2026-07-19",
    )
    second_reviewed_registry = BcchReviewedRegistry.create(
        registry_version="bcch-reviewed-series-v1",
        entries=(second_entry,),
    )
    second_release = _release(
        tmp_path / "second",
        identity_registry,
        get_raw=changed_get,
    )
    second = (
        BcchBdeAdapter(second_reviewed_registry)
        .normalize_series(
            spec=_spec(),
            registry_entry=second_entry,
            release=second_release,
            identity_registry=identity_registry,
        )
        .observations
    )

    assert list(second["fact_id"]) == list(first["fact_id"])
    assert set(second["release_id"]).isdisjoint(set(first["release_id"]))
    assert set(second["observation_id"]).isdisjoint(set(first["observation_id"]))
    assert second["value"].iloc[0] == 0.046
    assert first["value"].iloc[0] == 0.045


def test_failed_domain_validation_does_not_mutate_identity_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, registry = _adapter_entry_registry()
    identity_registry = IdentityRegistry()
    release = _release(tmp_path, identity_registry)
    before = identity_registry.snapshot()

    def fail_domain_validation(_frame: object) -> object:
        raise DataContractError("forced market-domain validation failure")

    monkeypatch.setattr(
        market_module,
        "validate_market_quotes",
        fail_domain_validation,
    )

    with pytest.raises(DataContractError, match="forced"):
        BcchBdeAdapter(registry).normalize_series(
            spec=_spec(),
            registry_entry=entry,
            release=release,
            identity_registry=identity_registry,
        )

    assert identity_registry.snapshot() == before


def test_monthly_bcch_index_uses_calendar_month_periods_without_rate_scaling(
    tmp_path: Path,
) -> None:
    spec = BcchSeriesSpec(
        series_id="F074.IPC.IND.TEST.M",
        variable="consumer_price_index",
        quote_type="published_index",
        tenor_years=None,
        currency="not_applicable",
        indexation="IPC",
        instrument_type="ipc_index",
        compounding=None,
        day_count=None,
        source_unit="index_points",
        canonical_unit="dimensionless",
    )
    get_raw = _monthly_index_get_raw()
    search_raw = _monthly_index_search_raw()
    entry = replace(
        _reviewed_entry(
            spec=spec,
            get_raw=get_raw,
            search_raw=search_raw,
        ),
        frequency_code="MONTHLY",
        spanish_title="Índice de precios al consumidor",
        english_title="Consumer price index",
        first_observation=date(2026, 1, 1),
        last_observation=date(2026, 3, 1),
        updated_on=date(2026, 4, 2),
        created_on=date(2026, 1, 1),
    )
    registry = BcchReviewedRegistry.create(
        registry_version="bcch-reviewed-series-v1",
        entries=(entry,),
    )
    identity_registry = IdentityRegistry()
    release = replace(
        _release(
            tmp_path,
            identity_registry,
            get_raw=get_raw,
            search_raw=search_raw,
        ),
        reference_end=date(2026, 4, 2),
    )

    observations = (
        BcchBdeAdapter(registry)
        .normalize_series(
            spec=spec,
            registry_entry=entry,
            release=release,
            identity_registry=identity_registry,
        )
        .observations
    )

    assert list(observations["period_end"]) == [
        date(2026, 2, 1),
        date(2026, 3, 1),
        date(2026, 4, 1),
    ]
    assert list(observations["value"]) == [100.0, 101.0, 102.0]
    assert set(observations["observation_role"]) == {"covariate"}
    assert observations["transformation_id"].isna().all()


def test_adapter_rejects_observations_outside_release_reference_interval(
    tmp_path: Path,
) -> None:
    entry, registry = _adapter_entry_registry()
    identity_registry = IdentityRegistry()
    release = replace(
        _release(tmp_path, identity_registry),
        reference_start=date(2026, 1, 2),
    )

    with pytest.raises(SourceContractError, match="reference"):
        BcchBdeAdapter(registry).normalize_series(
            spec=_spec(),
            registry_entry=entry,
            release=release,
            identity_registry=identity_registry,
        )
