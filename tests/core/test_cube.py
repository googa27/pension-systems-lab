from datetime import UTC, date, datetime

import pytest
import xarray as xr
from pydantic import ValidationError

from chile_demographic_pde.core.cube import DemographicCube
from chile_demographic_pde.core.errors import (
    DataContractError,
    PopulationBasisError,
    UnitMismatchError,
)
from chile_demographic_pde.core.provenance import SourceRef, VariableProvenance
from chile_demographic_pde.core.semantics import (
    SEMANTIC_COMPATIBILITY,
    PopulationBasis,
    SemanticKind,
    Unit,
    assert_semantic_compatible,
)

COMPATIBLE_METADATA = [
    (SemanticKind.COUNT, Unit.PERSON, PopulationBasis.CENSUS_ENUMERATED),
    (SemanticKind.COUNT, Unit.PERSON, PopulationBasis.RESIDENT_ESTIMATE),
    (SemanticKind.COUNT, Unit.PERSON, PopulationBasis.PENSIONER),
    (SemanticKind.COUNT, Unit.EVENT, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.COUNT, Unit.EVENT, PopulationBasis.CENSUS_ENUMERATED),
    (SemanticKind.COUNT, Unit.EVENT, PopulationBasis.RESIDENT_ESTIMATE),
    (SemanticKind.COUNT, Unit.EVENT, PopulationBasis.PENSIONER),
    (
        SemanticKind.EXPOSURE,
        Unit.PERSON_YEAR,
        PopulationBasis.RESIDENT_ESTIMATE,
    ),
    (SemanticKind.EXPOSURE, Unit.PERSON_YEAR, PopulationBasis.PERSON_TIME),
    (SemanticKind.EXPOSURE, Unit.PERSON_YEAR, PopulationBasis.PENSIONER),
    (
        SemanticKind.POPULATION,
        Unit.PERSON,
        PopulationBasis.CENSUS_ENUMERATED,
    ),
    (
        SemanticKind.POPULATION,
        Unit.PERSON,
        PopulationBasis.RESIDENT_ESTIMATE,
    ),
    (SemanticKind.POPULATION, Unit.PERSON, PopulationBasis.PENSIONER),
    (SemanticKind.RATE, Unit.PER_YEAR, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.RATE, Unit.PER_YEAR, PopulationBasis.CENSUS_ENUMERATED),
    (SemanticKind.RATE, Unit.PER_YEAR, PopulationBasis.RESIDENT_ESTIMATE),
    (SemanticKind.RATE, Unit.PER_YEAR, PopulationBasis.PERSON_TIME),
    (SemanticKind.RATE, Unit.PER_YEAR, PopulationBasis.PENSIONER),
    (SemanticKind.RATE, Unit.PER_MONTH, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.RATE, Unit.PER_MONTH, PopulationBasis.CENSUS_ENUMERATED),
    (SemanticKind.RATE, Unit.PER_MONTH, PopulationBasis.RESIDENT_ESTIMATE),
    (SemanticKind.RATE, Unit.PER_MONTH, PopulationBasis.PERSON_TIME),
    (SemanticKind.RATE, Unit.PER_MONTH, PopulationBasis.PENSIONER),
    (
        SemanticKind.PROBABILITY,
        Unit.DIMENSIONLESS,
        PopulationBasis.NOT_APPLICABLE,
    ),
    (
        SemanticKind.PROBABILITY,
        Unit.DIMENSIONLESS,
        PopulationBasis.RESIDENT_ESTIMATE,
    ),
    (
        SemanticKind.PROBABILITY,
        Unit.DIMENSIONLESS,
        PopulationBasis.PENSIONER,
    ),
    (SemanticKind.PROBABILITY, Unit.PERCENT, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.PROBABILITY, Unit.PERCENT, PopulationBasis.RESIDENT_ESTIMATE),
    (SemanticKind.PROBABILITY, Unit.PERCENT, PopulationBasis.PENSIONER),
    (SemanticKind.PRICE, Unit.CLP, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.PRICE, Unit.UF, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.AMOUNT, Unit.CLP, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.AMOUNT, Unit.UF, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.SIGNED_AMOUNT, Unit.CLP, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.SIGNED_AMOUNT, Unit.UF, PopulationBasis.NOT_APPLICABLE),
    (
        SemanticKind.DISCOUNT_FACTOR,
        Unit.DIMENSIONLESS,
        PopulationBasis.NOT_APPLICABLE,
    ),
    (SemanticKind.INDEX, Unit.DIMENSIONLESS, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.INDEX, Unit.PERCENT, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.INDEX, Unit.DAY, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.INDEX, Unit.DEGREE_CELSIUS, PopulationBasis.NOT_APPLICABLE),
    (SemanticKind.INDEX, Unit.MILLIMETER, PopulationBasis.NOT_APPLICABLE),
    (
        SemanticKind.INDEX,
        Unit.MICROGRAM_PER_CUBIC_METER,
        PopulationBasis.NOT_APPLICABLE,
    ),
]


def _source() -> SourceRef:
    return SourceRef(
        source_key="ine_fixture",
        name="INE fixture",
        url="https://www.ine.gob.cl/",
        retrieved_at=datetime(2026, 7, 18, tzinfo=UTC),
        sha256="a" * 64,
        vintage="2024-final",
        release_date=date(2025, 3, 28),
        release_missingness_reason=None,
        provisional=False,
    )


def _provenance(
    *,
    dimensions: tuple[str, ...] = ("age",),
) -> VariableProvenance:
    return VariableProvenance(
        sources=(_source(),),
        release_id="cldemopde:release:v1:" + "b" * 64,
        available_at=datetime(2026, 7, 18, tzinfo=UTC),
        observation_start=date(2024, 4, 1),
        observation_end=date(2024, 4, 30),
        dimensions=dimensions,
        aggregation_rules=("sum within published age bins",),
        missingness_reason="not_applicable_complete",
    )


def _annotated_data(
    values: list[int],
    *,
    ages: list[int] | None = None,
) -> xr.DataArray:
    data = xr.DataArray(
        values,
        dims=("age",),
        coords={"age": ages if ages is not None else list(range(len(values)))},
    )
    data.attrs.update(
        semantic_kind=SemanticKind.COUNT.value,
        unit=Unit.PERSON.value,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE.value,
        provenance=_provenance().model_dump(mode="json"),
    )
    return data


def _cube() -> DemographicCube:
    return DemographicCube(xr.Dataset({"deaths": _annotated_data([1, 2])}))


def test_cube_requires_semantics_and_provenance_for_every_variable() -> None:
    dataset = xr.Dataset({"deaths": ("age", [1, 2])}, coords={"age": [0, 1]})

    with pytest.raises(DataContractError, match="deaths"):
        DemographicCube(dataset)


@pytest.mark.parametrize(
    ("attribute", "invalid_value"),
    [
        ("semantic_kind", "not-a-semantic-kind"),
        ("unit", "not-a-unit"),
        ("population_basis", "not-a-population-basis"),
        ("provenance", {"sources": []}),
    ],
)
def test_cube_wraps_invalid_typed_metadata_with_variable_context(
    attribute: str,
    invalid_value: object,
) -> None:
    data = _annotated_data([1, 2])
    data.attrs[attribute] = invalid_value

    with pytest.raises(DataContractError, match="deaths"):
        DemographicCube(xr.Dataset({"deaths": data}))


@pytest.mark.parametrize(
    ("kind", "unit", "basis"),
    COMPATIBLE_METADATA,
)
def test_cube_accepts_every_compatible_semantic_contract(
    kind: SemanticKind,
    unit: Unit,
    basis: PopulationBasis,
) -> None:
    data = _annotated_data([1, 2])
    data.attrs.update(
        semantic_kind=kind.value,
        unit=unit.value,
        population_basis=basis.value,
    )

    cube = DemographicCube(xr.Dataset({"variable": data}))

    assert cube.dataset["variable"].attrs["semantic_kind"] == kind.value


def test_compatible_metadata_fixture_exactly_matches_the_production_table() -> None:
    production_metadata = {
        (kind, unit, basis)
        for kind, unit_contracts in SEMANTIC_COMPATIBILITY.items()
        for unit, bases in unit_contracts.items()
        for basis in bases
    }

    assert set(SEMANTIC_COMPATIBILITY) == set(SemanticKind)
    assert production_metadata == set(COMPATIBLE_METADATA)


def test_semantic_validator_preserves_domain_specific_error_classes() -> None:
    with pytest.raises(UnitMismatchError):
        assert_semantic_compatible(
            SemanticKind.COUNT,
            Unit.PERCENT,
            PopulationBasis.NOT_APPLICABLE,
        )
    with pytest.raises(PopulationBasisError):
        assert_semantic_compatible(
            SemanticKind.COUNT,
            Unit.PERSON,
            PopulationBasis.NOT_APPLICABLE,
        )


@pytest.mark.parametrize(
    ("kind", "unit"),
    [
        (SemanticKind.RATE, Unit.PERSON),
        (SemanticKind.EXPOSURE, Unit.PER_YEAR),
        (SemanticKind.COUNT, Unit.PERCENT),
        (SemanticKind.POPULATION, Unit.EVENT),
        (SemanticKind.PROBABILITY, Unit.CLP),
        (SemanticKind.PRICE, Unit.DIMENSIONLESS),
        (SemanticKind.DISCOUNT_FACTOR, Unit.PERCENT),
        (SemanticKind.INDEX, Unit.PER_MONTH),
    ],
)
def test_cube_rejects_incompatible_semantic_kind_and_unit(
    kind: SemanticKind,
    unit: Unit,
) -> None:
    data = _annotated_data([1, 2])
    data.attrs.update(semantic_kind=kind.value, unit=unit.value)

    with pytest.raises(DataContractError, match=r"variable.*unit"):
        DemographicCube(xr.Dataset({"variable": data}))


@pytest.mark.parametrize(
    ("kind", "unit"),
    [
        (SemanticKind.RATE, Unit.PERSON),
        (SemanticKind.EXPOSURE, Unit.PER_YEAR),
    ],
)
def test_with_variable_rejects_incompatible_semantic_kind_and_unit(
    kind: SemanticKind,
    unit: Unit,
) -> None:
    with pytest.raises(DataContractError, match=r"births.*unit"):
        _cube().with_variable(
            "births",
            xr.DataArray([3, 4], dims=("age",), coords={"age": [0, 1]}),
            kind=kind,
            unit=unit,
            population_basis=PopulationBasis.RESIDENT_ESTIMATE,
            provenance=_provenance(),
        )


@pytest.mark.parametrize(
    ("kind", "unit", "basis"),
    [
        (
            SemanticKind.COUNT,
            Unit.PERSON,
            PopulationBasis.NOT_APPLICABLE,
        ),
        (
            SemanticKind.EXPOSURE,
            Unit.PERSON_YEAR,
            PopulationBasis.CENSUS_ENUMERATED,
        ),
        (
            SemanticKind.POPULATION,
            Unit.PERSON,
            PopulationBasis.PERSON_TIME,
        ),
        (
            SemanticKind.PROBABILITY,
            Unit.DIMENSIONLESS,
            PopulationBasis.CENSUS_ENUMERATED,
        ),
        (SemanticKind.PRICE, Unit.CLP, PopulationBasis.RESIDENT_ESTIMATE),
        (
            SemanticKind.DISCOUNT_FACTOR,
            Unit.DIMENSIONLESS,
            PopulationBasis.PENSIONER,
        ),
        (SemanticKind.INDEX, Unit.DAY, PopulationBasis.RESIDENT_ESTIMATE),
    ],
)
def test_cube_rejects_incompatible_population_basis(
    kind: SemanticKind,
    unit: Unit,
    basis: PopulationBasis,
) -> None:
    data = _annotated_data([1, 2])
    data.attrs.update(
        semantic_kind=kind.value,
        unit=unit.value,
        population_basis=basis.value,
    )

    with pytest.raises(DataContractError, match=r"variable.*population basis"):
        DemographicCube(xr.Dataset({"variable": data}))


def test_with_variable_rejects_incompatible_population_basis() -> None:
    with pytest.raises(DataContractError, match=r"births.*population basis"):
        _cube().with_variable(
            "births",
            xr.DataArray([3, 4], dims=("age",), coords={"age": [0, 1]}),
            kind=SemanticKind.COUNT,
            unit=Unit.EVENT,
            population_basis=PopulationBasis.PERSON_TIME,
            provenance=_provenance(),
        )


def test_source_ref_requires_release_date_or_missingness_reason() -> None:
    payload = _source().model_dump()
    del payload["release_date"]

    with pytest.raises(ValidationError, match="release"):
        SourceRef.model_validate(payload)


def test_source_ref_requires_provisional_flag() -> None:
    payload = _source().model_dump()
    del payload["provisional"]

    with pytest.raises(ValidationError, match="provisional"):
        SourceRef.model_validate(payload)


def test_source_ref_requires_timezone_aware_retrieval_time() -> None:
    with pytest.raises(ValidationError, match="retrieved_at"):
        SourceRef(
            source_key="ine_fixture",
            name="INE fixture",
            url="https://www.ine.gob.cl/",
            retrieved_at=datetime(2026, 7, 18),
            sha256="a" * 64,
            vintage="2024-final",
            release_date=date(2025, 3, 28),
            release_missingness_reason=None,
            provisional=False,
        )


@pytest.mark.parametrize(
    "field",
    [
        "observation_start",
        "observation_end",
        "dimensions",
        "aggregation_rules",
        "missingness_reason",
    ],
)
def test_variable_provenance_requires_every_declaration(field: str) -> None:
    payload = _provenance().model_dump()
    del payload[field]

    with pytest.raises(ValidationError, match=field):
        VariableProvenance.model_validate(payload)


def test_variable_provenance_requires_ordered_observation_dates() -> None:
    payload = _provenance().model_dump()
    payload["observation_start"] = date(2024, 5, 1)
    payload["observation_end"] = date(2024, 4, 30)

    with pytest.raises(ValidationError, match="observation"):
        VariableProvenance.model_validate(payload)


@pytest.mark.parametrize(
    "dimensions",
    [
        ("age", "age"),
        ("age", ""),
        ("age", "   "),
    ],
)
def test_variable_provenance_rejects_malformed_dimensions(
    dimensions: tuple[str, ...],
) -> None:
    payload = _provenance().model_dump()
    payload["dimensions"] = dimensions

    with pytest.raises(ValidationError, match="dimensions"):
        VariableProvenance.model_validate(payload)


@pytest.mark.parametrize("aggregation_rules", [(), ("",), ("   ",)])
def test_variable_provenance_rejects_malformed_aggregation_rules(
    aggregation_rules: tuple[str, ...],
) -> None:
    payload = _provenance().model_dump()
    payload["aggregation_rules"] = aggregation_rules

    with pytest.raises(ValidationError, match="aggregation_rules"):
        VariableProvenance.model_validate(payload)


@pytest.mark.parametrize("missingness_reason", ["", "   "])
def test_variable_provenance_rejects_blank_missingness_reason(
    missingness_reason: str,
) -> None:
    payload = _provenance().model_dump()
    payload["missingness_reason"] = missingness_reason

    with pytest.raises(ValidationError, match="missingness_reason"):
        VariableProvenance.model_validate(payload)


def test_provenance_models_are_frozen_and_validate_sources() -> None:
    source = _source()
    provenance = _provenance()

    with pytest.raises(ValidationError, match="frozen"):
        source.name = "changed"
    with pytest.raises(ValidationError, match="frozen"):
        provenance.notes = ("changed",)
    with pytest.raises(ValidationError, match="sources"):
        VariableProvenance(
            sources=(),
            release_id="cldemopde:release:v1:" + "b" * 64,
            available_at=datetime(2026, 7, 18, tzinfo=UTC),
            observation_start=date(2024, 4, 1),
            observation_end=date(2024, 4, 30),
            dimensions=("age",),
            aggregation_rules=("sum within published age bins",),
            missingness_reason="not_applicable_complete",
        )
    with pytest.raises(ValidationError, match="sha256"):
        SourceRef(
            source_key="ine_fixture",
            name="INE fixture",
            url="https://www.ine.gob.cl/",
            retrieved_at=datetime(2026, 7, 18, tzinfo=UTC),
            sha256="not-a-checksum",
            vintage="2024-final",
            release_date=date(2025, 3, 28),
            release_missingness_reason=None,
            provisional=False,
        )


def test_provenance_models_round_trip_through_json() -> None:
    source = _source()
    provenance = _provenance()

    assert SourceRef.model_validate_json(source.model_dump_json()) == source
    assert VariableProvenance.model_validate_json(provenance.model_dump_json()) == provenance


def test_cube_rejects_provenance_dimension_mismatch() -> None:
    data = _annotated_data([1, 2])
    data.attrs["provenance"] = _provenance(dimensions=("time",)).model_dump(mode="json")

    with pytest.raises(DataContractError, match=r"deaths.*dimensions"):
        DemographicCube(xr.Dataset({"deaths": data}))


def test_with_variable_rejects_provenance_dimension_mismatch() -> None:
    with pytest.raises(DataContractError, match=r"births.*dimensions"):
        _cube().with_variable(
            "births",
            xr.DataArray([3, 4], dims=("age",), coords={"age": [0, 1]}),
            kind=SemanticKind.COUNT,
            unit=Unit.EVENT,
            population_basis=PopulationBasis.RESIDENT_ESTIMATE,
            provenance=_provenance(dimensions=("time",)),
        )


def test_cube_accepts_explicit_empty_dimensions_for_a_scalar_variable() -> None:
    scalar = xr.DataArray(42)
    scalar.attrs.update(
        semantic_kind=SemanticKind.COUNT.value,
        unit=Unit.EVENT.value,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE.value,
        provenance=_provenance(dimensions=()).model_dump(mode="json"),
    )

    result = DemographicCube(xr.Dataset({"total": scalar}))

    assert result.dataset["total"].dims == ()


def test_cube_string_lookup_returns_variable() -> None:
    deaths = _cube()["deaths"]

    assert isinstance(deaths, xr.DataArray)
    assert deaths.identical(_annotated_data([1, 2]).rename("deaths"))


def test_cube_dict_lookup_delegates_labeled_selection() -> None:
    selected = _cube()[{"age": 1}]

    assert isinstance(selected, xr.Dataset)
    assert int(selected["deaths"]) == 2


def test_with_variable_is_pure_and_attaches_typed_metadata() -> None:
    source = _cube()
    incoming = xr.DataArray(
        [3, 4],
        dims=("age",),
        coords={"age": [0, 1]},
        attrs={"description": "observed births"},
    )
    incoming_before = incoming.copy(deep=True)

    result = source.with_variable(
        "births",
        incoming,
        kind=SemanticKind.COUNT,
        unit=Unit.PERSON,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
        provenance=_provenance(),
    )

    assert "births" not in source.dataset
    assert incoming.identical(incoming_before)
    assert result.dataset["births"].attrs == {
        "description": "observed births",
        "semantic_kind": "count",
        "unit": "person",
        "population_basis": "resident_estimate",
        "provenance": _provenance().model_dump(mode="json"),
    }


@pytest.mark.parametrize(
    ("argument", "invalid_value"),
    [
        ("kind", object()),
        ("unit", object()),
        ("population_basis", object()),
        ("provenance", {"sources": []}),
    ],
)
def test_with_variable_wraps_invalid_runtime_metadata_with_variable_context(
    argument: str,
    invalid_value: object,
) -> None:
    metadata: dict[str, object] = {
        "kind": SemanticKind.COUNT,
        "unit": Unit.PERSON,
        "population_basis": PopulationBasis.RESIDENT_ESTIMATE,
        "provenance": _provenance(),
    }
    metadata[argument] = invalid_value

    with pytest.raises(DataContractError, match="births") as captured:
        _cube().with_variable(
            "births",
            xr.DataArray([3, 4], dims=("age",), coords={"age": [0, 1]}),
            **metadata,
        )

    assert captured.value.__cause__ is not None


def test_with_variable_normalizes_compatible_runtime_metadata_payloads() -> None:
    result = _cube().with_variable(
        "births",
        xr.DataArray([3, 4], dims=("age",), coords={"age": [0, 1]}),
        kind="count",
        unit="person",
        population_basis="resident_estimate",
        provenance=_provenance().model_dump(mode="json"),
    )

    assert result.dataset["births"].attrs["semantic_kind"] == "count"
    assert result.dataset["births"].attrs["unit"] == "person"
    assert result.dataset["births"].attrs["population_basis"] == "resident_estimate"
    assert result.dataset["births"].attrs["provenance"] == _provenance().model_dump(mode="json")


@pytest.mark.parametrize(
    "incoming",
    [
        xr.DataArray([3, 4], dims=("age",), coords={"age": [1, 2]}),
        xr.DataArray([3, 4], dims=("age",)),
        xr.DataArray(
            [3, 4],
            dims=("age",),
            coords={
                "age": [0, 1],
                "release": ("age", ["preliminary", "preliminary"]),
            },
        ),
    ],
)
def test_with_variable_rejects_implicit_coordinate_alignment(
    incoming: xr.DataArray,
) -> None:
    source_data = _annotated_data([1, 2]).assign_coords(release=("age", ["final", "final"]))
    source = DemographicCube(xr.Dataset({"deaths": source_data}))

    with pytest.raises(DataContractError, match=r"coordinate|age|release"):
        source.with_variable(
            "births",
            incoming,
            kind=SemanticKind.COUNT,
            unit=Unit.PERSON,
            population_basis=PopulationBasis.RESIDENT_ESTIMATE,
            provenance=_provenance(),
        )


def test_with_variable_allows_a_new_noncolliding_dimension() -> None:
    incoming = xr.DataArray(
        [3, 4],
        dims=("sex",),
        coords={"sex": ["female", "male"]},
    )

    result = _cube().with_variable(
        "births_by_sex",
        incoming,
        kind=SemanticKind.COUNT,
        unit=Unit.PERSON,
        population_basis=PopulationBasis.RESIDENT_ESTIMATE,
        provenance=_provenance(dimensions=("sex",)),
    )

    assert result.dataset["births_by_sex"].dims == ("sex",)
    assert result.dataset["births_by_sex"].coords["sex"].values.tolist() == [
        "female",
        "male",
    ]


def test_with_variable_rejects_different_shared_scalar_coordinates() -> None:
    source_data = _annotated_data([1, 2]).assign_coords(vintage="2024-final")
    source = DemographicCube(xr.Dataset({"deaths": source_data}))
    incoming = xr.DataArray(
        [3, 4],
        dims=("age",),
        coords={"age": [0, 1], "vintage": "2024-preliminary"},
    )

    with pytest.raises(DataContractError, match="vintage"):
        source.with_variable(
            "births",
            incoming,
            kind=SemanticKind.COUNT,
            unit=Unit.PERSON,
            population_basis=PopulationBasis.RESIDENT_ESTIMATE,
            provenance=_provenance(),
        )


def test_with_variable_rejects_a_scalar_coordinate_only_on_the_cube() -> None:
    source_data = _annotated_data([1, 2]).assign_coords(vintage="2024-final")
    source = DemographicCube(xr.Dataset({"deaths": source_data}))
    incoming = xr.DataArray(
        [3, 4],
        dims=("age",),
        coords={"age": [0, 1]},
    )

    with pytest.raises(DataContractError, match=r"vintage|scalar"):
        source.with_variable(
            "births",
            incoming,
            kind=SemanticKind.COUNT,
            unit=Unit.PERSON,
            population_basis=PopulationBasis.RESIDENT_ESTIMATE,
            provenance=_provenance(),
        )


def test_with_variable_rejects_a_scalar_coordinate_only_on_incoming_data() -> None:
    incoming = xr.DataArray(
        [3, 4],
        dims=("age",),
        coords={"age": [0, 1], "vintage": "2024-preliminary"},
    )

    with pytest.raises(DataContractError, match=r"vintage|scalar"):
        _cube().with_variable(
            "births",
            incoming,
            kind=SemanticKind.COUNT,
            unit=Unit.PERSON,
            population_basis=PopulationBasis.RESIDENT_ESTIMATE,
            provenance=_provenance(),
        )


def test_with_variable_rejects_a_name_that_is_an_incoming_coordinate() -> None:
    incoming = xr.DataArray(
        [3, 4],
        dims=("sex",),
        coords={"sex": ["female", "male"]},
    )

    with pytest.raises(DataContractError, match="sex"):
        _cube().with_variable(
            "sex",
            incoming,
            kind=SemanticKind.COUNT,
            unit=Unit.PERSON,
            population_basis=PopulationBasis.RESIDENT_ESTIMATE,
            provenance=_provenance(),
        )


@pytest.mark.parametrize("name", ["", "   "])
def test_with_variable_rejects_blank_names(name: str) -> None:
    with pytest.raises(DataContractError, match="name"):
        _cube().with_variable(
            name,
            xr.DataArray([3, 4], dims=("age",), coords={"age": [0, 1]}),
            kind=SemanticKind.COUNT,
            unit=Unit.PERSON,
            population_basis=PopulationBasis.RESIDENT_ESTIMATE,
            provenance=_provenance(),
        )


def test_cube_equality_is_xarray_aware_and_cubes_are_unhashable() -> None:
    class SpecializedCube(DemographicCube):
        pass

    cube = _cube()
    equal = _cube()
    different = DemographicCube(xr.Dataset({"deaths": _annotated_data([1, 3])}))
    specialized = SpecializedCube(xr.Dataset({"deaths": _annotated_data([1, 2])}))

    assert cube == equal
    assert cube != different
    assert cube.__eq__(object()) is NotImplemented
    assert cube.__eq__(specialized) is NotImplemented
    assert DemographicCube.__hash__ is None
    with pytest.raises(TypeError):
        hash(cube)
