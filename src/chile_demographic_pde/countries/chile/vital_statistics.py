"""Attested INE birth controls and non-mutating DEIS reconciliation."""

from __future__ import annotations

import json
import re
from datetime import date
from enum import StrEnum
from numbers import Integral
from typing import Final, cast

import pandas as pd
from pandera.errors import SchemaErrors

from chile_demographic_pde.core.errors import DataContractError
from chile_demographic_pde.core.provenance import SourceRef
from chile_demographic_pde.countries.chile.curated import (
    CuratedDomainRelease,
    CuratedTableContract,
    _attest_curated_release,
    _finalize_demographic_batch,
)
from chile_demographic_pde.data.domain_schemas import (
    DemographicObservationBatch,
    validate_demographic_observations,
)
from chile_demographic_pde.data.envelope import NormalizedDomainBatch
from chile_demographic_pde.data.identity import IdentityRegistry
from chile_demographic_pde.data.roles import ObservationDomain

INE_BIRTH_CONTROL_CONTRACT: Final = CuratedTableContract(
    contract_id="ine-birth-controls-curated-v1",
    columns=(
        "year",
        "month",
        "sex",
        "region",
        "geography_vintage",
        "births",
    ),
    logical_types=(
        "integer",
        "integer",
        "string",
        "string",
        "string",
        "integer",
    ),
)
INE_BIRTH_CONTROL_COLUMNS: Final = INE_BIRTH_CONTROL_CONTRACT.columns
_TRANSFORMATION_ID: Final = "ine-birth-control-normalize"
_TRANSFORMATION_VERSION: Final = "v1"
_REGION: Final = re.compile(r"^(?:CL|CL-(?:0[1-9]|1[0-6])|unknown)$")
_SEXES: Final = frozenset({"all", "male", "female", "indeterminate"})
_COMPARISON_KEY: Final = (
    "variable",
    "semantic_kind",
    "unit",
    "population_basis",
    "period_start",
    "period_end",
    "age_lower",
    "age_upper",
    "age_open",
    "sex",
    "region",
    "cause",
    "date_basis",
    "age_role",
    "age_measure_unit",
    "age_quantity",
    "sex_role",
    "commune",
    "geography_basis",
    "geography_vintage",
    "cause_code_system",
    "cause_role",
    "external_cause",
    "registration_start",
    "registration_end",
    "birth_order",
    "nationality",
    "nationality_role",
    "country_of_birth",
    "education_level",
    "residence_five_years_ago",
    "migration_status",
)


class IneBirthReleaseKind(StrEnum):
    """Publication status and period granularity of an INE birth release."""

    FINAL_ANNUAL = "final_annual"
    PROVISIONAL_ANNUAL = "provisional_annual"
    COYUNTURAL_MONTHLY = "coyuntural_monthly"

    @property
    def provisional(self) -> bool:
        """Whether the release is explicitly nonfinal."""

        return self is not IneBirthReleaseKind.FINAL_ANNUAL

    @property
    def monthly(self) -> bool:
        """Whether each row represents one occurrence month."""

        return self is IneBirthReleaseKind.COYUNTURAL_MONTHLY


def normalize_ine_birth_controls_curated(
    frame: pd.DataFrame,
    *,
    release: CuratedDomainRelease,
    identity_registry: IdentityRegistry,
) -> DemographicObservationBatch:
    """Map one checksum-attested curated INE control table to its own domain batch."""

    _attest_curated_release(frame, release)
    if release.table_contract != INE_BIRTH_CONTROL_CONTRACT:
        raise DataContractError(
            "INE birth controls require the exact reviewed physical table contract."
        )
    if tuple(frame.columns) != INE_BIRTH_CONTROL_COLUMNS:
        raise DataContractError(
            "INE birth controls require exact columns in this order: "
            f"{INE_BIRTH_CONTROL_COLUMNS!r}."
        )
    if not isinstance(identity_registry, IdentityRegistry):
        raise DataContractError("INE birth control normalization requires an IdentityRegistry.")

    source = _primary_source(release)
    kind = _release_kind(frame["month"], source=source)
    records = cast(
        list[dict[str, object]],
        frame.to_dict(orient="records"),
    )
    payload = pd.DataFrame(
        _normalize_ine_birth_control(
            record,
            release=release,
            source=source,
            kind=kind,
        )
        for record in records
    )
    duplicate_key = (
        "period_start",
        "period_end",
        "sex",
        "region",
        "geography_vintage",
    )
    if bool(payload.duplicated(subset=list(duplicate_key), keep=False).any()):
        raise DataContractError("INE birth controls contain duplicate period/sex/geography keys.")
    try:
        return _finalize_demographic_batch(
            payload,
            release=release,
            identity_registry=identity_registry,
            source_identity={
                "publisher": "Instituto Nacional de Estadisticas de Chile",
                "dataset": "published_birth_controls",
                "contract_id": release.table_contract.contract_id,
            },
        )
    except SchemaErrors as error:
        raise DataContractError(
            f"Normalized INE birth controls violate the demographic schema: {error}"
        ) from error


def compare_deis_births_to_ine_controls(
    deis: DemographicObservationBatch,
    controls: DemographicObservationBatch,
    *,
    control_release_kind: IneBirthReleaseKind,
) -> pd.DataFrame:
    """Reconcile matching concepts without revising either immutable input batch."""

    if not isinstance(control_release_kind, IneBirthReleaseKind):
        raise DataContractError("control_release_kind must be an IneBirthReleaseKind.")
    deis_frame = _validate_comparison_input(deis, name="DEIS births")
    control_frame = _validate_comparison_input(
        controls,
        name="INE birth controls",
    )
    _require_birth_control_semantics(deis_frame, name="DEIS births")
    _require_birth_control_semantics(
        control_frame,
        name="INE birth controls",
    )
    _require_role_and_parity(
        deis_frame,
        name="DEIS births",
        role="observed_fact",
        parity="unknown_order",
    )
    _require_role_and_parity(
        control_frame,
        name="INE birth controls",
        role="diagnostic_control",
        parity="all_orders",
    )
    _require_control_release_kind(control_frame, control_release_kind)
    _require_unique_comparison_keys(deis_frame, name="DEIS births")
    _require_unique_comparison_keys(
        control_frame,
        name="INE birth controls",
    )
    _require_compatible_concepts(deis_frame, control_frame)

    left = deis_frame.loc[:, [*_COMPARISON_KEY, "value"]].rename(columns={"value": "deis_value"})
    right = control_frame.loc[:, [*_COMPARISON_KEY, "value"]].rename(
        columns={"value": "control_value"}
    )
    diagnostic = left.merge(
        right,
        how="inner",
        on=list(_COMPARISON_KEY),
        validate="one_to_one",
        sort=False,
    )
    if diagnostic.empty:
        raise DataContractError("DEIS births and INE controls have no matched comparison keys.")
    diagnostic["difference"] = diagnostic["deis_value"] - diagnostic["control_value"]
    exact = diagnostic["difference"].eq(0)
    matched_status, different_status = _comparison_statuses(control_release_kind)
    diagnostic["status"] = different_status
    diagnostic.loc[exact, "status"] = matched_status
    diagnostic["control_release_kind"] = control_release_kind.value
    diagnostic["control_provisional"] = control_release_kind.provisional
    return diagnostic.loc[
        :,
        [
            *_COMPARISON_KEY,
            "deis_value",
            "control_value",
            "difference",
            "status",
            "control_release_kind",
            "control_provisional",
        ],
    ].reset_index(drop=True)


def _primary_source(release: CuratedDomainRelease) -> SourceRef:
    primary_key = release.manifest.primary_fact_source_key
    matching = tuple(
        source for source in release.provenance.sources if source.source_key == primary_key
    )
    if len(matching) != 1:
        raise DataContractError(
            "INE curated release requires exactly one primary provenance source."
        )
    return matching[0]


def _release_kind(
    months: pd.Series,
    *,
    source: SourceRef,
) -> IneBirthReleaseKind:
    missing = months.isna()
    if bool(missing.any()) and not bool(missing.all()):
        raise DataContractError("INE birth controls cannot mix annual and monthly period grains.")
    if bool(missing.all()):
        return (
            IneBirthReleaseKind.PROVISIONAL_ANNUAL
            if source.provisional
            else IneBirthReleaseKind.FINAL_ANNUAL
        )
    if not source.provisional:
        raise DataContractError("INE monthly birth controls must come from a provisional source.")
    return IneBirthReleaseKind.COYUNTURAL_MONTHLY


def _normalize_ine_birth_control(
    row: dict[str, object],
    *,
    release: CuratedDomainRelease,
    source: SourceRef,
    kind: IneBirthReleaseKind,
) -> dict[str, object]:
    year = _require_integer(row["year"], field="year", minimum=1)
    month = _normalize_month(row["month"], kind)
    births = _require_integer(row["births"], field="births", minimum=0)
    sex = _require_text(row["sex"], field="sex")
    if sex not in _SEXES:
        raise DataContractError(f"INE birth control sex must be one of {sorted(_SEXES)!r}.")
    region = _require_text(row["region"], field="region")
    if _REGION.fullmatch(region) is None:
        raise DataContractError(
            "INE birth control region must be CL, CL-01 through CL-16, or unknown."
        )
    geography_vintage = _normalize_geography_vintage(
        row["geography_vintage"],
        region=region,
    )
    period_start, period_end, date_basis = _control_period(
        year,
        month,
        kind,
    )
    if (
        period_start < release.provenance.observation_start
        or period_end > release.provenance.observation_end
    ):
        raise DataContractError("INE birth control period falls outside attested release coverage.")
    dimension_missingness = {
        "birth_order": "published_all_orders_total",
        "commune": "aggregated_to_published_geography",
        "maternal_age": "aggregated_all_maternal_ages",
        "maternal_nationality": "not_published_in_aggregate_control",
        "registration_interval": "not_published_at_control_row_level",
    }
    if geography_vintage is None:
        dimension_missingness["geography_vintage"] = "not_applicable_to_national_total"
    primary = release.manifest.primary_fact_asset
    return {
        "domain": ObservationDomain.DEMOGRAPHIC.value,
        "variable": "births",
        "value": births,
        "semantic_kind": "count",
        "unit": "event",
        "population_basis": "not_applicable",
        "observation_role": "diagnostic_control",
        "parity_scope": "all_orders",
        "period_start": period_start,
        "period_end": period_end,
        "source_key": primary.source_key,
        "vintage": release.vintage,
        "released_at": primary.released_at,
        "release_missingness_reason": (
            None
            if primary.release_missingness_reason is None
            else primary.release_missingness_reason.value
        ),
        "available_at": release.manifest.available_at,
        "provisional": source.provisional,
        "transformation_id": _TRANSFORMATION_ID,
        "transformation_version": _TRANSFORMATION_VERSION,
        "aggregation_rule": (
            "preserve the published INE observed birth count as a diagnostic "
            "control; never revise DEIS events"
        ),
        "missingness_reason": json.dumps(
            dimension_missingness,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "source_status_code": None,
        "age_lower": None,
        "age_upper": None,
        "age_open": False,
        "sex": sex,
        "region": region,
        "cause": "not_applicable",
        "date_basis": date_basis,
        "age_role": "mother",
        "age_measure_unit": "all_grouped_completed_years",
        "age_quantity": None,
        "sex_role": "newborn",
        "commune": None,
        "geography_basis": "maternal_residence",
        "geography_vintage": geography_vintage,
        "cause_code_system": None,
        "cause_role": None,
        "external_cause": None,
        "registration_start": None,
        "registration_end": None,
        "birth_order": None,
        "nationality": None,
        "nationality_role": "mother",
        "country_of_birth": None,
        "education_level": None,
        "residence_five_years_ago": None,
        "migration_status": None,
    }


def _normalize_month(
    value: object,
    kind: IneBirthReleaseKind,
) -> int | None:
    if _is_missing_scalar(value):
        if kind.monthly:
            raise DataContractError("INE coyuntural monthly controls require an integer month.")
        return None
    if not kind.monthly:
        raise DataContractError("INE annual birth controls require month to be missing.")
    return _require_integer(value, field="month", minimum=1, maximum=12)


def _control_period(
    year: int,
    month: int | None,
    kind: IneBirthReleaseKind,
) -> tuple[date, date, str]:
    if kind.monthly:
        if month is None:  # pragma: no cover - guarded by _normalize_month
            raise AssertionError("monthly INE control lacks a month")
        start = date(year, month, 1)
        end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        return start, end, "birth_occurrence_month"
    return (
        date(year, 1, 1),
        date(year + 1, 1, 1),
        "birth_occurrence_year",
    )


def _normalize_geography_vintage(
    value: object,
    *,
    region: str,
) -> str | None:
    if _is_missing_scalar(value):
        if region != "CL":
            raise DataContractError("Regional INE birth controls require geography_vintage.")
        return None
    return _require_text(value, field="geography_vintage")


def _require_integer(
    value: object,
    *,
    field: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise DataContractError(f"INE birth control {field} must be an integer.")
    normalized = int(value)
    if normalized < minimum or (maximum is not None and normalized > maximum):
        interval = (
            f" from {minimum} through {maximum}"
            if maximum is not None
            else f" greater than or equal to {minimum}"
        )
        raise DataContractError(f"INE birth control {field} must be an integer{interval}.")
    return normalized


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"INE birth control {field} must be a nonblank string.")
    if value != value.strip():
        raise DataContractError(
            f"INE birth control {field} must not contain surrounding whitespace."
        )
    return value


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float) and value != value


def _validate_comparison_input(
    batch: DemographicObservationBatch,
    *,
    name: str,
) -> pd.DataFrame:
    if not isinstance(batch, NormalizedDomainBatch):
        raise DataContractError(f"{name} must be a DemographicObservationBatch.")
    if batch.domain is not ObservationDomain.DEMOGRAPHIC:
        raise DataContractError(f"{name} must contain demographic observations.")
    try:
        return validate_demographic_observations(batch.observations)
    except SchemaErrors as error:
        raise DataContractError(
            f"{name} violates the demographic observation contract: {error}"
        ) from error


def _require_birth_control_semantics(
    frame: pd.DataFrame,
    *,
    name: str,
) -> None:
    expected = {
        "variable": {"births"},
        "semantic_kind": {"count"},
        "unit": {"event"},
    }
    for column, values in expected.items():
        if set(frame[column]) != values:
            raise DataContractError(f"{name} has incompatible birth {column} semantics.")


def _require_role_and_parity(
    frame: pd.DataFrame,
    *,
    name: str,
    role: str,
    parity: str,
) -> None:
    if set(frame["observation_role"]) != {role}:
        raise DataContractError(f"{name} has an incompatible observation role.")
    if set(frame["parity_scope"]) != {parity}:
        raise DataContractError(f"{name} has an incompatible parity scope.")


def _require_control_release_kind(
    controls: pd.DataFrame,
    kind: IneBirthReleaseKind,
) -> None:
    if set(controls["provisional"]) != {kind.provisional}:
        raise DataContractError(
            "INE control provisional flags conflict with the explicit release kind."
        )
    expected_basis = "birth_occurrence_month" if kind.monthly else "birth_occurrence_year"
    if set(controls["date_basis"]) != {expected_basis}:
        raise DataContractError(
            "INE control period grain conflicts with the explicit release kind."
        )
    intervals_are_exact = all(
        _period_matches_release_kind(start, end, kind)
        for start, end in controls.loc[
            :,
            ["period_start", "period_end"],
        ].itertuples(index=False, name=None)
    )
    if not intervals_are_exact:
        raise DataContractError(
            "INE control calendar interval shape conflicts with the explicit release kind."
        )


def _period_matches_release_kind(
    start: object,
    end: object,
    kind: IneBirthReleaseKind,
) -> bool:
    if type(start) is not date or type(end) is not date:
        return False
    if kind.monthly:
        if start.day != 1:
            return False
        expected_end = (
            date(start.year + 1, 1, 1)
            if start.month == 12
            else date(start.year, start.month + 1, 1)
        )
        return end == expected_end
    return start.month == 1 and start.day == 1 and end == date(start.year + 1, 1, 1)


def _require_unique_comparison_keys(
    frame: pd.DataFrame,
    *,
    name: str,
) -> None:
    if bool(frame.duplicated(subset=list(_COMPARISON_KEY), keep=False).any()):
        raise DataContractError(f"{name} contains duplicate comparison keys.")


def _require_compatible_concepts(
    deis: pd.DataFrame,
    controls: pd.DataFrame,
) -> None:
    if set(deis["date_basis"]) != set(controls["date_basis"]) or set(
        zip(deis["period_start"], deis["period_end"], strict=True)
    ) != set(zip(controls["period_start"], controls["period_end"], strict=True)):
        raise DataContractError("DEIS births and INE controls use incompatible period concepts.")
    if set(deis["geography_basis"]) != set(controls["geography_basis"]):
        raise DataContractError("DEIS births and INE controls use incompatible geography concepts.")
    if set(deis["sex_role"]) != set(controls["sex_role"]):
        raise DataContractError("DEIS births and INE controls use incompatible sex concepts.")
    if _comparison_domain(deis) != _comparison_domain(controls):
        raise DataContractError(
            "DEIS births and INE controls contain unmatched comparison keys; "
            "partial reconciliation is not permitted."
        )


def _comparison_domain(frame: pd.DataFrame) -> set[tuple[object, ...]]:
    return {
        tuple(
            ("__missing__", column) if _is_missing_scalar(value) else value
            for column, value in zip(_COMPARISON_KEY, values, strict=True)
        )
        for values in frame.loc[
            :,
            list(_COMPARISON_KEY),
        ].itertuples(index=False, name=None)
    }


def _comparison_statuses(kind: IneBirthReleaseKind) -> tuple[str, str]:
    if kind is IneBirthReleaseKind.FINAL_ANNUAL:
        return "reconciled_final", "different_final"
    if kind is IneBirthReleaseKind.PROVISIONAL_ANNUAL:
        return "matched_provisional", "different_provisional"
    return "matched_coyuntural", "different_coyuntural"
