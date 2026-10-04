from __future__ import annotations


class DemographicPDEError(Exception):
    """Base class for expected domain failures."""


class DataContractError(DemographicPDEError):
    """A labeled dataset violates a semantic contract."""


class UnitMismatchError(DataContractError):
    """Two operands have incompatible physical units."""


class SurfaceAlignmentError(DataContractError):
    """Labeled dimensions or coordinates do not match exactly."""


class SemanticKindError(DataContractError):
    """An operation combines incompatible semantic kinds."""


class PopulationBasisError(DataContractError):
    """An operation mixes incompatible definitions of population."""


class IdentityCollisionError(DataContractError):
    """Distinct observations resolve to the same canonical identity."""


class AmbiguousRevisionError(DataContractError):
    """A release history has no unique latest eligible revision."""


class ReportingProjectionInputError(DataContractError):
    """A reporting projection is used where an inferential input is required."""


class SourceContractError(DataContractError):
    """An official source violates its declared release contract."""


class ObservationOperatorError(DemographicPDEError):
    """An observation operator cannot be applied or composed."""


class ReconstructionError(DemographicPDEError):
    """A grouped-data reconstruction is infeasible or inaccurate."""


class MissingOfficialDataError(DemographicPDEError):
    """A required official asset is absent and cannot be fetched automatically."""


class ChecksumMismatchError(DemographicPDEError):
    """A downloaded or local asset does not match its declared checksum."""
