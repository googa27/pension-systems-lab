# Architecture and coding design

## Design principles

- **Single responsibility**: data parsing, observation operators, rate estimation, dynamics, PDE solving, calibration, and diagnostics live in separate modules.
- **Dependency inversion**: the demographic model consumes rate callables rather than concrete estimator classes.
- **Immutable configuration/results**: Pydantic configuration and frozen dataclasses prevent hidden mutation.
- **Explicit observation layer**: grouped data never masquerade as point observations.
- **Testability**: the FEM solver, integrator, estimators, coupled pipeline, Bayesian layer, forecasts, and interventions each have direct tests.
- **Reproducibility**: curated public CSVs, `uv.lock`, a cleared notebook, and source URLs are bundled. Raw Excel/workbook redistribution is unresolved and is intentionally excluded from this public copy.

## Canonical semantic algebra

`DemographicCube` is the canonical labeled `xarray.Dataset` container. Every
data variable declares a `semantic_kind`, `unit`, `population_basis`, and
`provenance`. The semantic contract is exhaustive:

| Semantic kind | Unit | Compatible population bases |
|---|---|---|
| `count` | `person` | `census_enumerated`, `resident_estimate`, `pensioner` |
| `count` | `event` | `not_applicable`, `census_enumerated`, `resident_estimate`, `pensioner` |
| `exposure` | `person * year` | `resident_estimate`, `person_time`, `pensioner` |
| `population` | `person` | `census_enumerated`, `resident_estimate`, `pensioner` |
| `rate` | `1 / year`, `1 / month` | `not_applicable`, `census_enumerated`, `resident_estimate`, `person_time`, `pensioner` |
| `probability` | `dimensionless`, `percent` | `not_applicable`, `resident_estimate`, `pensioner` |
| `price` | `CLP`, `UF` | `not_applicable` |
| `discount_factor` | `dimensionless` | `not_applicable` |
| `index` | `dimensionless`, `percent`, `day`, `degree_Celsius`, `millimeter`, `microgram / meter ** 3` | `not_applicable` |

The basis contract is resolved after the semantic kind and unit. This permits
source-independent event counts and market rates to use `not_applicable`
without permitting `person` counts to do so. No other semantic-kind, unit, or
population-basis combination is accepted.
Every source record explicitly declares its name, URL, release date,
timezone-aware retrieval timestamp, SHA-256 checksum, vintage, and
provisional/final flag. Every variable provenance record explicitly declares a
nonempty source set, ordered observation start and end dates, its unique
nonblank ordered dimension names, nonempty nonblank aggregation rules, and a
nonblank missingness reason. Transformation identifiers and notes remain
optional. The declared dimension tuple must exactly equal the variable's
ordered dimensions; an explicitly empty tuple is valid for a scalar. Adding a
variable also preserves exact shared dimension labels and coordinates, so a
cube cannot silently align different age, time, sex, or other coordinates.

The core keeps physical units, population definitions, and observed data roles
separate. Its primary labeled dimensions are normally `age`, `time`, and `sex`;
an observation layer replaces `age` with `age_bin` when representing published
groups. The dimension names are not restricted to that set, but any operation
requires its labels to agree exactly.

| Object | Meaning and required unit | Population basis |
|---|---|---|
| `RateSurface` | A continuous reciprocal-time rate, `1 / year` or `1 / month` | The underlying population definition, such as `resident_estimate` or `census_enumerated` |
| `ExposureSurface` | Person-time exposure, `person * year` | The population definition from which that exposure was constructed |
| `ExpectedCountSurface` | Expected event count, `person` | Inherited from the compatible rate and exposure; resident estimates are the default |
| Observed bin total | A published count in an `age_bin` (and, when applicable, `time` and `sex`) | Its source definition, such as Censo enumerated population; it is not an expected count |
| Censo reconstructed density | A descriptive continuous-age count density constrained by Censo bins | `census_enumerated`; it is neither observed single-age data nor a resident estimate |
| PDE population state | Continuous resident-population density | `resident_estimate` is the canonical target |

`RateSurface` arithmetic rejects even compatible-looking operands unless all
dimensions and coordinates match exactly, the population bases match, and the
units satisfy the requested operation. Addition also requires identical rate
units: month-to-year conversion is explicit. The sole implicit
surface-to-surface dimensional conversion is

$$
\color{yellow}{\text{rate}}
\;*\;
\color{cyan}{\text{exposure}}
\;\longrightarrow\;
\color{lime}{\text{expected count}}.
$$

In the implemented wrapper this is an annual `RateSurface` times a compatible
`ExposureSurface`; a monthly rate must be converted explicitly before that
operation. No arithmetic turns a grouped observed total into a rate, exposure,
or population state.

`ObservationOperator @ surface` is the explicit grouped linear functional, or
bin projection, from a labeled continuous field. The implemented surface
application accepts a `RateSurface` with the operator's exact input coordinate
and returns an untyped `xarray.DataArray` of age integrals indexed by
`age_bin`, preserving non-contracted dimensions such as `sex`. That bare
result is neither an expected count nor an observed count without compatible
exposure, a time interval, and the observation model. `p1_bin_integral_operator()`
exactly integrates the P1 field represented on its age nodes; there is no
midpoint approximation. Operators may also compose with `@` when their
intermediate dimensions and labels match exactly.

For Censo population groups, the same exact-bin operator supplies equality
constraints for a nonnegative, minimum-curvature P1 reconstruction. It exactly
constrains every supplied Censo total to numerical tolerance, including $85+$
after explicit truncation to $[85,105]$. P1 exactness does not identify
within-bin shape; the shape on that truncated interval is
regularization-dependent, and the tail shape and domain beyond 105 remain
unidentified unless a separately tested tail model is chosen.

The canonical PDE state is an official resident-estimate density. The current
reference pipeline has not yet received official resident single-age estimates:
it still propagates the exact-bin descriptive Censo-enumerated reconstruction
as an explicitly disclosed initial proxy. The official-data tranche must supply
the resident-estimate state before the pipeline can use that population basis.

## Data flow

```text
Current legacy reference flow
Curated public CSVs ──► validated `ProjectData`
        │
        ├──► exact age-bin operators ──► smooth base rates
        │
        ├──► Censo exact-bin descriptive reconstruction ──► current initial proxy
        │                                                     │
        └──► monthly totals ──► smooth dynamic factors        ▼
                                                   two-sex FEM PDE propagation
                                                              │
                                                     PDE-implied offsets
                                                              │
                                                              └── iterative likelihood update

Canonical typed-ingestion target (not yet the reference pipeline)
official source artifacts ──► `DemographicCube` with typed metadata
official resident single-age estimates ──► canonical resident-estimate state
```

## Extension points

- Replace the separable rate functions with tensor-product age–period splines when multiple age profiles become available.
- Supply age-specific migration source callables without changing the FEM solver.
- Replace `AgeTransportFEM` with another implementation satisfying the same `step` contract.
- Add parity, region, cause of death, or socioeconomic state as coupled population compartments.
- Replace block-coordinate fitting with differentiable sparse solves when a mature JAX FEM stack is justified.
