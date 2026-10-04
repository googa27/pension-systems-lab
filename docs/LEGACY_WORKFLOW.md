# Chile Demographic PDE 🇨🇱

A reproducible research project for estimating dynamic continuous-age fertility and mortality schedules from official Chilean grouped data and propagating them through a two-sex McKendrick–von Foerster PDE solved with finite elements.

## What is implemented

- **Actual public data only**: INE provisional vital statistics and Censo 2024 population counts are bundled under `data/`.
- **Typed demographic data core**: `DemographicCube` records each labeled
  variable's semantic kind, unit, population basis, and source/vintage
  provenance; rate arithmetic requires exact coordinate alignment.
- **Exact grouped-data observation operator**: published age-bin counts are modeled as integrals of continuous event intensities.
- **Exact Censo-bin reconstruction**: an optional descriptive within-bin curve
  preserves every enumerated age-bin total under explicit nonnegativity
  constraints and reports residual, conditioning, local-curvature, and actual
  coarse-versus-refined grid-comparison diagnostics.
- **Explicit population-basis boundary**: resident estimates are the canonical
  PDE state. Until official resident single-age estimates arrive, the reference
  pipeline explicitly uses the descriptive Censo-enumerated reconstruction as
  its initial proxy.
- **Penalized Poisson spline rates**: continuous fertility and sex-specific mortality schedules are estimated on the log scale; mortality is constrained to be nondecreasing after age 40 to prevent an unsupported tail reversal from the coarse $85+$ exposure bin.
- **Dynamic structural factors**: smooth monthly fertility and mortality multipliers are fitted by penalized Poisson likelihood.
- **PDE-constrained calibration**: rate updates alternate with finite-element propagation of the age distribution.
- **Finite elements**: P1 elements, implicit Euler, inflow boundary at age zero, natural old-age outflow, and transport stabilization are assembled with `scikit-fem`.
- **Bayesian model**: a damped Poisson local-linear-trend state-space model is fitted with NumPyro SVI in the notebook, with an optional NUTS implementation exposed by the package.
- **Diagnostics and forecasting**: Poisson deviance, Pearson residuals, Ljung–Box tests, rolling-origin forecasts, posterior intervals, and intervention scenarios.

## Mathematical core

For sex $s\in\{m,f\}$, age $a$ and calendar time $t$:

$$
\frac{\partial \color{cyan}{n_s}}{\partial t}
+
\frac{\partial \color{cyan}{n_s}}{\partial a}
=
-\color{orange}{\mu_s(a,t)}\color{cyan}{n_s(a,t)}
+\color{yellow}{\nu_s(a,t)}.
$$

The age-zero boundary is endogenous:

$$
\color{cyan}{n_s(0,t)}
=
\color{lime}{p_s(t)}
\int_0^{a_{\max}}
\color{yellow}{f(a,t)}\color{cyan}{n_f(a,t)}\,da.
$$

A published death count in age bin $A_i$ and time interval $T_j$ has mean

$$
\Lambda^D_{ijs}
=
\int_{T_j}\int_{A_i}
\color{orange}{\mu_s(a,t)}\color{cyan}{n_s(a,t)}\,da\,dt,
\qquad
D_{ijs}\sim\operatorname{Poisson}(\Lambda^D_{ijs}).
$$

Births by maternal age use the analogous integral of $\color{yellow}{f(a,t)}\color{cyan}{n_f(a,t)}$.

## Run it

```bash
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy
uv run pension-lab capabilities
uv run pension-lab demo
uv run pension-lab verify-data
```

The notebook is kept with outputs cleared and is not part of the bounded public smoke gate:

```text
notebooks/chile_demographic_pde.ipynb
```

## Repository map

```text
src/chile_demographic_pde/
├── calibration/       # PDE-constrained block-coordinate inference
├── core/              # cube metadata, provenance, units, and typed surfaces
├── data/              # validated loaders, age-bin operators, reconstruction
│   ├── observation.py # composable exact P1 age-bin observation operators
│   └── reconstruction.py # exact-bin descriptive Censo reconstruction
├── diagnostics/       # residual tests, backtests, interventions
├── dynamics/          # deterministic and Bayesian state-space trends
├── pde/               # scikit-fem transport solver and two-sex PDE
├── rates/             # grouped Poisson spline estimators
├── config.py          # immutable validated configuration
└── pipeline.py        # reference end-to-end fit
```

## Public semantic workflow

`DemographicCube` is the canonical container for variables with labeled
coordinates and required semantic/provenance metadata. `RateSurface` represents
a continuous reciprocal-time rate, `ExposureSurface` represents `person *
year`, and `ExpectedCountSurface` represents a `person` count. Their only
implicit dimensional conversion is

$$
\color{yellow}{\text{rate}}
\;\color{white}{\times}\;
\color{cyan}{\text{exposure}}
\;\longrightarrow\;
\color{lime}{\text{expected count}}.
$$

The operands must have identical labeled coordinates and the same population
basis; conversion between time units is explicit. `ObservationOperator @
surface` is the explicit grouped linear functional, or bin projection. For its
implemented `RateSurface` input, it returns an untyped `xarray.DataArray` of
age integrals, not an expected or observed count without compatible exposure, a
time interval, and the observation model. A Censo grouped total, its optional
descriptive within-bin density, and a resident-population state remain separate
objects. Every supplied Censo total, including $85+$ after explicit truncation
to $[85,105]$, is exactly constrained to numerical tolerance; its shape on the
truncated interval is regularization-dependent and its tail shape and domain
beyond 105 are not identified.

## Scientific scope and limitations

This is a serious structural prototype, not an official INE population projection, legal pension projection, actuarial certification, or regulatory forecast.

1. The bundled monthly series has 16 observations, January 2025–April 2026.
2. The detailed age schedules are observed for April 2026, so the identifiable dynamic specification is separable:

   $$
   f(a,t)=f_0(a)e^{\eta_f(t)},
   \qquad
   \mu_s(a,t)=\mu_{0,s}(a)e^{\eta_{\mu}(t)}.
   $$

   Arbitrary age–period interactions cannot be identified from one age profile.
3. Censo 2024 enumerated grouped counts and INE resident-population estimates
   are different population bases. The canonical PDE state is a resident
   estimate, but the current reference pipeline still uses the exact-bin
   descriptive Censo reconstruction as an explicitly disclosed initial proxy.
   Its interpolation exactly constrains every supplied official Censo total,
   including $85+$ after explicit truncation to $[85,105]$. The shape on that
   truncated interval is regularization-dependent, while the tail shape and
   domain beyond 105 are not identified. Official resident single-age estimates
   and person-year exposures are required before the pipeline can use that
   canonical basis.
4. Migration is present in the PDE interface but set to zero in the bundled scenario analysis because age-specific migration flows are not identified by these files.
5. The intervention forecasts are conditional structural scenarios, not causal estimates.

See `reports/RESULTS.md`, `docs/MATHEMATICS.md`, `docs/LIBRARY_COMPARISON.md`, and `docs/LIMITATIONS.md` for the full discussion.
