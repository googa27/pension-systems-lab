# Mathematical formulation

## 1. State, units, and characteristics

Let $\color{cyan}{n_s(a,t)}$ be population density in people per year of age. The transport field is $(1,1)$ in the age–time plane, so characteristics satisfy

$$
\frac{da}{dt}=1,
\qquad
t-a=\text{birth cohort}.
$$

The two-sex McKendrick–von Foerster system is

$$
\partial_t \color{cyan}{n_s}
+
\partial_a \color{cyan}{n_s}
=
-\color{orange}{\mu_s}\color{cyan}{n_s}
+\color{yellow}{\nu_s}.
$$

The nonlocal birth boundary is

$$
\color{cyan}{n_s(0,t)}
=
\color{lime}{p_s(t)}
\int_0^{a_{\max}}
\color{yellow}{f(a,t)}\color{cyan}{n_f(a,t)}\,da.
$$

## 2. Semantic data contract

The model distinguishes a continuous rate, a person-time exposure, an expected
event count, and an observed grouped total. On a labeled grid whose usual
dimensions are `age`, `time`, and `sex`, let $r$ be a rate, $E$ an exposure,
and $\Lambda$ an expected count. Their units are respectively $1/\text{year}$
(or an explicitly converted reciprocal-time unit), $\text{person}\cdot\text{year}$,
and `person`:

$$
\color{yellow}{r(a,t,s)}
\;\color{white}{\times}\;
\color{cyan}{E(a,t,s)}
=
\color{lime}{\Lambda(a,t,s)}.
$$

This is the only implicit dimensional conversion in the typed surface algebra.
`RateSurface + RateSurface` requires the same reciprocal-time unit, the same
population basis, and identical labeled dimensions and coordinates. Rate and
exposure multiplication has those same alignment and basis requirements;
monthly-to-annual conversion is explicit. `ExpectedCountSurface` is not an
observed count, and a published bin total is never silently reinterpreted as a
continuous rate, an exposure, or a PDE state.

Each `DemographicCube` variable records its semantic kind, unit, population
basis, and validated provenance. The admissible metadata combinations are:

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

Population basis is a unit-sensitive contract: an event count may be
source-independent and a market rate may have no population basis, while a
`person` count must identify its demographic population. A Censo enumerated
count density, an INE resident-population estimate, and a resident-derived
person-time exposure all have different meanings even when their numeric
magnitudes are similar.

Provenance is complete rather than inferred. A source declares its name, URL,
release date, timezone-aware retrieval timestamp, SHA-256 checksum, vintage,
and provisional/final status. A variable declares ordered observation start
and end dates, ordered unique nonblank dimensions that exactly equal its
`xarray` dimensions, nonempty nonblank aggregation rules, and a nonblank
missingness reason. Scalar variables declare the empty dimension tuple
explicitly. Transformation identifiers and notes are optional.

The canonical PDE state is a resident-estimate population density. Its normal
coordinates are `age`, `time`, and `sex`; a grouped observation uses `age_bin`
in place of the integrated age dimension. In the current reference pipeline,
however, official resident single-age estimates have not yet been supplied. It
therefore propagates an exact-bin, descriptive Censo-enumerated reconstruction
as an explicitly disclosed initial proxy. This is not a claim that the current
pipeline already uses an official resident single-age state.

## 3. Published bins are linear functionals

For an age bin $A_i=[a_i,a_{i+1})$ and period $T_j=[t_j,t_{j+1})$,

$$
\Lambda^B_{ij}
=
\int_{T_j}\int_{A_i}
\color{yellow}{f(a,t)}\color{cyan}{n_f(a,t)}\,da\,dt,
$$

$$
\Lambda^D_{ijs}
=
\int_{T_j}\int_{A_i}
\color{orange}{\mu_s(a,t)}\color{cyan}{n_s(a,t)}\,da\,dt.
$$

The observation model is

$$
B_{ij}\sim\operatorname{Poisson}(\Lambda^B_{ij}),
\qquad
D_{ijs}\sim\operatorname{Poisson}(\Lambda^D_{ijs}).
$$

After discretization, this becomes

$$
\color{lime}{\boldsymbol\lambda}
=
\underbrace{\color{yellow}{H}}_{\text{exact bin-integral operator}}
\underbrace{\color{cyan}{\boldsymbol q}}_{\text{fine continuous intensity}}.
$$

`p1_bin_integral_operator()` constructs an `ObservationOperator` whose labeled
matrix $H$ is exact for the represented P1 field; no midpoint approximation is
used. `ObservationOperator @ surface` is the explicit grouped linear
functional, or bin projection. The implemented surface application accepts a
`RateSurface` with exactly the operator's input `age` coordinate and returns an
untyped `xarray.DataArray` of age integrals indexed by `age_bin`, preserving
other dimensions such as `sex`. That bare result is neither an expected count
nor an observed count without compatible exposure, a time interval, and the
observation model. Operator composition also uses `@` and requires exact
intermediate labels.

## 4. Identifiability and regularization

Grouped totals do not identify arbitrary within-bin variation. If $H z=0$, then
$H(q+z)=Hq$. For the optional descriptive interpolation of Censo-enumerated
counts, the package selects a nonnegative minimum-curvature P1 representative:

$$
\min_{\boldsymbol n\geq0}
\frac12\boldsymbol n^\mathsf{T}
\color{yellow}{R}
\boldsymbol n
\quad\text{subject to}\quad
\color{cyan}{A}\boldsymbol n
=
\color{lime}{\boldsymbol N}_{\mathrm{Censo}}.
$$

The equality constraints are authoritative, so every supplied Censo total is
preserved to numerical tolerance, including $85+$ after explicit truncation to
$[85,105]$; regularization selects among exact-bin curves. The result is
descriptive Censo-enumerated count density, not observed single-age data and
not an official resident-population estimate. Its shape on $[85,105]$ is
regularization-dependent, while the tail shape and domain beyond 105 remain
unidentified unless a separately tested tail model is selected.

Grid dependence is assessed by reconstructing again after refining every age
interval, interpolating the refined exact-bin curve onto the original nodes,
and reporting

$$
\Delta_\infty
=
\max_j\left|n_h(a_j)-n_{h/2}(a_j)\right|,
\qquad
\Delta_{\infty,\mathrm{rel}}
=
\frac{\Delta_\infty}
{\max\left(\lVert n_h\rVert_\infty,\lVert n_{h/2}\rVert_\infty\right)}.
$$

This actual coarse-versus-refined comparison is distinct from the same-grid
`local_curvature_indicator`. Neither turns the descriptive curve into an
observation; the official grouped totals remain authoritative.

For continuous event rates, we instead select a smooth positive representative
using a log B-spline:

$$
\log r(a)=\sum_{k=1}^{K}\theta_k B_k(a).
$$

For bin exposure density $e(a)$,

$$
\lambda_i(\theta)
=
\int_{A_i}e(a)\exp\left(\sum_k\theta_kB_k(a)\right)da.
$$

The penalized objective is

$$
\widehat\theta
=
\arg\min_\theta
\left\{
\sum_i\left[\lambda_i(\theta)-y_i\log\lambda_i(\theta)\right]
+
\frac{\rho}{2}\|D_2\theta\|_2^2
\right\}.
$$

The penalty resolves the inverse problem by suppressing unsupported spline curvature; it does not manufacture observed single-age data.


For mortality, the final open-ended exposure group is $85+$. An unconstrained spline can use that broad cell to create an artificial decline at extreme ages. The reference fit therefore imposes the shape restriction

$$
\frac{\partial}{\partial a}\log \color{orange}{\mu_s(a)}\geq 0,
\qquad a\geq 40.
$$

Because the derivative of a B-spline is linear in its coefficients, SciPy enforces this through a linear inequality constraint at a dense age grid. This is a transparent regularizing assumption, not information observed at single ages.

## 5. Dynamic rate specification

Because the bundled age profile is available only for April 2026, the defensible dynamic model is separable:

$$
\color{yellow}{f(a,t)}
=
\color{yellow}{f_0(a)}\exp\{\eta_f(t)\},
$$

$$
\color{orange}{\mu_s(a,t)}
=
\color{orange}{\mu_{0,s}(a)}\exp\{\eta_\mu(t)\}.
$$

For monthly observed counts $Y_t$ and PDE-implied unit-rate offsets $O_t$,

$$
Y_t\sim\operatorname{Poisson}\left(O_t e^{\eta_t}\right),
$$

$$
\widehat{\boldsymbol\eta}
=
\arg\min_{\boldsymbol\eta}
\left\{
\sum_t\left[O_te^{\eta_t}-Y_t(\log O_t+\eta_t)\right]
+
\frac{\lambda}{2}\|D_2\boldsymbol\eta\|_2^2
\right\}.
$$

## 6. Weak form and finite elements

For a test function $v$, implicit Euler gives

$$
(v,n^{k+1})
+\Delta t\,(v,\partial_a n^{k+1})
+\Delta t\,(v,\mu^{k+1}n^{k+1})
+\Delta t\,\varepsilon(\partial_av,\partial_an^{k+1})
=
(v,n^k)+\Delta t\,(v,\nu^{k+1}).
$$

With P1 basis functions $\{\phi_i\}$:

$$
\left[
\color{cyan}{M}
+\Delta t\left(
\color{yellow}{C}
+\color{orange}{R(\mu)}
+\varepsilon\color{lime}{K}
\right)
\right]\boldsymbol n^{k+1}
=
\color{cyan}{M}\boldsymbol n^k
+\Delta t\,\color{cyan}{M}\boldsymbol\nu^{k+1}.
$$

Here $M$ is the mass matrix, $C$ the age-advection matrix, $R$ the mortality reaction matrix, and $K$ the stiffness matrix. `scikit-fem` assembles all four. The small artificial diffusion stabilizes central Galerkin transport; the code records this modeling choice explicitly.

## 7. PDE-constrained block-coordinate calibration

The coupled algorithm iterates:

1. solve the PDE using current time multipliers;
2. calculate unit-rate offsets from the resulting population path;
3. fit smooth Poisson multipliers conditional on those offsets;
4. damp the update in log space;
5. repeat until the largest multiplier change is below tolerance.

This is simultaneous statistical–PDE fitting in the sense that the statistical likelihood depends on the state generated by the PDE and the PDE boundary/reaction terms depend on the fitted rates. It is not a fully differentiable one-shot optimizer through every sparse solve.

## 8. Bayesian dynamic model

The NumPyro component uses a damped structural local-linear trend:

$$
b_t=\rho b_{t-1}+\sigma_b z_t,
\qquad
\eta_t=\eta_{t-1}+b_t,
\qquad
z_t\sim\mathcal N(0,1),
$$

$$
Y_t\sim\operatorname{Poisson}(O_te^{\eta_t}).
$$

The reference fit fixes $\rho=0.8$ because 16 observations do not identify a persistent-slope parameter reliably. The package supports fast mean-field SVI and an optional slower NUTS path. Forecast intervals propagate latent level, damped slope, and innovation uncertainty.

## 9. Diagnostics

Implemented checks include:

- Poisson deviance against the saturated model;
- Pearson residuals and a descriptive dispersion statistic;
- Ljung–Box residual autocorrelation test;
- rolling-origin one-step forecasts with no look-ahead;
- structural counterfactuals propagated through the PDE.

With only 16 monthly observations, formal time-series tests have low power and effective degrees of freedom are difficult to define under penalization. Their output is diagnostic, not definitive.
