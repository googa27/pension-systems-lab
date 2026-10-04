# Limitations and what would materially improve the model

## Data limitations

1. **One detailed age profile**: the bundled INE age distributions are for April 2026. Monthly dynamics therefore scale a fixed age shape. A series of age-by-month or age-by-year files would identify age–period interactions.
2. **Population basis and approximate exposures**: Censo enumerated grouped
   counts and INE resident-population estimates are different population bases.
   The canonical PDE state uses resident estimates. Until the official-data
   tranche supplies official resident single-age estimates, the current
   reference pipeline still propagates the exact-bin descriptive
   Censo-enumerated reconstruction as an explicitly disclosed initial proxy.
   It also uses Censo 2024 counts as approximate exposure anchors for
   2025–2026; they are not interchangeable with resident-derived person-time
   exposures.
3. **Coarse population bins**: Censo counts are in five-year ages and an open
   $85+$ group. The optional nonnegative P1 interpolation preserves every
   supplied official Censo total exactly to numerical tolerance, including
   $85+$ after explicit truncation to $[85,105]$, but it is still a descriptive
   within-bin interpolation, not observed single-age population. Its within-bin
   shape, especially on $[85,105]$, is regularization-dependent; the tail
   shape and domain beyond 105 remain unidentified unless a separately tested
   tail model is chosen. Its residual,
   conditioning, active-bound, local-curvature, and actual
   coarse-versus-refined grid-comparison diagnostics are exposed. Displaying
   the open $85+$ total as a density uses the disclosed model-domain interval
   $[85,105]$; Censo itself does not assert a finite endpoint at 105.
   The mortality tail is constrained to be nondecreasing after age 40 because
   the open-ended exposure bin cannot identify an unconstrained extreme-age
   shape.
4. **Infant mortality**: the Censo-aligned reference fit combines deaths below age five. A serious infant model should retain neonatal, post-neonatal, month, and day resolution.
5. **Migration**: the model interface includes migration, but the scenarios use zero age-specific migration. Long-horizon population totals are therefore conditional and not official projections.
6. **Provisional registrations**: monthly INE counts are provisional. A release-delay or vintage model would be required for real-time nowcasting.

## Statistical limitations

- Sixteen months are not enough to distinguish trend, seasonality, shocks, and registration revisions reliably.
- Penalized latent monthly factors can fit the observed totals closely; naïve residual degrees of freedom overstate certainty.
- Intervention multipliers are scenarios, not causal effects.
- No cause-specific competing risks, parity structure, region, education, or nationality are included.
- The legacy FEM transport solver uses small artificial diffusion and clips tiny negative numerical states after each solve. This is a pragmatic stabilization/guardrail, not a proof that the scheme is rigorously positivity preserving for arbitrary rates, grids, or time steps.
- The optional NumPyro MCMC path is suitable only as a short research diagnostic in this public candidate unless rerun with multiple chains, convergence diagnostics, and domain calibration review. A short single-chain run has undefined R-hat and is not an accepted policy calibration.

## Highest-value next data

1. Annual or monthly birth counts by exact maternal age and birth order.
2. Deaths by single age, sex, cause, region, and date of occurrence.
3. Official resident-population estimates and person-time exposures by single
   age, sex, region, and year.
4. Internal and international migration origin–destination flows by age and sex.
5. Provisional/final publication vintages to estimate registration delay.
6. Censo 2024 microdata variables for parity, migration, education, nationality, and commune.
