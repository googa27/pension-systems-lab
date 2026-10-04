# Data governance

## Publication data rule

This public copy bundles only curated public CSV files under `data/processed/`.
Raw Excel/workbook files are not published here because redistribution licensing is unresolved. The package must not silently replace missing official/public data with synthetic observations.

## Bundled files

- `data/processed/censo2024_chile_population_by_age_sex.csv`
- `data/processed/ine_monthly_births_deaths_2025_2026.csv`
- `data/processed/ine_april_2026_births_by_maternal_age.csv`
- `data/processed/ine_april_2026_deaths_by_age_sex.csv`

## Verification

Run:

```bash
pension-lab verify-data
```

The command emits JSON with observed totals, expected public baseline totals, SHA-256 checksums, and file byte sizes. `data/PUBLIC_MANIFEST.json` is the publication manifest for the curated CSV tranche; it hashes every `data/processed/*.csv` file and records source URLs, row counts, unknown retrieval-date status, and unverified licensing status.

## Privacy and security

- No private/person-level data is required or bundled.
- No passwords or credentials are required for the CLI smoke checks.
- No sibling repository imports or local path hacks are permitted for public integrations.
- Notebook outputs are cleared; costly full notebook execution is not part of the public smoke gate.
- Raw source workbooks and legacy private output notebooks are not part of the public publication artifact.

## Modeling caveats

- Censo 2024 grouped enumerated counts and resident population estimates are separate population bases.
- Current reference scenarios use an explicitly disclosed Censo reconstruction proxy until reviewed resident single-age estimates are available.
- Monthly vital-statistics counts are provisional.
- Scenario forecasts are conditional research outputs, not official projections or causal estimates.
