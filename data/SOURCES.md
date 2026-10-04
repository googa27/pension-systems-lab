# Data sources and provenance

No synthetic observations are used.

## Public-copy bundling policy

This publication copy bundles only curated national public aggregate CSV extracts under `data/processed/`.
Raw Excel/workbook artifacts and legacy private output notebooks are not redistributed here because redistribution licensing and privacy review have not been resolved. Rebuilds from raw source material must be performed by maintainers from the original public source portals under the applicable source terms.

`data/PUBLIC_MANIFEST.json` records SHA-256 hashes, byte sizes, row counts, source URLs from the metadata below, and release cautions for every bundled curated CSV. Retrieval dates are recorded as unknown/not recorded rather than backfilled. Licensing status is explicitly `not_verified`: public accessibility is not treated as a verified redistribution or reuse license.

## Chile source families represented by curated CSV

### INE provisional vital statistics — April 2026

- Official topic page: https://www.ine.gob.cl/estadisticas-por-tema/demografia-y-poblacion/estadisticas-vitales
- Bundled curated CSVs:
  - `data/processed/ine_april_2026_births_by_maternal_age.csv`
  - `data/processed/ine_april_2026_deaths_by_age_sex.csv`
- Used tables:
  - births by maternal age and newborn sex;
  - deaths by age and sex.

### Censo 2024 national population by age and sex

- Official portal: https://censo2024.ine.gob.cl/resultados/
- Bundled curated CSV: `data/processed/censo2024_chile_population_by_age_sex.csv`
- Used table: national population by five-year age group and sex.

### INE monthly births/deaths, January 2025 through April 2026

- Bundled curated CSV: `data/processed/ine_monthly_births_deaths_2025_2026.csv`
- The file contains one public INE press-release URL per row.

## Validation totals

- April 2026 births: 12,542 total; 12,512 with specified maternal age in the fitted range table.
- April 2026 deaths: 10,041.
- Censo 2024 population: 18,480,432.

These identities are enforced by automated tests and by `pension-lab verify-data`.

## Extraction note

The curated CSVs are deterministic extracts from public source material. The main package consumes only the committed CSVs, so smoke checks run offline under `uv` without relying on an Excel engine or publishing raw workbooks.
