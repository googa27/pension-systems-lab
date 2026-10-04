# Source registry catalog

## Status legend

- `bundled_curated_csv`: included under `data/processed/` in this public copy.
- `planned_not_ingested`: named for roadmap context only; no data is included or consumed.

## Chile sources bundled as curated CSV

| Registry key | Source | Status | Bundled curated file(s) | Retrieval date | License review | Notes |
|---|---|---|---|---|---|---|
| `cl_ine_censo_2024_age_sex` | Instituto Nacional de Estadísticas (INE), Censo 2024 results portal: https://censo2024.ine.gob.cl/resultados/ | `bundled_curated_csv` | `censo2024_chile_population_by_age_sex.csv` | Unknown/not recorded | Not verified | National grouped population by age and sex. Censo enumerated basis, not resident single-age estimates. |
| `cl_ine_vital_april_2026_births_age` | INE provisional vital statistics topic page and April 2026 press release | `bundled_curated_csv` | `ine_april_2026_births_by_maternal_age.csv` | Unknown/not recorded | Not verified | Births by maternal age for April 2026; excludes unspecified maternal age in the age-schedule fit. |
| `cl_ine_vital_april_2026_deaths_age_sex` | INE provisional vital statistics topic page and April 2026 press release | `bundled_curated_csv` | `ine_april_2026_deaths_by_age_sex.csv` | Unknown/not recorded | Not verified | Deaths by age and sex for April 2026. |
| `cl_ine_monthly_births_deaths_2025_2026` | INE monthly press releases; row-level URLs are listed in `data/processed/ine_monthly_births_deaths_2025_2026.csv` and `data/PUBLIC_MANIFEST.json` | `bundled_curated_csv` | `ine_monthly_births_deaths_2025_2026.csv` | Unknown/not recorded | Not verified | Monthly total births/deaths with one public source URL per row. |
| `cl_cmf_tm2020_mortality` | Comisión para el Mercado Financiero (CMF) TM-2020 regulatory mortality material | `planned_registry_adapter` | none in `data/processed/` | Not applicable | Not verified | Adapter code and tests are retained; public package must not imply bundled raw workbook redistribution or official projections. |
| `cl_bcch_market_series` | Banco Central de Chile public series APIs/pages | `planned_registry_adapter` | none in `data/processed/` | Not applicable | Not verified | Finance adapter contracts retained; no market data files are bundled in this CSV tranche. |

## International sources planned, not ingested

| Registry key | Source family | Status | Intended review before ingestion |
|---|---|---|---|
| `un_population` | United Nations population datasets/prospects | `planned_not_ingested` | License, vintage, geography harmonization, checksum registry, population-basis semantics. |
| `oecd_pensions_ageing` | OECD pension, ageing, and labour indicators | `planned_not_ingested` | Terms of use, indicator definitions, country-code mapping, release-vintage tracking. |
| `ilo_labour_social_security` | International Labour Organization labour and social-security indicators | `planned_not_ingested` | License, frequency, sector/coverage semantics, missingness rules. |
| `world_bank_indicators` | World Bank population, macro, and pension-relevant indicators | `planned_not_ingested` | API terms, indicator registry, unit normalization, vintage checks. |

No international source listed above is currently bundled, downloaded, or ingested by the package.
