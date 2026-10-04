# Roadmap

## Current public baseline

- Curated public Chile CSVs only.
- Continuous-age fertility/mortality prototype with two-sex FEM propagation.
- Chile CMF/finance/vital-statistics adapters retained for compatibility and tests.
- Optional public `afp_operations` integration by installed dependency.
- Public README assets, publication guard, CI workflow, and architecture/publication tests.

## Near term

1. Keep the default test suite CPU-pinned and reproducible in CI.
2. Split audited legacy modules only when a real feature requires touching them.
3. Add examples that consume `pension-lab verify-data` JSON without executing notebooks.
4. Publish clear data-source manifests for every bundled curated CSV.

## Medium term

1. Versioned data-source registry generated from public metadata.
2. Stable Python API policy with deprecation warnings for research internals.
3. Optional `afp_operations` extra once the public package is released and versioned.
4. Resident single-age estimate ingestion when redistribution and source contracts are reviewed.

## International planned, not ingested

The following are registry targets only; no data from them is bundled or ingested in this publication pass:

- United Nations population and demographic indicators.
- OECD pension and ageing indicators.
- ILO labour-market and social-security indicators.
- World Bank population, macroeconomic, and pension-relevant indicators.

Each future ingestion requires license review, source URL registry entries, checksums, data-basis semantics, and tests before becoming package input.
