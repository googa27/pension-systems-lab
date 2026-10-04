# Product Requirements Document (PRD)

## Purpose

Publish a hardened public research workbench for demographic and pension-system modeling while preserving the recovered `chile_demographic_pde` import name and serious existing numerical models.

## Non-goals

- No official INE, CMF, legal, actuarial, or regulatory projections.
- No private data, credentials, raw unpublished workbooks, or unresolved Excel redistribution.
- No GitHub writes or release automation from this hardening pass.
- No replacement of the legacy model with a scaffold.

## In scope

- Public package metadata and deterministic CLI entry point `pension-lab`.
- Smoke-scale two-sex FEM scenario for fast verification.
- JSON data verification for the bundled curated public CSVs.
- Optional public `afp_operations` integration via normal installed-package import only.
- Architecture guardrails for legacy size/fanout exceptions, import cycles, and layer boundaries.
- Documentation of public API, roadmap, data governance, and source registry.

## Users

- Researchers inspecting the demographic PDE prototype.
- Maintainers preparing a public repository.
- Downstream pension-system demos that need a stable public CLI and import surface.

## Success criteria

- `uv run pytest`, `uv run ruff check .`, and `uv run mypy` complete successfully.
- `pension-lab capabilities`, `pension-lab demo`, `pension-lab verify-data`, and `pension-lab pension-demo` run deterministically.
- Public docs do not claim raw Excel workbooks are bundled.
- Architecture tests fail on undocumented growth, fanout regressions, internal cycles, or import-boundary violations.
