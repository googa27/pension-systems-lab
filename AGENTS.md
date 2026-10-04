# Pension Systems Lab — agent/contributor contract

Public research workbench for Chilean/international pension-system questions, recovered demographic models, continuous-age McKendrick–von Foerster PDEs, numerical verification and Bayesian dynamics. Importable package: `chile_demographic_pde`; CLI: `pension-lab`. Research alpha, not an official demographic projection or pension rule engine.

## Canonical docs
- `docs/ARCHITECTURE.yaml`, `docs/ARCHITECTURE.md`: enforced architecture and legacy no-growth exceptions.
- `docs/THEORY.md`, `docs/LIMITATIONS.md`: equations, stabilization/clipping and inference limitations.
- `data/PUBLIC_MANIFEST.json`, `docs/SOURCE_REGISTRY.md`, `docs/DATA_GOVERNANCE.md`: aggregate fixtures, lineage, licensing/unknown vintage notes.
- `docs/ECOSYSTEM.md`, `docs/ROADMAP.md`: public dependency seams and international extension roadmap.

## Exact commands
```bash
uv sync --all-extras --dev
uv run pension-lab capabilities
uv run pension-lab verify-data
uv run pension-lab demo
JAX_PLATFORM_NAME=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
uv run python scripts/publication_guard.py --mode git
python scripts/scan_publication.py
```
The full suite includes actual MCMC and notebook execution; allow several minutes. Optional AFP integration: install the public `afp-operations` repository/package, then `uv run --no-sync pension-lab pension-demo`. No sibling source imports or path hacks.

## Architecture and extension workflow
- Domain semantic contracts own units/grain/classification; IO belongs behind acquisition adapters; demographic calibration and PDE backends remain distinct; notebooks/report builders are consumers of public contracts.
- Research maintained libraries and official sources before extension. Implement typed contracts, tests, registry metadata, then model/report changes.
- Keep `tests/unit`, `tests/integration`, `tests/e2e`, and `tests/architecture` meaningful. Architecture tests enforce import boundaries and exact scoped legacy size/dependency ratchets.
- Maximum 10 runtime `.py`/package entries per directory excluding `__init__.py`/metadata; <=500-line modules by default. Legacy exceptions are documented and may not grow.
- Synthetic scenarios must be labelled. Short-chain MCMC is software smoke, not accepted calibration. Negative-density clipping is disclosed, not a positivity proof.
- Curated national aggregates only; no credentials, person-level/member data, raw workbooks, portal exports, databases, agent histories, private repo names or machine paths. Never copy recovered notebook outputs to public artifacts.
- Definition of done: tests/lint/types/build, cleared notebook outputs, publication guard and redacted secret scan pass; docs and architecture agree. Review legal/data licensing status explicitly.
