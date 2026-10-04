# Ecosystem and integration contracts

**Public roadmap:** [Pension Systems — PDE & Bayesian Research](https://github.com/users/googa27/projects/36).

## Implemented connection

| Repository | Ownership | Connection |
|---|---|---|
| [pension-systems-lab](https://github.com/googa27/pension-systems-lab) | Country scenarios, demographic evidence, data/source contracts, comparative research | Imports the installed AFP public API through `pension-lab pension-demo` |
| [afp-operations](https://github.com/googa27/afp-operations) | Reusable policy fixtures, financial arithmetic, operations composition, transport and inference kernels | Exposes typed APIs plus JSON capability/demo/Bayesian smoke commands |

From a lab checkout, install the companion package (not yet on PyPI):

```bash
uv sync --dev
uv pip install "git+https://github.com/googa27/afp-operations.git"
uv run --no-sync pension-lab pension-demo
```

No `sys.path` changes, sibling private imports, credential requests, or live transfers are used. For reproducible research releases, pin the repository commit or use an audited wheel and record its hash.

## Verified public adjacent projects

- [finite_element_options](https://github.com/googa27/finite_element_options): public FEM method/evidence architecture; financial-option solvers are **not** silently treated as age-transport solvers.
- [finite_difference_options](https://github.com/googa27/finite_difference_options): public finite-difference method/evidence architecture and future solver-conformance comparison.

These are currently architectural/research links, not installed runtime dependencies. Sharing a discretization family does not establish interchangeable equation, boundary, regularity, or stability contracts.

## Canonical provider seams

| Role | Required public contract | State |
|---|---|---|
| Data platform | Source ID, classification, units, grain, vintage, quality, lineage, content identity and replay | Registry/contracts in lab; large-scale external ingestion is roadmap |
| Formulation provider | Equation/state identity, policy assumptions, reductions, solver-route compatibility | Typed local research contracts; external adapter roadmap |
| Numerical backend | Explicit grid/measure, inflow/outflow, mortality, stabilization, convergence and residual evidence | Local FV/P1-FEM implemented; extended backends roadmap |
| Artifact renderer | Audience, model/data provenance, calibration status, deterministic artifact QA | Reproducible README assets and lab notebook/report builders |

## Private connections

Private ecosystem ownership and dependency links are managed in a **private coordination issue/Project**, outside these public repositories. No private repository is made public. Private adapter names, paths, credentials, member records, portal exports and agent histories must never enter public manifests or CI artifacts.

Actual private-data operation requires a separately approved least-privilege adapter. A GitHub roadmap connection is not evidence that such an adapter is implemented or operational.
