# Public API

## Python package

The historical import name is retained:

```python
import chile_demographic_pde
```

Stable public surfaces for this publication pass:

- `chile_demographic_pde.ModelConfig`
- `chile_demographic_pde.data.loaders.load_project_data`
- `chile_demographic_pde.pde.fem.AgeTransportFEM`
- `chile_demographic_pde.pde.model.TwoSexDemographicModel`
- `chile_demographic_pde.cli.*_result` helpers for deterministic CLI payloads

Other modules remain available for compatibility but should be treated as research internals unless documented by tests and future API notes.

## CLI

All commands print one JSON object to stdout with sorted keys.

```bash
pension-lab capabilities
pension-lab demo
pension-lab verify-data
pension-lab pension-demo
```

- `capabilities`: public capability manifest, data policy, and model inventory.
- `demo`: a small actual two-sex FEM smoke scenario; it does not execute the costly full notebook.
- `verify-data`: validates curated CSV totals and emits file SHA-256 checksums.
- `pension-demo`: imports the optional public `afp_operations` package if installed and invokes one of its public demo callables (`public_demo`, `demo`, or `run_demo`). If the dependency is absent, it returns a deterministic JSON status rather than using sibling paths or private internals. The `afp` extra is reserved so a future public package pin can be added after release without changing the CLI name.

## Projection disclaimer

The API produces research scenarios and data checks. It does not provide official population projections, legal pension advice, regulatory mortality projections, or actuarial certification.
