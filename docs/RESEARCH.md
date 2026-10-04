# Research and maintained-library decisions

## Scientific scope

The canonical theory is the **McKendrick–von Foerster age-structured transport equation** with age-zero inflow, mortality loss and migration/source terms. Pension economics additionally requires contribution participation, wage/return distributions, benefit rules, fiscal accounting and behavioural assumptions. Demography alone does not solve a pension system.

Country/year/group partial pooling and time-varying log-mortality rates can propagate posterior uncertainty into PDE projections. Distinguish process uncertainty, parameter uncertainty, measurement error and numerical error. Statistical diagnostics and posterior predictive checks are mandatory before policy interpretation.

## Selected tools and boundaries

| Capability | Selected library | Alternatives / selection reason | Adapter boundary |
|---|---|---|---|
| Sparse linear algebra and reference numerics | NumPy / SciPy | Maintained numerical primitives; no handwritten sparse solver | Arrays, grids and residual reports |
| 1D finite elements | scikit-fem | Lightweight assembly with explicit weak forms; FEniCSx/DOLFINx is heavier for this slice | P1 assembly plus SciPy solve; stabilization is explicit |
| Hierarchical inference | NumPyro / JAX | Differentiable model/NUTS; PyMC and Stan are valid alternative backends | Optional AFP imports; posterior arrays and diagnostics |
| MCMC diagnostics | ArviZ | Standard R-hat/ESS/PPC vocabulary | Planned acceptance gate; a smoke run is not accepted calibration |
| Aggregate tables and typed data | pandas / Pandera / Pydantic / xarray (lab) | Maintained schema/semantic primitives rather than a new dataframe framework | Source-acquisition and canonical-data contracts |
| Artifacts | Matplotlib / nbformat / nbclient (lab) | Reproducible figures and executable notebooks | Read-only model/data consumer |
| Monetary arithmetic | Python Decimal | Base-10 deterministic rounding; not binary float for account amounts | Research policy/account domain |

Domain code owns policy fixtures, explicit units and classifications, boundary assumptions, workflow composition, and evidence. It does **not** own a new inference engine, dataframe, sparse solver or FEM assembly framework. AFP's custom upwind finite-volume composition is justified by the small transparent age-transport contract and is tested against characteristic and physical mass-balance controls.

## Primary software sources

- scikit-fem documentation: <https://scikit-fem.readthedocs.io/>
- scikit-fem maintained project and license: <https://github.com/kinnala/scikit-fem>
- SciPy documentation/license: <https://scipy.org/>
- NumPyro documentation/project: <https://num.pyro.ai/> and <https://github.com/pyro-ppl/numpyro>
- JAX documentation: <https://docs.jax.dev/>
- ArviZ diagnostics: <https://python.arviz.org/>
- Pydantic / Pandera: <https://docs.pydantic.dev/> and <https://pandera.readthedocs.io/>
- Decimal semantics: <https://docs.python.org/3/library/decimal.html>

Review exact locked versions, wheel-platform support, upstream licenses, public API stability and deprecations at each release; repository visibility or an active package does not by itself prove suitability.

## Public evidence and international expansion

- [INE Chile](https://www.ine.gob.cl/): national population and vital-statistics aggregates.
- [Superintendencia de Pensiones](https://www.spensiones.cl/): regulation and administrative aggregates.
- [CMF Chile](https://www.cmfchile.cl/): relevant mortality/actuarial source releases; source identity does not confer unrestricted redistribution.
- [UN World Population Prospects](https://population.un.org/wpp/): international demography; modelled estimates are not interchangeable with administrative observations.
- [OECD Pensions](https://www.oecd.org/en/topics/pensions.html): international system structures and pension indicators.
- [ILOSTAT](https://ilostat.ilo.org/): labour and coverage denominators.
- [World Bank data](https://data.worldbank.org/): macroeconomic/coverage covariates and metadata.
- [Human Mortality Database](https://www.mortality.org/): potentially valuable mortality comparison; registration and terms must be reviewed before any ingestion/redistribution.

International acquisition is a roadmap, not a bundled database. Every source needs a licensing decision, harmonized units/grain, effective/vintage/retrieval times, lineage, hashes, replay rules and reference checks. Licensing and retrieval dates of recovered CSV fixtures remain explicitly unverified; unknown fields are not invented.
