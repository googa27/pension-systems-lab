# Library choices and alternatives

## PDE and finite elements

| Library | Strengths | Costs / limitations | Decision |
|---|---|---|---|
| **scikit-fem** | Lightweight, pure Python, explicit sparse matrices, easy P1 weak forms, straightforward `uv` installation | Less automation than FEniCSx; transport stabilization must be designed explicitly | **Chosen** for a transparent 1D FEM research model |
| FEniCSx | Industrial variational-form language, advanced meshes, PETSc/MPI scalability | Heavy native stack; less portable as a self-contained `uv` ZIP | Best when moving to large multiregional or multidimensional PDE systems |
| Firedrake | Excellent automated FEM and adjoints, PETSc performance | Heavy installation and platform requirements | Strong research alternative for PDE-constrained optimization at scale |
| FiPy | Mature finite-volume PDE library, good conservative transport tools | It is finite volume rather than FEM; less direct connection to the requested weak-form implementation | Good benchmark solver |
| JAX-FEM / custom JAX sparse solves | Differentiability and accelerator support | Smaller ecosystem, more custom numerical work, sparse autodiff maturity is a constraint | Future route for end-to-end gradient inference |

## Statistical inference

| Library | Strengths | Costs / limitations | Decision |
|---|---|---|---|
| **SciPy** | Stable optimizers, quadrature, sparse solves, spline primitives | Requires us to expose the demographic structure explicitly | **Chosen** for deterministic estimation |
| **NumPyro** | NUTS/SVI on JAX, composable structural state-space models, fast small Bayesian models | Sparse FEM solves are not automatically differentiable through SciPy | **Chosen** for Bayesian time factors |
| PyMC | Excellent modeling ergonomics, diagnostics and broad community | Heavier stack; no particular advantage for this small JAX-oriented layer | Strong alternative |
| Stan / CmdStanPy | Robust HMC and mature diagnostics | More friction for custom Python-side PDE coupling | Strong alternative if the dynamic layer is separated from the PDE |
| statsmodels | Reliable diagnostic tests and conventional time-series tooling | Not a PDE inference framework | **Chosen** for Ljung–Box diagnostics |

## Demography-specific alternatives

R has a more mature specialist ecosystem for some subproblems:

- `ungroup`: penalized composite-link ungrouping;
- `MortalitySmooth`: P-spline mortality surfaces;
- `StMoMo`: stochastic mortality models;
- `demography`: functional demographic forecasting.

Those packages are valuable benchmarks. The Python implementation here was selected because it enables one repository containing custom observation operators, FEM propagation, Bayesian state-space models, tests, and extension hooks.
