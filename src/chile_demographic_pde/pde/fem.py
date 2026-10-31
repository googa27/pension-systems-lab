"""Finite-element solver for the age-transport McKendrick equation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix
from skfem import Basis, BilinearForm, ElementLineP1, MeshLine, asm, condense, solve
from skfem.helpers import dot, grad

FloatArray = NDArray[np.float64]


@BilinearForm
def _mass(u: object, v: object, _w: object) -> object:
    return u * v  # type: ignore[operator]


@BilinearForm
def _advection(u: object, v: object, _w: object) -> object:
    return u.grad[0] * v  # type: ignore[attr-defined]


@BilinearForm
def _diffusion(u: object, v: object, _w: object) -> object:
    return dot(grad(u), grad(v))


@BilinearForm
def _reaction(u: object, v: object, w: object) -> object:
    return w.mu * u * v  # type: ignore[attr-defined]


class AgeTransportFEM:
    """P1 FEM with implicit Euler and optional artificial diffusion.

    It solves ``n_t + n_a = -mu n + source`` on ``[0, maximum_age]``.
    The age-zero inflow is imposed strongly as a Dirichlet condition.  The
    right boundary is natural outflow.  Small negative values caused by the
    unstabilized transport operator are clipped after solving; the provided
    diffusion parameter should be chosen large enough that clipping is tiny.
    """

    def __init__(
        self,
        maximum_age: float = 105.0,
        nodes: int = 211,
        artificial_diffusion: float = 0.015,
    ) -> None:
        if not np.all(np.isfinite([maximum_age, artificial_diffusion])):
            raise ValueError("FEM configuration must be finite")
        if maximum_age <= 0.0 or nodes < 3 or artificial_diffusion < 0.0:
            raise ValueError("Invalid FEM configuration")
        self.mesh = MeshLine(np.linspace(0.0, maximum_age, nodes, dtype=float))
        self.basis = Basis(self.mesh, ElementLineP1())
        self.age = np.asarray(self.basis.doflocs[0], dtype=float)
        self.artificial_diffusion = artificial_diffusion
        self._mass = csc_matrix(asm(_mass, self.basis))
        self._advection = csc_matrix(asm(_advection, self.basis))
        self._diffusion = csc_matrix(asm(_diffusion, self.basis))

    def step(
        self,
        state: ArrayLike,
        mortality: ArrayLike,
        inflow: float,
        dt: float,
        source: ArrayLike | float = 0.0,
    ) -> FloatArray:
        population = np.asarray(state, dtype=float)
        hazard = np.asarray(mortality, dtype=float)
        if population.shape != self.age.shape or hazard.shape != self.age.shape:
            raise ValueError("state and mortality must match the FEM age grid")
        source_vector = np.broadcast_to(np.asarray(source, dtype=float), self.age.shape)
        if not (
            np.all(np.isfinite(population))
            and np.all(np.isfinite(hazard))
            and np.all(np.isfinite(source_vector))
            and np.all(np.isfinite([inflow, dt]))
        ):
            raise ValueError("state, mortality, source, inflow, and dt must be finite")
        if np.any(hazard < 0.0) or inflow < 0.0 or dt <= 0.0:
            raise ValueError("mortality and inflow must be nonnegative; dt must be positive")
        reaction = csc_matrix(asm(_reaction, self.basis, mu=self.basis.interpolate(hazard)))
        operator = self._mass + dt * (
            self._advection + reaction + self.artificial_diffusion * self._diffusion
        )
        rhs = np.asarray(self._mass @ population + dt * (self._mass @ source_vector), dtype=float)

        # Delegate nonzero Dirichlet elimination and expansion to scikit-fem.
        boundary_values = np.zeros_like(population)
        boundary_values[0] = inflow
        inflow_dofs = self.basis.get_dofs(lambda x: x[0] == 0.0)
        solution = np.asarray(
            solve(*condense(operator, rhs, x=boundary_values, D=inflow_dofs)), dtype=float
        )
        return np.maximum(solution, 0.0)
