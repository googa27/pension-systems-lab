"""Bayesian structural time-series layer implemented with NumPyro."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax import lax
from numpy.typing import ArrayLike, NDArray
from numpyro.diagnostics import summary as posterior_summary
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import Adam

from chile_demographic_pde.dynamics.penalized_trend import PenalizedPoissonTrend

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True, eq=False)
class BayesianForecast:
    __hash__ = None  # type: ignore[assignment]

    mean: FloatArray
    median: FloatArray
    lower: FloatArray
    upper: FloatArray
    draws: FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class BayesianTrendResult:
    __hash__ = None  # type: ignore[assignment]

    posterior_mean: FloatArray
    posterior_lower: FloatArray
    posterior_upper: FloatArray
    samples: dict[str, FloatArray]
    divergences: int
    maximum_r_hat: float
    minimum_effective_sample_size: float
    inference_method: str
    final_loss: float
    trend_damping: float

    def forecast(
        self,
        horizon: int,
        *,
        random_seed: int = 0,
        confidence: float = 0.95,
    ) -> BayesianForecast:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        rng = np.random.default_rng(random_seed)
        levels = np.asarray(self.samples["levels"], dtype=float)
        slopes = np.asarray(self.samples["slopes"], dtype=float)
        sigma_curvature = np.asarray(self.samples["sigma_curvature"], dtype=float)
        draws = np.empty((levels.shape[0], horizon), dtype=float)
        current_level = levels[:, -1].copy()
        current_slope = slopes[:, -1].copy()
        for step in range(horizon):
            current_slope = self.trend_damping * current_slope + rng.normal(0.0, sigma_curvature)
            current_level += current_slope
            draws[:, step] = np.exp(np.clip(current_level, -30.0, 30.0))
        alpha = 0.5 * (1.0 - confidence)
        return BayesianForecast(
            mean=np.mean(draws, axis=0),
            median=np.median(draws, axis=0),
            lower=np.quantile(draws, alpha, axis=0),
            upper=np.quantile(draws, 1.0 - alpha, axis=0),
            draws=draws,
        )


def _local_linear_trend_model(y: jax.Array, offset: jax.Array, trend_damping: float = 0.8) -> None:
    """Non-centred second-order random walk for the latent log multiplier."""

    observations = y.shape[0]
    baseline = jnp.log(jnp.sum(y) / jnp.sum(offset))
    initial_level = numpyro.sample("initial_level", dist.Normal(baseline, 0.10))
    initial_slope = numpyro.sample("initial_slope", dist.Normal(0.0, 0.05))
    sigma_curvature = numpyro.sample("sigma_curvature", dist.HalfNormal(0.05))
    curvature_z = numpyro.sample("curvature_z", dist.Normal(0.0, 1.0).expand((observations - 2,)))

    second_level = initial_level + initial_slope

    def transition(
        carry: tuple[jax.Array, jax.Array], innovation: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        previous_level, previous_slope = carry
        current_slope = trend_damping * previous_slope + sigma_curvature * innovation
        current_level = previous_level + current_slope
        return (current_level, current_slope), (current_level, current_slope)

    _, (later_levels, later_slopes) = lax.scan(
        transition,
        (second_level, initial_slope),
        curvature_z,
    )
    levels = jnp.concatenate((jnp.asarray([initial_level, second_level]), later_levels))
    slopes = jnp.concatenate((jnp.asarray([initial_slope, initial_slope]), later_slopes))
    numpyro.deterministic("levels", levels)
    numpyro.deterministic("slopes", slopes)
    numpyro.sample("observed", dist.Poisson(offset * jnp.exp(levels)), obs=y)


def fit_bayesian_local_linear_trend(
    observations: ArrayLike,
    *,
    offset: ArrayLike | None = None,
    num_warmup: int = 250,
    num_samples: int = 400,
    chains: int = 1,
    random_seed: int = 0,
    inference_method: str = "svi",
    svi_steps: int = 2_500,
    trend_damping: float = 0.8,
) -> BayesianTrendResult:
    """Fit a Bayesian Poisson local-linear trend.

    ``inference_method="svi"`` uses NumPyro's automatic-normal variational
    guide and is the portable notebook default. ``"nuts"`` runs exact HMC
    sampling and reports R-hat, effective sample size, and divergences; it is
    slower for the tightly identified high-count series used here.
    """

    y = np.asarray(observations, dtype=float)
    exposure = np.ones_like(y) if offset is None else np.asarray(offset, dtype=float)
    if y.ndim != 1 or y.size < 3 or np.any(y < 0.0):
        raise ValueError("observations must be a nonnegative vector of length at least three")
    if exposure.shape != y.shape or np.any(exposure <= 0.0):
        raise ValueError("offset must be positive and match observations")
    if inference_method not in {"svi", "nuts"}:
        raise ValueError("inference_method must be 'svi' or 'nuts'")
    if not 0.0 <= trend_damping < 1.0:
        raise ValueError("trend_damping must be in [0, 1)")

    y_jax = jnp.asarray(y)
    exposure_jax = jnp.asarray(exposure)
    key = jax.random.key(random_seed)

    if inference_method == "svi":
        if svi_steps < 1:
            raise ValueError("svi_steps must be positive")
        deterministic_initial = (
            PenalizedPoissonTrend(penalty=30.0).fit(y, offset=exposure).log_fitted
        )
        initial_slopes = np.diff(deterministic_initial)
        initial_curvature = initial_slopes[1:] - trend_damping * initial_slopes[:-1]
        initial_sigma = max(float(np.sqrt(np.mean(initial_curvature**2))), 0.01)
        initial_values = {
            "initial_level": float(deterministic_initial[0]),
            "initial_slope": float(deterministic_initial[1] - deterministic_initial[0]),
            "sigma_curvature": initial_sigma,
            "curvature_z": initial_curvature / initial_sigma,
        }
        guide = AutoNormal(
            _local_linear_trend_model,
            init_loc_fn=init_to_value(values=initial_values),
            init_scale=0.02,
        )
        svi = SVI(
            _local_linear_trend_model,
            guide,
            Adam(step_size=0.005),
            Trace_ELBO(),
        )
        svi_result = svi.run(
            key,
            svi_steps,
            y=y_jax,
            offset=exposure_jax,
            trend_damping=trend_damping,
            progress_bar=False,
        )
        posterior_key, predictive_key = jax.random.split(jax.random.fold_in(key, 1))
        latent_samples = guide.sample_posterior(
            posterior_key,
            svi_result.params,
            sample_shape=(num_samples,),
            y=y_jax,
            offset=exposure_jax,
            trend_damping=trend_damping,
        )
        deterministic = Predictive(
            _local_linear_trend_model,
            posterior_samples=latent_samples,
            return_sites=["levels", "slopes"],
        )(
            predictive_key,
            y=y_jax,
            offset=exposure_jax,
            trend_damping=trend_damping,
        )
        merged = {**latent_samples, **deterministic}
        samples = {name: np.asarray(value, dtype=float) for name, value in merged.items()}
        divergences = 0
        maximum_r_hat = float("nan")
        minimum_effective_sample_size = float("nan")
        final_loss = float(np.asarray(svi_result.losses)[-1])
    else:
        kernel = NUTS(
            _local_linear_trend_model,
            target_accept_prob=0.95,
            max_tree_depth=12,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=chains,
            progress_bar=False,
            chain_method="sequential",
        )
        mcmc.run(key, y=y_jax, offset=exposure_jax, trend_damping=trend_damping)
        chain_samples = mcmc.get_samples(group_by_chain=True)
        raw_samples = mcmc.get_samples(group_by_chain=False)
        samples = {name: np.asarray(value, dtype=float) for name, value in raw_samples.items()}
        diagnostics = posterior_summary(chain_samples, group_by_chain=True)
        r_hats = np.concatenate(
            [np.asarray(stats["r_hat"], dtype=float).reshape(-1) for stats in diagnostics.values()]
        )
        effective_sizes = np.concatenate(
            [np.asarray(stats["n_eff"], dtype=float).reshape(-1) for stats in diagnostics.values()]
        )
        divergences = int(
            np.asarray(mcmc.get_extra_fields(group_by_chain=False)["diverging"]).sum()
        )
        maximum_r_hat = float(np.nanmax(r_hats))
        minimum_effective_sample_size = float(np.nanmin(effective_sizes))
        final_loss = float("nan")

    rate_draws = np.exp(np.clip(samples["levels"], -30.0, 30.0))
    posterior_draws = rate_draws * exposure[None, :]
    return BayesianTrendResult(
        posterior_mean=np.mean(posterior_draws, axis=0),
        posterior_lower=np.quantile(posterior_draws, 0.025, axis=0),
        posterior_upper=np.quantile(posterior_draws, 0.975, axis=0),
        samples=samples,
        divergences=divergences,
        maximum_r_hat=maximum_r_hat,
        minimum_effective_sample_size=minimum_effective_sample_size,
        inference_method=inference_method,
        final_loss=final_loss,
        trend_damping=trend_damping,
    )
