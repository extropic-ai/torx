"""Custom overdamped-Langevin gates (ULA / MALA) and validation utilities.

`LangevinGate` is a first-class `AbstractContinuousGate` whose `sample` runs a
real gradient-drift step on the soft-spin quartic Ising energy $V(x)$, so the
gate genuinely descends the potential it defines and samples the continuous
Boltzmann law $\\pi(x)\\propto e^{-V(x)/T}$. Two modes:

- unadjusted Langevin (ULA): the raw discretised step, biased at $O(\\varepsilon)$;
- Metropolis-adjusted Langevin (MALA): a Metropolis accept/reject on top of the
  same proposal, asymptotically exact for $e^{-V/T}$.

Driven generically by `HybridSampleSimulator`, which calls `gate.sample`
without any affineness assumption. Because the drift $\\nabla V$ is nonlinear,
this gate has no closed-form affine-Gaussian channel (`affine_parameters`), so
the exact-moment simulator does not apply; correctness is checked against a
tractable fine-grid quadrature reference instead.
"""

from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from _notebook_style import (
    COARSE_COLOR,
    EXACT_COLOR,
    FIGSIZE_CONVERGENCE,
    FIGSIZE_SINGLE,
    NEUTRAL_GRAY,
    TORX_COLOR,
)
from jaxtyping import Array, Float, Key, PyTree

from torx.psc import AbstractContinuousGate, HybridSites


def _continuous_sites_converter(x: HybridSites | int | list[int]) -> HybridSites:
    """Normalise a continuous-site spec to a `HybridSites` dict (no discrete sites)."""
    if isinstance(x, dict):
        return x
    return {"discrete": [], "continuous": [x] if not isinstance(x, list) else x}


class LangevinGate(AbstractContinuousGate[dict[str, Array], tuple[int, ...]]):
    r"""One overdamped-Langevin step on the soft-spin quartic Ising energy.

    The gate descends the potential

    $$V(x) = -\tfrac{\beta}{2}\, s^\top A_{\mathrm{adj}}\, s \;-\; h^\top s
    \;+\; \tfrac{\lambda_q}{4}\sum_i (x_i^2-1)^2,\qquad s_i=\tanh(x_i),$$

    with the update

    $$x' = x - \varepsilon\,\nabla V(x) + \sqrt{2 T \varepsilon}\,\xi,\qquad
    \xi\sim\mathcal N(0, I),$$

    The continuous-time limit of this dynamics has invariant law the Boltzmann
    measure $\pi(x)\propto e^{-V(x)/T}$. The discrete update above (ULA,
    ``metropolis=False``) only approximates $\pi$, carrying an
    $O(\varepsilon)$ step-size bias; with ``metropolis=True`` a
    Metropolis-Hastings accept/reject for that target removes the bias and
    makes the chain asymptotically exact for $\pi$.

    The energy coefficients and integrator constants travel in ``theta``:
    ``adj`` ($A_{\mathrm{adj}}$), ``field`` ($h$), ``beta`` ($\beta$),
    ``lam_quartic`` ($\lambda_q$), ``step`` ($\varepsilon$), ``temperature``
    ($T$). Only ``metropolis`` is structural (static).
    """

    _draw_label = "Langevin"

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]
    metropolis: bool = eqx.field(static=True, default=False)

    def energy(
        self, x: Float[Array, " d"], theta: dict[str, Array]
    ) -> Float[Array, ""]:
        """Soft-spin quartic Ising energy $V(x)$ that the gate descends."""
        s = jnp.tanh(x)
        coupling = -0.5 * theta["beta"] * (s @ (theta["adj"] @ s))
        bias = -(theta["field"] @ s)
        quartic = 0.25 * theta["lam_quartic"] * jnp.sum((x * x - 1.0) ** 2)
        return coupling + bias + quartic

    def init_params(self, key: Key[Array, ""]) -> dict[str, Array]:
        """Default ``theta``: a decoupled unit quartic well at unit temperature.

        The notebook overrides these with physics-derived values, mirroring
        `AffineGaussianGate`; this only fixes shapes and a valid fallback target.
        """
        d = sum(self.dims)
        return {
            "adj": jnp.zeros((d, d)),
            "field": jnp.zeros(d),
            "beta": jnp.asarray(0.0),
            "lam_quartic": jnp.asarray(1.0),
            "step": jnp.asarray(0.01),
            "temperature": jnp.asarray(1.0),
        }

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: dict[str, Array],
        info: Any = None,
        site_info: Any = None,
        return_aux: bool = False,
    ):
        """Advance the continuous state by one (optionally adjusted) Langevin step."""
        x = inputs["continuous"]
        eps = params["step"]
        temperature = params["temperature"]
        grad_x = jax.grad(self.energy)(x, params)

        noise_key, accept_key = jax.random.split(key)
        if info is None:
            noise = jax.random.normal(noise_key, x.shape, dtype=x.dtype)
        else:
            noise = info.normal(noise_key, x.shape, dtype=x.dtype)

        proposal = x - eps * grad_x + jnp.sqrt(2.0 * temperature * eps) * noise

        if not self.metropolis:
            output = proposal
        else:
            grad_prop = jax.grad(self.energy)(proposal, params)
            # log target ratio: -(V(x') - V(x)) / T
            log_target = (
                -(self.energy(proposal, params) - self.energy(x, params)) / temperature
            )
            # log proposal ratio log q(x | x') - log q(x' | x); the Gaussian
            # proposal has mean y - eps * grad V(y) and covariance 2 T eps I.
            fwd = proposal - (x - eps * grad_x)
            bwd = x - (proposal - eps * grad_prop)
            log_proposal = (jnp.sum(fwd**2) - jnp.sum(bwd**2)) / (
                4.0 * temperature * eps
            )
            accept_prob = jnp.exp(jnp.minimum(0.0, log_target + log_proposal))
            if info is None:
                accepted = jax.random.bernoulli(accept_key, accept_prob)
            else:
                accepted = info.bernoulli(accept_key, accept_prob).astype(bool)
            output = jnp.where(accepted, proposal, x)

        return (output, None) if return_aux else output


def langevin_theta(
    *,
    adj,
    field,
    beta: float,
    lam_quartic: float,
    step: float,
    temperature: float,
    dtype=jnp.float32,
) -> dict[str, Array]:
    """Pack the energy coefficients and integrator constants into a `theta` dict."""
    return {
        "adj": jnp.asarray(adj, dtype=dtype),
        "field": jnp.asarray(field, dtype=dtype),
        "beta": jnp.asarray(beta, dtype=dtype),
        "lam_quartic": jnp.asarray(lam_quartic, dtype=dtype),
        "step": jnp.asarray(step, dtype=dtype),
        "temperature": jnp.asarray(temperature, dtype=dtype),
    }


def boltzmann_1d_reference(gate: LangevinGate, theta: dict[str, Array], grid):
    """Exact 1-D Boltzmann density and CDF for the gate's own energy, by quadrature.

    Evaluates $V$ through ``gate.energy`` on a fine grid so the reference target
    is provably the same potential the gate descends, normalises
    $e^{-V(x)/T}$ by the trapezoidal rule, and returns ``(density, cdf)`` on the
    grid. This is an independent, tractable quadrature reference for the 1-D marginal.
    """
    grid = np.asarray(grid, dtype=float)
    temperature = float(theta["temperature"])
    energies = np.array(
        [float(gate.energy(jnp.asarray([xi], dtype=jnp.float32), theta)) for xi in grid]
    )
    weights = np.exp(-(energies - energies.min()) / temperature)
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    norm = trapezoid(weights, grid)
    density = weights / norm
    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))]
    )
    cdf = cdf / cdf[-1]
    return density, cdf


def ks_distance(samples, grid, ref_cdf) -> float:
    """One-sample Kolmogorov-Smirnov statistic against a reference CDF.

    ``ref_cdf`` is the reference CDF tabulated on ``grid`` (as from
    `boltzmann_1d_reference`); it is linearly interpolated onto the sorted
    samples, so the supremum is taken at the sample points where the empirical
    CDF jumps (the true KS supremum locations), not on the fixed grid. Returns
    the two-sided $\\sup_x |F_{\\mathrm{emp}}(x) - F_{\\mathrm{ref}}(x)|$.
    """
    xs = np.sort(np.asarray(samples).ravel())
    n = xs.size
    ref_at = np.interp(xs, np.asarray(grid, float), np.asarray(ref_cdf, float))
    emp_above = np.arange(1, n + 1) / n
    emp_below = np.arange(0, n) / n
    return float(max(np.max(emp_above - ref_at), np.max(ref_at - emp_below)))


def plot_mala_reference(grid, density, mala_samples, *, ks: float, temperature: float):
    """MALA terminal histogram over the exact quadrature Boltzmann density.

    The orange histogram is MALA's terminal marginal after the configured run; the slate curve is the
    exact $\\pi(x)\\propto e^{-V(x)/T}$ from `boltzmann_1d_reference`. Their
    overlap (small KS distance, annotated) is the correctness certificate.
    """
    grid = np.asarray(grid, dtype=float)
    mala_samples = np.asarray(mala_samples).ravel()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.hist(
        mala_samples,
        bins=60,
        density=True,
        color=TORX_COLOR,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.3,
        label="MALA samples",
    )
    ax.plot(
        grid,
        density,
        color=EXACT_COLOR,
        lw=2.0,
        label=r"exact $\pi(x)\propto e^{-V(x)/T}$",
    )
    ax.set_xlabel(r"soft-spin coordinate $x$")
    ax.set_ylabel("density")
    ax.set_title(rf"MALA vs exact Boltzmann marginal  ($T={temperature:g}$)")
    ax.annotate(
        f"KS distance = {ks:.3f}",
        xy=(0.03, 0.96),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color=EXACT_COLOR,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_ula_bias(
    grid,
    density,
    ula_samples,
    mala_samples,
    *,
    step: float,
    ks_ula: float,
    ks_mala: float,
):
    """ULA's step-size bias against the exact density, with MALA as the fix.

    Same 1-D target as `plot_mala_reference`. The gold histogram is ULA at a
    coarse step: it underweights the main mode and overfills the trough
    relative to the exact slate curve (large KS). The orange histogram is MALA
    on the identical proposal, which the accept/reject step pulls back onto the
    exact density (small KS).
    """
    grid = np.asarray(grid, dtype=float)
    ula_samples = np.asarray(ula_samples).ravel()
    mala_samples = np.asarray(mala_samples).ravel()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.hist(
        ula_samples,
        bins=60,
        density=True,
        color=COARSE_COLOR,
        alpha=0.55,
        edgecolor="white",
        linewidth=0.3,
        label=f"ULA  (KS = {ks_ula:.3f})",
    )
    ax.hist(
        mala_samples,
        bins=60,
        density=True,
        histtype="step",
        color=TORX_COLOR,
        linewidth=1.8,
        label=f"MALA  (KS = {ks_mala:.3f})",
    )
    ax.plot(
        grid,
        density,
        color=EXACT_COLOR,
        lw=2.0,
        label=r"exact $\pi(x)$",
    )
    ax.set_xlabel(r"soft-spin coordinate $x$")
    ax.set_ylabel("density")
    ax.set_title(rf"ULA step-size bias removed by MALA  ($\varepsilon={step:g}$)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_ks_vs_step(steps, ks_ula, ks_mala):
    """KS distance to the exact marginal vs step size, for ULA and MALA.

    ULA's KS grows with the step $\\varepsilon$ (its discretisation bias is
    $O(\\varepsilon)$); MALA stays flat near the Monte Carlo floor because the
    accept/reject step corrects the bias at every $\\varepsilon$.
    """
    steps = np.asarray(steps, dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE_CONVERGENCE)
    ax.plot(
        steps,
        ks_ula,
        marker="o",
        color=COARSE_COLOR,
        lw=1.9,
        label="ULA (unadjusted)",
    )
    ax.plot(
        steps,
        ks_mala,
        marker="s",
        color=TORX_COLOR,
        lw=1.9,
        label="MALA (adjusted)",
    )
    ax.set_xlabel(r"Langevin step size  $\varepsilon$")
    ax.set_ylabel("KS distance to exact marginal")
    ax.set_title("Step-size bias: ULA grows, MALA stays flat")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_site_mean_parity(
    mu_short, mu_long, *, n: int, reps_short: int, reps_long: int
):
    """Per-site mean soft spin from a short MALA run vs a long-run MALA reference.

    Points on the diagonal show the deployed (short) MALA chain has reached the
    same per-site statistics as a much longer MALA reference on the identical
    energy: a convergence check on the graph, using long-run MALA as ground
    truth (never a discrete-Gibbs comparison).
    """
    mu_short = np.asarray(mu_short)
    mu_long = np.asarray(mu_long)
    diff = float(np.max(np.abs(mu_short - mu_long)))
    lo = float(min(mu_short.min(), mu_long.min()))
    hi = float(max(mu_short.max(), mu_long.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lims = (lo - pad, hi + pad)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.plot(lims, lims, color=NEUTRAL_GRAY, lw=0.9, ls="--", label=r"$y=x$")
    ax.scatter(
        mu_long,
        mu_short,
        s=42,
        color=TORX_COLOR,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
        label=f"site mean ({n} sites)",
    )
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal")
    ax.set_xlabel(
        rf"long-run MALA mean  $\langle\tanh x_i\rangle$  ({reps_long} steps)"
    )
    ax.set_ylabel(rf"deployed MALA mean  ({reps_short} steps)")
    ax.set_title("Deployed vs long-run MALA per-site mean")
    ax.annotate(
        f"max abs diff = {diff:.3f}",
        xy=(0.04, 0.96),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9,
        color=EXACT_COLOR,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig
