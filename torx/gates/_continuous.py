"""Continuous and hybrid probabilistic gates."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ._base import (
    AbstractContinuousGate,
    AbstractControlledContinuousGate,
    HybridSites,
)


def _continuous_sites_converter(x: HybridSites | list[int]) -> HybridSites:
    if isinstance(x, dict):
        return x
    return {"discrete": [], "continuous": [x] if not isinstance(x, list) else x}


def _hybrid_sites_converter(
    x: HybridSites | tuple[int | list[int], int | list[int]],
) -> HybridSites:
    if isinstance(x, dict):
        return x
    discrete, continuous = x
    return {
        "discrete": [discrete] if isinstance(discrete, int) else list(discrete),
        "continuous": [continuous] if isinstance(continuous, int) else list(continuous),
    }


class GaussianNoiseGate(AbstractContinuousGate[Float[Array, " d"], tuple[int, ...]]):
    r"""
    Additive Gaussian noise gate.

    This gate adds independent Gaussian noise to the continuous state:
    $x' = x + \mathcal{N}(0, \exp(\theta))$

    The variance is parameterized as $\exp(\theta)$ to ensure positivity.
    """

    _draw_label = "Gauss"

    theta: Float[Array, " d"]
    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def sample(
        self,
        substate: dict[str, Array],
        key: Key[Array, ""],
    ) -> Float[Array, " d"]:
        x = substate["continuous"]
        var = jnp.exp(self.theta)
        return x + jax.random.normal(key, x.shape) * jnp.sqrt(var)


class AffineGaussianGate(AbstractContinuousGate[dict[str, Array], tuple[int, ...]]):
    r"""
    Affine transformation with Gaussian noise.

    This gate applies an affine transformation followed by Gaussian noise:
    $x' = Ax + b + \mathcal{N}(0, \sigma^2)$ where $\sigma^2 = \exp$(`log_var`)

    The theta dict contains:

    - `A`: Transformation matrix of shape (d, d)
    - `b`: Bias vector of shape (d,)
    - `log_var`: Log-variance of shape (d,)
    """

    _draw_label = "AffineGauss"

    theta: dict[str, Array]
    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def sample(
        self,
        substate: dict[str, Array],
        key: Key[Array, ""],
    ) -> Float[Array, " d"]:
        x = substate["continuous"]
        A, b = self.theta["A"], self.theta["b"]
        var = jnp.exp(self.theta["log_var"])
        noise = jax.random.normal(key, x.shape) * jnp.sqrt(var)
        return A @ x + b + noise


class MixtureGaussianGate(
    AbstractControlledContinuousGate[dict[str, Array], tuple[int, ...]]
):
    r"""
    Mixture Gaussian gate controlled by a discrete site.

    This gate samples from one of K Gaussian components, where the component
    is selected by the discrete control site. Given discrete state $k$:
    $x' = x + \mu_k + \mathcal{N}(0, \sigma_k^2)$ where
    $\sigma_k^2 = \exp$(`log_vars[k]`)

    The discrete site should take values in $\{0, 1, ..., K-1\}$ where K is
    the number of mixture components.

    !!! note

        dims only specifies continuous dimensions since we only support
        control by a single discrete site for now.

    The theta dict contains:

    - `means`: Component means of shape (K, d)
    - `log_vars`: Component log-variances of shape (K, d)
    """

    _draw_label = "MoG"

    theta: dict[str, Array]
    sites: HybridSites = eqx.field(converter=_hybrid_sites_converter)
    dims: tuple[int, ...]

    def sample(
        self,
        substate: dict[str, Array],
        key: Key[Array, ""],
    ) -> Float[Array, " d"]:
        x = substate["continuous"]
        k = self._get_discrete_control(substate)
        var = jnp.exp(self.theta["log_vars"][k])
        mean = self.theta["means"][k]
        return x + mean + jax.random.normal(key, x.shape) * jnp.sqrt(var)


class JumpDiffusionGate(
    AbstractControlledContinuousGate[dict[str, Array], tuple[int, ...]]
):
    r"""
    Jump-diffusion gate controlled by a discrete firing site.

    This gate applies continuous diffusion, and conditionally applies a jump
    based on the discrete control site. Given discrete state $j \in \{0, 1\}$:

    $$
    x' = x + \mathcal{N}(0, \sigma_d^2)
    + j \cdot (\mu_j + \mathcal{N}(0, \sigma_j^2))
    $$

    where $\sigma_d^2 = \exp$(`diff_log_var`) and
    $\sigma_j^2 = \exp$(`jump_log_var`).

    When $j = 0$, only diffusion is applied. When $j = 1$, both diffusion and
    the jump are applied.

    !!! note

        dims only specifies continuous dimensions since we only support
        control by a single discrete site for now.

    The theta dict contains:

    - `diff_log_var`: Diffusion log-variance of shape (d,)
    - `jump_mean`: Jump mean of shape (d,)
    - `jump_log_var`: Jump log-variance of shape (d,)
    """

    _draw_label = "JumpDiff"

    theta: dict[str, Array]
    sites: HybridSites = eqx.field(converter=_hybrid_sites_converter)
    dims: tuple[int, ...]

    def sample(
        self,
        substate: dict[str, Array],
        key: Key[Array, ""],
    ) -> Float[Array, " d"]:
        x = substate["continuous"]
        do_jump = self._get_discrete_control(substate)
        k1, k2 = jax.random.split(key)
        diff_var = jnp.exp(self.theta["diff_log_var"])
        x = x + jax.random.normal(k1, x.shape) * jnp.sqrt(diff_var)
        jump_var = jnp.exp(self.theta["jump_log_var"])
        jump = self.theta["jump_mean"] + jax.random.normal(k2, x.shape) * jnp.sqrt(
            jump_var
        )
        return x + do_jump.astype(x.dtype) * jump


GaussianNoiseGate.__init__.__doc__ = (
    """See [`torx.AbstractContinuousGate.__init__`][]."""
)
AffineGaussianGate.__init__.__doc__ = (
    """See [`torx.AbstractContinuousGate.__init__`][]."""
)
MixtureGaussianGate.__init__.__doc__ = (
    """See [`torx.AbstractControlledContinuousGate.__init__`][]."""
)
JumpDiffusionGate.__init__.__doc__ = (
    """See [`torx.AbstractControlledContinuousGate.__init__`][]."""
)
