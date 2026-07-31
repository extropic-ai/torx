"""Continuous and hybrid probabilistic gates."""

from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Key, PyTree

from ...factor import _SampleOutput, InfoTree
from ._base import (
    AbstractAffineGaussianGate,
    AbstractControlledContinuousGate,
    HybridSites,
)


def _draw_normal(
    sampler, key: Key[Array, ""], shape: tuple[int, ...], dtype: DTypeLike = float
) -> Float[Array, "..."]:
    # Draw noise in the dtype it will be scaled by, so a consistent fp32 circuit
    # sampled under jax_enable_x64 does not pull in fp64 noise (which strict
    # dtype promotion would then reject).
    if sampler is None:
        return jax.random.normal(key, shape, dtype=dtype)
    return sampler.normal(key, shape, dtype=dtype)


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


class GaussianNoiseGate(
    AbstractAffineGaussianGate[Float[Array, " d"], tuple[int, ...]]
):
    r"""
    Additive Gaussian noise gate.

    This gate adds independent Gaussian noise to the continuous state:
    $x' = x + \mathcal{N}(0, \exp(\theta))$

    The variance is parameterized as $\exp(\theta)$ to ensure positivity.
    """

    _draw_label = "Gauss"

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        """Initial ``theta`` (log-variance), zeros over the continuous dims."""
        return jnp.zeros(sum(self.dims))

    def affine_parameters(
        self,
        theta: Float[Array, " d"],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the additive Gaussian channel as ``(A, b, log_var)``."""
        dim = sum(self.dims)
        log_var = jnp.broadcast_to(theta, (dim,))
        return (
            jnp.eye(dim, dtype=log_var.dtype),
            jnp.zeros(dim, dtype=log_var.dtype),
            log_var,
        )


class AffineGaussianGate(AbstractAffineGaussianGate[dict[str, Array], tuple[int, ...]]):
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

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> dict[str, Array]:
        """Initial identity affine: ``A = I``, ``b = 0``, ``log_var = 0``."""
        d = sum(self.dims)
        return {"A": jnp.eye(d), "b": jnp.zeros(d), "log_var": jnp.zeros(d)}

    def affine_parameters(
        self,
        theta: dict[str, Array],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return ``(A, b, log_var)`` for this affine-Gaussian channel."""
        return theta["A"], theta["b"], theta["log_var"]


def _deterministic_log_var(dim: int, dtype) -> Array:
    """Log-variance of a noise-free channel (zero variance)."""
    return jnp.full(dim, -jnp.inf, dtype=dtype)


class Displace(AbstractAffineGaussianGate[Float[Array, " d"], tuple[int, ...]]):
    r"""Displace continuous sites by ``theta``.

    This deterministic affine-Gaussian channel applies

    $$
    x' = x + \theta.
    $$
    """

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        """Initial displacement ``theta`` of zeros over the continuous dims."""
        return jnp.zeros(sum(self.dims))

    def affine_parameters(
        self,
        theta: Float[Array, " d"],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the displacement channel as ``(A, b, log_var)``."""
        dim = sum(self.dims)
        return (
            jnp.eye(dim, dtype=theta.dtype),
            theta,
            _deterministic_log_var(dim, theta.dtype),
        )


class Scale(AbstractAffineGaussianGate[Float[Array, " d"], tuple[int, ...]]):
    r"""Scale continuous sites by $\exp(\theta)$.

    This deterministic affine-Gaussian channel applies

    $$
    x'_i = \exp(\theta_i) x_i.
    $$
    """

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        """Initial log-scale ``theta`` of zeros (i.e. unit scale)."""
        return jnp.zeros(sum(self.dims))

    def affine_parameters(
        self,
        theta: Float[Array, " d"],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the scaling channel as ``(A, b, log_var)``."""
        dim = sum(self.dims)
        return (
            jnp.diag(jnp.exp(theta)),
            jnp.zeros(dim, dtype=theta.dtype),
            _deterministic_log_var(dim, theta.dtype),
        )


class Mix(AbstractAffineGaussianGate[Float[Array, ""], tuple[int, ...]]):
    r"""Rotate exactly two scalar continuous sites by angle ``theta``.

    This deterministic affine-Gaussian channel applies

    $$
    \begin{bmatrix}x'_0\\x'_1\end{bmatrix}
    =
    \begin{bmatrix}
      \cos\theta & -\sin\theta\\
      \sin\theta & \cos\theta
    \end{bmatrix}
    \begin{bmatrix}x_0\\x_1\end{bmatrix}.
    $$

    Expects two scalar continuous sites (``dims = (1, 1)``).
    """

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, ""]:
        """Initial rotation angle ``theta`` of zero."""
        return jnp.zeros(())

    def affine_parameters(
        self,
        theta: Float[Array, ""],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the rotation channel as ``(A, b, log_var)``."""
        c = jnp.cos(theta)
        s = jnp.sin(theta)
        matrix = jnp.array([[c, -s], [s, c]])
        return (
            matrix,
            jnp.zeros(2, dtype=matrix.dtype),
            _deterministic_log_var(2, matrix.dtype),
        )


class Diffuse(AbstractAffineGaussianGate[Float[Array, " d"], tuple[int, ...]]):
    r"""Apply Brownian diffusion with log-variance ``theta``.

    This additive affine-Gaussian channel applies

    $$
    x' = x + \epsilon,\qquad
    \epsilon \sim \mathcal{N}(0, \exp(\theta)).
    $$

    For a diffusion coefficient $D$ over time $t$, set ``theta`` to
    $\log(2 D t)$. Functionally this matches
    [`torx.psc.GaussianNoiseGate`][]; it is provided as a named
    Brownian-diffusion channel.
    """

    sites: HybridSites = eqx.field(converter=_continuous_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> Float[Array, " d"]:
        """Initial log-variance ``theta`` of zeros over the continuous dims."""
        return jnp.zeros(sum(self.dims))

    def affine_parameters(
        self,
        theta: Float[Array, " d"],
    ) -> tuple[
        Float[Array, " local_dim local_dim"],
        Float[Array, " local_dim"],
        Float[Array, " local_dim"],
    ]:
        """Return the Brownian diffusion channel as ``(A, b, log_var)``."""
        dim = sum(self.dims)
        log_var = jnp.broadcast_to(theta, (dim,))
        return (
            jnp.eye(dim, dtype=log_var.dtype),
            jnp.zeros(dim, dtype=log_var.dtype),
            log_var,
        )


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

    sites: HybridSites = eqx.field(converter=_hybrid_sites_converter)
    dims: tuple[int, ...]
    num_components: int = eqx.field(static=True)

    def init_params(self, key: Key[Array, ""]) -> dict[str, Array]:
        """Initial ``means`` and ``log_vars`` of zeros, shape ``(K, d)``."""
        zeros = jnp.zeros((self.num_components, sum(self.dims)))
        return {"means": zeros, "log_vars": zeros}

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: dict[str, Array],
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        x = inputs["continuous"]
        k = self._get_discrete_control(inputs)
        var = jnp.exp(params["log_vars"][k])
        mean = params["means"][k]
        noise = _draw_normal(info, key, x.shape, dtype=var.dtype)
        output = x + mean + noise * jnp.sqrt(var)
        return (output, None) if return_aux else output


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

    sites: HybridSites = eqx.field(converter=_hybrid_sites_converter)
    dims: tuple[int, ...]

    def init_params(self, key: Key[Array, ""]) -> dict[str, Array]:
        """Initial ``diff_log_var``/``jump_mean``/``jump_log_var`` of zeros."""
        d = sum(self.dims)
        return {
            "diff_log_var": jnp.zeros(d),
            "jump_mean": jnp.zeros(d),
            "jump_log_var": jnp.zeros(d),
        }

    def sample(
        self,
        key: Key[Array, ""],
        inputs: Mapping[str, PyTree[Array]],
        params: dict[str, Array],
        info: InfoTree = None,
        site_info: Any = None,
        return_aux: bool = False,
    ) -> _SampleOutput:
        x = inputs["continuous"]
        do_jump = self._get_discrete_control(inputs)
        k1, k2 = jax.random.split(key)
        diff_var = jnp.exp(params["diff_log_var"])
        diff_noise = _draw_normal(info, k1, x.shape, dtype=diff_var.dtype)
        x = x + diff_noise * jnp.sqrt(diff_var)
        jump_var = jnp.exp(params["jump_log_var"])
        jump_noise = _draw_normal(info, k2, x.shape, dtype=jump_var.dtype)
        jump = params["jump_mean"] + jump_noise * jnp.sqrt(jump_var)
        output = x + do_jump.astype(x.dtype) * jump
        return (output, None) if return_aux else output


GaussianNoiseGate.__init__.__doc__ = (
    """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
)
AffineGaussianGate.__init__.__doc__ = (
    """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
)
Displace.__init__.__doc__ = (
    """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
)
Scale.__init__.__doc__ = """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
Mix.__init__.__doc__ = """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
Diffuse.__init__.__doc__ = """See [`torx.psc.AbstractAffineGaussianGate.__init__`][]."""
MixtureGaussianGate.__init__.__doc__ = (
    """See [`torx.psc.AbstractControlledContinuousGate.__init__`][]."""
)
JumpDiffusionGate.__init__.__doc__ = (
    """See [`torx.psc.AbstractControlledContinuousGate.__init__`][]."""
)
