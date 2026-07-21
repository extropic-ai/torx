"""Notebook-friendly wrappers around Torx affine-Gaussian simulation.

The continuous gates used by the pmode notebooks (``Displace``, ``Scale``,
``Mix``, ``Diffuse``, ``AffineGaussianGate``) are public ``torx.psc`` gates and
are constructed directly in the notebooks. This module only provides density
helpers for the Gaussian moments produced by ``AffineGaussianSimulator``.
"""

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss


def _promote_points(points, dim):
    """Promote a 1-D ``(M,)`` grid to ``(M, 1)`` when the density is 1-D."""
    if dim == 1 and points.ndim == 1:
        return points.reshape(-1, 1)
    return points


def gaussian_density(moments, *, site=None, grid):
    """Evaluate a Gaussian density on a grid.

    **Arguments:**

    - `moments`: Gaussian moments or posterior moments.
    - `site`: Optional continuous site to marginalize before evaluation.
    - `grid`: Points with shape `(M,)` for 1-D marginals or `(..., d)` otherwise.

    **Returns:**

    Density values with the leading shape of `grid`.
    """
    if site is None:
        mu = moments.mean
        cov = moments.covariance
    else:
        mu, cov = moments.site_moments(site)

    mu = jnp.atleast_1d(mu)
    cov = jnp.atleast_2d(cov)
    points = _promote_points(grid, mu.size)
    return jss.multivariate_normal.pdf(points, mu, cov)


def mixture_density(means, log_vars, weights, grid):
    """Evaluate a diagonal Gaussian mixture density on a grid.

    **Arguments:**

    - `means`: Component means with shape `(K, d)`.
    - `log_vars`: Component log-variances with shape `(K, d)`.
    - `weights`: Component weights with shape `(K,)`, summing to 1.
    - `grid`: Points with shape `(M,)` for 1-D mixtures or `(..., d)` otherwise.

    **Returns:**

    Mixture density values with the leading shape of `grid`.
    """
    points = _promote_points(grid, means.shape[-1])

    def component_log_pdf(mean, log_var):
        scale = jnp.exp(0.5 * log_var)
        return jnp.sum(jss.norm.logpdf(points, loc=mean, scale=scale), axis=-1)

    log_components = jax.vmap(component_log_pdf)(means, log_vars)
    weight_shape = (-1,) + (1,) * (log_components.ndim - 1)
    weighted = jnp.log(weights).reshape(weight_shape) + log_components
    return jnp.exp(jax.nn.logsumexp(weighted, axis=0))
