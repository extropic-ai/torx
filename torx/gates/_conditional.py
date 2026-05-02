"""Conditional Bernoulli gates for sampled probabilistic circuits."""

import itertools

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ._base import AbstractDiscreteGate


class PConditionalBernoulliLayer(
    AbstractDiscreteGate[list[int], Float[Array, " targets"], tuple[int, ...]]
):
    """A vectorized conditional Bernoulli update layer.

    Each target pbit is sampled independently with probability
    ``sigmoid(theta + sum(control_weights * control_values))``.

    This gate is intended for the sample-based simulator. It avoids materializing
    a branch table over every control configuration, which keeps high-fan-in
    Gibbs updates compact.

    **Arguments:**

    - `theta`: Bias logits, one scalar per target.
    - `target_sites`: Global pbit indices to update.
    - `control_sites`: Rectangular ``[num_targets, num_controls]`` site indices.
    - `control_weights`: Weights with the same shape as ``control_sites``.
    """

    _draw_label = "PCOND"

    theta: Float[Array, " targets"]
    sites: list[int]
    target_sites: list[int]
    control_sites: list[list[int]]
    control_weights: Float[Array, "targets controls"]

    def __init__(
        self,
        theta: Float[Array, " targets"],
        target_sites: list[int],
        control_sites: list[list[int]],
        control_weights: Float[Array, "targets controls"],
    ):
        param_dtype = jnp.result_type(theta, control_weights, 0.0)
        theta = jnp.atleast_1d(jnp.asarray(theta, dtype=param_dtype))
        target_sites = [int(site) for site in target_sites]
        control_sites = [
            [int(site) for site in per_target] for per_target in control_sites
        ]
        control_weights = jnp.asarray(control_weights, dtype=param_dtype)

        if not target_sites:
            raise ValueError("target_sites must contain at least one site.")
        if len(set(target_sites)) != len(target_sites):
            raise ValueError("target_sites must be unique.")
        all_sites = target_sites + [site for row in control_sites for site in row]
        if any(site < 0 for site in all_sites):
            raise ValueError("sites must be non-negative.")
        if theta.shape != (len(target_sites),):
            raise ValueError("theta must have one entry per target site.")
        if len(control_sites) != len(target_sites):
            raise ValueError("control_sites must have one row per target site.")

        num_controls = len(control_sites[0]) if control_sites else 0
        if any(len(row) != num_controls for row in control_sites):
            raise ValueError("control_sites must be rectangular.")
        expected_shape = (len(target_sites), num_controls)
        if control_weights.shape != expected_shape:
            raise ValueError(
                "control_weights must have shape "
                f"{expected_shape}, got {control_weights.shape}."
            )

        self.theta = theta
        self.target_sites = target_sites
        self.control_sites = control_sites
        self.control_weights = control_weights

        sites = set(self.target_sites)
        for per_target in self.control_sites:
            sites.update(per_target)
        self.sites = sorted(sites)

    @property
    def dims(self) -> tuple[int, ...]:  # type: ignore[override]
        return (2,) * len(self.sites)

    def get_matrix(self) -> Float[Array, "dim dim"]:
        """Return an exact local transition matrix.

        This is practical only for small layers and is primarily useful for
        tests and exact state-vector checks. Large high-fan-in layers should use
        ``SampleSimulator``.
        """
        num_sites = len(self.sites)
        dim = 2**num_sites
        basis = [tuple(bits) for bits in itertools.product([0, 1], repeat=num_sites)]
        site_to_local = {site: idx for idx, site in enumerate(self.sites)}
        target_local = [site_to_local[site] for site in self.target_sites]
        num_targets = len(target_local)
        control_local = [
            [site_to_local[site] for site in per_target]
            for per_target in self.control_sites
        ]
        strides = jnp.array([2 ** (num_sites - 1 - idx) for idx in range(num_sites)])

        matrix = jnp.zeros((dim, dim), dtype=self.theta.dtype)
        for col, input_bits_tuple in enumerate(basis):
            input_bits = jnp.array(input_bits_tuple)
            controls = jnp.array(
                [
                    [input_bits[idx] for idx in per_target]
                    for per_target in control_local
                ]
            ).astype(self.control_weights.dtype)
            logits = self.theta + jnp.sum(controls * self.control_weights, axis=1)
            probs = jax.nn.sigmoid(logits)

            for target_values in itertools.product([0, 1], repeat=num_targets):
                output_bits = input_bits
                target_values_arr = jnp.array(target_values)
                for local_idx, value in zip(target_local, target_values_arr):
                    output_bits = output_bits.at[local_idx].set(value)
                row = jnp.sum(output_bits * strides).astype(jnp.int32)
                prob = jnp.prod(jnp.where(target_values_arr == 1, probs, 1.0 - probs))
                matrix = matrix.at[row, col].add(prob)
        return matrix
