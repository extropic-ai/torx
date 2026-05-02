"""Filter-based gradients for sample-based probabilistic circuits."""

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Key, PyTree

from .._custom_types import BitString
from ._sampled_compile import CompiledSamplePCircuit
from ._sampled_forward import _check_supports_gradient, _expval_all, sample_circuit


@eqx.filter_custom_vjp
def sample_expval_all_param_shift_filter(
    circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""], num_samples: int
) -> Float[Array, " num_pbits"]:
    r"""
    Estimate the expectation value of all pbits after circuit execution.

    This is a wrapper function for `_expval_all` so that a custom VJP rule can
    be defined without affecting core functionality. The VJP uses the
    parameter shift rule with deterministic gates, but with no additional
    circuit executions. For K-branch gates with softmax probabilities, the
    gradient with respect to logits is:

    $$\frac{\partial \mathbb{E}[O]}{\partial \theta_k} =
        p_k \left( \mathbb{E}[O|k] - \mathbb{E}[O] \right)$$

    where $p_k = \text{softmax}([0, \theta])_k$ is the probability of branch $k$.
    The difference between this VJP rule and `param_shift_inf` above is that,
    for this rule, the conditional expectations $\mathbb{E}[O|k]$ are estimated
    directly from the primal execution by filtering samples based on which
    branch was selected at each gate.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain to estimate the expval

    **Returns:**

    An array containing the expectation values of all the pbits.
    """
    return _expval_all(circuit, x, key, num_samples)


@sample_expval_all_param_shift_filter.def_fwd
def sample_expval_all_param_shift_filter_fwd(
    perturbed: CompiledSamplePCircuit,
    circuit: CompiledSamplePCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
) -> tuple[
    Float[Array, " pbits"],
    tuple[Int[Array, " samples pbits"], Int[Array, "reps num_gates num_samples"]],
]:
    """Perform the forward execution of sample_expval_all_param_shift_filter."""
    samples, branch_indices = sample_circuit(circuit, x, key, num_samples)
    val = jnp.mean(samples.astype(jnp.float32), axis=0)
    return val, (samples, branch_indices)


@sample_expval_all_param_shift_filter.def_bwd
def sample_expval_all_param_shift_filter_bwd(
    residuals: PyTree,
    grad_out: Float[Array, " num_pbits"],
    perturbed: CompiledSamplePCircuit,
    circuit: CompiledSamplePCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
) -> CompiledSamplePCircuit:
    """Perform the backward execution of sample_expval_all_param_shift_filter."""
    _check_supports_gradient(circuit)
    # (num_samples, num_pbits), (reps, num_gates, num_samples)
    primal, branch_indices = residuals

    num_gates = circuit.thetas.shape[0]
    max_branches = circuit.max_branches

    if max_branches == 2:
        sig = jax.nn.sigmoid(circuit.thetas[:, 0])
        probs = jnp.stack([1 - sig, sig], axis=-1)  # (num_gates, 2)
    else:
        padded_logits = jnp.concatenate(
            [jnp.zeros((num_gates, 1)), circuit.thetas], axis=1
        )  # (num_gates, max_branches)
        probs = jax.nn.softmax(padded_logits, axis=-1)

    # For each gate g and branch k, compute the count and conditional expectation
    # branch_indices: (reps, num_gates, num_samples)
    # primal: (num_samples, num_pbits)

    # Create one-hot encoding of branch indices
    # (reps, num_gates, num_samples, max_branches)
    branch_one_hot = jax.nn.one_hot(branch_indices, max_branches)

    # Count samples per branch: (reps, num_gates, max_branches)
    branch_counts = jnp.sum(branch_one_hot, axis=2)

    # Compute conditional expectation E[O | branch=k] for each branch
    # We need to sum primal values weighted by indicator that branch k was selected

    # For each rep r, gate g, branch k:
    # E[O | branch=k] = sum_s (primal[s] * I[branch_indices[r,g,s] == k]) / count[r,g,k]

    # (reps, num_gates, max_branches, num_pbits)
    # Sum primal values where branch k was selected
    weighted_sum = jnp.einsum(
        "sp,rgsk->rgkp", primal.astype(branch_one_hot.dtype), branch_one_hot
    )

    safe_counts = jnp.where(branch_counts > 0, branch_counts, 1.0)

    # (reps, num_gates, max_branches, num_pbits)
    expval_per_branch = weighted_sum / safe_counts[:, :, :, None]

    # Set expval to 0 where count is 0
    expval_per_branch = jnp.where(
        branch_counts[:, :, :, None] > 0, expval_per_branch, 0.0
    )

    # (num_pbits,)
    expval_overall = jnp.mean(primal.astype(branch_one_hot.dtype), axis=0)

    # grad_theta[g, j] = p_{j+1} * (E[O | branch=j+1] - E[O]) @ grad_out
    # Average over reps: (num_gates, max_branches, num_pbits)
    expval_per_branch_avg = jnp.mean(expval_per_branch, axis=0)

    # diff[g, k] = E[O | branch=k] - E[O]  for k = 1..K-1 (branches 1 to K-1)
    # (num_gates, max_branches-1, num_pbits)
    diff = expval_per_branch_avg[:, 1:, :] - expval_overall[None, None, :]

    # grad[g, j] = p[g, j+1] * diff[g, j] @ grad_out
    # (num_gates, max_branches-1)
    grads_in = probs[:, 1:] * jnp.einsum("gjp,p->gj", diff, grad_out)

    trainable = eqx.filter(circuit, perturbed)
    grad_circuit = jax.tree.unflatten(jax.tree.structure(trainable), (grads_in,))

    return grad_circuit
