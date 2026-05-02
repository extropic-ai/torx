"""Parameter-shift gradients for sample-based probabilistic circuits."""

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Key, PyTree

from .._custom_types import BitString
from ._sampled_compile import CompiledSamplePCircuit
from ._sampled_forward import _check_supports_gradient, _expval_all


@eqx.filter_custom_vjp
def sample_expval_all_param_shift_inf(
    circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""], num_samples: int
) -> Float[Array, " num_pbits"]:
    r"""
    Estimate the expectation value of all pbits after circuit execution.

    This is a wrapper function for `_expval_all` so that a custom VJP rule can
    be defined without affecting core functionality. The VJP uses the
    parameter shift rule with deterministic gates. For K-branch gates with
    softmax probabilities, the gradient with respect to logits is:

    $$\frac{\partial \mathbb{E}[O]}{\partial \theta_k} =
        p_k \left( \mathbb{E}[O|k] - \mathbb{E}[O] \right)$$

    where $p_k = \text{softmax}([0, \theta])_k$ is the probability of branch $k$,
    and $\mathbb{E}[O|k]$ is the expected observable when branch $k$ is selected
    deterministically.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain to estimate the expval

    **Returns:**

    An array containing the expectation values of all the pbits.
    """
    return _expval_all(circuit, x, key, num_samples)


@sample_expval_all_param_shift_inf.def_fwd
def sample_expval_all_param_shift_inf_fwd(
    perturbed: CompiledSamplePCircuit,
    circuit: CompiledSamplePCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
) -> tuple[Float[Array, " num_pbits"], None]:
    """Perform the forward execution of sample_expval_all_param_shift_inf."""
    return _expval_all(circuit, x, key, num_samples), None


@sample_expval_all_param_shift_inf.def_bwd
def sample_expval_all_param_shift_inf_bwd(
    residuals: PyTree,
    grad_out: Float[Array, " num_pbits"],
    perturbed: CompiledSamplePCircuit,
    circuit: CompiledSamplePCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
) -> CompiledSamplePCircuit:
    """Perform the backward execution of sample_expval_all_param_shift_inf."""
    _check_supports_gradient(circuit)
    if circuit.reps != 1:
        raise NotImplementedError(
            "Deterministic variant of parameter-shift for multiple repetitions "
            "is not yet implemented. For computing gradients of circuits with "
            "multiple repetitions, use 'param_shift_filter' method instead."
        )

    trainable = eqx.filter(circuit, perturbed)
    num_gates = circuit.thetas.shape[0]
    max_branches = circuit.max_branches

    # For each gate g and branch k, we need to evaluate the circuit with
    # gate g set to deterministically select branch k.
    # This requires num_gates * max_branches evaluations.
    def make_deterministic_theta(g, k, base_thetas):
        new_thetas = base_thetas.copy()
        new_thetas = new_thetas.at[g, :].set(-jnp.inf)
        if k != 0:
            new_thetas = new_thetas.at[g, k - 1].set(jnp.inf)
        return new_thetas

    thetas_list = []
    for g in range(num_gates):
        for k in range(max_branches):
            thetas_list.append(make_deterministic_theta(g, k, circuit.thetas))

    # Stack into (num_gates * max_branches, num_gates, max_branches-1)
    thetas_deterministic = jnp.stack(thetas_list)

    circuit_deterministic = jax.tree.unflatten(
        jax.tree.structure(trainable), (thetas_deterministic,)
    )
    circuit_deterministic = eqx.combine(circuit_deterministic, circuit)

    keys = jax.random.split(key, num_gates * max_branches)
    vmap_axes = (
        eqx.filter(trainable, perturbed, inverse=True, replace=0),
        None,
        0,
        None,
    )

    # (num_gates * max_branches, num_pbits)
    expvals_deterministic = eqx.filter_vmap(_expval_all, in_axes=vmap_axes)(
        circuit_deterministic, x, keys, num_samples
    )

    # Reshape to (num_gates, max_branches, num_pbits)
    expvals_per_branch = expvals_deterministic.reshape(num_gates, max_branches, -1)

    if max_branches == 2:
        sig = jax.nn.sigmoid(circuit.thetas[:, 0])
        # (num_gates, 2)
        probs = jnp.stack([1 - sig, sig], axis=-1)
    else:
        padded_logits = jnp.concatenate(
            [jnp.zeros((num_gates, 1)), circuit.thetas], axis=1
        )  # (num_gates, max_branches)
        probs = jax.nn.softmax(padded_logits, axis=-1)

    # grad_theta[g, j] = sum_k dp_k/dtheta_j * (expval_branch[g,k] @ grad_out)
    # where dp_k/dtheta_j = p_k * (delta_{k,j+1} - p_{j+1})
    # grad_theta[g, j] = p_{j+1} * (expval_branch[g,j+1] - expval_overall[g]) @ grad_out

    # weighted over branches
    # (num_gates, num_pbits)
    expval_overall = jnp.einsum("gk,gkp->gp", probs, expvals_per_branch)

    # gradient for each theta parameter
    # (num_gates, max_branches-1)
    # grad[g, j] = p[g, j+1] * (expval_branch[g, j+1] - expval_overall[g]) @ grad_out
    diff = (
        expvals_per_branch[:, 1:, :] - expval_overall[:, None, :]
    )  # (num_gates, max_branches-1, num_pbits)
    grads_in = probs[:, 1:] * jnp.einsum(
        "gjp,p->gj", diff, grad_out
    )  # (num_gates, max_branches-1)

    grad_circuit = jax.tree.unflatten(jax.tree.structure(trainable), (grads_in,))

    return grad_circuit


@eqx.filter_custom_vjp
def sample_expval_all_param_shift_single(
    circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""], num_samples: int
) -> Float[Array, " num_pbits"]:
    r"""
    Estimate the expectation value of all pbits after circuit execution.

    This is a wrapper function for `_expval` so that a custom VJP rule can be
    defined without affecting core functionality. The VJP uses the
    parameter shift rule with reuse of the primal. This rule is given by:

    $$\frac{\partial U(\theta)}{\partial\theta} = U(\theta) -
        U\left( -\ln\left[(1 + \exp(-\theta))^2 - 1 \right]  \right)$$

    where $U(\theta)$ taken from the primal.

    **Note:** This method only supports 2-branch (K=2) gates. For circuits
    with K>2 gates, use `param_shift_inf` or `param_shift_filter` instead.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain to estimate the expval

    **Returns:**

    An array containing the expectation values of all the pbits.
    """
    return _expval_all(circuit, x, key, num_samples)


@sample_expval_all_param_shift_single.def_fwd
def sample_expval_all_param_shift_single_fwd(
    perturbed: CompiledSamplePCircuit,
    circuit: CompiledSamplePCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
) -> tuple[Float[Array, " num_pbits"], Float[Array, " num_pbits"]]:
    """Perform the forward execution of sample_expval_all_param_shift_single."""
    expval = _expval_all(circuit, x, key, num_samples)
    return expval, expval


@sample_expval_all_param_shift_single.def_bwd
def sample_expval_all_param_shift_single_bwd(
    residuals: PyTree,
    grad_out: Float[Array, " num_pbits"],
    perturbed: CompiledSamplePCircuit,
    circuit: CompiledSamplePCircuit,
    x: BitString,
    key: Key[Array, ""],
    num_samples: int,
) -> CompiledSamplePCircuit:
    """Perform the backward execution of sample_expval_all_param_shift_single."""
    _check_supports_gradient(circuit)
    # (num_pbits,)
    expval_primal = residuals

    if circuit.reps != 1:
        raise NotImplementedError(
            "Single-shift variant of parameter-shift for multiple repetitions "
            "is not yet implemented. For computing gradients of circuits with "
            "multiple repetitions, use 'param_shift_filter' method instead."
        )

    # Check if any gate has K > 2 branches
    if circuit.max_branches > 2:
        raise NotImplementedError(
            "param_shift_single only supports 2-branch gates. "
            "Use 'param_shift_inf' or 'param_shift_filter' for K-branch gates."
        )

    trainable = eqx.filter(circuit, perturbed)
    num_gates = circuit.thetas.shape[0]

    # For K=2 gates, thetas has shape (num_gates, 1)
    theta_flat = circuit.thetas[:, 0]  # (num_gates,)

    # (num_gates, num_gates, 1)
    shifted_thetas = jnp.tile(circuit.thetas[None, :, :], (num_gates, 1, 1))
    shift_values = -jnp.log((1 + jnp.exp(-theta_flat)) ** 2 - 1)  # (num_gates,)
    shifted_thetas = shifted_thetas.at[
        jnp.arange(num_gates), jnp.arange(num_gates), 0
    ].set(shift_values)

    param_shift_circuit = jax.tree.unflatten(
        jax.tree.structure(trainable), (shifted_thetas,)
    )
    param_shift_circuit = eqx.combine(param_shift_circuit, circuit)

    keys = jax.random.split(key, num_gates)
    vmap_axes = (
        eqx.filter(trainable, perturbed, inverse=True, replace=0),
        None,
        0,
        None,
    )

    # (num_gates, num_pbits)
    expvals = eqx.filter_vmap(_expval_all, in_axes=vmap_axes)(
        param_shift_circuit, x, keys, num_samples
    )

    # (num_gates,)
    grad_flat = (expval_primal - expvals) @ grad_out

    # Reshape to (num_gates, 1) to match thetas shape
    grads_in = grad_flat[:, None]

    grad_circuit = jax.tree.unflatten(jax.tree.structure(trainable), (grads_in,))

    return grad_circuit
