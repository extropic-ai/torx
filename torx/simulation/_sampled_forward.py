"""Forward sampling kernels for sample-based probabilistic circuits."""

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Key

from .._custom_types import BitString
from ._sampled_compile import CompiledSamplePCircuit


def sample_circuit(
    circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""], num_samples: int
) -> tuple[
    Int[Array, "num_samples num_pbits"], Int[Array, "reps num_gates num_samples"]
]:
    if circuit.has_conditional:
        return _sample_conditional_circuit(circuit, x, key, num_samples)
    return _sample_branch_circuit(circuit, x, key, num_samples)


def _sample_branch_indices(
    circuit: CompiledSamplePCircuit,
    key: Key[Array, ""],
    num_samples: int,
) -> Int[Array, "reps num_gates num_samples"]:
    num_gates = circuit.thetas.shape[0]
    shape = (circuit.reps, num_gates, num_samples)
    if circuit.max_branches == 2:
        probs = jax.nn.sigmoid(circuit.thetas[:, 0])
        return jax.random.bernoulli(key, probs[None, :, None], shape=shape).astype(
            jnp.int32
        )

    padded_logits = jnp.concatenate([jnp.zeros((num_gates, 1)), circuit.thetas], axis=1)
    return jax.random.categorical(
        key,
        padded_logits[None, :, None, :],
        axis=-1,
        shape=shape,
    )


def _index_dtype():
    return (
        jnp.int64
        if jax.config.jax_enable_x64  # pyright: ignore[reportAttributeAccessIssue]
        else jnp.int32
    )


def _apply_branch_gate(
    state: Int[Array, " num_pbits"],
    branch_ops: Int[Array, "K B l"],
    sites: Int[Array, " l"],
    dims: Int[Array, " l"],
    branch_idx: Int[Array, ""],
) -> Int[Array, " num_pbits"]:
    dtype = _index_dtype()
    substate = state[sites].astype(dtype)
    strides = jnp.cumprod(dims[::-1])[::-1]
    strides = jnp.concatenate([strides[1:], jnp.array([1], dtype=dtype)])
    index = jnp.sum(substate * strides)
    return state.at[sites].set(branch_ops[branch_idx, index])


def _sample_branch_circuit(
    circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""], num_samples: int
) -> tuple[
    Int[Array, "num_samples num_pbits"], Int[Array, "reps num_gates num_samples"]
]:
    """Sample a circuit containing only branch-table gates."""
    state = jnp.tile(x[None, :], (num_samples, 1))

    def inner_body_fn(
        carry: Int[Array, "num_samples num_pbits"], x: tuple
    ) -> tuple[Int[Array, "num_samples num_pbits"], None]:
        state = carry
        branch_ops, sites, dims, branch_idx = x
        state = jax.vmap(
            _apply_branch_gate,
            in_axes=(0, None, None, None, 0),
        )(state, branch_ops, sites, dims, branch_idx)
        return state, None

    def outer_body_fn(
        carry: Int[Array, "num_samples num_pbits"],
        branch_indices: Int[Array, "num_gates num_samples"],
    ) -> tuple[Int[Array, "num_samples num_pbits"], None]:
        state = jax.lax.scan(
            inner_body_fn,
            carry,
            (
                circuit.branch_ops,
                circuit.sites,
                circuit.dims,
                branch_indices,
            ),
        )[0]
        return state, None

    branch_indices = _sample_branch_indices(circuit, key, num_samples)
    state = jax.lax.scan(outer_body_fn, state, branch_indices)[0]
    return state, branch_indices


def _sample_conditional_circuit(
    circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""], num_samples: int
) -> tuple[
    Int[Array, "num_samples num_pbits"], Int[Array, "reps num_gates num_samples"]
]:
    """
    Obtain samples from the final distribution of the probabilistic circuit.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain

    **Returns:**

    A tuple of:
    - An integer array with shape (num_samples, num_pbits) containing the
      computational basis state samples.
    - An integer array with shape (reps, num_gates, num_samples) containing
      the branch indices that were sampled for each gate.
    """
    # (num_samples, num_pbits)
    state = jnp.tile(x[None, :], (num_samples, 1))
    conditional = circuit.conditional_data
    if conditional is None:
        raise ValueError("Compiled circuit does not contain conditional data.")

    def _apply_gate(
        state: Int[Array, " num_pbits"],
        gate_type: Int[Array, ""],
        branch_ops: Int[Array, "K B l"],
        sites: Int[Array, " l"],
        dims: Int[Array, " l"],
        branch_idx: Int[Array, ""],
        cond_target_sites: Int[Array, " T"],
        cond_control_sites: Int[Array, "T C"],
        cond_control_weights: Float[Array, "T C"],
        cond_logits: Float[Array, " T"],
        cond_num_targets: Int[Array, ""],
        cond_num_controls: Int[Array, " T"],
        cond_random: Float[Array, " T"],
    ) -> Int[Array, " num_pbits"]:
        def apply_branch(st):
            return _apply_branch_gate(st, branch_ops, sites, dims, branch_idx)

        def apply_conditional(st):
            control_weights = cond_control_weights
            # Read controls before scattering target updates. This gives stable
            # Gibbs semantics even when a target also appears as a control.
            control_vals = st[cond_control_sites].astype(control_weights.dtype)
            controls = jnp.arange(cond_control_sites.shape[1])
            control_mask = controls[None, :] < cond_num_controls[:, None]
            weighted_controls = control_vals * control_weights
            field = cond_logits + jnp.sum(
                jnp.where(
                    control_mask,
                    weighted_controls,
                    jnp.zeros_like(weighted_controls),
                ),
                axis=1,
            )
            draws = (cond_random < jax.nn.sigmoid(field)).astype(st.dtype)
            target_mask = jnp.arange(cond_target_sites.shape[0]) < cond_num_targets
            current = st[cond_target_sites]
            updates = jnp.where(target_mask, draws, current)
            return st.at[cond_target_sites].set(updates)

        return jax.lax.cond(gate_type == 0, apply_branch, apply_conditional, state)

    # inner loop scans over the gates of the circuit
    def inner_body_fn(
        carry: Int[Array, "num_samples num_pbits"], x: tuple
    ) -> tuple[Int[Array, "num_samples num_pbits"], None]:
        # (num_samples, num_pbits)
        state = carry

        (
            gate_types,
            branch_ops,
            sites,
            dims,
            branch_idx,
            cond_target_sites,
            cond_control_sites,
            cond_control_weights,
            cond_logits,
            cond_num_targets,
            cond_num_controls,
            cond_random,
        ) = x

        # (num_samples, num_pbits)
        state = jax.vmap(
            _apply_gate,
            in_axes=(
                0,
                None,
                None,
                None,
                None,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
            ),
        )(
            state,
            gate_types,
            branch_ops,
            sites,
            dims,
            branch_idx,
            cond_target_sites,
            cond_control_sites,
            cond_control_weights,
            cond_logits,
            cond_num_targets,
            cond_num_controls,
            cond_random,
        )
        return state, None

    # outer loop scans over the number of repetitions
    def outer_body_fn(
        carry: Int[Array, "num_samples num_pbits"],
        x: tuple[
            Int[Array, "num_gates num_samples"],
            Float[Array, "num_gates num_samples max_targets"],
        ],
    ) -> tuple[Int[Array, "num_samples num_pbits"], None]:
        # (num_samples, num_pbits)
        state = carry

        # (num_gates, num_samples)
        branch_indices, cond_random = x

        # (num_samples, num_pbits)
        state = jax.lax.scan(
            inner_body_fn,
            state,
            (
                gate_types,
                branch_ops,
                sites,
                dims,
                branch_indices,
                cond_target_sites,
                cond_control_sites,
                cond_control_weights,
                cond_logits,
                cond_num_targets,
                cond_num_controls,
                cond_random,
            ),
        )[0]
        return state, None

    gate_types = conditional.gate_types

    # (num_gates, max_branches, B, locality)
    branch_ops = circuit.branch_ops

    # (num_gates, locality)
    sites = circuit.sites
    dims = circuit.dims
    # Conditional data is stored compactly by conditional gate, but the scan
    # iterates over all circuit gates. Expand views here, not in the compiled
    # circuit object, so branch-only circuits keep the old payload shape.
    cond_target_sites = conditional.target_sites[conditional.gate_cond_indices]
    cond_control_sites = conditional.control_sites[conditional.gate_cond_indices]
    cond_control_weights = conditional.control_weights[conditional.gate_cond_indices]
    cond_logits = conditional.logits[conditional.gate_cond_indices]
    cond_num_targets = conditional.num_targets[conditional.gate_cond_indices]
    cond_num_controls = conditional.num_controls[conditional.gate_cond_indices]

    branch_key, cond_key = jax.random.split(key)
    branch_indices = _sample_branch_indices(circuit, branch_key, num_samples)
    cond_random = jax.random.uniform(
        cond_key,
        shape=(
            circuit.reps,
            conditional.target_sites.shape[0],
            num_samples,
            conditional.target_sites.shape[1],
        ),
        dtype=conditional.logits.dtype,
    )
    cond_random_by_gate = cond_random[:, conditional.gate_cond_indices]

    # (num_samples, num_pbits)
    state = jax.lax.scan(outer_body_fn, state, (branch_indices, cond_random_by_gate))[0]

    return state, branch_indices


def _expval_all(
    circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""], num_samples: int
) -> Float[Array, " pbits"]:
    """
    Estimate the expectation value of all pbits after circuit execution.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain to estimate the expval

    **Returns:**

    An array containing the expectation values of all the pbits.
    """
    samples = sample_circuit(circuit, x, key, num_samples)[0]
    val = jnp.mean(samples.astype(jnp.float32), axis=0)
    return val


def _check_supports_gradient(circuit: CompiledSamplePCircuit):
    if circuit.has_conditional:
        raise NotImplementedError(
            "SampleSimulator gradients for conditional sample gates are not "
            "implemented yet. Use SampleSimulator.sample/expval_all for forward "
            "sampling, or differentiate a branch-table circuit."
        )
