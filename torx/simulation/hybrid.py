"""Hybrid sample-based simulator for discrete + continuous gates."""

import warnings
from typing import ClassVar, Type, TypedDict

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Key
from typing_extensions import Self

from .._circuit import HybridGateType, HybridPCircuit
from ..gates import AbstractDiscreteGate, AbstractHybridGate, AbstractKBranchGate
from .base import AbstractCompiledPCircuit, AbstractSimulator


class HybridState(TypedDict):
    """State representation for hybrid circuits."""

    discrete: Int[Array, "... num_discrete_sites"]
    continuous: Float[Array, "... continuous_dim"]


class CompiledHybridPCircuit(AbstractCompiledPCircuit[HybridPCircuit]):
    """Compiled hybrid circuit for the hybrid sample simulator."""

    gates: list[HybridGateType]
    discrete_dims: tuple[int, ...]
    continuous_dims: tuple[int, ...]
    reps: int

    @classmethod
    def from_pcircuit(cls, circuit: HybridPCircuit) -> Self:
        """Compile a hybrid circuit.

        **Arguments:**

        - `circuit`: The hybrid circuit to compile.

        **Returns:**

        The compiled circuit.
        """
        return cls(
            circuit.gates,
            circuit.discrete_dims,
            circuit.continuous_dims,
            circuit.reps,
        )

    def to_pcircuit(self, structure: HybridPCircuit) -> HybridPCircuit:
        """Create a new circuit with parameters from this compiled circuit.

        **Arguments:**

        - `structure`: The circuit structure to use.

        **Returns:**

        A new hybrid circuit with updated parameters.
        """
        return HybridPCircuit(self.gates, self.reps)


class HybridSampleSimulator(AbstractSimulator):
    """Sample-based simulator for hybrid circuits.

    Handles both discrete gates (via branch sampling) and continuous/hybrid
    gates (via their sample() method).

    !!! warning "Compilation time"

        This simulator unrolls the full loop over gates during tracing, so
        compilation time scales poorly with the number of gates.
    """

    circuit_backend: ClassVar[Type[AbstractCompiledPCircuit]] = CompiledHybridPCircuit
    num_samples: int

    def sample(
        self,
        circuit: HybridPCircuit,
        initial_state: HybridState,
        key: Key[Array, ""],
    ) -> HybridState:
        """Run circuit and return samples of final state.

        **Arguments:**

        - `circuit`: The hybrid circuit to execute.
        - `initial_state`: Initial state with "discrete" and "continuous" arrays.
        - `key`: JAX random key.

        **Returns:**

        Dict with "discrete" and "continuous" final state arrays.
        """
        return sample_hybrid_circuit(
            circuit,
            initial_state,
            key,
            self.num_samples,
        )

    def expval(
        self,
        circuit: HybridPCircuit,
        initial_state: HybridState,
        site: int,
        key: Key[Array, ""],
        site_type: str,
    ) -> Float[Array, ""]:
        """Estimate expectation value of a site.

        **Arguments:**

        - `circuit`: The hybrid circuit to execute.
        - `initial_state`: Initial state with "discrete" and "continuous" arrays.
        - `site`: Index of the site.
        - `key`: JAX random key.
        - `site_type`: Either "discrete" or "continuous".

        **Returns:**

        The expectation value of the site.
        """
        return self.expval_all(circuit, initial_state, key)[site_type][site]

    def expval_all(
        self,
        circuit: HybridPCircuit,
        initial_state: HybridState,
        key: Key[Array, ""],
    ) -> HybridState:
        """Estimate expectation values of all sites.

        **Arguments:**

        - `circuit`: The hybrid circuit to execute.
        - `initial_state`: Initial state with "discrete" and "continuous" arrays.
        - `key`: JAX random key.

        **Returns:**

        Dict with "discrete" and "continuous" expectation value arrays.
        """
        result = self.sample(circuit, initial_state, key)
        return {
            "discrete": jnp.mean(result["discrete"], axis=0).astype(jnp.float32),
            "continuous": jnp.mean(result["continuous"], axis=0),
        }


def sample_hybrid_circuit(
    circuit: HybridPCircuit,
    initial_state: HybridState,
    key: Key[Array, ""],
    num_samples: int,
) -> HybridState:
    """Sample-based simulation of hybrid circuit.

    **Arguments:**

    - `circuit`: The hybrid circuit.
    - `initial_state`: Initial state with "discrete" and "continuous" arrays.
    - `key`: JAX random key.
    - `num_samples`: Number of samples to generate.

    **Returns:**

    Dict with "discrete" and "continuous" arrays of shape (num_samples, ...).
    """
    if len(circuit.continuous_dims) == 0:
        warnings.warn(
            "HybridPCircuit has no continuous sites. "
            "Consider using SampleSimulator with DiscretePCircuit instead.",
            UserWarning,
        )

    cont_offsets = (0,) + tuple(
        sum(circuit.continuous_dims[: i + 1])
        for i in range(len(circuit.continuous_dims))
    )

    def _sample_single(
        sample_key: Key[Array, ""],
    ) -> tuple:
        discrete = initial_state["discrete"]
        continuous = initial_state["continuous"]

        rep_keys = jax.random.split(sample_key, circuit.reps)

        def _body_fn(
            rep: int,
            state: tuple,
        ) -> tuple:
            discrete, continuous = state
            gate_keys = jax.random.split(rep_keys[rep], len(circuit.gates))

            for gate_idx, gate in enumerate(circuit.gates):
                gate_key = gate_keys[gate_idx]

                if isinstance(gate, AbstractDiscreteGate):
                    discrete = _apply_discrete_gate(gate, discrete, gate_key)
                else:
                    continuous = _apply_hybrid_gate(
                        gate,
                        discrete,
                        continuous,
                        gate_key,
                        cont_offsets,
                        circuit.continuous_dims,
                    )

            return discrete, continuous

        return jax.lax.fori_loop(0, circuit.reps, _body_fn, (discrete, continuous))

    sample_keys = jax.random.split(key, num_samples)
    discrete, continuous = jax.vmap(_sample_single)(sample_keys)

    return {"discrete": discrete, "continuous": continuous}


def _apply_discrete_gate(
    gate: AbstractDiscreteGate,
    discrete: Int[Array, " num_discrete"],
    key: Key[Array, ""],
) -> Int[Array, " num_discrete"]:
    sites = [gate.sites] if isinstance(gate.sites, int) else list(gate.sites)
    sites_arr = jnp.array(sites)

    if not isinstance(gate, AbstractKBranchGate):
        raise ValueError(f"Discrete gate {gate} must be AbstractKBranchGate")

    branches = gate.branches  # (K, basis_size, num_sites)

    # Compute strides for indexing: strides[i] = prod(dims[i+1:])
    # e.g. dims=[2,3,4] -> strides=[3*4, 4, 1] = [12,4,1]
    # so flat index = s0*12 + s1*4 + s2
    gate_dims = jnp.array(gate.dims)
    strides = jnp.concatenate([jnp.cumprod(gate_dims[::-1])[::-1][1:], jnp.array([1])])

    if gate.num_branches == 2:
        branch_idx = jax.random.bernoulli(key, gate.prob).astype(jnp.int32)
    else:
        padded_logits = jnp.pad(jnp.atleast_1d(gate.theta), (1, 0))
        branch_idx = jax.random.categorical(key, padded_logits)

    substate = discrete[sites_arr]
    index = jnp.sum(substate * strides)
    new_substate = branches[branch_idx, index]
    return discrete.at[sites_arr].set(new_substate)


def _apply_hybrid_gate(
    gate: AbstractHybridGate,
    discrete: Int[Array, " num_discrete"],
    continuous: Float[Array, " continuous_dim"],
    key: Key[Array, ""],
    cont_offsets: tuple[int, ...],
    continuous_dims: tuple[int, ...],
) -> Float[Array, " continuous_dim"]:
    gate_sites = gate.sites
    cont_sites = gate_sites.get("continuous", [])

    disc_sites_arr = (
        jnp.array(gate_sites.get("discrete", []))
        if gate_sites.get("discrete", [])
        else jnp.array([], dtype=jnp.int32)
    )
    cont_sites_list = list(cont_sites) if cont_sites else []

    disc_sub = (
        discrete[disc_sites_arr]
        if len(disc_sites_arr) > 0
        else jnp.array([], dtype=jnp.int32)
    )

    # probably suboptimal compile time, but naive approach
    # for specific workloads we could construct a more optimized simulator
    # e.g. if we know certain shapes are padding compatible, we can pad out some
    # of the params/sites/etc.
    if cont_sites_list:
        cont_parts = [
            continuous[cont_offsets[s] : cont_offsets[s + 1]] for s in cont_sites_list
        ]
        cont_sub = jnp.concatenate(cont_parts)
    else:
        cont_sub = jnp.array([])

    substate = {"discrete": disc_sub, "continuous": cont_sub}
    new_cont_sub = gate.sample(substate, key)

    new_continuous = continuous
    idx = 0
    for s in cont_sites_list:
        site_dim = continuous_dims[s]
        start = cont_offsets[s]
        end = start + site_dim
        new_continuous = new_continuous.at[start:end].set(
            new_cont_sub[idx : idx + site_dim]
        )
        idx += site_dim

    return new_continuous
