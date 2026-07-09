"""Statevector simulators."""

from typing import ClassVar, Type

import jax
from jax import numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import Self

from .._circuit import DiscretePCircuit
from .._custom_types import StateVector
from ..gates import AbstractDiscreteGate
from .base import AbstractCompiledPCircuit, AbstractSimulator


class CompiledStateVectorPCircuit(AbstractCompiledPCircuit[DiscretePCircuit]):
    """Compiled probabilistic circuit class for the state vector simulator."""

    gates: list[AbstractDiscreteGate]
    thetas: list[Float[Array, "..."]]
    num_pdits: int
    dims: tuple[int, ...]
    reps: int

    @classmethod
    def from_pcircuit(
        cls, circuit: DiscretePCircuit, thetas: list[Float[Array, "..."]]
    ) -> Self:
        """
        Compile the given probabilistic circuit with the given parameters.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compile
        - `thetas`: Per-gate parameters aligned with `circuit.gates`

        **Returns:**

        The compiled circuit.
        """
        if len(thetas) != len(circuit.gates):
            raise ValueError(
                f"expected {len(circuit.gates)} thetas aligned with "
                f"circuit.gates, got {len(thetas)}."
            )
        return cls(circuit.gates, thetas, circuit.num_pdits, circuit.dims, circuit.reps)

    def to_pcircuit(self, structure: DiscretePCircuit) -> DiscretePCircuit:
        """Return a `DiscretePCircuit` with this compiled circuit's gates and reps."""
        return DiscretePCircuit(self.gates, self.reps)


# todo: as the DFG matures, we can probably merge this into the PSC DFG .sample
class StateVectorSimulator(
    AbstractSimulator[DiscretePCircuit, CompiledStateVectorPCircuit]
):
    """Simulator for exact state vectors representing probability distributions."""

    circuit_backend: ClassVar[Type[AbstractCompiledPCircuit]] = (
        CompiledStateVectorPCircuit
    )

    def build_circuit(
        self, circuit: DiscretePCircuit, thetas: list[Float[Array, "..."]]
    ) -> CompiledStateVectorPCircuit:
        return CompiledStateVectorPCircuit.from_pcircuit(circuit, thetas)

    @staticmethod
    def apply_gate(
        state: StateVector,
        gate: AbstractDiscreteGate,
        theta: Float[Array, "..."],
        dims: tuple[int, ...],
    ) -> StateVector:
        """
        Apply gate to the state StateVector and return the resulting state.

        **Arguments:**

        - `state`: The state to apply the gate to
        - `gate`: The gate to apply
        - `theta`: The gate's parameters
        - `dims`: The dimensions of all sites in the circuit

        **Returns:**

        The state after applying the gate.
        """
        # Get site indices
        sites = gate.sites
        if isinstance(sites, int):
            sites = [sites]
        else:
            sites = list(sites)

        # Get dimensions for the sites this gate acts on
        sub_dims = tuple(dims[s] for s in sites)

        matrix = gate.get_matrix(theta)

        # Reshape state to tensor form
        reshaped_state = state.reshape(dims)

        # Reshape matrix from (prod(sub_dims), prod(sub_dims)) to
        # (d0, d1, ..., d0', ...)
        # First half are output dims, second half are input dims
        matrix_shape = sub_dims + sub_dims
        reshaped_matrix = matrix.reshape(matrix_shape)

        num_sites = len(sites)
        matrix_contract_axes = list(range(num_sites, 2 * num_sites))

        result = jnp.tensordot(
            reshaped_matrix, reshaped_state, axes=(matrix_contract_axes, sites)
        )

        # tensordot puts output axes first, move them back to original positions
        # result shape is (d0, d1, ..., remaining_dims...)
        result = jnp.moveaxis(result, list(range(num_sites)), sites)

        return result.reshape(-1)

    def density(
        self, circuit: CompiledStateVectorPCircuit, x: StateVector
    ) -> StateVector:
        """
        Compute the final distribution over computational basis states.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compute the distribution of
        - `x`: The initial state vector, also the initial distribution

        **Returns:**

        The final distribution of the circuit.
        """
        dims = circuit.dims
        state = x.reshape(dims)

        def body_fn(_: int, state: StateVector) -> StateVector:
            # todo: we could also allow an option to pad the gates out
            # this would reduce compilation time, but potentially increase memory
            for gate, theta in zip(circuit.gates, circuit.thetas):
                state = self.apply_gate(state.reshape(-1), gate, theta, dims).reshape(
                    dims
                )

            return state

        state = jax.lax.fori_loop(0, circuit.reps, body_fn, state)

        return state.reshape(-1)

    def expval(
        self, circuit: CompiledStateVectorPCircuit, x: StateVector, pbit: int
    ) -> Float[Array, ""]:
        r"""
        Compute the expectation value of the given discrete site after circuit
        execution.

        For a site with basis values $0, \ldots, d - 1$, this returns
        $\sum_i i \, P(x = i)$. For binary sites, this is equivalent to the
        probability of measuring ``1``.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial state vector, also the initial distribution
        - `pbit`: The index of the site to compute the final expectation value of

        **Returns:**

        The expectation value of the given site after circuit execution.
        """
        dims = circuit.dims
        state = self.density(circuit, x).reshape(dims)

        sum_axes = list(range(circuit.num_pdits))
        sum_axes.remove(pbit)
        marginal = jnp.sum(state, axis=tuple(sum_axes))
        values = jnp.arange(dims[pbit], dtype=marginal.dtype)
        return jnp.sum(values * marginal)

    def expval_all(
        self, circuit: CompiledStateVectorPCircuit, x: StateVector
    ) -> Float[Array, " pbits"]:
        r"""
        Compute the expectation value of all discrete sites after circuit execution.

        For each site with basis values $0, \ldots, d - 1$, this returns
        $\sum_i i \, P(x = i)$. For binary sites, this is equivalent to the
        probability of measuring ``1``.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial state vector, also the initial distribution

        **Returns:**

        An array containing the expectation values of all discrete sites.
        """
        dims = circuit.dims
        state = self.density(circuit, x).reshape(dims)

        expvals = []
        for pbit in range(circuit.num_pdits):
            sum_axes = list(range(circuit.num_pdits))
            sum_axes.remove(pbit)
            marginal = jnp.sum(state, axis=tuple(sum_axes))
            values = jnp.arange(dims[pbit], dtype=marginal.dtype)
            expvals.append(jnp.sum(values * marginal))

        return jnp.stack(expvals)
