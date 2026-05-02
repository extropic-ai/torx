"""Compiled data model for sample-based probabilistic circuits."""

import functools

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int
from typing_extensions import Self

from .._circuit import DiscretePCircuit
from .._custom_meta import AbstractStrictModule
from ..gates import AbstractConditionalSampleGate, AbstractKBranchGate
from .base import AbstractCompiledPCircuit


class CompiledConditionalSampleData(AbstractStrictModule):
    """Compact conditional-gate arrays for the sample simulator.

    Conditional payloads are stored only once per conditional gate. During the
    forward scan, `gate_cond_indices` expands them back to circuit-gate order.
    """

    gate_types: Int[Array, " num_gates"]  # 0 = K-branch, 1 = conditional sample
    gate_cond_indices: Int[Array, " num_gates"]
    target_sites: Int[Array, "num_cond_gates max_targets"]
    control_sites: Int[Array, "num_cond_gates max_targets max_controls"]
    control_weights: Float[Array, "num_cond_gates max_targets max_controls"]
    logits: Float[Array, "num_cond_gates max_targets"]
    num_targets: Int[Array, " num_cond_gates"]
    num_controls: Int[Array, "num_cond_gates max_targets"]


class CompiledSamplePCircuit(AbstractCompiledPCircuit[DiscretePCircuit]):
    """Compiled probabilistic circuit class for the sample-based simulator."""

    num_pdits: int
    reps: int
    max_branches: int

    branch_ops: Int[Array, "num_gates max_branches max_basis max_l"]
    num_branches: Int[Array, " num_gates"]  # K for each gate
    sites: Int[Array, "num_gates l"]  # index of sites (e.g. wires operated on)
    dims: Int[Array, "num_gates l"]  # per-site dimensions, padded
    basis_sizes: Int[Array, " num_gates"]  # prod(dims) for each gate
    thetas: Float[
        Array, "num_gates max_branches_minus_1"
    ]  # K-1 params per gate, padded with -inf
    conditional_data: CompiledConditionalSampleData | None

    @property
    def has_conditional(self) -> bool:
        return self.conditional_data is not None

    @classmethod
    def from_pcircuit(cls, circuit: DiscretePCircuit) -> Self:
        """Compile a probabilistic circuit into array-backed sampling data."""
        dtype = cls._index_dtype()
        gates = cls._validated_gates(circuit.gates)
        branch_gates = [gate for gate in gates if isinstance(gate, AbstractKBranchGate)]
        cond_gates = [
            gate for gate in gates if isinstance(gate, AbstractConditionalSampleGate)
        ]

        locality = max((cls._branch_locality(gate) for gate in branch_gates), default=1)
        max_branches = max((gate.num_branches for gate in branch_gates), default=2)

        gate_dims, dims_array, basis_sizes, max_basis = cls._gate_dimensions(
            gates, locality, dtype
        )
        branch_ops, num_branches = cls._compile_branch_ops(
            gates, gate_dims, locality, max_basis, max_branches, dtype
        )
        sites = cls._compile_sites(gates, locality, circuit.num_pdits, dtype)
        thetas = cls._compile_thetas(gates, max_branches)
        conditional_data = cls._compile_conditional_data(gates, cond_gates, dtype)

        return cls(
            circuit.num_pdits,
            circuit.reps,
            max_branches,
            branch_ops,
            num_branches,
            sites,
            dims_array,
            basis_sizes,
            thetas,
            conditional_data,
        )

    @staticmethod
    def _index_dtype():
        return (
            jnp.int64
            if jax.config.jax_enable_x64  # pyright: ignore[reportAttributeAccessIssue]
            else jnp.int32
        )

    @staticmethod
    def _validated_gates(gates) -> list:
        compiled_gates = list(gates)
        for gate in compiled_gates:
            if not isinstance(
                gate, (AbstractKBranchGate, AbstractConditionalSampleGate)
            ):
                raise ValueError(
                    "SampleSimulator only supports AbstractKBranchGate and "
                    "AbstractConditionalSampleGate gates."
                )
        return compiled_gates

    @staticmethod
    def _branch_locality(gate: AbstractKBranchGate) -> int:
        return len(gate.sites) if isinstance(gate.sites, list) else 1

    @staticmethod
    def _num_cond_targets(gate: AbstractConditionalSampleGate) -> int:
        return gate.num_targets

    @staticmethod
    def _num_cond_controls(gate: AbstractConditionalSampleGate) -> int:
        return gate.num_controls

    @classmethod
    def _gate_dimensions(cls, gates, locality: int, dtype):
        gate_dims = []
        basis_sizes = []
        for gate in gates:
            if isinstance(gate, AbstractKBranchGate):
                dims = list(gate.dims)
                dims_padded = dims + [2] * (locality - len(dims))
            else:
                dims_padded = [2] * locality
            gate_dims.append(dims_padded)
            basis_sizes.append(cls._basis_size(dims_padded))
        return (
            gate_dims,
            jnp.array(gate_dims, dtype=dtype),
            jnp.array(basis_sizes, dtype=dtype),
            max(basis_sizes),
        )

    @staticmethod
    def _basis_size(dims: list[int]) -> int:
        basis_size = 1
        for dim in dims:
            basis_size *= int(dim)
        return basis_size

    @classmethod
    def _compile_branch_ops(
        cls,
        gates,
        gate_dims: list[list[int]],
        locality: int,
        max_basis: int,
        max_branches: int,
        dtype,
    ):
        branch_ops = []
        num_branches = []
        for i, gate in enumerate(gates):
            if isinstance(gate, AbstractKBranchGate):
                num_branches.append(gate.num_branches)
                extended = jnp.stack(
                    [
                        cls._extend_op(
                            gate.branches[k], locality, max_basis, gate_dims[i]
                        )
                        for k in range(gate.num_branches)
                    ]
                )
                if gate.num_branches < max_branches:
                    padding = jnp.tile(
                        extended[0:1], (max_branches - gate.num_branches, 1, 1)
                    )
                    extended = jnp.concatenate([extended, padding], axis=0)
            else:
                num_branches.append(1)
                extended = jnp.zeros((max_branches, max_basis, locality), dtype=dtype)
            branch_ops.append(extended)
        return (
            jnp.stack(branch_ops).astype(dtype),
            jnp.array(num_branches, dtype=dtype),
        )

    @classmethod
    def _compile_sites(cls, gates, locality: int, num_pdits: int, dtype):
        return jnp.stack(
            [
                (
                    cls._extend_sites(gate.sites, locality, num_pdits)
                    if isinstance(gate, AbstractKBranchGate)
                    else jnp.zeros(locality, dtype=dtype)
                )
                for gate in gates
            ]
        ).astype(dtype)

    @staticmethod
    def _compile_thetas(gates, max_branches: int):
        thetas = []
        for gate in gates:
            padded = jnp.full(max_branches - 1, -jnp.inf)
            if isinstance(gate, AbstractKBranchGate):
                theta_vals = jnp.atleast_1d(gate.theta)
                padded = padded.at[: gate.num_branches - 1].set(
                    theta_vals[: gate.num_branches - 1]
                )
            thetas.append(padded)
        return jnp.stack(thetas)

    @classmethod
    def _compile_conditional_data(
        cls,
        gates,
        cond_gates,
        dtype,
    ) -> CompiledConditionalSampleData | None:
        if not cond_gates:
            return None

        max_targets = max(cls._num_cond_targets(gate) for gate in cond_gates)
        max_controls = max(cls._num_cond_controls(gate) for gate in cond_gates)
        cond_float_dtype = cls._conditional_float_dtype(cond_gates, dtype)
        cond_indices = {id(gate): idx for idx, gate in enumerate(cond_gates)}
        gate_types = []
        gate_cond_indices = []
        targets = []
        controls = []
        weights = []
        logits = []
        num_targets = []
        num_controls = []

        for gate in gates:
            if isinstance(gate, AbstractConditionalSampleGate):
                gate_types.append(1)
                gate_cond_indices.append(cond_indices[id(gate)])
            else:
                gate_types.append(0)
                # Branch gates never read conditional arrays; index 0 is a
                # harmless placeholder that keeps the scan inputs rectangular.
                gate_cond_indices.append(0)

        for gate in cond_gates:
            target_sites = jnp.zeros(max_targets, dtype=dtype)
            control_sites = jnp.zeros((max_targets, max_controls), dtype=dtype)
            control_weights = jnp.zeros(
                (max_targets, max_controls), dtype=cond_float_dtype
            )
            cond_logits = jnp.zeros(max_targets, dtype=cond_float_dtype)
            controls_per_target = jnp.zeros(max_targets, dtype=dtype)

            nt = cls._num_cond_targets(gate)
            nc = cls._num_cond_controls(gate)
            target_sites = target_sites.at[:nt].set(
                jnp.array(gate.target_sites, dtype=dtype)
            )
            control_sites = control_sites.at[:nt, :nc].set(
                jnp.array(gate.control_sites, dtype=dtype)
            )
            control_weights = control_weights.at[:nt, :nc].set(
                jnp.asarray(gate.conditional_weights, dtype=cond_float_dtype)
            )
            cond_logits = cond_logits.at[:nt].set(
                jnp.asarray(gate.conditional_logits, dtype=cond_float_dtype)
            )
            controls_per_target = controls_per_target.at[:nt].set(nc)
            num_targets.append(nt)

            targets.append(target_sites)
            controls.append(control_sites)
            weights.append(control_weights)
            logits.append(cond_logits)
            num_controls.append(controls_per_target)

        return CompiledConditionalSampleData(
            jnp.array(gate_types, dtype=dtype),
            jnp.array(gate_cond_indices, dtype=dtype),
            jnp.stack(targets),
            jnp.stack(controls),
            jnp.stack(weights),
            jnp.stack(logits),
            jnp.array(num_targets, dtype=dtype),
            jnp.stack(num_controls),
        )

    @staticmethod
    def _conditional_float_dtype(cond_gates, dtype):
        if not cond_gates:
            return dtype
        return functools.reduce(
            jnp.result_type,
            [
                item
                for gate in cond_gates
                for item in (gate.conditional_logits, gate.conditional_weights)
            ],
        )

    def to_pcircuit(self, structure: DiscretePCircuit) -> DiscretePCircuit:
        """
        Create a new circuit with the same structure as the given circuit.

        **Arguments:**

        - `structure`: The probabilistic circuit to reparametrize

        **Returns:**

        A probabilistic circuit with the parameters of the compiled circuit.
        """
        new_gates = []
        conditional = self.conditional_data

        for i, old_gate in enumerate(structure):
            if isinstance(old_gate, AbstractConditionalSampleGate):
                if conditional is None:
                    raise ValueError(
                        "Compiled circuit does not contain conditional data."
                    )
                cond_idx = int(conditional.gate_cond_indices[i])
                nt = old_gate.num_targets
                nc = self._num_cond_controls(old_gate)
                new_gate = eqx.tree_at(
                    lambda g: (g.theta, g.control_weights),
                    old_gate,
                    (
                        conditional.logits[cond_idx, :nt],
                        conditional.control_weights[cond_idx, :nt, :nc],
                    ),
                )
            else:
                K = self.num_branches[i]
                new_param = self.thetas[i, : K - 1]
                # if K=2, theta should be scalar
                if K == 2:
                    new_param = new_param[0]
                new_gate = eqx.tree_at(lambda g: g.theta, old_gate, new_param)
            new_gates.append(new_gate)

        return DiscretePCircuit(new_gates, self.reps)

    @staticmethod
    def _extend_op(
        op: Int[Array, "basis_size sites"],
        locality: int,
        max_basis: int,
        dims_padded: list[int],
    ) -> Int[Array, "max_basis locality"]:
        """Pad a branch lookup table to the shared sample-kernel shape."""
        num_sites = op.shape[1]
        repeat_factor = int(jnp.prod(jnp.array(dims_padded[num_sites:])))
        new_op = jnp.repeat(op, repeat_factor, axis=0)
        new_op = jnp.pad(new_op, [(0, 0), (0, locality - num_sites)])
        current_rows = new_op.shape[0]
        if current_rows < max_basis:
            new_op = jnp.pad(new_op, [(0, max_basis - current_rows), (0, 0)])

        return new_op

    @staticmethod
    def _extend_sites(
        sites: int | list[int], locality: int, num_sites: int
    ) -> Int[Array, " locality"]:
        """Pad gate sites with virtual indices up to the shared locality."""
        sts = jnp.array([sites]) if isinstance(sites, int) else jnp.array(sites)
        new_sites = jnp.concatenate(
            [sts, jnp.arange(num_sites, num_sites + locality - len(sts))]
        )
        return new_sites
