"""Sample based simulators."""

import functools
import math
from typing import ClassVar, Literal, Type

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Key, PyTree
from typing_extensions import Self

from .._circuit import DiscretePCircuit
from .._custom_types import BitString
from .._sampler import _resolve, AbstractSampler
from ..gates import AbstractGeneratorGate, AbstractKBranchGate
from .base import AbstractCompiledPCircuit, AbstractSimulator


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
    sampler: AbstractSampler  # entropy source for branch decisions

    @classmethod
    def from_pcircuit(
        cls,
        circuit: DiscretePCircuit,
        thetas: list[Float[Array, "..."]],
        sampler: AbstractSampler | None = None,
    ) -> Self:
        r"""
        Compile the given probabilistic circuit with the given parameters.

        This compiled form removes the list of probabilistic gates in favour
        of a more JIT-friendly representation. This representation uses the
        matrix forms of the branches of the probabilistic gates.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compile
        - `thetas`: Per-gate parameters aligned with `circuit.gates`; stacked
            and padded to `(num_gates, max_branches - 1)` with `-inf`.
        - `sampler`: Optional entropy source for branch decisions.

        **Returns:**

        The compiled circuit.
        """
        if len(thetas) != len(circuit.gates):
            raise ValueError(
                f"expected {len(circuit.gates)} thetas aligned with "
                f"circuit.gates, got {len(thetas)}."
            )
        if not circuit.gates:
            raise ValueError("SampleSimulator requires at least one gate.")

        # the maximum number of pdits that any single gate in the circuit acts on
        def _f(a, b):
            return max(a, len(b.sites) if isinstance(b.sites, list) else 1)

        locality = functools.reduce(
            _f,
            circuit.gates,
            1,
        )

        dtype = (
            jnp.int64
            if jax.config.jax_enable_x64  # pyright: ignore[reportAttributeAccessIssue]
            else jnp.int32
        )

        gates = [
            gate for gate in circuit.gates if isinstance(gate, AbstractKBranchGate)
        ]
        if len(gates) != len(circuit.gates):
            unsupported_gates = [
                gate
                for gate in circuit.gates
                if not isinstance(gate, AbstractKBranchGate)
            ]
            unsupported = [type(gate).__name__ for gate in unsupported_gates]
            generator_gates = [
                type(gate).__name__
                for gate in unsupported_gates
                if isinstance(gate, AbstractGeneratorGate)
            ]
            generator_hint = (
                " Generator gates require StateVectorSimulator: "
                f"{', '.join(generator_gates)}."
                if generator_gates
                else ""
            )
            raise ValueError(
                "SampleSimulator only supports AbstractKBranchGate lookup-table "
                f"gates; unsupported gates: {', '.join(unsupported)}."
                f"{generator_hint}"
            )

        max_branches = max(gate.num_branches for gate in gates)

        # compute per-gate dims and basis sizes
        gate_dims = []
        gate_basis_sizes = []
        for gate in gates:
            dims = list(gate.dims)
            num_sites = len(dims)
            # virtual padding sites
            dims_padded = dims + [2] * (locality - num_sites)
            gate_dims.append(dims_padded)
            gate_basis_sizes.append(math.prod(dims_padded))

        max_basis = max(gate_basis_sizes)
        dims_array = jnp.array(gate_dims, dtype=dtype)
        basis_sizes = jnp.array(gate_basis_sizes, dtype=dtype)

        # extend branch ops to max basis size, locality, and max_branches
        branch_ops_list = []
        num_branches_list = []
        for i, gate in enumerate(gates):
            K = gate.num_branches
            num_branches_list.append(K)

            # gate.branches has shape (K, basis_size, num_sites)
            # extend each branch to (max_basis, locality)
            extended_branches = jnp.stack(
                [
                    cls._extend_op(gate.branches[k], locality, max_basis, gate_dims[i])
                    for k in range(K)
                ]
            )

            # Padded branches are unreachable in normal sampling because their
            # logits are -inf. Deterministic gradient probes may index them to
            # keep vmaps rectangular, but their probabilities stay zero, so any
            # finite branch table is equivalent.
            if K < max_branches:
                padding = jnp.tile(extended_branches[0:1], (max_branches - K, 1, 1))
                extended_branches = jnp.concatenate(
                    [extended_branches, padding], axis=0
                )

            branch_ops_list.append(extended_branches)

        branch_ops = jnp.stack(branch_ops_list).astype(dtype)
        num_branches = jnp.array(num_branches_list, dtype=dtype)

        # the sites that the gates act on, extended to the appropriate locality
        sites = jnp.stack(
            [
                cls._extend_sites(gate.sites, locality, circuit.num_pdits)
                for gate in gates
            ]
        ).astype(dtype)

        # pad thetas to (num_gates, max_branches - 1) with -inf
        thetas_list = []
        for gate, theta in zip(gates, thetas):
            theta_vals = jnp.atleast_1d(theta)
            K = gate.num_branches
            # theta has K-1 values, pad to max_branches - 1 with -inf
            padded_theta = jnp.full(max_branches - 1, -jnp.inf)
            padded_theta = padded_theta.at[: K - 1].set(theta_vals[: K - 1])
            thetas_list.append(padded_theta)
        padded_thetas = jnp.stack(thetas_list)

        return cls(
            circuit.num_pdits,
            circuit.reps,
            max_branches,
            branch_ops,
            num_branches,
            sites,
            dims_array,
            basis_sizes,
            padded_thetas,
            _resolve(sampler),
        )

    def to_pcircuit(self, structure: DiscretePCircuit) -> DiscretePCircuit:
        """Return a `DiscretePCircuit` with `structure`'s gates and compiled reps."""
        return DiscretePCircuit(structure.gates, self.reps)

    def to_thetas(self) -> list[Float[Array, "..."]]:
        """Recover the per-gate parameter list from the padded `thetas`.

        Inverse of the stacking done in `from_pcircuit`: each gate's theta is
        the first `K - 1` entries of its padded row.

        **Returns:**

        A list of per-gate thetas, aligned with the original gates.
        """
        num_gates = self.thetas.shape[0]
        return [
            self.thetas[i, : int(self.num_branches[i]) - 1] for i in range(num_gates)
        ]

    @staticmethod
    def _extend_op(
        op: Int[Array, "basis_size sites"],
        locality: int,
        max_basis: int,
        dims_padded: list[int],
    ) -> Int[Array, "max_basis locality"]:
        """
        Extend the dimensions of the lookup table of a gate.

        "Locality" in this context means the maximum number of pdits that any
        single gate in the circuit acts on. Every gate that acts on less
        than this number of pdits must be padded with identity operators.

        This function is necessary since only matrices with the same shape can
        be stacked with each other. For example, a PNOT gate has the lookup
        table [[1], [0]] but a PCNOT gate has [[0, 0], [0, 1], [1, 1], [1, 0]],
        which is a different shape. In this case, it is necessary to extend
        the PNOT to 2 pdits (with an identity operator on the second pdit).

        The number of rows is prod(dims). Rows are padded to max_basis with
        zeros (these rows should never be indexed if basis_sizes is used
        correctly).

        **Arguments:**

        - `op`: the lookup table of a probabilistic gate
        - `locality`: the number of pdits to extend the gate to
        - `max_basis`: the maximum basis size across all gates
        - `dims_padded`: the dimensions of each site, padded to locality with 1s

        **Returns:**

        The new lookup table with shape (max_basis, locality).
        """
        num_sites = op.shape[1]
        # basis_size = op.shape[0]

        # compute the repeat factor from padded dims
        # for each padded site, we need to repeat the rows
        repeat_factor = math.prod(dims_padded[num_sites:])

        # repeat rows for the padded dimensions
        new_op = jnp.repeat(op, repeat_factor, axis=0)

        # pad columns for extra sites
        new_op = jnp.pad(new_op, [(0, 0), (0, locality - num_sites)])

        # pad rows to max_basis (these should never be accessed)
        current_rows = new_op.shape[0]
        if current_rows < max_basis:
            new_op = jnp.pad(new_op, [(0, max_basis - current_rows), (0, 0)])

        return new_op

    @staticmethod
    def _extend_sites(
        sites: int | list[int], locality: int, num_sites: int
    ) -> Int[Array, " locality"]:
        """
        Extend the number of sites of a gate to the given locality.

        "Locality" in this context means the maximum number of sites that any
        single gate in the circuit acts on. Every gate that acts on less
        than this number of sites must be padded with identity operators.

        This function is necessary since only arrays with the same length can
        be stacked with each other. For example, a PNOT gate has a site
        list of length 1 but a PCNOT gate has a site list of length 2. Thus
        it is necessary to extend the PNOT to 2 sites (with an identity
        operator on the second site).

        **Arguments:**

        - `sites`: the sites that the probabilistic gate acts on
        - `locality`: the number of sites to extend the gate to
        - `num_sites`: the total number of sites in the circuit

        **Returns:**

        The new sites.
        """
        sts = jnp.array([sites]) if isinstance(sites, int) else jnp.array(sites)
        new_sites = jnp.concatenate(
            [sts, jnp.arange(num_sites, num_sites + locality - len(sts))]
        )
        return new_sites


_DiffMethod = Literal["param_shift_inf", "param_shift_single", "param_shift_filter"]


class SampleSimulator(AbstractSimulator[DiscretePCircuit, CompiledSamplePCircuit]):
    r"""
    A sample-based simulator for probabilistic circuits.

    Instead of storing full state vectors, it samples from distributions.

    ??? tip "Choosing a differentiation method"

        Three differentiation methods are available:

        - `"param_shift_inf"`: Uses the parameter shift rule with deterministic
          gates ($\theta \to \pm\infty$). Requires $2N$ circuit evaluations for
          $N$ parameters.

        - `"param_shift_single"`: Uses the parameter shift rule with primal reuse.
          Requires $N$ circuit evaluations.

        - `"param_shift_filter"`: Estimates gradients from a single forward pass
          by filtering samples based on which branch was taken at each gate. For
          each gate, samples are partitioned into those that applied the gate vs.
          those that did not, and the gradient is estimated from the difference
          in expectation values between these groups.

    """

    diff_method: _DiffMethod
    num_samples: int
    sampler: AbstractSampler

    circuit_backend: ClassVar[Type[AbstractCompiledPCircuit]] = CompiledSamplePCircuit

    def __init__(
        self,
        diff_method: _DiffMethod = "param_shift_inf",
        num_samples: int = 1,
        sampler: AbstractSampler | None = None,
    ):
        """
        Initialize the sample simulator.

        **Arguments:**

        - `diff_method`: method used for differentiating circuit parameters
        - `num_samples`: number of samples used to estimate expectation values
        - `sampler`: source of randomness for branch decisions.
        """
        if diff_method not in [
            "param_shift_inf",
            "param_shift_single",
            "param_shift_filter",
        ]:
            raise ValueError(
                "diff_method must be param_shift_inf, param_shift_single, "
                "or param_shift_filter"
            )
        self.num_samples = num_samples
        self.diff_method = diff_method
        self.sampler = _resolve(sampler)

    def sample(
        self, circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""]
    ) -> Int[Array, "num_samples num_pbits"]:
        """
        Obtain samples from the final distribution of the probabilistic circuit.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial computational basis state of the circuit
        - `key`: The random key to use to obtain samples

        **Returns:**

        An integer array with shape (num_samples, num_pbits) containing
        the computational basis state samples.
        """
        if x.shape[0] != circuit.num_pdits:
            raise ValueError(
                f"Malformed bitstring, shape {x.shape[0]} should match "
                f"number of pbits {circuit.num_pdits}"
            )
        return sample_circuit(circuit, x, key, num_samples=self.num_samples)[0]

    def expval(
        self,
        circuit: CompiledSamplePCircuit,
        x: BitString,
        pbit: int,
        key: Key[Array, ""],
    ) -> Float[Array, ""]:
        r"""
        Estimate the expectation value of the given discrete site after circuit
        execution.

        For a site with basis values $0, \ldots, d - 1$, this estimates
        $\mathbb{E}[x] = \sum_i i \, P(x = i)$.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial computational basis state of the circuit
        - `pbit`: The index of the site to estimate the final expectation value of
        - `key`: The random key to use to obtain samples

        **Returns:**

        The expectation value of the given site after circuit execution.
        """
        return self.expval_all(circuit, x, key)[pbit]

    def expval_all(
        self, circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""]
    ) -> Float[Array, " num_pbits"]:
        r"""
        Estimate the expectation value of all discrete sites after circuit execution.

        For each site with basis values $0, \ldots, d - 1$, this estimates
        $\mathbb{E}[x] = \sum_i i \, P(x = i)$.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial computational basis state of the circuit
        - `key`: The random key to use to obtain samples

        **Returns:**

        An array containing the expectation values of all discrete sites.
        """
        # we don't currently support differentiation of circuits with multiple
        # repetitions with the param_shift_inf and param_shift_single variants,
        # because this would involve unrolling the scan in the sampling function
        # function, which would lead to large compilation times
        if self.diff_method == "param_shift_inf":
            return sample_expval_all_param_shift_inf(
                circuit, x, key, num_samples=self.num_samples
            )

        if self.diff_method == "param_shift_single":
            return sample_expval_all_param_shift_single(
                circuit, x, key, num_samples=self.num_samples
            )

        return sample_expval_all_param_shift_filter(
            circuit, x, key, num_samples=self.num_samples
        )

    def build_circuit(
        self, circuit: DiscretePCircuit, thetas: list[Float[Array, "..."]]
    ) -> CompiledSamplePCircuit:
        """Compile ``circuit`` with this simulator's branch sampler attached.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compile.
        - `thetas`: Per-gate parameters aligned with `circuit.gates`.

        **Returns:**

        The compiled circuit.
        """
        return CompiledSamplePCircuit.from_pcircuit(circuit, thetas, self.sampler)


def sample_circuit(
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

    def _apply_gate(
        state: Int[Array, " num_pbits"],
        branch_ops: Int[Array, "K B l"],
        sites: Int[Array, " l"],
        dims: Int[Array, " l"],
        branch_idx: Int[Array, ""],
    ) -> Int[Array, " num_pbits"]:
        dtype = (
            jnp.int64
            if jax.config.jax_enable_x64  # pyright: ignore[reportAttributeAccessIssue]
            else jnp.int32
        )

        # (locality,)
        # Sites >= num_pbits are virtual padding sites
        # append zeros before indexing
        virtual_state = jnp.zeros((sites.shape[0],), dtype=state.dtype)
        padded_state = jnp.concatenate([state, virtual_state])
        substate = padded_state[sites].astype(dtype)

        # index = sum_i substate[i] * prod(dims[i+1:])
        # compute strides: [prod(dims[1:]), prod(dims[2:]), ..., 1]
        strides = jnp.cumprod(dims[::-1])[::-1]
        strides = jnp.concatenate([strides[1:], jnp.array([1], dtype=dtype)])
        index = jnp.sum(substate * strides)

        # select the branch operation based on branch_idx
        selected_op = branch_ops[branch_idx]
        new_substate = selected_op[index]

        return state.at[sites].set(new_substate)

    # inner loop scans over the gates of the circuit
    def inner_body_fn(
        carry: Int[Array, "num_samples num_pbits"], x: tuple
    ) -> tuple[Int[Array, "num_samples num_pbits"], None]:
        # (num_samples, num_pbits)
        state = carry

        # (K, B, locality), (locality,), (locality,), (num_samples,)
        branch_ops, sites, dims, branch_idx = x

        # (num_samples, num_pbits)
        state = jax.vmap(_apply_gate, in_axes=(0, None, None, None, 0))(
            state, branch_ops, sites, dims, branch_idx
        )
        return state, None

    # outer loop scans over the number of repetitions
    def outer_body_fn(
        carry: Int[Array, "num_samples num_pbits"],
        x: Int[Array, "num_gates num_samples"],
    ) -> tuple[Int[Array, "num_samples num_pbits"], None]:
        # (num_samples, num_pbits)
        state = carry

        # (num_gates, num_samples)
        branch_indices = x

        # (num_samples, num_pbits)
        state = jax.lax.scan(
            inner_body_fn, state, (branch_ops, sites, dims, branch_indices)
        )[0]
        return state, None

    # (num_gates, max_branches, B, locality)
    branch_ops = circuit.branch_ops

    # (num_gates, locality)
    sites = circuit.sites
    dims = circuit.dims

    num_gates = circuit.thetas.shape[0]

    if circuit.max_branches == 2:
        probs = jax.nn.sigmoid(circuit.thetas[:, 0])
        branch_indices = circuit.sampler.bernoulli(
            key,
            probs[None, :, None],
            shape=(circuit.reps, num_gates, num_samples),
        )
    else:
        padded_logits = jnp.concatenate(
            [jnp.zeros((num_gates, 1)), circuit.thetas], axis=1
        )
        branch_indices = circuit.sampler.categorical(
            key,
            padded_logits[None, :, None, :],
            axis=-1,
            shape=(circuit.reps, num_gates, num_samples),
        )

    # (num_samples, num_pbits)
    state = jax.lax.scan(outer_body_fn, state, branch_indices)[0]

    return state, branch_indices


def _expval_all(
    circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""], num_samples: int
) -> Float[Array, " pbits"]:
    r"""
    Estimate the expectation value of all discrete sites after circuit execution.

    For each site with basis values $0, \ldots, d - 1$, this estimates
    $\mathbb{E}[x] = \sum_i i \, P(x = i)$.

    **Arguments:**

    - `circuit`: The probabilistic circuit to execute
    - `x`: The initial computational basis state of the circuit
    - `key`: The random key to use to obtain samples
    - `num_samples`: The number of samples to obtain to estimate the expval

    **Returns:**

    An array containing the expectation values of all discrete sites.
    """
    samples = sample_circuit(circuit, x, key, num_samples)[0]
    val = jnp.mean(samples.astype(jnp.float32), axis=0)
    return val


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
        if k == 0:
            new_thetas = new_thetas.at[g, :].set(-jnp.inf)
        else:
            new_thetas = new_thetas.at[g, :].set(-jnp.inf)
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

    # Shared gate parameters are reused in every repetition
    # sum over reps
    num_reps = branch_indices.shape[0]
    expval_per_branch_sum = jnp.sum(expval_per_branch, axis=0)

    # diff[g, k] = sum_r (E[O | branch_{r,g}=k] - E[O]) for branches 1..K-1.
    # (num_gates, max_branches-1, num_pbits)
    diff = expval_per_branch_sum[:, 1:, :] - num_reps * expval_overall[None, None, :]

    # grad[g, j] = p[g, j+1] * diff[g, j] @ grad_out
    # (num_gates, max_branches-1)
    grads_in = probs[:, 1:] * jnp.einsum("gjp,p->gj", diff, grad_out)

    trainable = eqx.filter(circuit, perturbed)
    grad_circuit = jax.tree.unflatten(jax.tree.structure(trainable), (grads_in,))

    return grad_circuit
