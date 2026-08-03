"""Probabilistic circuit classes."""

import functools
import itertools
from abc import abstractmethod
from typing import (
    Any,
    Generic,
    Iterator,
    Mapping,
    NamedTuple,
    Sequence,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, Key, PyTree

from ..dfg import AbstractDFG, Site
from ..factor import PortSpec
from ..tractable_prob_factors import DeterministicFactor
from .gates import AbstractDiscreteGate, AbstractHybridGate, AbstractPGate

_GateType = TypeVar("_GateType", bound=AbstractPGate)


class _Src(NamedTuple):
    """A wire's current source.

    ``values[writer][start : start + length]``.
    """

    writer: str
    start: int
    length: int


class _Gather(NamedTuple):
    """A slice of one parent's output.

    ``parent_outputs[parent_index][start : start + length]``.
    """

    parent_index: int
    start: int
    length: int


class _GateWiring(NamedTuple):
    """One gate's wire-level I/O.

    `reads` maps each of the gate's input ports to the wires gathered into it,
    `write_wires` are the wires the gate writes, `write_lens` their lengths
    within that output.
    """

    reads: Mapping[str, Sequence[Any]]
    write_wires: Sequence[Any]
    write_lens: Sequence[int]


class _GatherPorting(eqx.Module):
    """A `Site.porting_fn` that gathers per-wire slices from parent outputs.

    `spec` describes how to build each input port, including the dtype used
    when a port gathers no wires. Concretely it is:

        ( (port_name, (gather, gather, ...), dtype),
          (port_name, (gather, ...), dtype),
          ... )

    """

    spec: tuple[tuple[str, tuple[_Gather, ...], Any], ...] = eqx.field(static=True)

    def __call__(
        self, parent_outputs: Sequence[PyTree[Array]]
    ) -> Mapping[str, PyTree[Array]]:
        inputs = {}
        for port, gathers, dtype in self.spec:
            parts = [parent_outputs[pi][s : s + n] for pi, s, n in gathers]
            inputs[port] = jnp.concatenate(parts) if parts else jnp.zeros((0,), dtype)
        return inputs


def _wire_sites(
    gates: Sequence[AbstractPGate],
    wirings: Sequence[_GateWiring],
    initial_src: Mapping[Any, _Src],
    output_wires: Sequence[Any] | Mapping[str, Sequence[Any]],
    output_spec: PortSpec,
    param_keys: Sequence[int],
) -> tuple[Site, ...]:
    """Build one executable `Site` per gate plus a terminal output site."""

    src = dict(initial_src)

    def parents_of(wire_lists):
        parents = []
        for wires in wire_lists:
            for w in wires:
                addr = src[w].writer
                if addr not in parents:
                    parents.append(addr)
        return tuple(parents)

    def gather(parents, wires):
        return tuple(
            _Gather(parents.index(src[w].writer), src[w].start, src[w].length)
            for w in wires
        )

    def porting(parents, reads, factor_input_ports):
        return _GatherPorting(
            tuple(
                (
                    port,
                    gather(parents, wires),
                    jax.tree.leaves(factor_input_ports[port])[0].dtype,
                )
                for port, wires in reads.items()
            )
        )

    sites = []
    for i, (gate, (reads, write_wires, write_lens), param_key) in enumerate(
        zip(gates, wirings, param_keys)
    ):
        name = f"g{i}"
        parents = parents_of(list(reads.values()))
        sites.append(
            Site(
                name=name,
                factor=gate,
                parents=parents,
                porting_fn=porting(parents, reads, gate.input_ports),
                param_key=param_key,
                info_key=None,
                site_info=None,
            )
        )
        offset = 0
        for w, length in zip(write_wires, write_lens):
            src[w] = _Src(name, offset, length)
            offset += length

    if isinstance(output_wires, Mapping):
        output_reads = output_wires
        output_input_ports = output_spec
        output_fn = lambda inputs, site_info: dict(inputs)
    else:
        output_reads = {"in": output_wires}
        output_input_ports = {"in": output_spec}
        output_fn = lambda inputs, site_info: inputs["in"]

    out_parents = parents_of(list(output_reads.values()))
    sites.append(
        Site(
            name="out",
            factor=DeterministicFactor(output_fn, output_input_ports, output_spec),
            parents=out_parents,
            porting_fn=porting(out_parents, output_reads, output_input_ports),
            param_key=None,
            info_key=None,
            site_info=None,
        )
    )
    return tuple(sites)


class AbstractPCircuit(AbstractDFG, Generic[_GateType]):
    """Abstract base class for probabilistic circuits."""

    gates: eqx.AbstractVar[list[_GateType]]
    reps: eqx.AbstractVar[int]

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Return iterator over circuit gates."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return number of gates."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> _GateType:
        """Return gate at a particular index in the circuit order."""
        raise NotImplementedError

    def init_params(self, key: Key[Array, ""]) -> list[PyTree[Array]]:
        """Default per-gate parameters, aligned with the circuit's gates.

        Splits `key` and calls each gate's `init_params`.

        **Arguments:**

        - `key`: PRNG key.

        **Returns:**

        A list of per-gate parameter pytrees, one per gate in circuit order.
        """
        keys = jax.random.split(key, max(len(self), 1))
        return [gate.init_params(k) for gate, k in zip(self, keys)]


class DiscretePCircuit(AbstractPCircuit[AbstractDiscreteGate]):
    """Probabilistic circuit for discrete gates."""

    gates: list[AbstractDiscreteGate]
    num_pdits: int
    dims: tuple[int, ...]
    reps: int
    sites: tuple[Site, ...]
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)
    output_name: str = eqx.field(static=True)
    topological_order: tuple[int, ...] = eqx.field(static=True)
    sites_by_name: Mapping[str, int] = eqx.field(static=True)

    def __init__(self, gates: list[AbstractDiscreteGate], reps: int = 1):
        """
        Initialize a probabilistic circuit.

        **Arguments:**

        - `gates`: the list of gates to be applied in the circuit
        - `reps`: the number of times the circuit is applied to the initial input
        """
        if reps < 1:
            raise ValueError("reps must be at least 1.")

        self.gates = gates
        self.reps = reps

        self.num_pdits = int(
            functools.reduce(
                lambda a, b: max(a, np.max(b.sites)),
                gates,
                -1,
            )
            + 1
        )

        dims_dict = {}
        for gate in gates:
            sites = gate.sites
            if isinstance(sites, int):
                sites = [sites]
            for i, site_idx in enumerate(sites):
                gate_dim = gate.dims[i]
                if site_idx in dims_dict and dims_dict[site_idx] != gate_dim:
                    raise ValueError(
                        f"Dimension mismatch at site {site_idx}: "
                        f"expected {dims_dict[site_idx]}, got {gate_dim}"
                    )
                dims_dict[site_idx] = gate_dim

        self.dims = tuple(dims_dict.get(i, 2) for i in range(self.num_pdits))

        # Each pdit is a wire holding a single value
        wirings = []
        for g in gates:
            wires = [g.sites] if isinstance(g.sites, int) else list(g.sites)
            wirings.append(_GateWiring({"in": wires}, wires, [1] * len(wires)))
        repeated_gates = list(gates) * reps
        repeated_wirings = wirings * reps
        param_keys = list(range(len(gates))) * reps
        dfg_sites = _wire_sites(
            repeated_gates,
            repeated_wirings,
            {p: _Src("in", p, 1) for p in range(self.num_pdits)},
            list(range(self.num_pdits)),
            jax.ShapeDtypeStruct((self.num_pdits,), jnp.int32),
            param_keys,
        )
        input_ports, sites_by_name, topological_order, output_spec = self._derive(
            dfg_sites,
            {"in": jax.ShapeDtypeStruct((self.num_pdits,), jnp.int32)},
            "out",
        )
        self.sites = dfg_sites
        self.input_ports = input_ports
        self.output_name = "out"
        self.sites_by_name = sites_by_name
        self.topological_order = topological_order
        self.output_spec = output_spec

    def __iter__(self) -> Iterator:
        """Return iterator over circuit gates."""
        return iter(self.gates)

    def __len__(self) -> int:
        """Return number of gates."""
        return len(self.gates)

    def __getitem__(self, index: int) -> AbstractDiscreteGate:
        """Return gate at a particular index in the circuit order."""
        return self.gates[index]

    def __add__(self, other):
        """Append other circuit's gates to this circuit's gates."""
        return DiscretePCircuit(self.gates + other.gates, reps=self.reps)


_HybridGateType: TypeAlias = AbstractDiscreteGate | AbstractHybridGate


class HybridState(TypedDict):
    """Discrete and continuous registers of a hybrid circuit."""

    discrete: Int[Array, "... num_discrete_sites"]
    continuous: Float[Array, "... continuous_dim"]


class HybridPCircuit(AbstractPCircuit[_HybridGateType]):
    """Circuit supporting discrete, continuous, and hybrid gates.

    Stores both discrete and continuous dimensions separately.
    """

    gates: list[_HybridGateType]
    discrete_dims: tuple[int, ...]
    continuous_dims: tuple[int, ...]
    reps: int
    sites: tuple[Site, ...]
    input_ports: Mapping[str, PortSpec] = eqx.field(static=True)
    output_spec: PortSpec = eqx.field(static=True)
    output_name: str = eqx.field(static=True)
    topological_order: tuple[int, ...] = eqx.field(static=True)
    sites_by_name: Mapping[str, int] = eqx.field(static=True)

    def __init__(
        self,
        gates: list[_HybridGateType],
        reps: int = 1,
    ):
        """
        Initialize a hybrid circuit.

        **Arguments:**

        - `gates`: List of gates (discrete, continuous, or hybrid).
        - `reps`: Number of times the circuit is repeated.
        """
        if reps < 1:
            raise ValueError("reps must be at least 1.")

        self.gates = gates
        self.reps = reps

        self.discrete_dims, self.continuous_dims = self._infer_dims(gates)

        # Discrete sites are width-1 wires; continuous site s is a width-
        # continuous_dims[s] wire at offset sum(continuous_dims[:s]) in the
        # continuous register.
        cont_dims = self.continuous_dims
        cont_off = self._continuous_starts

        initial_src = {
            ("d", p): _Src("discrete", p, 1) for p in range(len(self.discrete_dims))
        }
        initial_src.update(
            {
                ("c", s): _Src("continuous", cont_off[s], cont_dims[s])
                for s in range(len(cont_dims))
            }
        )

        def _wiring(gate: _HybridGateType) -> _GateWiring:
            if isinstance(gate, AbstractDiscreteGate):
                s = gate.sites
                pdits = [s] if isinstance(s, int) else list(s)
                dwires = [("d", p) for p in pdits]
                return _GateWiring({"in": dwires}, dwires, [1] * len(dwires))
            ds = gate.sites.get("discrete", [])
            cs = gate.sites.get("continuous", [])
            reads = {
                "discrete": [("d", p) for p in ds],
                "continuous": [("c", s) for s in cs],
            }
            cwires = [("c", s) for s in cs]
            return _GateWiring(reads, cwires, [cont_dims[s] for s in cs])

        repeated_gates = list(gates) * reps
        base_wirings = [_wiring(g) for g in gates]
        repeated_wirings = base_wirings * reps
        param_keys = list(range(len(gates))) * reps
        input_ports = {
            "discrete": jax.ShapeDtypeStruct((len(self.discrete_dims),), jnp.int32),
            "continuous": jax.ShapeDtypeStruct(
                (sum(self.continuous_dims),), jnp.float32
            ),
        }
        dfg_sites = _wire_sites(
            repeated_gates,
            repeated_wirings,
            initial_src,
            {
                "discrete": [("d", p) for p in range(len(self.discrete_dims))],
                "continuous": [("c", s) for s in range(len(cont_dims))],
            },
            input_ports,
            param_keys,
        )
        input_ports, sites_by_name, topological_order, output_spec = self._derive(
            dfg_sites, input_ports, "out"
        )
        self.sites = dfg_sites
        self.input_ports = input_ports
        self.output_name = "out"
        self.sites_by_name = sites_by_name
        self.topological_order = topological_order
        self.output_spec = output_spec

    @staticmethod
    def _infer_dims(
        gates: list[_HybridGateType],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        discrete_dims_dict: dict[int, int] = {}
        continuous_dims_dict: dict[int, int] = {}

        for gate in gates:
            if isinstance(gate, AbstractDiscreteGate):
                sites = gate.sites
                if isinstance(sites, int):
                    sites = [sites]
                for i, site_idx in enumerate(sites):
                    gate_dim = gate.dims[i]
                    if (
                        site_idx in discrete_dims_dict
                        and discrete_dims_dict[site_idx] != gate_dim
                    ):
                        raise ValueError(
                            f"Discrete dimension mismatch at site {site_idx}: "
                            f"expected {discrete_dims_dict[site_idx]}, got {gate_dim}"
                        )
                    discrete_dims_dict[site_idx] = gate_dim
            elif isinstance(gate, AbstractHybridGate):
                gate_sites = gate.sites
                disc_sites = gate_sites.get("discrete", [])
                cont_sites = gate_sites.get("continuous", [])

                for site_idx in disc_sites:
                    discrete_dims_dict.setdefault(site_idx, 2)

                for i, site_idx in enumerate(cont_sites):
                    gate_dim = gate.dims[i] if i < len(gate.dims) else 1
                    if (
                        site_idx in continuous_dims_dict
                        and continuous_dims_dict[site_idx] != gate_dim
                    ):
                        raise ValueError(
                            f"Continuous dimension mismatch at site {site_idx}: "
                            f"expected {continuous_dims_dict[site_idx]}, got {gate_dim}"
                        )
                    continuous_dims_dict[site_idx] = gate_dim

        num_discrete = max(discrete_dims_dict.keys(), default=-1) + 1
        num_continuous = max(continuous_dims_dict.keys(), default=-1) + 1

        discrete_dims = tuple(discrete_dims_dict.get(i, 2) for i in range(num_discrete))
        continuous_dims = tuple(
            continuous_dims_dict.get(i, 1) for i in range(num_continuous)
        )

        return discrete_dims, continuous_dims

    @property
    def num_discrete_sites(self) -> int:
        """Number of discrete sites in the circuit."""
        return len(self.discrete_dims)

    @property
    def num_continuous_sites(self) -> int:
        """Number of continuous sites in the circuit."""
        return len(self.continuous_dims)

    @property
    def continuous_state_dim(self) -> int:
        """Total dimension of continuous state = sum(continuous_dims)."""
        return sum(self.continuous_dims)

    @property
    def _continuous_starts(self) -> tuple[int, ...]:
        """Cumulative start offset of each continuous site in the register.

        Each start is the running sum of the preceding sites' dims, so site `i`
        occupies register slots `[start_i, start_i + dim_i)`. For
        `continuous_dims = (2, 1, 3)` this returns `(0, 2, 3)`: site 0 spans
        slots 0-1, site 1 slot 2, site 2 slots 3-5.
        """
        return tuple(itertools.accumulate(self.continuous_dims, initial=0))[:-1]

    @property
    def continuous_offsets(self) -> tuple[tuple[int, int, int], ...]:
        """Per continuous site `(site, start, stop)` slice into the register."""
        return tuple(
            (site, start, start + dim)
            for site, (start, dim) in enumerate(
                zip(self._continuous_starts, self.continuous_dims)
            )
        )

    def __iter__(self):
        """Return iterator over circuit gates."""
        return iter(self.gates)

    def __len__(self) -> int:
        """Return number of gates."""
        return len(self.gates)

    def __getitem__(self, index: int) -> _HybridGateType:
        """Return gate at a particular index in the circuit order."""
        return self.gates[index]
