"""Probabilistic circuit classes."""

import functools
from abc import abstractmethod
from typing import Generic, Iterator, TypeVar

import numpy as np

from ._custom_meta import AbstractStrictModule
from .gates import AbstractDiscreteGate, AbstractHybridGate, AbstractPGate

_GateType = TypeVar("_GateType", bound=AbstractPGate)


class AbstractPCircuit(AbstractStrictModule, Generic[_GateType]):
    """Abstract base class for probabilistic circuits."""

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


class DiscretePCircuit(AbstractPCircuit[AbstractDiscreteGate]):
    """Probabilistic circuit for discrete gates."""

    gates: list[AbstractDiscreteGate]
    num_pdits: int
    dims: tuple[int, ...]
    reps: int

    def __init__(self, gates: list[AbstractDiscreteGate], reps: int = 1):
        """
        Initialize a probabilistic circuit.

        **Arguments:**

        - `gates`: the list of gates to be applied in the circuit
        - `reps`: the number of times the circuit is applied to the initial input
        """
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

        dims_dict: dict[int, int] = {}
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


HybridGateType = AbstractDiscreteGate | AbstractHybridGate


class HybridPCircuit(AbstractPCircuit[HybridGateType]):
    """Circuit supporting discrete, continuous, and hybrid gates.

    Stores both discrete and continuous dimensions separately.
    """

    gates: list[HybridGateType]
    discrete_dims: tuple[int, ...]
    continuous_dims: tuple[int, ...]
    reps: int

    def __init__(
        self,
        gates: list[HybridGateType],
        reps: int = 1,
    ):
        """
        Initialize a hybrid circuit.

        **Arguments:**

        - `gates`: List of gates (discrete, continuous, or hybrid).
        - `reps`: Number of times the circuit is repeated.
        """
        self.gates = gates
        self.reps = reps

        self.discrete_dims, self.continuous_dims = self._infer_dims(gates)

    @staticmethod
    def _infer_dims(
        gates: list[HybridGateType],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        discrete_dims_dict: dict[int, int] = {}
        continuous_dims_dict: dict[int, int] = {}

        for gate in gates:
            if isinstance(gate, AbstractDiscreteGate):
                # Discrete gate: sites is int | list[int]
                sites = gate.sites if isinstance(gate.sites, list) else [gate.sites]
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
                cont_sites = gate_sites.get("continuous", [])

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

    def __iter__(self):
        """Return iterator over circuit gates."""
        return iter(self.gates)

    def __len__(self) -> int:
        """Return number of gates."""
        return len(self.gates)

    def __getitem__(self, index: int) -> HybridGateType:
        """Return gate at a particular index in the circuit order."""
        return self.gates[index]
