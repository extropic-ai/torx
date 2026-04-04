"""Abstract base classes for simulators and compiled circuits."""

from abc import abstractmethod
from typing import Generic, Type, TYPE_CHECKING, TypeVar

from typing_extensions import Self

from .._circuit import AbstractPCircuit, DiscretePCircuit
from .._custom_meta import AbstractStrictModule

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


_CircuitType = TypeVar("_CircuitType", bound=AbstractPCircuit)


class AbstractCompiledPCircuit(AbstractStrictModule, Generic[_CircuitType]):
    """Abstract parent class for probabilistic circuits built for a specific backend."""

    @classmethod
    @abstractmethod
    def from_pcircuit(cls, circuit: _CircuitType) -> Self:
        """
        Compile the given probabilistic circuit.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compile

        **Returns:**

        The compiled circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def to_pcircuit(self, structure: _CircuitType) -> _CircuitType:
        """
        Create a new circuit with the same structure as the given circuit.

        Creates a circuit with parameters from the compiled circuit.

        **Arguments:**

        - `structure`: The probabilistic circuit to reparametrize

        **Returns:**

        A probabilistic circuit with the parameters of the compiled circuit.
        """
        raise NotImplementedError


class AbstractSimulator(AbstractStrictModule):
    """Abstract parent class for probabilistic circuit simulators."""

    circuit_backend: AbstractClassVar[Type[AbstractCompiledPCircuit]]

    @abstractmethod
    def expval(self, *args, **kwargs):
        """
        Compute the expectation value of the given index after circuit execution.
        """
        raise NotImplementedError

    @abstractmethod
    def expval_all(self, *args, **kwargs):
        """Compute the expectation value of all indices after circuit execution."""
        raise NotImplementedError

    @classmethod
    def build_circuit(cls, circuit: DiscretePCircuit) -> AbstractCompiledPCircuit:
        """
        Build the circuit for this simulator.

        **Arguments:**

        - `circuit`: The circuit to build

        **Returns:**

        The built circuit.
        """
        return cls.circuit_backend.from_pcircuit(circuit)
