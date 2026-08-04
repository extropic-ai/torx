"""Abstract base classes for simulators and compiled circuits."""

from abc import abstractmethod
from typing import Generic, Type, TYPE_CHECKING, TypeVar

from ihoop.eqx import AbstractStrictModule
from jaxtyping import Array, PyTree
from typing_extensions import Self

from .._circuit import AbstractPCircuit

if TYPE_CHECKING:
    from typing import ClassVar as AbstractClassVar
else:
    from equinox import AbstractClassVar


_CircuitType = TypeVar("_CircuitType", bound=AbstractPCircuit)


class AbstractCompiledPCircuit(AbstractStrictModule, Generic[_CircuitType]):
    """Abstract parent class for probabilistic circuits built for a specific backend."""

    @classmethod
    @abstractmethod
    def from_pcircuit(cls, circuit: _CircuitType, thetas: PyTree[Array]) -> Self:
        """
        Compile the given probabilistic circuit with the given parameters.

        **Arguments:**

        - `circuit`: The probabilistic circuit to compile
        - `thetas`: The parameters to bake into the compiled circuit.

        **Returns:**

        The compiled circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def to_pcircuit(self, structure: _CircuitType) -> _CircuitType:
        """Return a circuit structure reconstructed by this compiled backend."""
        raise NotImplementedError


_CompiledType = TypeVar("_CompiledType", bound=AbstractCompiledPCircuit)


class AbstractSimulator(AbstractStrictModule, Generic[_CircuitType, _CompiledType]):
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

    @abstractmethod
    def build_circuit(
        self, circuit: _CircuitType, thetas: PyTree[Array]
    ) -> _CompiledType:
        """
        Build the circuit for this simulator with the given parameters.

        **Arguments:**

        - `circuit`: The circuit to build
        - `thetas`: The parameters to bake into the compiled circuit.

        **Returns:**

        The built circuit.
        """
        raise NotImplementedError
