"""Sample based simulators."""

from typing import ClassVar, Literal, Type

from jaxtyping import Array, Float, Int, Key

from .._custom_types import BitString
from ._sampled_compile import CompiledConditionalSampleData, CompiledSamplePCircuit
from ._sampled_filter import sample_expval_all_param_shift_filter
from ._sampled_forward import sample_circuit
from ._sampled_param_shift import (
    sample_expval_all_param_shift_inf,
    sample_expval_all_param_shift_single,
)
from .base import AbstractCompiledPCircuit, AbstractSimulator

_DiffMethod = Literal["param_shift_inf", "param_shift_single", "param_shift_filter"]


__all__ = [
    "CompiledSamplePCircuit",
    "CompiledConditionalSampleData",
    "SampleSimulator",
    "sample_circuit",
    "sample_expval_all_param_shift_filter",
    "sample_expval_all_param_shift_inf",
    "sample_expval_all_param_shift_single",
]


class SampleSimulator(AbstractSimulator):
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

    circuit_backend: ClassVar[Type[AbstractCompiledPCircuit]] = CompiledSamplePCircuit

    def __init__(
        self, diff_method: _DiffMethod = "param_shift_inf", num_samples: int = 1
    ):
        """
        Initialize the sample simulator.

        **Arguments:**

        - `diff_method`: method used for differentiating circuit parameters
        - `num_samples`: number of samples used to estimate expectation values
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
        """
        Estimate the expectation value of the given pbit after circuit execution.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial computational basis state of the circuit
        - `pbit`: The index of the pbit to estimate the final expectation value of
        - `key`: The random key to use to obtain samples

        **Returns:**

        The expectation value of the given pbit after circuit execution.
        """
        return self.expval_all(circuit, x, key)[pbit]

    def expval_all(
        self, circuit: CompiledSamplePCircuit, x: BitString, key: Key[Array, ""]
    ) -> Float[Array, " num_pbits"]:
        """
        Estimate the expectation value of all pbits after circuit execution.

        **Arguments:**

        - `circuit`: The probabilistic circuit to execute
        - `x`: The initial computational basis state of the circuit
        - `key`: The random key to use to obtain samples

        **Returns:**

        An array containing the expectation values of all the pbits.
        """
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
