"""Binary probabilistic gates."""

import itertools

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._base import AbstractMultiBinaryPGate, AbstractSingleBinaryPGate


class PNOT(AbstractSingleBinaryPGate):
    r"""
    A probabilistic NOT gate.

    This gate flips the value of a given pbit with probability $p = \sigma(\theta)$
    where $\theta$ is the gate parameter, and otherwise does nothing.

    The transition matrix is:

    $$\begin{pmatrix} 1-p & p \\ p & 1-p \end{pmatrix}$$
    """

    theta: Float[Array, ""]
    sites: int

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 2 1"]:
        """
        branches[0] is identity: 0 -> 0 and 1 -> 1.
        branches[1] is bit flip: 0 -> 1 and 1 -> 0.
        """
        return jnp.array(
            [
                [[0], [1]],  # identity
                [[1], [0]],  # flip
            ]
        )

    def get_matrix(self) -> Float[Array, "2 2"]:
        """See `AbstractPGate.get_matrix` for documentation."""
        return jnp.array([[1 - self.prob, self.prob], [self.prob, 1 - self.prob]])


class PCNOT(AbstractMultiBinaryPGate):
    r"""
    A probabilistic CNOT gate.

    This gate performs a controlled-NOT operation on two given pbits with
    probability $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and
    otherwise does nothing. A controlled-NOT operation flips the value of
    the second pbit if the first pbit has the value `1`, and does nothing
    if the first pbit has the value `0`.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1-p & p \\ 0 & 0 & p & 1-p
    \end{pmatrix}$$
    """

    _control_indices = (0,)

    theta: Float[Array, ""]
    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00 -> 00, 01 -> 01, 10 -> 10, 11 -> 11.
        branches[1] is CNOT: 00->00, 01->01, 10->11, 11->10.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [0, 1], [1, 1], [1, 0]],  # CNOT
            ]
        )

    def get_matrix(self) -> Float[Array, "4 4"]:
        """See `AbstractPGate.get_matrix` for documentation."""
        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 - self.prob, self.prob],
                [0, 0, self.prob, 1 - self.prob],
            ]
        )


class PSWAP(AbstractMultiBinaryPGate):
    r"""
    A probabilistic SWAP gate.

    This gate swaps the values of two given pbits with probability
    $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and otherwise
    does nothing.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 \\ 0 & 1-p & p & 0 \\ 0 & p & 1-p & 0 \\ 0 & 0 & 0 & 1
    \end{pmatrix}$$
    """

    theta: Float[Array, ""]
    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00 -> 00, 01 -> 01, 10 -> 10, 11 -> 11.
        branches[1] is swap: 00->00, 01->10, 10->01, 11->11.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [1, 0], [0, 1], [1, 1]],  # swap
            ]
        )

    def get_matrix(self) -> Float[Array, "4 4"]:
        """See `AbstractPGate.get_matrix` for documentation."""
        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1 - self.prob, self.prob, 0],
                [0, self.prob, 1 - self.prob, 0],
                [0, 0, 0, 1],
            ]
        )


class PJUMP(AbstractMultiBinaryPGate):
    r"""
    A probabilistic JUMP gate.

    This gate performs a "jump" operation on two given pbits with probability
    $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and otherwise does
    nothing. A jump operation moves probability from $|10)$ to $|01)$
    if the first pbit has the value `1`.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 \\ 0 & 1 & p & 0 \\ 0 & 0 & 1-p & 0 \\ 0 & 0 & 0 & 1
    \end{pmatrix}$$
    """

    theta: Float[Array, ""]
    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00 -> 00, 01 -> 01, 10 -> 10, 11 -> 11.
        branches[1] is JUMP: 00->00, 01->01, 10->01, 11->11.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [0, 1], [0, 1], [1, 1]],  # JUMP
            ]
        )

    def get_matrix(self) -> Float[Array, "4 4"]:
        """See `AbstractPGate.get_matrix` for documentation."""
        return jnp.array(
            [[1, 0, 0, 0], [0, 1, self.prob, 0], [0, 0, 1 - self.prob, 0], [0, 0, 0, 1]]
        )


class PMultiCNOT(AbstractMultiBinaryPGate):
    r"""
    A probabilistic multi-controlled-NOT gate.

    This gate performs a multi-controlled-NOT operation on any number of given pbits
    with probability $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and
    otherwise does nothing. A multi-controlled-NOT operation flips the value
    of the last pbit if all the other pbits have the value `1`, and does
    nothing otherwise.

    The transition matrix is identity except for the last two rows/columns which
    swap with probability $p$.
    """

    _control_indices = "all_but_last"
    _draw_label = "PMCNOT"

    theta: Float[Array, ""]
    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 2**d d"]:
        """
        branches[0] is identity.
        branches[1] is multi-controlled-NOT: flips last bit when all others are 1.
        """
        identity = jnp.array(list(itertools.product([0, 1], repeat=len(self.sites))))
        op = identity.at[-2, -1].set(1).at[-1, -1].set(0)
        return jnp.stack([identity, op])

    def get_matrix(self) -> Float[Array, "d d"]:
        """See `AbstractPGate.get_matrix` for documentation."""
        return (
            jnp.eye(2 ** len(self.sites))
            .at[-2, -2]
            .set(1 - self.prob)
            .at[-2, -1]
            .set(self.prob)
            .at[-1, -2]
            .set(self.prob)
            .at[-1, -1]
            .set(1 - self.prob)
        )


class PDEMUX(AbstractMultiBinaryPGate):
    r"""
    A probabilistic demultiplexor.

    This gate has one input pbit and one "work" pbit. It either (i) copies the
    input to the work pbit and sets the input to `0`, or (ii) does nothing to
    the input and sets the work pbit to `0`. The probability of the former
    occurring is $p = \sigma(\theta)$ where $\theta$ is the gate parameter.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 1 & 0 & 0 \\ 0 & 0 & p & p \\ 0 & 0 & 1-p & 1-p \\ 0 & 0 & 0 & 0
    \end{pmatrix}$$
    """

    _control_indices = (0,)

    theta: Float[Array, ""]
    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] resets second pbit: 00 -> 00, 01 -> 00, 10 -> 10, 11 -> 10.
        branches[1] copies and resets: 00->00, 01->00, 10->01, 11->01.
        """
        return jnp.array(
            [
                [[0, 0], [0, 0], [1, 0], [1, 0]],  # resets second pbit
                [[0, 0], [0, 0], [0, 1], [0, 1]],  # copies and resets
            ]
        )

    def get_matrix(self) -> Float[Array, "4 4"]:
        """See `AbstractPGateApply.get_matrix` for documentation."""
        return jnp.array(
            [
                [1, 1, 0, 0],
                [0, 0, self.prob, self.prob],
                [0, 0, 1 - self.prob, 1 - self.prob],
                [0, 0, 0, 0],
            ]
        )


class POR(AbstractMultiBinaryPGate):
    r"""
    A probabilistic OR gate.

    This gate performs a modified OR operation on two given pbits with probability
    $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and otherwise does
    nothing. A modified OR operation sets the first pbit to the OR of the two input
    pbits, and sets the second pbit to `0`.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 \\ 0 & 1-p & 0 & 0 \\ 0 & p & 1 & p \\ 0 & 0 & 0 & 1-p
    \end{pmatrix}$$
    """

    _control_indices = "all_but_last"

    theta: Float[Array, ""]
    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00 -> 00, 01 -> 01, 10 -> 10, 11 -> 11.
        branches[1] is OR: 00->00, 01->10, 10->10, 11->10.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [1, 0], [1, 0], [1, 0]],  # OR
            ]
        )

    def get_matrix(self) -> Float[Array, "4 4"]:
        """See `AbstractPGate.get_matrix` for documentation."""
        # 00 -> 00, 01 -> 10, 10 -> 10, 11 -> 10

        return jnp.array(
            [
                [1, 0, 0, 0],
                [0, 1 - self.prob, 0, 0],
                [0, self.prob, 1, self.prob],
                [0, 0, 0, 1 - self.prob],
            ]
        )


class PReset(AbstractSingleBinaryPGate):
    r"""
    A probabilistic reset gate.

    This gate sets the given pbit to `0` with probability $p = \sigma(\theta)$ where
    $\theta$ is the gate parameter, and otherwise does nothing.

    The transition matrix is:

    $$\begin{pmatrix} 1 & p \\ 0 & 1-p \end{pmatrix}$$
    """

    theta: Float[Array, ""]
    sites: int

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 2 1"]:
        """
        branches[0] is identity: 0 -> 0 and 1 -> 1.
        branches[1] resets to 0: 0 -> 0 and 1 -> 0.
        """
        return jnp.array(
            [
                [[0], [1]],  # identity
                [[0], [0]],  # reset
            ]
        )

    def get_matrix(self) -> Float[Array, "2 2"]:
        """See `AbstractPGate.get_matrix` for documentation."""
        return jnp.array([[1, self.prob], [0, 1 - self.prob]])


class PCopy(AbstractMultiBinaryPGate):
    r"""
    A probabilistic copy gate.

    This gate copies the first pbit to the second pbit with probability
    $p = \sigma(\theta)$ where $\theta$ is the gate parameter, and otherwise
    does nothing.

    Transition matrix:

    $$\begin{pmatrix}
    1 & p & 0 & 0 \\ 0 & 1-p & 0 & 0 \\ 0 & 0 & 1-p & 0 \\ 0 & 0 & p & 1
    \end{pmatrix}$$
    """

    _control_indices = (0,)

    theta: Float[Array, ""]
    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 4 2"]:
        """
        branches[0] is identity: 00->00, 01->01, 10->10, 11->11.
        branches[1] copies first to second: 00->00, 01->00, 10->11, 11->11.
        """
        return jnp.array(
            [
                [[0, 0], [0, 1], [1, 0], [1, 1]],  # identity
                [[0, 0], [0, 0], [1, 1], [1, 1]],  # copy
            ]
        )

    def get_matrix(self) -> Float[Array, "4 4"]:
        """See `AbstractPGate.get_matrix` for documentation."""
        return jnp.array(
            [
                [1, self.prob, 0, 0],
                [0, 1 - self.prob, 0, 0],
                [0, 0, 1 - self.prob, 0],
                [0, 0, self.prob, 1],
            ]
        )


class PCSWAP(AbstractMultiBinaryPGate):
    r"""
    A probabilistic controlled-SWAP gate (Fredkin gate).

    This gate swaps the values of two target pbits only when the control pbit
    is in state $|1)$, with probability $p = \sigma(\theta)$ where $\theta$
    is the gate parameter. If the control is $|0)$ or the gate is not applied
    (with probability $1-p$), nothing happens.

    The first site is the control, the second and third sites are the targets
    to be swapped.

    Transition matrix:

    $$\begin{pmatrix}
    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1-p & p & 0 \\
    0 & 0 & 0 & 0 & 0 & p & 1-p & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
    \end{pmatrix}$$
    """

    _control_indices = (0,)

    theta: Float[Array, ""]
    sites: list[int]

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 8 3"]:
        """
        branches[0] is identity.
        branches[1] is controlled-SWAP: swap targets if control=1.
        """
        # Input states in order: 000, 001, 010, 011, 100, 101, 110, 111
        identity = jnp.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        # Output when control=1: swap the last two bits
        cswap = jnp.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
            ]
        )
        return jnp.stack([identity, cswap])

    def get_matrix(self) -> Float[Array, "8 8"]:
        """See `AbstractPGate.get_matrix` for documentation."""
        p = self.prob
        return jnp.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1 - p, p, 0],
                [0, 0, 0, 0, 0, p, 1 - p, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )


PNOT.__init__.__doc__ = """See [`torx.AbstractSingleBinaryPGate.__init__`][]."""
PReset.__init__.__doc__ = """See [`torx.AbstractSingleBinaryPGate.__init__`][]."""
PCNOT.__init__.__doc__ = """See [`torx.AbstractMultiBinaryPGate.__init__`][]."""
PSWAP.__init__.__doc__ = """See [`torx.AbstractMultiBinaryPGate.__init__`][]."""
PJUMP.__init__.__doc__ = """See [`torx.AbstractMultiBinaryPGate.__init__`][]."""
PDEMUX.__init__.__doc__ = """See [`torx.AbstractMultiBinaryPGate.__init__`][]."""
POR.__init__.__doc__ = """See [`torx.AbstractMultiBinaryPGate.__init__`][]."""
PCopy.__init__.__doc__ = """See [`torx.AbstractMultiBinaryPGate.__init__`][]."""
PMultiCNOT.__init__.__doc__ = """See [`torx.AbstractMultiBinaryPGate.__init__`][]."""
PCSWAP.__init__.__doc__ = """See [`torx.AbstractMultiBinaryPGate.__init__`][]."""
