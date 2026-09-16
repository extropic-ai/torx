"""Pdit probabilistic gates."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ._base import AbstractMultiPditGate, AbstractSinglePditGate


class PditShift(AbstractSinglePditGate):
    r"""
    A probabilistic cyclic shift gate for a single k-dimensional pdit.

    This gate shifts the state cyclically: i -> (i+1) mod k with probability
    $p = \sigma(\theta)$, and does nothing with probability $1-p$.

    The transition matrix for dimension k is:
    $(1 - p) I + p C$ where $C$ is the cyclic permutation matrix.
    """

    sites: int
    dims: tuple[int, ...] = eqx.field(
        converter=lambda d: (d,) if isinstance(d, int) else tuple(d)
    )

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 d 1"]:
        """
        branches[0] is identity: i -> i.
        branches[1] is cyclic shift: i -> (i+1) mod k.
        """
        d = self.dims[0]
        identity = jnp.array([[i] for i in range(d)])
        shift = jnp.array([[(i + 1) % d] for i in range(d)])
        return jnp.stack([identity, shift])

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "d d"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        d = self.dims[0]
        p = self.prob(theta)
        identity = jnp.eye(d, dtype=p.dtype)
        cyclic = jnp.roll(identity, shift=1, axis=0)
        return (1 - p) * identity + p * cyclic


class PditSWAP(AbstractMultiPditGate):
    r"""
    A probabilistic SWAP gate for two k-dimensional pdits.

    This gate swaps the values of two pdits with probability $p = \sigma(\theta)$,
    and does nothing with probability $1-p$. Both pdits must have the same dimension.

    For dimension k, the state space is $k^2$ and the SWAP permutes basis states
    $|i,j) \to |j,i)$.
    """

    sites: list[int]
    dims: tuple[int, ...] = eqx.field(
        converter=lambda d: (d, d) if isinstance(d, int) else tuple(d)
    )

    @property
    def num_branches(self) -> int:
        return 2

    @property
    def branches(self) -> Int[Array, "2 d2 2"]:
        """
        branches[0] is identity: |i,j) -> |i,j).
        branches[1] is SWAP: |i,j) -> |j,i).
        """
        d = self.dims[0]
        identity = jnp.array([[i, j] for i in range(d) for j in range(d)])
        swap = jnp.array([[j, i] for i in range(d) for j in range(d)])
        return jnp.stack([identity, swap])

    def get_matrix(self, theta: Float[Array, "1"]) -> Float[Array, "d2 d2"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        d = self.dims[0]
        d2 = d * d
        p = self.prob(theta)

        # Build SWAP matrix: |i,j> -> |j,i>
        swap = jnp.zeros((d2, d2), dtype=p.dtype)
        for i in range(d):
            for j in range(d):
                src = i * d + j
                dst = j * d + i
                swap = swap.at[dst, src].set(1.0)

        identity = jnp.eye(d2, dtype=p.dtype)
        return (1 - p) * identity + p * swap


class PditCycle(AbstractSinglePditGate):
    r"""
    A probabilistic 3-branch cyclic gate for a single d-dimensional pdit.

    This gate implements a random walk on a ring: with some probability stay
    in place, shift forward, or shift backward. Useful for modeling diffusion
    or random walks on discrete rings.

    This gate has 3 branches:
    - Branch 0: Identity (i -> i)
    - Branch 1: Forward cycle (i -> (i+1) mod d)
    - Branch 2: Backward cycle (i -> (i-1) mod d)

    The probabilities are determined by softmax([0, theta[0], theta[1]]).

    The transition matrix is:
    $p_0 I + p_1 C_{forward} + p_2 C_{backward}$

    where $C_{forward}$ is the forward cyclic permutation and $C_{backward}$
    is the backward cyclic permutation.

    """

    sites: int
    dims: tuple[int, ...] = eqx.field(
        converter=lambda d: (d,) if isinstance(d, int) else tuple(d)
    )

    @property
    def num_branches(self) -> int:
        return 3

    @property
    def branches(self) -> Int[Array, "3 d 1"]:
        """
        branches[0] is identity: i -> i.
        branches[1] is forward cycle: i -> (i+1) mod d.
        branches[2] is backward cycle: i -> (i-1) mod d.
        """
        d = self.dims[0]
        identity = jnp.array([[i] for i in range(d)])
        forward = jnp.array([[(i + 1) % d] for i in range(d)])
        backward = jnp.array([[(i - 1) % d] for i in range(d)])
        return jnp.stack([identity, forward, backward])

    def get_matrix(self, theta: Float[Array, "2"]) -> Float[Array, "d d"]:
        """See [`torx.psc.AbstractDiscreteGate.get_matrix`][] for documentation."""
        d = self.dims[0]
        p = self.probs(theta)  # [p_identity, p_forward, p_backward]
        identity = jnp.eye(d, dtype=p.dtype)
        forward = jnp.roll(identity, shift=1, axis=0)
        backward = jnp.roll(identity, shift=-1, axis=0)
        return p[0] * identity + p[1] * forward + p[2] * backward


PditShift.__init__.__doc__ = """See [`torx.psc.AbstractSinglePditGate.__init__`][]."""
PditSWAP.__init__.__doc__ = """See [`torx.psc.AbstractMultiPditGate.__init__`][]."""
PditCycle.__init__.__doc__ = """See [`torx.psc.AbstractSinglePditGate.__init__`][]."""
