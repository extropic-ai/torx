"""Generator-defined gates for torx circuits."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from ._base import AbstractGeneratorGate


class PISING(AbstractGeneratorGate[list[int], Float[Array, "5"], tuple[int, int]]):
    r"""
    A probabilistic Ising interaction gate.

    Unlike the Boolean gates (PNOT, PCNOT, PSWAP, ...) which interpolate between
    identity and a logic operation, PISING is an **energy-based** gate: it drives
    two pbits toward the thermal equilibrium of a pairwise Ising interaction.
    Applying it repeatedly thermalizes the joint distribution toward the
    Boltzmann distribution $\pi_i \propto e^{-\beta E_i}$.

    !!! note "Simulator support"

        `PISING` is a generator-defined dense-matrix gate. Use it with
        [`StateVectorSimulator`][torx.psc.StateVectorSimulator]; it is not
        compatible with [`SampleSimulator`][torx.psc.SampleSimulator], which
        currently requires branch lookup-table gates.

    The Ising energy for two pbits $\sigma_1, \sigma_2 \in \{0, 1\}$ is:

    $$E(\sigma_1, \sigma_2) = -J \, s_1 s_2 - h_1 s_1 - h_2 s_2$$

    where $s_i = 2\sigma_i - 1 \in \{-1, +1\}$.

    The transition matrix $P = \exp(Q \cdot \Delta t)$ is the matrix exponential
    of a Markov generator $Q$ whose off-diagonal entries are Glauber
    single-spin-flip rates:

    $$Q_{ji} = \frac{1}{1 + e^{\beta \, \Delta E_{ij}}}$$

    Only single-spin-flip transitions are nonzero.

    **Parameters**:

    - `theta[0]` — $J$: coupling strength
      ($J > 0$ ferromagnetic, $J < 0$ antiferromagnetic)
    - `theta[1]` — $h_1$: external field on first pbit
    - `theta[2]` — $h_2$: external field on second pbit
    - `theta[3]` — $\beta$: inverse temperature
      ($\beta \to 0$: uniform, $\beta \to \infty$: ground state)
    - `theta[4]` — $\Delta t$: continuous-time step
      ($0$: identity, larger: closer to equilibrium)
    """

    sites: list[int]

    @property
    def dims(self) -> tuple[int, int]:  # type: ignore[override]
        return (2, 2)

    def init_params(self, key: Key[Array, ""]) -> Float[Array, "5"]:
        """Initial ``theta = [J, h1, h2, beta, dt]`` (zeros).

        See [`torx.psc.AbstractPGate.init_params`][].
        """
        return jnp.zeros(5)

    def J(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[0]

    def h1(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[1]

    def h2(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[2]

    def beta(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[3]

    def dt(self, theta: Float[Array, "5"]) -> Float[Array, ""]:
        return theta[4]

    @classmethod
    def _glauber_rate(cls, dE: Float[Array, ""], beta: Float[Array, ""]):
        """Glauber single-spin flip rate for energy change dE."""
        return 1.0 / (1.0 + jnp.exp(beta * dE))

    def _energies(self, theta: Float[Array, "5"]) -> Float[Array, "4"]:
        """Ising energies for each basis configuration.

        Maps {0,1} -> {-1,+1} via s = 2*sigma - 1.

        Returns energies for |00), |01), |10), |11).
        """
        J, h1, h2 = self.J(theta), self.h1(theta), self.h2(theta)
        return jnp.array(
            [
                -J * (-1) * (-1) - h1 * (-1) - h2 * (-1),  # |00): s=(-1,-1)
                -J * (-1) * (+1) - h1 * (-1) - h2 * (+1),  # |01): s=(-1,+1)
                -J * (+1) * (-1) - h1 * (+1) - h2 * (-1),  # |10): s=(+1,-1)
                -J * (+1) * (+1) - h1 * (+1) - h2 * (+1),  # |11): s=(+1,+1)
            ]
        )

    def get_generator(self, theta: Float[Array, "5"]) -> Float[Array, "4 4"]:
        r"""Build the 4x4 Glauber-dynamics Markov generator $Q$.

        $Q$ is a rate matrix with zero column sums and non-negative off-diagonal
        entries. Only single-spin-flip transitions are nonzero:
        $|00) \leftrightarrow |01)$,
        $|00) \leftrightarrow |10)$,
        $|01) \leftrightarrow |11)$,
        $|10) \leftrightarrow |11)$.

        The generator is independent of $\Delta t$.
        """
        E = self._energies(theta)
        beta = self.beta(theta)

        transitions = jnp.array([[0, 1], [0, 2], [1, 3], [2, 3]])

        Q = jnp.zeros((4, 4))
        for idx in range(4):
            i, j = transitions[idx, 0], transitions[idx, 1]
            dE_ij = E[j] - E[i]
            dE_ji = E[i] - E[j]
            Q = Q.at[j, i].add(self._glauber_rate(dE_ij, beta))
            Q = Q.at[i, j].add(self._glauber_rate(dE_ji, beta))

        Q = Q.at[jnp.diag_indices(4)].set(0.0)
        Q = Q - jnp.diag(Q.sum(axis=0))

        return Q


PISING.__init__.__doc__ = """See [`torx.psc.AbstractDiscreteGate.__init__`][]."""
