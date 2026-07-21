"""Test the PISING gate."""

import unittest

import jax.numpy as jnp
from jax.scipy.linalg import expm

from torx.psc import (
    DiscretePCircuit,
    PISING,
    PNOT,
    SampleSimulator,
    StateVectorSimulator,
)


class TestPISINGProperties(unittest.TestCase):

    def test_attributes(self):
        sites = [0, 1]
        gate = PISING(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.dims, (2, 2))

    def test_different_sites(self):
        theta = jnp.array([1.0, 0.0, 0.0, 1.0, 1.0])
        gate = PISING([2, 5])

        self.assertEqual(gate.sites, [2, 5])
        self.assertEqual(gate.dims, (2, 2))
        P = gate.get_matrix(theta)
        self.assertEqual(P.shape, (4, 4))


class TestPISINGGenerator(unittest.TestCase):

    def _make_gate(self, J=1.0, h1=0.0, h2=0.0, beta=1.0, dt=1.0):
        theta = jnp.array([J, h1, h2, beta, dt])
        return PISING([0, 1]), theta

    def test_generator_shape(self):
        gate, theta = self._make_gate()
        Q = gate.get_generator(theta)
        self.assertEqual(Q.shape, (4, 4))

    def test_generator_columns_sum_to_zero(self):
        """Markov generator must have zero column sums."""
        for J in [-1.0, 0.0, 1.0, 2.0]:
            for h1 in [-0.5, 0.0, 0.5]:
                for h2 in [-0.5, 0.0, 0.5]:
                    gate, theta = self._make_gate(J=J, h1=h1, h2=h2, beta=1.5, dt=1.0)
                    Q = gate.get_generator(theta)
                    col_sums = Q.sum(axis=0)
                    self.assertTrue(
                        jnp.allclose(col_sums, 0.0, atol=1e-6),
                        f"Column sums not zero for J={J}, h1={h1}, h2={h2}: {col_sums}",
                    )

    def test_generator_nonnegative_offdiagonal(self):
        """Off-diagonal entries of a Markov generator must be non-negative."""
        for J in [-1.0, 0.0, 1.0]:
            for h1 in [-1.0, 0.0, 1.0]:
                gate, theta = self._make_gate(J=J, h1=h1, h2=0.5, beta=2.0, dt=1.0)
                Q = gate.get_generator(theta)
                offdiag = Q - jnp.diag(jnp.diag(Q))
                self.assertTrue(
                    jnp.all(offdiag >= -1e-10),
                    f"Negative off-diagonal for J={J}, h1={h1}",
                )

    def test_generator_only_single_spin_flips(self):
        """Only single-spin-flip transitions should be nonzero.

        |00) <-> |01), |00) <-> |10), |01) <-> |11), |10) <-> |11).
        Multi-spin flips |00) <-> |11) and |01) <-> |10) must be zero.
        """
        gate, theta = self._make_gate(J=1.5, h1=0.3, h2=-0.7, beta=2.0, dt=1.0)
        Q = gate.get_generator(theta)

        # |00)=0, |01)=1, |10)=2, |11)=3
        # Multi-spin flip positions: (0,3), (3,0), (1,2), (2,1)
        self.assertTrue(jnp.isclose(Q[3, 0], 0.0))
        self.assertTrue(jnp.isclose(Q[0, 3], 0.0))
        self.assertTrue(jnp.isclose(Q[2, 1], 0.0))
        self.assertTrue(jnp.isclose(Q[1, 2], 0.0))

    def test_generator_independent_of_dt(self):
        """Generator should be a unit-rate matrix independent of dt."""
        gate1, theta1 = self._make_gate(J=1.0, h1=0.5, h2=-0.3, beta=1.0, dt=1.0)
        gate2, theta2 = self._make_gate(J=1.0, h1=0.5, h2=-0.3, beta=1.0, dt=2.0)
        Q1 = gate1.get_generator(theta1)
        Q2 = gate2.get_generator(theta2)
        self.assertTrue(jnp.allclose(Q1, Q2, atol=1e-6))


class TestPISINGMatrix(unittest.TestCase):

    def _make_gate(self, J=1.0, h1=0.0, h2=0.0, beta=1.0, dt=1.0):
        theta = jnp.array([J, h1, h2, beta, dt])
        return PISING([0, 1]), theta

    def test_matrix_shape(self):
        gate, theta = self._make_gate()
        P = gate.get_matrix(theta)
        self.assertEqual(P.shape, (4, 4))

    def test_matrix_is_stochastic(self):
        test_params = [
            dict(J=1.0, h1=0.0, h2=0.0, beta=1.0, dt=1.0),
            dict(J=-1.0, h1=0.5, h2=-0.5, beta=2.0, dt=0.5),
            dict(J=0.0, h1=1.0, h2=1.0, beta=0.1, dt=3.0),
            dict(J=2.0, h1=-1.0, h2=0.3, beta=5.0, dt=0.1),
            dict(J=0.5, h1=0.0, h2=0.0, beta=10.0, dt=1.0),
        ]
        for params in test_params:
            with self.subTest(**params):
                gate, theta = self._make_gate(**params)
                P = gate.get_matrix(theta)

                # Non-negative
                self.assertTrue(
                    jnp.all(P >= -1e-10),
                    f"Negative entries in P for {params}: min={P.min()}",
                )

                # Column-stochastic
                col_sums = P.sum(axis=0)
                self.assertTrue(
                    jnp.allclose(col_sums, 1.0, atol=1e-6),
                    f"Columns don't sum to 1 for {params}: {col_sums}",
                )

    def test_zero_coupling_factorizes(self):
        """With J=0, the two spins are independent.

        The 4x4 matrix should be the Kronecker product of two 2x2 single-spin
        Glauber matrices.
        """
        h1, h2, beta, dt = 0.5, -0.3, 1.5, 1.0
        gate, theta = self._make_gate(J=0.0, h1=h1, h2=h2, beta=beta, dt=dt)
        P = gate.get_matrix(theta)

        # Build single-spin Glauber propagators independently
        # For spin with field h: E(0) = +h (s=-1), E(1) = -h (s=+1)
        # rate 0->1: 1 / (1 + exp(beta * (E1 - E0))) = 1 / (1 + exp(beta * -2h))
        # rate 1->0: 1 / (1 + exp(beta * (E0 - E1))) = 1 / (1 + exp(beta * 2h))
        def single_spin_propagator(h):
            r01 = 1.0 / (1.0 + jnp.exp(beta * (-2 * h)))
            r10 = 1.0 / (1.0 + jnp.exp(beta * (2 * h)))
            Q = jnp.array([[-r01, r10], [r01, -r10]])

            return expm(Q * dt)

        P1 = single_spin_propagator(h1)
        P2 = single_spin_propagator(h2)
        P_expected = jnp.kron(P1, P2)

        self.assertTrue(
            jnp.allclose(P, P_expected, atol=1e-5),
            f"J=0 matrix doesn't factorize:\nP=\n{P}\nP1⊗P2=\n{P_expected}",
        )


class TestPISINGSimulator(unittest.TestCase):

    def test_single_gate_circuit(self):
        theta = jnp.array([1.0, 0.0, 0.0, 1.0, 1.0])
        gate = PISING([0, 1])
        circuit = DiscretePCircuit([gate])
        thetas = [theta]

        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit, thetas)
        initial = jnp.array([1.0, 0.0, 0.0, 0.0])
        result = sim.density(compiled, initial)

        self.assertEqual(result.shape, (4,))
        self.assertTrue(jnp.allclose(result.sum(), 1.0, atol=1e-6))
        self.assertTrue(jnp.all(result >= -1e-10))

    def test_matches_manual_matmul(self):
        theta = jnp.array([1.0, 0.5, -0.3, 2.0, 1.0])
        gate = PISING([0, 1])
        P = gate.get_matrix(theta)

        circuit = DiscretePCircuit([gate])
        thetas = [theta]
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit, thetas)

        initial = jnp.array([0.25, 0.25, 0.25, 0.25])
        result = sim.density(compiled, initial)
        expected = P @ initial

        self.assertTrue(jnp.allclose(result, expected, atol=1e-6))

    def test_mixed_circuit_with_pnot(self):
        ising_theta = jnp.array([1.0, 0.0, 0.0, 1.0, 1.0])
        circuit = DiscretePCircuit(
            [
                PNOT(0),
                PISING([0, 1]),
                PNOT(1),
            ]
        )
        thetas = [
            jnp.atleast_1d(1.0),
            ising_theta,
            jnp.atleast_1d(0.5),
        ]

        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit, thetas)
        initial = jnp.array([1.0, 0.0, 0.0, 0.0])
        result = sim.density(compiled, initial)

        self.assertEqual(result.shape, (4,))
        self.assertTrue(jnp.allclose(result.sum(), 1.0, atol=1e-6))
        self.assertTrue(jnp.all(result >= -1e-10))

    def test_sample_simulator_error_mentions_statevector(self):
        circuit = DiscretePCircuit([PISING([0, 1])])

        with self.assertRaisesRegex(ValueError, "PISING.*StateVectorSimulator"):
            SampleSimulator().build_circuit(circuit, [jnp.zeros(5)])


if __name__ == "__main__":
    unittest.main()
