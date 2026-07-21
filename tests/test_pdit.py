"""Tests for pdit gates and mixed-dimension circuits."""

import unittest

import jax.numpy as jnp

from torx.psc import (
    DiscretePCircuit,
    PditShift,
    PditSWAP,
    PNOT,
    StateVectorSimulator,
)


class TestPditGates(unittest.TestCase):
    def test_pdit_shift_dims(self):
        gate = PditShift(sites=0, dims=3)
        self.assertEqual(gate.dims, (3,))

        gate = PditShift(sites=0, dims=(5,))
        self.assertEqual(gate.dims, (5,))

    def test_pdit_shift_identity(self):
        gate = PditShift(sites=0, dims=3)
        matrix = gate.get_matrix(jnp.atleast_1d(-jnp.inf))
        self.assertTrue(jnp.allclose(matrix, jnp.eye(3)))

    def test_pdit_shift_deterministic(self):
        gate = PditShift(sites=0, dims=3)
        matrix = gate.get_matrix(jnp.atleast_1d(jnp.inf))
        # Cyclic shift: |0) -> |1), |1) -> |2), |2) -> |0)
        expected = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(matrix, expected))

    def test_pdit_shift_probabilistic(self):
        gate = PditShift(sites=0, dims=3)
        matrix = gate.get_matrix(jnp.atleast_1d(0.0))
        expected = 0.5 * jnp.eye(3) + 0.5 * jnp.array(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32
        )
        self.assertTrue(jnp.allclose(matrix, expected))

    def test_pdit_swap_dims(self):
        gate = PditSWAP(sites=[0, 1], dims=3)
        self.assertEqual(gate.dims, (3, 3))

        gate = PditSWAP(sites=[0, 1], dims=(4, 4))
        self.assertEqual(gate.dims, (4, 4))

    def test_pdit_swap_identity(self):
        gate = PditSWAP(sites=[0, 1], dims=2)
        matrix = gate.get_matrix(jnp.atleast_1d(-jnp.inf))
        self.assertTrue(jnp.allclose(matrix, jnp.eye(4)))

    def test_pdit_swap_deterministic(self):
        gate = PditSWAP(sites=[0, 1], dims=2)
        matrix = gate.get_matrix(jnp.atleast_1d(jnp.inf))
        # SWAP: |00) -> |00), |01) -> |10), |10) -> |01), |11) -> |11)
        expected = jnp.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.float32
        )
        self.assertTrue(jnp.allclose(matrix, expected))


class TestMixedDimCircuit(unittest.TestCase):
    def test_single_pdit_circuit_dims(self):
        circuit = DiscretePCircuit([PditShift(sites=0, dims=3)])
        self.assertEqual(circuit.dims, (3,))

    def test_multi_pdit_circuit_dims(self):
        circuit = DiscretePCircuit(
            [
                PditShift(sites=0, dims=3),
                PditSWAP(sites=[1, 2], dims=4),
            ]
        )
        self.assertEqual(circuit.dims, (3, 4, 4))

    def test_mixed_binary_pdit_dims(self):
        circuit = DiscretePCircuit(
            [
                PNOT(0),
                PditShift(sites=1, dims=3),
            ]
        )
        self.assertEqual(circuit.dims, (2, 3))

    def test_pdit_density_identity(self):
        circuit = DiscretePCircuit([PditShift(sites=0, dims=3)])
        thetas = [jnp.atleast_1d(-jnp.inf)]
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit, thetas)

        x = jnp.array([1.0, 0.0, 0.0])
        result = sim.density(compiled, x)
        self.assertTrue(jnp.allclose(result, x))

        x = jnp.array([0.0, 1.0, 0.0])
        result = sim.density(compiled, x)
        self.assertTrue(jnp.allclose(result, x))

    def test_pdit_density_shift(self):
        circuit = DiscretePCircuit([PditShift(sites=0, dims=3)])
        thetas = [jnp.atleast_1d(jnp.inf)]
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit, thetas)

        # |0) -> |1)
        x = jnp.array([1.0, 0.0, 0.0])
        result = sim.density(compiled, x)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0, 1.0, 0.0])))

        # |1) -> |2)
        x = jnp.array([0.0, 1.0, 0.0])
        result = sim.density(compiled, x)
        self.assertTrue(jnp.allclose(result, jnp.array([0.0, 0.0, 1.0])))

        # |2) -> |0)
        x = jnp.array([0.0, 0.0, 1.0])
        result = sim.density(compiled, x)
        self.assertTrue(jnp.allclose(result, jnp.array([1.0, 0.0, 0.0])))

    def test_mixed_circuit_density(self):
        circuit = DiscretePCircuit(
            [
                PNOT(0),  # Deterministic flip on site 0
                PditShift(sites=1, dims=3),  # Identity on site 1
            ]
        )
        thetas = [
            jnp.atleast_1d(jnp.inf),  # Deterministic flip on site 0
            jnp.atleast_1d(-jnp.inf),  # Identity on site 1
        ]
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit, thetas)

        # 2 * 3 = 6
        x = jnp.zeros(6).at[0].set(1.0)
        result = sim.density(compiled, x)

        # After PNOT flip: (1*3 + 0 = 3)
        expected = jnp.zeros(6).at[3].set(1.0)
        self.assertTrue(jnp.allclose(result, expected))

    def test_pdit_swap_circuit(self):
        circuit = DiscretePCircuit([PditSWAP(sites=[0, 1], dims=3)])
        thetas = [jnp.atleast_1d(jnp.inf)]
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit, thetas)

        # State space: 3 * 3 = 9
        # |1,2) = index 1*3 + 2 = 5
        x = jnp.zeros(9).at[5].set(1.0)
        result = sim.density(compiled, x)

        # |2,1) = index 2*3 + 1 = 7
        expected = jnp.zeros(9).at[7].set(1.0)
        self.assertTrue(jnp.allclose(result, expected))


class TestDimensionMismatch(unittest.TestCase):
    def test_dimension_mismatch_error(self):
        with self.assertRaises(ValueError):
            DiscretePCircuit(
                [
                    PditShift(sites=0, dims=3),
                    PditShift(sites=0, dims=5),  # Different dim on same site
                ]
            )


if __name__ == "__main__":
    unittest.main()
