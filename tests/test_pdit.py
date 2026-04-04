"""Tests for pdit gates and mixed-dimension circuits."""

import unittest

import jax.numpy as jnp

from torx import (
    DiscretePCircuit,
    PditShift,
    PditSWAP,
    PNOT,
    StateVectorSimulator,
)


class TestPditGates(unittest.TestCase):
    def test_pdit_shift_dims(self):
        gate = PditShift(0.0, sites=0, dims=3)
        self.assertEqual(gate.dims, (3,))

        gate = PditShift(0.0, sites=0, dims=(5,))
        self.assertEqual(gate.dims, (5,))

    def test_pdit_shift_identity(self):
        gate = PditShift(-jnp.inf, sites=0, dims=3)
        matrix = gate.get_matrix()
        self.assertTrue(jnp.allclose(matrix, jnp.eye(3)))

    def test_pdit_shift_deterministic(self):
        gate = PditShift(jnp.inf, sites=0, dims=3)
        matrix = gate.get_matrix()
        # Cyclic shift: |0) -> |1), |1) -> |2), |2) -> |0)
        expected = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(matrix, expected))

    def test_pdit_shift_probabilistic(self):
        gate = PditShift(0.0, sites=0, dims=3)
        matrix = gate.get_matrix()
        expected = 0.5 * jnp.eye(3) + 0.5 * jnp.array(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32
        )
        self.assertTrue(jnp.allclose(matrix, expected))

    def test_pdit_swap_dims(self):
        gate = PditSWAP(0.0, sites=[0, 1], dims=3)
        self.assertEqual(gate.dims, (3, 3))

        gate = PditSWAP(0.0, sites=[0, 1], dims=(4, 4))
        self.assertEqual(gate.dims, (4, 4))

    def test_pdit_swap_identity(self):
        gate = PditSWAP(-jnp.inf, sites=[0, 1], dims=2)
        matrix = gate.get_matrix()
        self.assertTrue(jnp.allclose(matrix, jnp.eye(4)))

    def test_pdit_swap_deterministic(self):
        gate = PditSWAP(jnp.inf, sites=[0, 1], dims=2)
        matrix = gate.get_matrix()
        # SWAP: |00) -> |00), |01) -> |10), |10) -> |01), |11) -> |11)
        expected = jnp.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.float32
        )
        self.assertTrue(jnp.allclose(matrix, expected))


class TestMixedDimCircuit(unittest.TestCase):
    def test_single_pdit_circuit_dims(self):
        circuit = DiscretePCircuit([PditShift(0.0, sites=0, dims=3)])
        self.assertEqual(circuit.dims, (3,))

    def test_multi_pdit_circuit_dims(self):
        circuit = DiscretePCircuit(
            [
                PditShift(0.0, sites=0, dims=3),
                PditSWAP(0.0, sites=[1, 2], dims=4),
            ]
        )
        self.assertEqual(circuit.dims, (3, 4, 4))

    def test_mixed_binary_pdit_dims(self):
        circuit = DiscretePCircuit(
            [
                PNOT(0.0, 0),
                PditShift(0.0, sites=1, dims=3),
            ]
        )
        self.assertEqual(circuit.dims, (2, 3))

    def test_pdit_density_identity(self):
        circuit = DiscretePCircuit([PditShift(-jnp.inf, sites=0, dims=3)])
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit)

        x = jnp.array([1.0, 0.0, 0.0])
        result = sim.density(compiled, x)
        self.assertTrue(jnp.allclose(result, x))

        x = jnp.array([0.0, 1.0, 0.0])
        result = sim.density(compiled, x)
        self.assertTrue(jnp.allclose(result, x))

    def test_pdit_density_shift(self):
        circuit = DiscretePCircuit([PditShift(jnp.inf, sites=0, dims=3)])
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit)

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
                PNOT(jnp.inf, 0),  # Deterministic flip on site 0
                PditShift(-jnp.inf, sites=1, dims=3),  # Identity on site 1
            ]
        )
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit)

        # 2 * 3 = 6
        x = jnp.zeros(6).at[0].set(1.0)
        result = sim.density(compiled, x)

        # After PNOT flip: (1*3 + 0 = 3)
        expected = jnp.zeros(6).at[3].set(1.0)
        self.assertTrue(jnp.allclose(result, expected))

    def test_pdit_swap_circuit(self):
        circuit = DiscretePCircuit([PditSWAP(jnp.inf, sites=[0, 1], dims=3)])
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit)

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
                    PditShift(0.0, sites=0, dims=3),
                    PditShift(0.0, sites=0, dims=5),  # Different dim on same site
                ]
            )


if __name__ == "__main__":
    unittest.main()
