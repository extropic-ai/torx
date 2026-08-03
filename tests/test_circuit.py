"""Test the probabilistic circuit in `torx.psc`."""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp

from torx.psc import DiscretePCircuit, HybridPCircuit, MixtureGaussianGate, PCNOT, PNOT


class TestDiscretePCircuit(unittest.TestCase):
    def test_properties(self):
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        circuit = DiscretePCircuit(gates)

        self.assertEqual(circuit.gates, gates)
        self.assertEqual(circuit.num_pdits, 2)
        self.assertEqual(circuit.reps, 1)

    def test_iteration(self):
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        circuit = DiscretePCircuit(gates)

        for gate, cgate in zip(gates, circuit):
            self.assertEqual(gate, cgate)

    def test_len(self):
        gates1 = [PNOT(0)]
        circuit1 = DiscretePCircuit(gates1)
        self.assertEqual(len(circuit1), 1)

        gates2 = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        circuit2 = DiscretePCircuit(gates2)
        self.assertEqual(len(circuit2), 3)

    def test_indexing(self):
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        circuit = DiscretePCircuit(gates)

        for i, gate in enumerate(gates):
            self.assertEqual(circuit[i], gate)

    def test_add(self):
        gates1 = [PNOT(0)]
        gates2 = [PCNOT([0, 1]), PNOT(1)]

        circuit1 = DiscretePCircuit(gates1)
        circuit2 = DiscretePCircuit(gates2)

        circuit_sum = circuit1 + circuit2
        self.assertEqual(circuit_sum.gates, gates1 + gates2)

        circuit_sum: DiscretePCircuit = sum(  # pyright: ignore[reportAssignmentType]
            [circuit1, circuit2], start=DiscretePCircuit([])
        )
        self.assertEqual(circuit_sum.gates, gates1 + gates2)

    def test_sample_applies_reps(self):
        circuit = DiscretePCircuit([PNOT(0)], reps=2)

        sample = eqx.filter_jit(circuit.sample)(
            jax.random.key(0),
            {"in": jnp.array([0])},
            [jnp.array([jnp.inf])],
        )

        self.assertTrue(jnp.array_equal(sample, jnp.array([0])))


class TestHybridPCircuit(unittest.TestCase):
    def test_sample_applies_reps_with_discrete_feedback(self):
        circuit = HybridPCircuit(
            [
                PNOT(0),
                MixtureGaussianGate(
                    sites=([0], [0]),
                    dims=(1,),
                    num_components=2,
                ),
            ],
            reps=2,
        )
        params = [
            jnp.array([jnp.inf]),
            {
                "means": jnp.array([[1.0], [10.0]]),
                "log_vars": jnp.full((2, 1), -jnp.inf),
            },
        ]

        sample = eqx.filter_jit(circuit.sample)(
            jax.random.key(0),
            {"discrete": jnp.array([0]), "continuous": jnp.array([0.0])},
            params,
        )

        self.assertTrue(jnp.array_equal(sample["discrete"], jnp.array([0])))
        self.assertTrue(jnp.allclose(sample["continuous"], jnp.array([11.0])))
