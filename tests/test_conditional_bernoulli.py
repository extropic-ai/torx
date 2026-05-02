import unittest

import equinox as eqx
import jax
import jax.numpy as jnp

import torx


class TestPConditionalBernoulliLayer(unittest.TestCase):
    def test_validates_shapes(self):
        with self.assertRaisesRegex(ValueError, "one entry per target"):
            torx.PConditionalBernoulliLayer(
                jnp.zeros(1),
                [0, 1],
                [[2], [2]],
                jnp.zeros((2, 1)),
            )

        with self.assertRaisesRegex(ValueError, "rectangular"):
            torx.PConditionalBernoulliLayer(
                jnp.zeros(2),
                [0, 1],
                [[2], [2, 3]],
                jnp.zeros((2, 2)),
            )

    def test_matrix_is_stochastic_and_exact(self):
        gate = torx.PConditionalBernoulliLayer(
            jnp.array([-0.5]),
            [2],
            [[0, 1]],
            jnp.array([[2.0, -1.0]]),
        )
        matrix = gate.get_matrix()
        self.assertEqual(matrix.shape, (8, 8))
        self.assertTrue(jnp.allclose(jnp.sum(matrix, axis=0), 1.0))

        circuit = torx.DiscretePCircuit([gate])
        sim = torx.StateVectorSimulator()
        compiled = sim.build_circuit(circuit)
        initial = jnp.zeros(8).at[4].set(1.0)
        expected = jax.nn.sigmoid(jnp.array(1.5))
        self.assertTrue(jnp.allclose(matrix[5, 4], expected))
        self.assertTrue(jnp.allclose(sim.expval(compiled, initial, 2), expected))


class TestConditionalSampleSimulator(unittest.TestCase):
    def test_high_fan_in_does_not_materialize_branch_table(self):
        control_sites = [list(range(10))]
        gate = torx.PConditionalBernoulliLayer(
            jnp.array([20.0]),
            [10],
            control_sites,
            jnp.zeros((1, 10)),
        )
        circuit = torx.DiscretePCircuit([gate])
        compiled = torx.SampleSimulator(num_samples=8).build_circuit(circuit)

        self.assertTrue(compiled.has_conditional)
        self.assertLess(compiled.branch_ops.shape[2], 2 ** len(control_sites[0]))

        initial = jnp.zeros(11, dtype=jnp.int32)
        samples = torx.SampleSimulator(num_samples=8).sample(
            compiled,
            initial,
            jax.random.key(0),
        )
        self.assertTrue(jnp.all(samples[:, 10] == 1))

    def test_mixed_branch_and_conditional_sampling_is_jittable(self):
        conditional = torx.PConditionalBernoulliLayer(
            jnp.array([-20.0]),
            [2],
            [[0, 1]],
            jnp.array([[40.0, 0.0]]),
        )
        circuit = torx.DiscretePCircuit([torx.PNOT(jnp.inf, 0), conditional])
        sim = torx.SampleSimulator(num_samples=4)
        compiled = sim.build_circuit(circuit)
        initial = jnp.array([0, 1, 0], dtype=jnp.int32)

        @eqx.filter_jit
        def run(compiled_circuit):
            return sim.sample(compiled_circuit, initial, jax.random.key(1))

        samples = run(compiled)
        self.assertEqual(samples.shape, (4, 3))
        self.assertTrue(jnp.all(samples[:, 0] == 1))
        self.assertTrue(jnp.all(samples[:, 2] == 1))

    def test_gradients_raise_clear_error(self):
        gate = torx.PConditionalBernoulliLayer(
            jnp.array([0.25]),
            [1],
            [[0]],
            jnp.array([[0.5]]),
        )
        compiled = torx.SampleSimulator(num_samples=16).build_circuit(
            torx.DiscretePCircuit([gate])
        )
        initial = jnp.array([1, 0], dtype=jnp.int32)
        sim = torx.SampleSimulator(num_samples=16)

        def loss(compiled_circuit):
            return jnp.sum(sim.expval_all(compiled_circuit, initial, jax.random.key(2)))

        with self.assertRaisesRegex(NotImplementedError, "PConditionalBernoulliLayer"):
            eqx.filter_grad(loss)(compiled)
