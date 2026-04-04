"""Tests for hybrid gates and HybridSampleSimulator."""

import unittest

import jax
import jax.numpy as jnp

from torx import (
    AffineGaussianGate,
    GaussianNoiseGate,
    HybridPCircuit,
    HybridSampleSimulator,
    JumpDiffusionGate,
    MixtureGaussianGate,
    PNOT,
)


class TestGaussianNoiseGate(unittest.TestCase):
    def test_output_shape(self):
        gate = GaussianNoiseGate(
            theta=jnp.zeros(3),
            sites=[0],
            dims=(3,),
        )
        key = jax.random.key(0)
        result = gate.sample({"continuous": jnp.zeros(3)}, key)
        self.assertEqual(result.shape, (3,))

    def test_variance(self):
        # log_var = 0 -> var = 1
        gate = GaussianNoiseGate(
            theta=jnp.zeros(4),
            sites=[0],
            dims=(4,),
        )

        key = jax.random.key(42)
        keys = jax.random.split(key, 10000)
        samples = jax.vmap(lambda k: gate.sample({"continuous": jnp.zeros(4)}, k))(keys)

        self.assertTrue(jnp.allclose(jnp.mean(samples, axis=0), 0, atol=0.05))
        self.assertTrue(jnp.allclose(jnp.std(samples, axis=0), 1, atol=0.05))


class TestAffineGaussianGate(unittest.TestCase):
    def test_affine_transform(self):
        d = 3
        gate = AffineGaussianGate(
            theta={
                "A": jnp.eye(d),
                "b": jnp.zeros(d),
                "log_var": jnp.full(d, -100),
            },
            sites=[0],
            dims=(d,),
        )
        key = jax.random.key(0)
        x = jnp.array([1.0, 2.0, 3.0])
        result = gate.sample({"continuous": x}, key)
        self.assertTrue(jnp.allclose(result, x, atol=1e-5))

    def test_affine_with_bias(self):
        d = 2
        gate = AffineGaussianGate(
            theta={
                "A": jnp.eye(d),
                "b": jnp.array([1.0, -1.0]),
                "log_var": jnp.full(d, -100),
            },
            sites=[0],
            dims=(d,),
        )
        key = jax.random.key(0)
        x = jnp.array([0.0, 0.0])
        result = gate.sample({"continuous": x}, key)
        self.assertTrue(jnp.allclose(result, jnp.array([1.0, -1.0]), atol=1e-5))


class TestMixtureGaussianGate(unittest.TestCase):
    def test_mode_selection(self):
        gate = MixtureGaussianGate(
            theta={
                "means": jnp.array([[10.0], [-10.0]]),
                "log_vars": jnp.array([[-100.0], [-100.0]]),
            },
            sites=([0], [0]),  # (discrete_site, continuous_site)
            dims=(1,),
        )

        key = jax.random.key(0)

        result0 = gate.sample(
            {"discrete": jnp.array([0]), "continuous": jnp.zeros(1)}, key
        )
        self.assertTrue(jnp.allclose(result0, 10.0, atol=0.1))

        result1 = gate.sample(
            {"discrete": jnp.array([1]), "continuous": jnp.zeros(1)}, key
        )
        self.assertTrue(jnp.allclose(result1, -10.0, atol=0.1))


class TestJumpDiffusionGate(unittest.TestCase):
    def test_diffusion_only(self):
        gate = JumpDiffusionGate(
            theta={
                "diff_log_var": jnp.zeros(2),
                "jump_mean": jnp.array([100.0, 100.0]),
                "jump_log_var": jnp.zeros(2),
            },
            sites=([0], [0]),
            dims=(2,),
        )

        key = jax.random.key(42)
        keys = jax.random.split(key, 1000)
        # no jump
        samples = jax.vmap(
            lambda k: gate.sample(
                {"discrete": jnp.array([0]), "continuous": jnp.zeros(2)}, k
            )
        )(keys)

        self.assertTrue(jnp.allclose(jnp.mean(samples, axis=0), 0, atol=0.1))
        self.assertTrue(jnp.allclose(jnp.std(samples, axis=0), 1, atol=0.1))

    def test_jump_fires(self):
        gate = JumpDiffusionGate(
            theta={
                "diff_log_var": jnp.full(2, -100),
                "jump_mean": jnp.array([5.0, -5.0]),
                "jump_log_var": jnp.full(2, -100),
            },
            sites=([0], [0]),
            dims=(2,),
        )

        key = jax.random.key(42)

        result = gate.sample(
            {"discrete": jnp.array([1]), "continuous": jnp.zeros(2)}, key
        )

        self.assertTrue(jnp.allclose(result, jnp.array([5.0, -5.0]), atol=0.1))


class TestHybridPCircuit(unittest.TestCase):
    def test_pure_continuous_circuit(self):
        gate = GaussianNoiseGate(
            theta=jnp.zeros(3),
            sites=[0],
            dims=(3,),
        )
        circuit = HybridPCircuit([gate])
        self.assertEqual(circuit.num_discrete_sites, 0)
        self.assertEqual(circuit.num_continuous_sites, 1)
        self.assertEqual(circuit.continuous_state_dim, 3)

    def test_mixed_circuit(self):
        pnot = PNOT(jnp.array(0.0), sites=0)
        noise = GaussianNoiseGate(
            theta=jnp.zeros(2),
            sites=[0],
            dims=(2,),
        )
        circuit = HybridPCircuit([pnot, noise])
        self.assertEqual(circuit.num_discrete_sites, 1)
        self.assertEqual(circuit.num_continuous_sites, 1)
        self.assertEqual(len(circuit), 2)


class TestHybridSampleSimulator(unittest.TestCase):
    def test_pure_continuous_simulation(self):
        gate = GaussianNoiseGate(
            theta=jnp.zeros(3),
            sites=[0],
            dims=(3,),
        )
        circuit = HybridPCircuit([gate])
        sim = HybridSampleSimulator(num_samples=100)

        key = jax.random.key(42)
        initial_state = {
            "discrete": jnp.array([], dtype=jnp.int32),
            "continuous": jnp.zeros(3),
        }
        result = sim.sample(circuit, initial_state, key)

        self.assertEqual(result["discrete"].shape, (100, 0))
        self.assertEqual(result["continuous"].shape, (100, 3))

    def test_mixed_simulation(self):
        pnot = PNOT(jnp.array(0.0), sites=0)  # p=0.5
        noise = GaussianNoiseGate(
            theta=jnp.zeros(2),
            sites=[0],
            dims=(2,),
        )
        circuit = HybridPCircuit([pnot, noise])
        sim = HybridSampleSimulator(num_samples=1000)

        key = jax.random.key(42)
        initial_state = {"discrete": jnp.array([0]), "continuous": jnp.zeros(2)}
        result = sim.sample(circuit, initial_state, key)

        discrete_mean = jnp.mean(result["discrete"].astype(float))
        self.assertTrue(0.4 < discrete_mean < 0.6)

        self.assertTrue(
            jnp.allclose(jnp.mean(result["continuous"], axis=0), 0, atol=0.1)
        )
        self.assertTrue(
            jnp.allclose(jnp.std(result["continuous"], axis=0), 1, atol=0.1)
        )

    def test_expval_all(self):
        gate = GaussianNoiseGate(
            theta=jnp.zeros(2),
            sites=[0],
            dims=(2,),
        )
        circuit = HybridPCircuit([gate])
        sim = HybridSampleSimulator(num_samples=10000)

        key = jax.random.key(42)
        initial_state = {
            "discrete": jnp.array([], dtype=jnp.int32),
            "continuous": jnp.array([1.0, 2.0]),
        }
        expval = sim.expval_all(circuit, initial_state, key)

        self.assertTrue(
            jnp.allclose(expval["continuous"], initial_state["continuous"], atol=0.05)
        )
