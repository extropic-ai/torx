"""Tests for hybrid gates and HybridSampleSimulator."""

import unittest

import jax
import jax.numpy as jnp

from torx.psc import (
    AbstractSampler,
    AffineGaussianGate,
    GaussianNoiseGate,
    HybridPCircuit,
    HybridSampleSimulator,
    JaxPRNGSampler,
    JumpDiffusionGate,
    MixtureGaussianGate,
    PNOT,
)


class TestGaussianNoiseGate(unittest.TestCase):
    def test_output_shape(self):
        gate = GaussianNoiseGate(sites=[0], dims=(3,))
        theta = jnp.zeros(3)
        key = jax.random.key(0)
        result = gate.sample(key, {"continuous": jnp.zeros(3)}, theta)
        self.assertEqual(result.shape, (3,))

    def test_variance(self):
        # log_var = 0 -> var = 1
        gate = GaussianNoiseGate(sites=[0], dims=(4,))
        theta = jnp.zeros(4)

        key = jax.random.key(42)
        keys = jax.random.split(key, 10000)
        samples = jax.vmap(
            lambda k: gate.sample(k, {"continuous": jnp.zeros(4)}, theta)
        )(keys)

        self.assertTrue(jnp.allclose(jnp.mean(samples, axis=0), 0, atol=0.05))
        self.assertTrue(jnp.allclose(jnp.std(samples, axis=0), 1, atol=0.05))


class TestAffineGaussianGate(unittest.TestCase):
    def test_affine_transform(self):
        d = 3
        gate = AffineGaussianGate(sites=[0], dims=(d,))
        theta = {
            "A": jnp.eye(d),
            "b": jnp.zeros(d),
            "log_var": jnp.full(d, -100.0),
        }
        key = jax.random.key(0)
        x = jnp.array([1.0, 2.0, 3.0])
        result = gate.sample(key, {"continuous": x}, theta)
        self.assertTrue(jnp.allclose(result, x, atol=1e-5))

    def test_affine_with_bias(self):
        d = 2
        gate = AffineGaussianGate(sites=[0], dims=(d,))
        theta = {
            "A": jnp.eye(d),
            "b": jnp.array([1.0, -1.0]),
            "log_var": jnp.full(d, -100.0),
        }
        key = jax.random.key(0)
        x = jnp.array([0.0, 0.0])
        result = gate.sample(key, {"continuous": x}, theta)
        self.assertTrue(jnp.allclose(result, jnp.array([1.0, -1.0]), atol=1e-5))


class TestMixtureGaussianGate(unittest.TestCase):
    def test_mode_selection(self):
        gate = MixtureGaussianGate(
            sites=([0], [0]),  # (discrete_site, continuous_site)
            dims=(1,),
            num_components=2,
        )
        theta = {
            "means": jnp.array([[10.0], [-10.0]]),
            "log_vars": jnp.array([[-100.0], [-100.0]]),
        }

        key = jax.random.key(0)

        result0 = gate.sample(
            key, {"discrete": jnp.array([0]), "continuous": jnp.zeros(1)}, theta
        )
        self.assertTrue(jnp.allclose(result0, 10.0, atol=0.1))

        result1 = gate.sample(
            key, {"discrete": jnp.array([1]), "continuous": jnp.zeros(1)}, theta
        )
        self.assertTrue(jnp.allclose(result1, -10.0, atol=0.1))


class TestJumpDiffusionGate(unittest.TestCase):
    def test_diffusion_only(self):
        gate = JumpDiffusionGate(sites=([0], [0]), dims=(2,))
        theta = {
            "diff_log_var": jnp.zeros(2),
            "jump_mean": jnp.array([100.0, 100.0]),
            "jump_log_var": jnp.zeros(2),
        }

        key = jax.random.key(42)
        keys = jax.random.split(key, 1000)
        # no jump
        samples = jax.vmap(
            lambda k: gate.sample(
                k, {"discrete": jnp.array([0]), "continuous": jnp.zeros(2)}, theta
            )
        )(keys)

        self.assertTrue(jnp.allclose(jnp.mean(samples, axis=0), 0, atol=0.1))
        self.assertTrue(jnp.allclose(jnp.std(samples, axis=0), 1, atol=0.1))

    def test_jump_fires(self):
        gate = JumpDiffusionGate(sites=([0], [0]), dims=(2,))
        theta = {
            "diff_log_var": jnp.full(2, -100.0),
            "jump_mean": jnp.array([5.0, -5.0]),
            "jump_log_var": jnp.full(2, -100.0),
        }

        key = jax.random.key(42)

        result = gate.sample(
            key, {"discrete": jnp.array([1]), "continuous": jnp.zeros(2)}, theta
        )

        self.assertTrue(jnp.allclose(result, jnp.array([5.0, -5.0]), atol=0.1))


class TestHybridPCircuit(unittest.TestCase):
    def test_pure_continuous_circuit(self):
        gate = GaussianNoiseGate(sites=[0], dims=(3,))
        circuit = HybridPCircuit([gate])
        self.assertEqual(circuit.num_discrete_sites, 0)
        self.assertEqual(circuit.num_continuous_sites, 1)
        self.assertEqual(circuit.continuous_state_dim, 3)

    def test_mixed_circuit(self):
        pnot = PNOT(sites=0)
        noise = GaussianNoiseGate(sites=[0], dims=(2,))
        circuit = HybridPCircuit([pnot, noise])
        self.assertEqual(circuit.num_discrete_sites, 1)
        self.assertEqual(circuit.num_continuous_sites, 1)
        self.assertEqual(len(circuit), 2)

    def test_discrete_only_circuit_has_no_continuous_starts(self):
        # A discrete-only circuit must report no continuous starts/offsets, not
        # a phantom (0,) from the cumulative-offset helper.
        circuit = HybridPCircuit([PNOT(sites=0)])
        self.assertEqual(circuit.num_continuous_sites, 0)
        self.assertEqual(circuit._continuous_starts, ())
        self.assertEqual(circuit.continuous_offsets, ())

    def test_controlled_continuous_gate_infers_discrete_control_site(self):
        circuit = HybridPCircuit(
            [MixtureGaussianGate(sites=([0], [0]), dims=(1,), num_components=2)]
        )

        self.assertEqual(circuit.num_discrete_sites, 1)
        self.assertEqual(circuit.num_continuous_sites, 1)


class TestHybridSampleSimulator(unittest.TestCase):
    def test_pure_continuous_simulation(self):
        gate = GaussianNoiseGate(sites=[0], dims=(3,))
        circuit = HybridPCircuit([gate])
        thetas = circuit.init_params(jax.random.key(0))
        sim = HybridSampleSimulator(num_samples=100)
        compiled = sim.build_circuit(circuit, thetas)

        key = jax.random.key(42)
        initial_state = {
            "discrete": jnp.array([], dtype=jnp.int32),
            "continuous": jnp.zeros(3),
        }
        result = sim.sample(compiled, initial_state, key)

        self.assertEqual(result["discrete"].shape, (100, 0))
        self.assertEqual(result["continuous"].shape, (100, 3))

    def test_mixed_simulation(self):
        pnot = PNOT(sites=0)  # p=0.5 at the default theta
        noise = GaussianNoiseGate(sites=[0], dims=(2,))
        circuit = HybridPCircuit([pnot, noise])
        thetas = circuit.init_params(jax.random.key(0))
        sim = HybridSampleSimulator(num_samples=1000)
        compiled = sim.build_circuit(circuit, thetas)

        key = jax.random.key(42)
        initial_state = {"discrete": jnp.array([0]), "continuous": jnp.zeros(2)}
        result = sim.sample(compiled, initial_state, key)

        discrete_mean = jnp.mean(result["discrete"].astype(float))
        self.assertTrue(0.4 < discrete_mean < 0.6)

        self.assertTrue(
            jnp.allclose(jnp.mean(result["continuous"], axis=0), 0, atol=0.1)
        )
        self.assertTrue(
            jnp.allclose(jnp.std(result["continuous"], axis=0), 1, atol=0.1)
        )

    def test_expval_all(self):
        gate = GaussianNoiseGate(sites=[0], dims=(2,))
        circuit = HybridPCircuit([gate])
        thetas = circuit.init_params(jax.random.key(0))
        sim = HybridSampleSimulator(num_samples=10000)
        compiled = sim.build_circuit(circuit, thetas)

        key = jax.random.key(42)
        initial_state = {
            "discrete": jnp.array([], dtype=jnp.int32),
            "continuous": jnp.array([1.0, 2.0]),
        }
        expval = sim.expval_all(compiled, initial_state, key)

        self.assertTrue(
            jnp.allclose(expval["continuous"], initial_state["continuous"], atol=0.05)
        )

    def test_expval_maps_continuous_site_to_slice(self):
        class _ZeroSampler(AbstractSampler):
            def bernoulli(self, key, p, shape=None):
                return jnp.zeros(shape if shape is not None else (), dtype=jnp.int32)

            def categorical(self, key, logits, axis=-1, shape=None):
                return jnp.zeros(shape if shape is not None else (), dtype=jnp.int32)

            def normal(self, key, shape=(), dtype=jnp.float32):
                return jnp.zeros(shape, dtype=dtype)

        gate = GaussianNoiseGate(sites=[0, 1], dims=(2, 1))
        circuit = HybridPCircuit([gate])
        sim = HybridSampleSimulator(num_samples=5, sampler=_ZeroSampler())
        compiled = sim.build_circuit(circuit, [jnp.zeros(3)])
        initial_state = {
            "discrete": jnp.array([], dtype=jnp.int32),
            "continuous": jnp.array([1.0, 2.0, 3.0]),
        }

        expval = sim.expval(
            compiled,
            initial_state,
            site=1,
            key=jax.random.key(0),
            site_type="continuous",
        )

        self.assertTrue(jnp.allclose(expval, jnp.array([3.0])))

    def test_custom_sampler_drives_continuous_noise(self):
        class _ZeroSampler(AbstractSampler):
            def bernoulli(self, key, p, shape=None):
                return jnp.zeros(shape if shape is not None else (), dtype=jnp.int32)

            def categorical(self, key, logits, axis=-1, shape=None):
                return jnp.zeros(shape if shape is not None else (), dtype=jnp.int32)

            def normal(self, key, shape=(), dtype=jnp.float32):
                return jnp.zeros(shape, dtype=dtype)

        gate = GaussianNoiseGate(sites=[0], dims=(2,))
        circuit = HybridPCircuit([gate])
        thetas = circuit.init_params(jax.random.key(0))
        sim = HybridSampleSimulator(num_samples=5, sampler=_ZeroSampler())
        compiled = sim.build_circuit(circuit, thetas)
        state = {
            "discrete": jnp.array([], dtype=jnp.int32),
            "continuous": jnp.array([1.0, 2.0]),
        }
        result = sim.sample(compiled, state, jax.random.key(0))
        self.assertTrue(
            jnp.allclose(result["continuous"], jnp.tile(state["continuous"], (5, 1)))
        )

    def test_fp32_circuit_samples_under_x64(self):
        # A consistent fp32 circuit sampled under x64 used to crash: the sampler
        # drew fp64 noise that strict promotion rejected against fp32 params.
        # getattr: jax types its dynamic config attrs differently per platform, so a
        # direct access needs a pyright ignore in some envs and flags it as
        # unnecessary in others; getattr is quiet everywhere.
        prev = getattr(jax.config, "jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:
            circuit = HybridPCircuit([GaussianNoiseGate(sites=[0], dims=(2,))])
            sim = HybridSampleSimulator(num_samples=8)
            compiled = sim.build_circuit(circuit, [jnp.zeros(2, dtype=jnp.float32)])
            state = {
                "discrete": jnp.array([], dtype=jnp.int32),
                "continuous": jnp.zeros(2, dtype=jnp.float32),
            }

            result = sim.sample(compiled, state, jax.random.key(1))

            self.assertEqual(result["continuous"].dtype, jnp.float32)
        finally:
            jax.config.update("jax_enable_x64", prev)


class TestJaxPRNGSampler(unittest.TestCase):
    def test_normal_default_dtype_honors_x64(self):
        # The default `float` dtype follows jax_enable_x64 (float32 off, float64
        # on), matching the affine-Gaussian simulator's dtype policy.
        key = jax.random.key(0)
        self.assertEqual(JaxPRNGSampler().normal(key, (2,)).dtype, jnp.float32)

        # getattr: jax types its dynamic config attrs differently per platform, so a
        # direct access needs a pyright ignore in some envs and flags it as
        # unnecessary in others; getattr is quiet everywhere.
        prev = getattr(jax.config, "jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:
            self.assertEqual(JaxPRNGSampler().normal(key, (2,)).dtype, jnp.float64)
        finally:
            jax.config.update("jax_enable_x64", prev)
