"""Tests for exact affine-Gaussian simulation."""

import unittest
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from torx.psc import (
    AbstractPGate,
    AffineGaussianGate,
    AffineGaussianSimulator,
    Diffuse,
    Displace,
    GaussianNoiseGate,
    HybridPCircuit,
    Mix,
    MixtureGaussianGate,
    PNOT,
    Scale,
)

Spec = tuple[AbstractPGate, PyTree]


def _build(*specs: Spec, reps: int = 1):
    """Assemble a circuit + aligned thetas from (gate, theta) specs."""
    gates = [gate for gate, _ in specs]
    thetas = [theta for _, theta in specs]
    circuit = HybridPCircuit(gates, reps=reps)
    sim = AffineGaussianSimulator()
    return sim, sim.build_circuit(circuit, thetas)


def _zeros(compiled) -> Float[Array, " continuous_dim"]:
    """Zero initial continuous state sized to the compiled circuit."""
    dim = compiled.site_offsets[-1][2] if compiled.site_offsets else 0
    return jnp.zeros(dim)


def _shift(
    value,
    sites: int | Sequence[int] = 0,
    dims: tuple[int, ...] | None = None,
) -> Spec:
    b = jnp.asarray(value).reshape(-1)
    dims = (b.size,) if dims is None else dims
    gate = AffineGaussianGate(sites=sites, dims=dims)
    theta = {
        "A": jnp.eye(b.size, dtype=b.dtype),
        "b": b,
        "log_var": jnp.full(b.size, -jnp.inf, dtype=b.dtype),
    }
    return gate, theta


def _noise(
    variance,
    sites: int | Sequence[int] = 0,
    dims: tuple[int, ...] | None = None,
) -> Spec:
    variance = jnp.asarray(variance).reshape(-1)
    dims = (variance.size,) if dims is None else dims
    gate = GaussianNoiseGate(sites=sites, dims=dims)
    return gate, jnp.log(variance)


def _linear_gaussian_specs() -> tuple[Spec, Spec]:
    noise = (
        GaussianNoiseGate(sites=[0], dims=(1,)),
        jnp.log(jnp.array([4.0])),
    )
    affine = (
        AffineGaussianGate(sites=[0, 1], dims=(1, 1)),
        {
            "A": jnp.array([[1.0, 0.0], [2.0, 0.0]]),
            "b": jnp.array([0.0, 1.0]),
            "log_var": jnp.array([-jnp.inf, jnp.log(0.25)]),
        },
    )
    return noise, affine


class TestAffineGaussianSimulator(unittest.TestCase):
    def test_affine_gaussian_sequence_matches_moment_formulas(self):
        sim, compiled = _build(
            (Diffuse(sites=0, dims=(1,)), jnp.log(jnp.array([2.0]))),
            (Displace(sites=0, dims=(1,)), jnp.array([1.0])),
            (Scale(sites=0, dims=(1,)), jnp.array([jnp.log(2.0)])),
            (Mix(sites=[0, 1], dims=(1, 1)), jnp.asarray(jnp.pi / 2.0)),
        )

        distribution = sim.propagate(compiled, _zeros(compiled))

        self.assertTrue(
            jnp.allclose(distribution.mean, jnp.array([0.0, 2.0]), atol=1e-6)
        )
        self.assertTrue(
            jnp.allclose(
                distribution.covariance,
                jnp.array([[0.0, 0.0], [0.0, 8.0]]),
                atol=1e-6,
            )
        )
        site_mean, site_covariance = distribution.site_moments(1)
        self.assertTrue(jnp.allclose(site_mean, distribution.mean[1:2]))
        self.assertTrue(
            jnp.allclose(site_covariance, distribution.covariance[1:2, 1:2])
        )

    def test_linear_gaussian_moments_conditioning_and_expvals(self):
        sim, compiled = _build(*_linear_gaussian_specs())
        initial = _zeros(compiled)

        distribution = sim.propagate(compiled, initial)

        self.assertTrue(jnp.allclose(distribution.mean, jnp.array([0.0, 1.0])))
        self.assertTrue(
            jnp.allclose(
                distribution.covariance,
                jnp.array([[4.0, 8.0], [8.0, 16.25]]),
            )
        )
        self.assertEqual(distribution.sites, (0, 1))
        self.assertEqual(distribution.observed_sites, ())

        posterior = sim.condition(
            compiled,
            {1: jnp.array([3.0])},
            initial_continuous=initial,
            query_sites=[0],
        )

        self.assertTrue(jnp.allclose(posterior.mean, jnp.array([8.0 / 16.25 * 2.0])))
        self.assertTrue(
            jnp.allclose(posterior.covariance, jnp.array([[4.0 - 8.0 * 8.0 / 16.25]]))
        )
        self.assertEqual(posterior.sites, (0,))
        self.assertEqual(posterior.observed_sites, (1,))
        self.assertTrue(
            jnp.allclose(sim.expval(compiled, initial, 1), jnp.array([1.0]))
        )
        self.assertTrue(
            jnp.allclose(sim.expval_all(compiled, initial), jnp.array([0.0, 1.0]))
        )

    def test_conditioning_packs_multidimensional_query_sites_in_requested_order(self):
        dims = (2, 1, 1)
        sim, compiled = _build(
            _noise(jnp.array([1.0, 2.0, 3.0, 4.0]), sites=[0, 1, 2], dims=dims),
            _shift(jnp.array([10.0, 20.0, 30.0, 40.0]), sites=[0, 1, 2], dims=dims),
        )

        posterior = sim.condition(
            compiled,
            {0: jnp.array([11.0, 18.0])},
            initial_continuous=_zeros(compiled),
            query_sites=[2, 1],
        )

        self.assertTrue(jnp.allclose(posterior.mean, jnp.array([40.0, 30.0])))
        self.assertTrue(
            jnp.allclose(posterior.covariance, jnp.diag(jnp.array([4.0, 3.0])))
        )
        self.assertEqual(posterior.sites, (2, 1))
        self.assertEqual(posterior.observed_sites, (0,))

    def test_conditioning_degenerate_cases(self):
        sim, compiled = _build(
            _noise(jnp.array([1.0, 2.0]), sites=0),
            _shift(jnp.array([1.0, -1.0]), sites=0),
        )

        prior_slice = sim.condition(
            compiled, {}, initial_continuous=_zeros(compiled), query_sites=[0]
        )

        self.assertTrue(jnp.allclose(prior_slice.mean, jnp.array([1.0, -1.0])))
        self.assertTrue(
            jnp.allclose(prior_slice.covariance, jnp.diag(jnp.array([1.0, 2.0])))
        )
        self.assertEqual(prior_slice.observed_sites, ())

        empty = sim.condition(
            compiled, {0: jnp.array([1.0, -1.0])}, initial_continuous=_zeros(compiled)
        )
        self.assertEqual(empty.mean.shape, (0,))
        self.assertEqual(empty.covariance.shape, (0, 0))
        self.assertEqual(empty.sites, ())
        self.assertEqual(empty.observed_sites, (0,))

    def test_reps_apply_expected_number_of_times(self):
        cases = [
            (
                "zero",
                _shift(2.0, sites=0),
                0,
                jnp.array([5.0]),
                jnp.array([5.0]),
                jnp.zeros((1, 1)),
            ),
            (
                "multidim",
                _shift(jnp.array([1.0, -2.0]), sites=0),
                3,
                jnp.zeros(2),
                jnp.array([3.0, -6.0]),
                jnp.zeros((2, 2)),
            ),
        ]

        for name, spec, reps, initial, expected_mean, expected_covariance in cases:
            with self.subTest(name=name):
                sim, compiled = _build(spec, reps=reps)
                distribution = sim.propagate(compiled, initial)
                self.assertTrue(jnp.allclose(distribution.mean, expected_mean))
                self.assertTrue(
                    jnp.allclose(distribution.covariance, expected_covariance)
                )

    def test_rejects_unsupported_gates(self):
        gate = MixtureGaussianGate(sites=([0], [0]), dims=(1,), num_components=2)
        mixture_theta = {
            "means": jnp.array([[0.0], [1.0]]),
            "log_vars": jnp.zeros((2, 1)),
        }
        cases = [
            ("PNOT", (PNOT(sites=0), jnp.zeros(1)), _shift(1.0, sites=0)),
            (
                "MixtureGaussianGate",
                (gate, mixture_theta),
                (PNOT(sites=0), jnp.zeros(1)),
            ),
        ]

        for message, *specs in cases:
            with self.subTest(message=message):
                sim, compiled = _build(*specs)
                with self.assertRaisesRegex(ValueError, message):
                    sim.propagate(compiled, _zeros(compiled))

    def test_condition_requires_valid_jitter_and_pd_observed_covariance(self):
        sim, compiled = _build(_shift(jnp.array([1.0, 2.0]), sites=[0, 1], dims=(1, 1)))

        with self.assertRaisesRegex(RuntimeError, "nonnegative"):
            sim.condition(
                compiled,
                {0: jnp.array([1.0])},
                initial_continuous=_zeros(compiled),
                jitter=-1.0,
            ).mean.block_until_ready()

        with self.assertRaisesRegex(RuntimeError, "not positive definite"):
            sim.condition(
                compiled, {0: jnp.array([1.0])}, initial_continuous=_zeros(compiled)
            ).mean.block_until_ready()

        posterior = sim.condition(
            compiled,
            {0: jnp.array([1.0])},
            initial_continuous=_zeros(compiled),
            jitter=1e-6,
        )
        self.assertTrue(jnp.allclose(posterior.mean, jnp.array([2.0])))

    def test_condition_rejects_invalid_site_arguments(self):
        sim, compiled = _build(
            _noise(jnp.array([1.0, 2.0, 3.0]), sites=[0, 1], dims=(2, 1))
        )
        initial = _zeros(compiled)
        cases = [
            ("overlap", "disjoint", {0: jnp.array([1.0, 2.0])}, [0, 1]),
            ("short observation", "site 0 has 1 value", {0: jnp.array([1.0])}, [1]),
            ("unknown observation", "not continuous sites", {5: jnp.array([1.0])}, [1]),
            ("unknown query", "not continuous sites", {0: jnp.array([1.0, 2.0])}, [7]),
        ]

        for name, message, observations, query_sites in cases:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                sim.condition(
                    compiled,
                    observations,
                    initial_continuous=initial,
                    query_sites=query_sites,
                )

    def test_rejects_theta_gate_count_mismatch(self):
        # zip-truncation used to silently drop a gate/theta on a length mismatch
        # and return wrong moments; build_circuit must reject it.
        sim = AffineGaussianSimulator()
        circuit = HybridPCircuit(
            [
                GaussianNoiseGate(sites=0, dims=(1,)),
                GaussianNoiseGate(sites=0, dims=(1,)),
            ],
            reps=1,
        )

        with self.assertRaisesRegex(ValueError, "thetas aligned"):
            sim.build_circuit(circuit, [jnp.log(jnp.array([4.0]))])

    def test_grad_through_conditioning(self):
        sim, compiled = _build(*_linear_gaussian_specs())

        def loss(obs):
            posterior = sim.condition(
                compiled, {1: obs}, initial_continuous=_zeros(compiled), query_sites=[0]
            )
            return jnp.sum(posterior.mean**2)

        # mu_post(y) = (cov_qo / var_o) * (y - mu_o) = (8/16.25)*(y - 1)
        # d/dy [mu_post^2] at y=3 = 2 * mu_post * (8/16.25) = 4 * (8/16.25)^2
        g = eqx.filter_jit(eqx.filter_grad(loss))(jnp.array([3.0]))
        c = 8.0 / 16.25
        expected = jnp.array([4.0 * c * c])
        self.assertTrue(jnp.allclose(g, expected))

    def test_jit_with_traced_jitter_affects_posterior(self):
        sim, compiled = _build(*_linear_gaussian_specs())

        @eqx.filter_jit
        def run(jitter):
            return sim.condition(
                compiled,
                {1: jnp.array([3.0])},
                initial_continuous=_zeros(compiled),
                query_sites=[0],
                jitter=jitter,
            ).mean

        no_jitter = run(jnp.array(0.0))
        with_jitter = run(jnp.array(1.0))

        self.assertTrue(jnp.allclose(no_jitter, jnp.array([8.0 / 16.25 * 2.0])))
        self.assertTrue(jnp.allclose(with_jitter, jnp.array([8.0 / 17.25 * 2.0])))
        self.assertFalse(jnp.allclose(no_jitter, with_jitter))

    def test_empty_circuit_returns_empty_state(self):
        sim = AffineGaussianSimulator()
        compiled = sim.build_circuit(HybridPCircuit([]), [])

        prior = sim.propagate(compiled, _zeros(compiled))
        self.assertEqual(prior.mean.shape, (0,))
        self.assertEqual(prior.covariance.shape, (0, 0))
        self.assertEqual(prior.sites, ())

        posterior = sim.condition(compiled, {}, initial_continuous=_zeros(compiled))
        self.assertEqual(posterior.mean.shape, (0,))
        self.assertEqual(posterior.covariance.shape, (0, 0))
        self.assertEqual(posterior.observed_sites, ())


if __name__ == "__main__":
    unittest.main()
