"""Tests for the `affine_parameters` channel the affine-Gaussian simulator reads."""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
from parameterized import parameterized

from torx.psc import (
    AbstractAffineGaussianGate,
    AffineGaussianGate,
    Diffuse,
    Displace,
    GaussianNoiseGate,
    Mix,
    Scale,
)


class TestAffineGaussianGates(unittest.TestCase):
    def test_named_gates_share_affine_abstraction(self):
        gates = [
            Displace(sites=0, dims=(1,)),
            Scale(sites=0, dims=(1,)),
            Mix(sites=[0, 1], dims=(1, 1)),
            Diffuse(sites=0, dims=(1,)),
        ]

        key = jax.random.key(0)
        for gate in gates:
            self.assertIsInstance(gate, AbstractAffineGaussianGate)
            theta = gate.init_params(key)
            A, b, log_var = gate.affine_parameters(theta)
            local_dim = sum(gate.dims)
            self.assertEqual(A.shape, (local_dim, local_dim))
            self.assertEqual(b.shape, (local_dim,))
            self.assertEqual(log_var.shape, (local_dim,))

    @parameterized.expand(
        [
            (
                "displace",
                Displace(sites=0, dims=(2,)),
                jnp.array([1.0, -2.0]),
                jnp.eye(2),
                jnp.array([1.0, -2.0]),
                jnp.full(2, -jnp.inf),
            ),
            (
                "scale",
                Scale(sites=0, dims=(2,)),
                jnp.log(jnp.array([2.0, 0.5])),
                jnp.diag(jnp.array([2.0, 0.5])),
                jnp.zeros(2),
                jnp.full(2, -jnp.inf),
            ),
            (
                "mix",
                Mix(sites=[0, 1], dims=(1, 1)),
                jnp.asarray(jnp.pi / 2.0),
                jnp.array([[0.0, -1.0], [1.0, 0.0]]),
                jnp.zeros(2),
                jnp.full(2, -jnp.inf),
            ),
            (
                "diffuse",
                Diffuse(sites=0, dims=(2,)),
                jnp.log(jnp.array([2.0, 4.0])),
                jnp.eye(2),
                jnp.zeros(2),
                jnp.log(jnp.array([2.0, 4.0])),
            ),
            (
                "noise",
                GaussianNoiseGate(sites=[0], dims=(3,)),
                jnp.array(-2.0),
                jnp.eye(3),
                jnp.zeros(3),
                jnp.full(3, -2.0),
            ),
            (
                "affine",
                AffineGaussianGate(sites=[0], dims=(2,)),
                {
                    "A": jnp.array([[1.0, 2.0], [0.0, 1.0]]),
                    "b": jnp.array([0.5, -1.0]),
                    "log_var": jnp.array([-3.0, -4.0]),
                },
                jnp.array([[1.0, 2.0], [0.0, 1.0]]),
                jnp.array([0.5, -1.0]),
                jnp.array([-3.0, -4.0]),
            ),
        ]
    )
    def test_affine_parameters(
        self, _name, gate, theta, expected_A, expected_b, expected_log_var
    ):
        A, b, log_var = gate.affine_parameters(theta)

        self.assertIsInstance(gate, AbstractAffineGaussianGate)
        self.assertTrue(jnp.allclose(A, expected_A, atol=1e-6))
        self.assertTrue(jnp.allclose(b, expected_b, atol=1e-6))
        self.assertTrue(jnp.allclose(log_var, expected_log_var, atol=1e-6))

    def test_diffuse_nan_log_var_propagates(self):
        gate = Diffuse(sites=0, dims=(1,))

        _, _, log_var = gate.affine_parameters(jnp.array([jnp.nan]))

        self.assertTrue(jnp.all(jnp.isnan(log_var)))

    @parameterized.expand(
        [
            ("displace", lambda: Displace(sites=0, dims=(2,))),
            ("scale", lambda: Scale(sites=0, dims=(2,))),
            ("mix", lambda: Mix(sites=[0, 1], dims=(1, 1))),
            ("diffuse", lambda: Diffuse(sites=0, dims=(2,))),
            ("noise", lambda: GaussianNoiseGate(sites=[0], dims=(2,))),
            ("affine", lambda: AffineGaussianGate(sites=[0], dims=(2,))),
        ]
    )
    def test_named_gate_affine_parameters_jit_matches_eager(self, _name, make_gate):
        """Each named gate's affine channel matches eager under filter_jit."""
        gate = make_gate()
        theta = gate.init_params(jax.random.key(0))

        eager = gate.affine_parameters(theta)
        jitted = eqx.filter_jit(gate.affine_parameters)(theta)

        for e, j in zip(eager, jitted):
            self.assertTrue(jnp.allclose(e, j))


if __name__ == "__main__":
    unittest.main()
