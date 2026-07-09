import unittest
from typing import Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from torx import ChainFactor, TiledFactor
from torx.factor import _PortSpec, AbstractReferenceFactor

_KEY = jax.random.key(0)


class _SumShift(AbstractReferenceFactor):
    """Output = sum of all input ports + `params` (+ `info` when given)."""

    port_names: tuple[str, ...] = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    input_ports: Mapping[str, _PortSpec] = eqx.field(static=True)
    output_spec: _PortSpec = eqx.field(static=True)

    def __init__(self, port_names, dim):
        self.port_names = tuple(port_names)
        self.dim = dim
        self.input_ports = {
            n: jax.ShapeDtypeStruct((dim,), jnp.float32) for n in self.port_names
        }
        self.output_spec = jax.ShapeDtypeStruct((dim,), jnp.float32)

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        total = params
        if info is not None:
            total = total + info
        for n in self.port_names:
            total = total + inputs[n]
        return (total, {"bias": params}) if return_aux else total

    def init_params(self, key) -> PyTree[Array]:
        return jax.random.normal(key, (self.dim,))


class TestTiledFactor(unittest.TestCase):
    def setUp(self):
        self.dim, self.n_tiles = 3, 4
        self.base = _SumShift(("x",), self.dim)
        self.x = jax.random.normal(_KEY, (self.n_tiles, self.dim))

    def test_param_shapes_tied_vs_untied(self):
        tied = TiledFactor(self.base, self.n_tiles, weight_tied=True)
        untied = TiledFactor(self.base, self.n_tiles, weight_tied=False)
        self.assertEqual(tied.init_params(_KEY).shape, (self.dim,))
        self.assertEqual(untied.init_params(_KEY).shape, (self.n_tiles, self.dim))

    def test_sample_values_tied_and_untied(self):
        tied = TiledFactor(self.base, self.n_tiles, weight_tied=True)
        params = self.base.init_params(_KEY)
        out = tied.sample(_KEY, {"x": self.x}, params)
        self.assertEqual(out.shape, (self.n_tiles, self.dim))
        self.assertTrue(jnp.allclose(out, self.x + params[None, :]))

        untied = TiledFactor(self.base, self.n_tiles, weight_tied=False)
        params_u = untied.init_params(_KEY)  # (n_tiles, dim)
        out_u = untied.sample(_KEY, {"x": self.x}, params_u)
        self.assertTrue(jnp.allclose(out_u, self.x + params_u))


class TestChainFactor(unittest.TestCase):
    def setUp(self):
        self.dim, self.n_steps = 3, 2

    def test_str_feedback_with_broadcast_port(self):
        # "x" is fed back; "c" is a constant broadcast context port.
        base = _SumShift(("x", "c"), self.dim)
        chain = ChainFactor(base, self.n_steps, "x", weight_tied=True)
        self.assertEqual(chain.feedback_ports, ("x",))
        self.assertEqual(chain.all_step_input_ports, ("c",))

        params = base.init_params(_KEY)
        x0 = jax.random.normal(_KEY, (self.dim,))
        c = jnp.ones((self.dim,))
        out = chain.sample(_KEY, {"x": x0, "c": c}, params)
        # main0 = p + x0 + c ; main1 = p + main0 + c = 2p + x0 + 2c.
        self.assertEqual(chain.init_params(_KEY).shape, (self.dim,))
        self.assertTrue(jnp.allclose(out, 2 * params + x0 + 2 * c))

    def test_construction_errors(self):
        base = _SumShift(("x",), self.dim)
        # Unknown feedback port name.
        with self.assertRaisesRegex(ValueError, "not an input port"):
            ChainFactor(base, 2, "nope", weight_tied=True)
        # Callable producing a port not in base.input_ports.
        with self.assertRaisesRegex(ValueError, "not in"):
            ChainFactor(base, 2, lambda m: {"z": m}, weight_tied=True)
        # Callable producing a feedback value of the wrong shape.
        with self.assertRaisesRegex(ValueError, "does not match"):
            ChainFactor(base, 2, lambda m: {"x": m[:-1]}, weight_tied=True)


if __name__ == "__main__":
    unittest.main()
