import unittest

import jax
import jax.numpy as jnp

from torx import DFG, DFGInfo, Site, TiledFactor
from torx.tractable_prob_factors import DeterministicFactor

_KEY = jax.random.key(0)
_SPEC = jax.ShapeDtypeStruct((2,), jnp.float32)


def _increment_factor():
    return DeterministicFactor(
        lambda inputs, site_info: inputs["x"] + 1.0,
        {"x": _SPEC},
        _SPEC,
    )


def _increment_dfg():
    return DFG(
        (
            Site(
                name="inc",
                factor=_increment_factor(),
                parents=("input",),
                porting_fn=("x",),
                param_key=None,
                info_key=None,
                site_info=None,
            ),
        ),
        {"input": _SPEC},
        "inc",
    )


class TestDFGReferences(unittest.TestCase):
    def test_exposed_site_outputs_have_reference_axis(self):
        dfg = _increment_dfg()
        x = jnp.array([1.0, 2.0])

        main, aux = dfg.sample_with_references(
            _KEY,
            {"input": x},
            {},
            DFGInfo(expose_site_outputs=True),
            n_references=2,
        )

        site_outputs, per_site_auxes = aux
        expected = x + 1.0
        self.assertTrue(jnp.allclose(main, expected))
        self.assertEqual(site_outputs["inc"].shape, (3, 2))
        self.assertTrue(
            jnp.allclose(site_outputs["inc"], jnp.broadcast_to(expected, (3, 2)))
        )
        self.assertEqual(per_site_auxes, (None,))

    def test_exposed_site_outputs_compose_with_tiled_factor(self):
        tiled = TiledFactor(_increment_dfg(), n_tiles=4, weight_tied=True)
        x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)

        main, aux = tiled.sample_with_references(
            _KEY,
            {"input": x},
            {},
            DFGInfo(expose_site_outputs=True),
            n_references=2,
        )

        site_outputs, _per_site_auxes = aux
        expected = x + 1.0
        self.assertTrue(jnp.allclose(main, expected))
        self.assertEqual(site_outputs["inc"].shape, (3, 4, 2))
        self.assertTrue(
            jnp.allclose(site_outputs["inc"], jnp.broadcast_to(expected, (3, 4, 2)))
        )


class TestDFGValidation(unittest.TestCase):
    def test_unknown_parent_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "unknown parent"):
            DFG(
                (
                    Site(
                        name="inc",
                        factor=_increment_factor(),
                        parents=("typo",),
                        porting_fn=("x",),
                        param_key=None,
                        info_key=None,
                        site_info=None,
                    ),
                ),
                {"input": _SPEC},
                "inc",
            )

    def test_unknown_output_name_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "output_name"):
            DFG(
                (
                    Site(
                        name="inc",
                        factor=_increment_factor(),
                        parents=("input",),
                        porting_fn=("x",),
                        param_key=None,
                        info_key=None,
                        site_info=None,
                    ),
                ),
                {"input": _SPEC},
                "typo",
            )

    def test_tuple_porting_typo_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "porting_fn"):
            DFG(
                (
                    Site(
                        name="inc",
                        factor=_increment_factor(),
                        parents=("input",),
                        porting_fn=("typo",),
                        param_key=None,
                        info_key=None,
                        site_info=None,
                    ),
                ),
                {"input": _SPEC},
                "inc",
            )


if __name__ == "__main__":
    unittest.main()
