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


class TestDFGPortingSpecValidation(unittest.TestCase):
    """A site's porting must produce the factor's input ports, with the specs
    the factor declares.
    """

    def _dfg(self, porting_fn, input_spec=_SPEC):
        return DFG(
            (
                Site(
                    name="inc",
                    factor=_increment_factor(),
                    parents=("input",),
                    porting_fn=porting_fn,
                    param_key=None,
                    info_key=None,
                    site_info=None,
                ),
            ),
            {"input": input_spec},
            "inc",
        )

    def test_correct_callable_porting_is_accepted(self):
        dfg = self._dfg(lambda parents: {"x": parents[0]})
        x = jnp.array([1.0, 2.0])
        self.assertTrue(jnp.allclose(dfg.sample(_KEY, {"input": x}, {}), x + 1.0))

    def test_callable_porting_to_the_wrong_port_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "extra"):
            self._dfg(lambda parents: {"typo": parents[0]})

    def test_callable_porting_returning_a_non_mapping_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "must return a mapping"):
            self._dfg(lambda parents: parents[0])

    def test_callable_porting_that_raises_fails_at_construction(self):
        def bad_porting(parents):
            raise RuntimeError("cannot route this")

        with self.assertRaisesRegex(ValueError, "cannot route this"):
            self._dfg(bad_porting)

    def test_callable_porting_with_the_wrong_shape_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "shapes and dtypes"):
            self._dfg(lambda parents: {"x": parents[0][:1]})

    def test_tuple_porting_with_the_wrong_shape_fails_at_construction(self):
        wide = jax.ShapeDtypeStruct((5,), jnp.float32)
        with self.assertRaisesRegex(ValueError, "shapes and dtypes"):
            self._dfg(("x",), input_spec=wide)

    def test_tuple_porting_with_the_wrong_dtype_fails_at_construction(self):
        ints = jax.ShapeDtypeStruct((2,), jnp.int32)
        with self.assertRaisesRegex(ValueError, "shapes and dtypes"):
            self._dfg(("x",), input_spec=ints)

    def test_a_site_parent_spec_is_checked_too(self):
        # `wide` produces (5,) but the second site's port declares (2,).
        wide_spec = jax.ShapeDtypeStruct((5,), jnp.float32)
        wide = DeterministicFactor(
            lambda inputs, site_info: jnp.zeros((5,), jnp.float32),
            {"x": _SPEC},
            wide_spec,
        )
        with self.assertRaisesRegex(ValueError, "shapes and dtypes"):
            DFG(
                (
                    Site(
                        name="wide",
                        factor=wide,
                        parents=("input",),
                        porting_fn=("x",),
                        param_key=None,
                        info_key=None,
                        site_info=None,
                    ),
                    Site(
                        name="inc",
                        factor=_increment_factor(),
                        parents=("wide",),
                        porting_fn=("x",),
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
