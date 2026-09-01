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


class TestSiteDefaults(unittest.TestCase):
    def test_the_optional_fields_default_to_none(self):
        site = Site(
            name="inc",
            factor=_increment_factor(),
            parents=("input",),
            porting_fn=("x",),
        )
        self.assertIsNone(site.param_key)
        self.assertIsNone(site.info_key)
        self.assertIsNone(site.site_info)

    def test_a_defaulted_site_still_builds_and_runs(self):
        dfg = DFG(
            (
                Site(
                    name="inc",
                    factor=_increment_factor(),
                    parents=("input",),
                    porting_fn=("x",),
                ),
            ),
            {"input": _SPEC},
            "inc",
        )
        x = jnp.array([1.0, 2.0])
        self.assertTrue(jnp.allclose(dfg.sample(_KEY, {"input": x}, {}), x + 1.0))
        self.assertEqual(dfg.init_params(_KEY), {})

    def test_the_optional_fields_can_still_be_given_positionally(self):
        site = Site("inc", _increment_factor(), ("input",), ("x",), "p", "i", "s")
        self.assertEqual(
            (site.param_key, site.info_key, site.site_info), ("p", "i", "s")
        )


class TestDFGInfoDefaults(unittest.TestCase):
    def test_both_fields_default(self):
        info = DFGInfo()
        self.assertFalse(info.expose_site_outputs)
        self.assertEqual(dict(info.entries), {})

    def test_entries_alone_is_enough(self):
        info = DFGInfo(entries={"a": 1})
        self.assertFalse(info.expose_site_outputs)
        self.assertEqual(dict(info.entries), {"a": 1})

    def test_a_positional_argument_is_refused(self):
        # Keyword-only, so a bare `DFGInfo(entries_dict)` cannot silently land in
        # `expose_site_outputs`.
        with self.assertRaises(TypeError):
            DFGInfo({"a": 1})  # pyright: ignore[reportCallIssue]

    def test_a_defaulted_info_leaves_the_aux_alone(self):
        dfg = _increment_dfg()
        x = jnp.array([1.0, 2.0])
        _main, aux = dfg.sample(_KEY, {"input": x}, {}, DFGInfo(), return_aux=True)
        self.assertEqual(aux, (None,))


if __name__ == "__main__":
    unittest.main()
