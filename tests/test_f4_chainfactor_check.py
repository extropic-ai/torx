"""Regression tests for F4 (torx): ChainFactor n_steps validation.

Run with the torx venv:
    venv-torx/bin/python -m pytest tests/test_f4_chainfactor_check.py -q

Before the fix, ChainFactor(n_steps=0) silently returned an all-zeros
"sample" (the scan's initial carry; no step ever ran). After:
construction raises ValueError("n_steps must be at least 1.").
Legitimate counts must keep working.
"""

import jax
import jax.numpy as jnp
import pytest

from torx import ChainFactor, DeterministicFactor

SDS = jax.ShapeDtypeStruct
KEY = jax.random.key(0)


def identity_factor():
    return DeterministicFactor(
        fn=lambda inputs, site_info: inputs["x"],
        input_ports={"x": SDS((), jnp.int32)},
        output_spec=SDS((), jnp.int32),
    )


def test_zero_steps_raises_at_construction():
    with pytest.raises(ValueError, match=r"n_steps must be at least 1\."):
        ChainFactor(
            identity_factor(), n_steps=0, feedback_porting_fn="x", weight_tied=True
        )


def test_legitimate_steps_return_input():
    chain = ChainFactor(
        identity_factor(), n_steps=1, feedback_porting_fn="x", weight_tied=True
    )
    out = chain.sample(KEY, {"x": jnp.array(5, dtype=jnp.int32)}, None)
    assert int(out) == 5
