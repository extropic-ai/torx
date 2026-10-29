"""Regression tests for F4 (torx): ChainFactor n_steps validation.

Run with the torx venv:
    venv-torx/bin/python -m pytest tests/test_f4_chainfactor_check.py -q

Before the fix, ChainFactor(n_steps=0) silently returned an all-zeros
"sample" (the scan's initial carry; no step ever ran), and n_steps=-1
failed late with a cryptic MLIRError from inside jax.lax.scan. After:
construction raises ValueError("n_steps must be at least 1."), mirroring
the circuit classes' reps check. Legitimate counts must keep working.
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


def test_negative_steps_raises_valueerror_at_construction():
    # Before: MLIRError from jax.lax.scan at sample time. After: a clear
    # ValueError at construction time (asserted by raising around the
    # constructor only).
    with pytest.raises(ValueError, match=r"n_steps must be at least 1\."):
        ChainFactor(
            identity_factor(), n_steps=-1, feedback_porting_fn="x", weight_tied=True
        )


def test_callable_porting_fn_zero_steps_raises():
    with pytest.raises(ValueError, match=r"n_steps must be at least 1\."):
        ChainFactor(
            identity_factor(),
            n_steps=0,
            feedback_porting_fn=lambda m: {"x": m},
            weight_tied=True,
        )


@pytest.mark.parametrize("n_steps", [1, 3, 1000])
def test_legitimate_steps_return_input(n_steps):
    chain = ChainFactor(
        identity_factor(), n_steps=n_steps, feedback_porting_fn="x", weight_tied=True
    )
    out = chain.sample(KEY, {"x": jnp.array(5, dtype=jnp.int32)}, None)
    assert int(out) == 5
