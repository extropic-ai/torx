"""Regression tests for F3 (torx): HybridPCircuit discrete-dim conflict check.

Run with the torx venv:
    venv-torx/bin/python -m pytest tests/test_f3_hybrid_dims_check.py -q

Before the fix, two hybrid gates disagreeing on a discrete (control)
site's dimension were silently accepted (last gate won) and out-of-range
control values later clamped silently onto the wrong mixture component.
After: construction raises ValueError with the same message shape as the
pre-existing discrete-gate branch. The discrete and continuous branches
must behave byte-identically to before (their messages are asserted
verbatim), and agreeing gates must keep constructing and sampling.
"""

import jax
import jax.numpy as jnp
import pytest

from torx.psc import HybridPCircuit, MixtureGaussianGate, PditShift


def _mix(cont_site, n_comp, dim=1):
    return MixtureGaussianGate(sites=(0, cont_site), dims=(dim,), num_components=n_comp)


def test_hybrid_discrete_conflict_raises():
    with pytest.raises(
        ValueError, match=r"Discrete dimension mismatch at site 0: expected 2, got 3"
    ):
        HybridPCircuit([_mix(0, 2), _mix(1, 3)])


def test_hybrid_discrete_conflict_reversed_order_raises():
    # F3 fix red-team (2026-10-15): the pre-fix bug was order-dependent
    # (last gate won), so the conflict must raise regardless of listing order.
    with pytest.raises(
        ValueError, match=r"Discrete dimension mismatch at site 0: expected 3, got 2"
    ):
        HybridPCircuit([_mix(1, 3), _mix(0, 2)])


def test_three_hybrid_gates_one_conflict_raises():
    with pytest.raises(ValueError, match="Discrete dimension mismatch at site 0"):
        HybridPCircuit([_mix(0, 2), _mix(1, 2), _mix(2, 3)])


def test_hybrid_vs_discrete_both_orders_raise():
    with pytest.raises(ValueError, match="Discrete dimension mismatch at site 0"):
        HybridPCircuit([PditShift(0, 3), _mix(0, 2)])
    with pytest.raises(ValueError, match="Discrete dimension mismatch at site 0"):
        HybridPCircuit([_mix(0, 2), PditShift(0, 3)])


def test_discrete_branch_message_unchanged():
    # Verbatim against the BEFORE capture (tw4 case (a)).
    with pytest.raises(
        ValueError, match=r"Discrete dimension mismatch at site 0: expected 2, got 3"
    ):
        HybridPCircuit([PditShift(0, 2), PditShift(0, 3)])


def test_continuous_branch_still_raises():
    g_a = MixtureGaussianGate(sites=(0, 0), dims=(2,), num_components=2)
    g_b = MixtureGaussianGate(sites=(1, 0), dims=(3,), num_components=2)
    with pytest.raises(
        ValueError, match=r"Continuous dimension mismatch at site 0: expected 2, got 3"
    ):
        HybridPCircuit([g_a, g_b])


def test_agreeing_hybrid_gates_construct_and_sample():
    circ = HybridPCircuit([_mix(0, 2), _mix(1, 2)])
    assert circ.discrete_dims == (2,)
    params = circ.init_params(jax.random.key(0))
    for control in (0, 1):
        out = circ.sample(
            jax.random.key(1),
            {
                "discrete": jnp.array([control], dtype=jnp.int32),
                "continuous": jnp.zeros(2, dtype=jnp.float32),
            },
            params,
        )
        assert out["continuous"].shape == (2,)


def test_agreeing_gates_apply_correct_components():
    # tw4b's crafted-params scenario on an AGREEING (legal) circuit:
    # control 0 -> shift ~0, control 1 -> shift ~+10, as designed.
    circ = HybridPCircuit([_mix(0, 2), _mix(1, 2)])
    params = dict(circ.init_params(jax.random.key(0)))
    params[0] = {
        "means": jnp.array([[0.0], [10.0]], dtype=jnp.float32),
        "log_vars": jnp.array([[-20.0], [-20.0]], dtype=jnp.float32),
    }
    params[1] = {
        "means": jnp.zeros((2, 1), dtype=jnp.float32),
        "log_vars": jnp.full((2, 1), -20.0, dtype=jnp.float32),
    }
    shifts = {}
    for control in (0, 1):
        out = circ.sample(
            jax.random.key(1),
            {
                "discrete": jnp.array([control], dtype=jnp.int32),
                "continuous": jnp.zeros(2, dtype=jnp.float32),
            },
            params,
        )
        shifts[control] = float(out["continuous"][0])
    assert abs(shifts[0] - 0.0) < 0.01
    assert abs(shifts[1] - 10.0) < 0.01
