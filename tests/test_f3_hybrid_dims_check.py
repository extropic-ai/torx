"""Regression tests: hybrid gates conflicting on a discrete site's dim.

Two hybrid gates disagreeing on a discrete (control) site's dimension
were silently accepted (last gate won). Construction now raises
ValueError, mirroring the pre-existing discrete-gate check.
"""

import jax
import jax.numpy as jnp
import pytest

from torx.psc import HybridPCircuit, MixtureGaussianGate, PditShift


def _mix(cont_site, n_comp):
    return MixtureGaussianGate(sites=(0, cont_site), dims=(1,), num_components=n_comp)


def test_hybrid_discrete_conflict_raises():
    # Both orders: the pre-fix bug was order-dependent (last gate won).
    with pytest.raises(
        ValueError, match=r"Discrete dimension mismatch at site 0: expected 2, got 3"
    ):
        HybridPCircuit([_mix(0, 2), _mix(1, 3)])
    with pytest.raises(
        ValueError, match=r"Discrete dimension mismatch at site 0: expected 3, got 2"
    ):
        HybridPCircuit([_mix(1, 3), _mix(0, 2)])


def test_hybrid_vs_discrete_conflict_raises():
    # A hybrid gate must not overwrite a dim a discrete gate already set,
    # and vice versa.
    with pytest.raises(ValueError, match="Discrete dimension mismatch at site 0"):
        HybridPCircuit([PditShift(0, 3), _mix(0, 2)])
    with pytest.raises(ValueError, match="Discrete dimension mismatch at site 0"):
        HybridPCircuit([_mix(0, 2), PditShift(0, 3)])


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
