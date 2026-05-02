import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import torx


def test_conditional_gate_validates_shapes():
    with pytest.raises(ValueError, match="one entry per target"):
        torx.PConditionalBernoulli(
            jnp.zeros(1),
            [0, 1],
            [[2], [2]],
            jnp.zeros((2, 1)),
        )

    with pytest.raises(ValueError, match="rectangular"):
        torx.PConditionalBernoulli(
            jnp.zeros(2),
            [0, 1],
            [[2], [2, 3]],
            jnp.zeros((2, 2)),
        )

    with pytest.raises(ValueError, match="unique"):
        torx.PConditionalBernoulli(
            jnp.zeros(2),
            [0, 0],
            [[1], [1]],
            jnp.zeros((2, 1)),
        )

    with pytest.raises(ValueError, match="non-negative"):
        torx.PConditionalBernoulli(
            jnp.zeros(1),
            [0],
            [[-1]],
            jnp.zeros((1, 1)),
        )


def test_conditional_matrix_is_stochastic_and_exact():
    gate = torx.PConditionalBernoulli(
        jnp.array([0.0]),
        [2],
        [[0, 1]],
        jnp.array([[0.0, 0.0]]),
    )
    matrix = gate.get_matrix()
    assert matrix.shape == (8, 8)
    assert jnp.allclose(jnp.sum(matrix, axis=0), 1.0)

    circuit = torx.DiscretePCircuit([gate])
    sim = torx.StateVectorSimulator()
    compiled = sim.build_circuit(circuit)
    initial = jnp.zeros(8).at[0].set(1.0)
    assert jnp.allclose(sim.expval(compiled, initial, 2), 0.5)


def test_conditional_matrix_reads_controls_before_target_updates():
    gate = torx.PConditionalBernoulli(
        jnp.array([-20.0, -20.0]),
        [0, 1],
        [[1], [0]],
        jnp.array([[40.0], [40.0]]),
    )
    circuit = torx.DiscretePCircuit([gate])
    sim = torx.StateVectorSimulator()
    compiled = sim.build_circuit(circuit)
    initial = jnp.zeros(4).at[1].set(1.0)
    density = sim.density(compiled, initial)

    assert jnp.argmax(density) == 2
    assert density[2] > 0.999


def test_conditional_sampling_handles_high_fan_in_without_branch_table():
    control_sites = [list(range(10))]
    gate = torx.PConditionalBernoulli(
        jnp.array([20.0]),
        [10],
        control_sites,
        jnp.zeros((1, 10)),
    )
    circuit = torx.DiscretePCircuit([gate])
    compiled = torx.SampleSimulator(num_samples=8).build_circuit(circuit)

    assert compiled.has_conditional
    assert compiled.branch_ops.shape[2] == 2
    assert compiled.conditional_data is not None
    assert compiled.conditional_data.target_sites.shape == (1, 1)

    initial = jnp.zeros(11, dtype=jnp.int32)
    samples = torx.SampleSimulator(num_samples=8).sample(
        compiled,
        initial,
        jax.random.key(0),
    )
    assert jnp.all(samples[:, 10] == 1)


def test_multi_target_conditional_sampling_and_padding():
    gate = torx.PConditionalBernoulli(
        jnp.array([20.0, -20.0]),
        [2, 3],
        [[0, 1], [0, 1]],
        jnp.zeros((2, 2)),
    )
    circuit = torx.DiscretePCircuit([gate])
    sim = torx.SampleSimulator(num_samples=8)
    compiled = sim.build_circuit(circuit)
    assert compiled.conditional_data is not None
    assert compiled.conditional_data.target_sites.shape == (1, 2)
    assert compiled.conditional_data.control_sites.shape == (1, 2, 2)

    initial = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    samples = sim.sample(compiled, initial, jax.random.key(4))

    assert jnp.all(samples[:, 2] == 1)
    assert jnp.all(samples[:, 3] == 0)


def test_conditional_sampling_with_repetitions():
    gate = torx.PConditionalBernoulli(
        jnp.array([20.0]),
        [0],
        [[]],
        jnp.zeros((1, 0)),
    )
    circuit = torx.DiscretePCircuit([gate], reps=3)
    sim = torx.SampleSimulator(num_samples=8)
    compiled = sim.build_circuit(circuit)

    samples = sim.sample(compiled, jnp.array([0], dtype=jnp.int32), jax.random.key(5))

    assert jnp.all(samples[:, 0] == 1)


def test_mixed_branch_and_conditional_sampling_is_jittable():
    conditional = torx.PConditionalBernoulli(
        jnp.array([20.0]),
        [2],
        [[0, 1]],
        jnp.zeros((1, 2)),
    )
    circuit = torx.DiscretePCircuit([torx.PNOT(jnp.inf, 0), conditional])
    sim = torx.SampleSimulator(num_samples=4)
    compiled = sim.build_circuit(circuit)
    initial = jnp.array([0, 1, 0], dtype=jnp.int32)

    @eqx.filter_jit
    def run(compiled_circuit):
        return sim.sample(compiled_circuit, initial, jax.random.key(1))

    samples = run(compiled)
    assert samples.shape == (4, 3)
    assert jnp.all(samples[:, 0] == 1)
    assert jnp.all(samples[:, 2] == 1)


def test_mixed_k_branch_and_conditional_sampling_uses_categorical_path():
    conditional = torx.PConditionalBernoulli(
        jnp.array([20.0]),
        [2],
        [[1]],
        jnp.zeros((1, 1)),
    )
    circuit = torx.DiscretePCircuit(
        [torx.PditCycle(jnp.array([0.0, 4.0]), 0, 3), conditional],
    )
    sim = torx.SampleSimulator(num_samples=16)
    compiled = sim.build_circuit(circuit)

    assert compiled.max_branches == 3
    samples = sim.sample(compiled, jnp.array([0, 1, 0]), jax.random.key(6))

    assert samples.shape == (16, 3)
    assert jnp.all(samples[:, 2] == 1)


def test_conditional_sample_matches_statevector_on_small_circuit():
    circuit = torx.DiscretePCircuit(
        [
            torx.PNOT(jnp.array(-0.2), 0),
            torx.PConditionalBernoulli(
                jnp.array([0.3, -0.4]),
                [1, 2],
                [[0], [1]],
                jnp.array([[0.7], [-0.5]]),
            ),
        ]
    )
    initial_density = jnp.zeros(8).at[0].set(1.0)
    exact = torx.StateVectorSimulator().expval_all(
        torx.StateVectorSimulator().build_circuit(circuit),
        initial_density,
    )

    sim = torx.SampleSimulator(num_samples=20_000)
    sampled = sim.expval_all(
        sim.build_circuit(circuit),
        jnp.array([0, 0, 0], dtype=jnp.int32),
        jax.random.key(7),
    )

    assert jnp.allclose(sampled, exact, atol=0.02)


def test_conditional_to_pcircuit_round_trip_updates_parameters():
    gate = torx.PConditionalBernoulli(
        jnp.array([0.25]),
        [1],
        [[0]],
        jnp.array([[0.5]]),
    )
    circuit = torx.DiscretePCircuit([torx.PNOT(jnp.array(0.1), 0), gate])
    compiled = torx.SampleSimulator(num_samples=4).build_circuit(circuit)
    assert compiled.conditional_data is not None

    new_conditional = eqx.tree_at(
        lambda data: (data.logits, data.control_weights),
        compiled.conditional_data,
        (
            compiled.conditional_data.logits.at[0, 0].set(1.25),
            compiled.conditional_data.control_weights.at[0, 0, 0].set(-0.75),
        ),
    )
    compiled = eqx.tree_at(lambda c: c.conditional_data, compiled, new_conditional)
    restored = compiled.to_pcircuit(circuit)

    assert jnp.allclose(restored.gates[1].theta, jnp.array([1.25]))
    assert jnp.allclose(restored.gates[1].control_weights, jnp.array([[-0.75]]))


@pytest.mark.parametrize(
    "diff_method", ["param_shift_inf", "param_shift_single", "param_shift_filter"]
)
def test_conditional_gradients_raise_clear_error(diff_method):
    gate = torx.PConditionalBernoulli(
        jnp.array([0.25]),
        [1],
        [[0]],
        jnp.array([[0.5]]),
    )
    compiled = torx.SampleSimulator(num_samples=16).build_circuit(
        torx.DiscretePCircuit([gate])
    )
    initial = jnp.array([1, 0], dtype=jnp.int32)
    sim = torx.SampleSimulator(num_samples=16, diff_method=diff_method)

    def loss(compiled_circuit):
        return jnp.sum(sim.expval_all(compiled_circuit, initial, jax.random.key(2)))

    with pytest.raises(NotImplementedError, match="conditional sample gates"):
        eqx.filter_grad(loss)(compiled)


def test_hybrid_simulator_reports_conditional_gate_limitation():
    circuit = torx.HybridPCircuit(
        [
            torx.PConditionalBernoulli(
                jnp.array([0.25]),
                [1],
                [[0]],
                jnp.array([[0.5]]),
            )
        ]
    )
    initial = {
        "discrete": jnp.array([1, 0], dtype=jnp.int32),
        "continuous": jnp.array([], dtype=jnp.float32),
    }
    sim = torx.HybridSampleSimulator(num_samples=4)

    with (
        pytest.warns(UserWarning, match="no continuous"),
        pytest.raises(ValueError, match="conditional sample gates"),
    ):
        sim.sample(circuit, initial, jax.random.key(8))
