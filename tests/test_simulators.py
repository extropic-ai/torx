"""Test the simulators in `simulators.py`"""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
from parameterized import parameterized

from torx.psc import (
    CompiledSamplePCircuit,
    CompiledStateVectorPCircuit,
    DiscretePCircuit,
    PCNOT,
    PditShift,
    PditSWAP,
    PMultiCNOT,
    PNOT,
    PSWAP,
    SampleSimulator,
    StateVectorSimulator,
)


def _thetas(*values):
    """Build a per-gate thetas list aligned with the circuit gates.

    Each gate's theta has shape ``(num_branches - 1,)``; for the 2-branch gates
    used here that is ``(1,)``.
    """
    return [jnp.atleast_1d(v) for v in values]


class TestCompiledStateVectorPCircuit(unittest.TestCase):
    def test_from_pcircuit(self):
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates, reps=3)

        built_circuit = CompiledStateVectorPCircuit.from_pcircuit(circuit, thetas)

        self.assertIsInstance(built_circuit, CompiledStateVectorPCircuit)
        self.assertEqual(built_circuit.gates, gates)
        self.assertEqual(built_circuit.num_pdits, 2)
        self.assertEqual(built_circuit.reps, 3)

    def test_to_pcircuit(self):
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates, reps=3)

        new_circuit = CompiledStateVectorPCircuit.from_pcircuit(
            circuit, thetas
        ).to_pcircuit(circuit)

        self.assertIsInstance(new_circuit, DiscretePCircuit)
        self.assertEqual(new_circuit.gates, gates)
        self.assertEqual(new_circuit.num_pdits, 2)
        self.assertEqual(new_circuit.reps, 3)


class TestStateVectorSimulator(unittest.TestCase):
    def test_attributes(self):
        sim = StateVectorSimulator()
        self.assertIs(sim.circuit_backend, CompiledStateVectorPCircuit)

    def test_build_circuit(self):
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        sim = StateVectorSimulator()
        built_circuit = sim.build_circuit(circuit, thetas)

        self.assertIsInstance(built_circuit, CompiledStateVectorPCircuit)

    def test_apply_gate_one_site(self):
        theta = jnp.atleast_1d(1.0)
        p = 1 / (1 + jnp.exp(-1))

        gate = PNOT(0)
        sim = StateVectorSimulator()
        dims = (2,)

        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(jnp.array([1.0, 0.0]), gate, theta, dims),
                jnp.array([1 - p, p]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(jnp.array([0.0, 1.0]), gate, theta, dims),
                jnp.array([p, 1 - p]),
            )
        )

    def test_apply_gate_two_sites(self):
        theta = jnp.atleast_1d(1.0)
        p = 1 / (1 + jnp.exp(-1))

        gate = PCNOT([0, 1])
        sim = StateVectorSimulator()
        dims = (2, 2)

        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[1.0, 0.0], [0.0, 0.0]]), gate, theta, dims
                ).reshape(-1),
                jnp.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[0.0, 1.0], [0.0, 0.0]]), gate, theta, dims
                ).reshape(-1),
                jnp.array([0.0, 1.0, 0.0, 0.0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[0.0, 0.0], [1.0, 0.0]]), gate, theta, dims
                ).reshape(-1),
                jnp.array([0.0, 0.0, 1 - p, p]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[0.0, 0.0], [0.0, 1.0]]), gate, theta, dims
                ).reshape(-1),
                jnp.array([0.0, 0.0, p, 1 - p]),
            )
        )

        gate = PCNOT([1, 0])
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[1.0, 0.0], [0.0, 0.0]]), gate, theta, dims
                ).reshape(-1),
                jnp.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[0.0, 1.0], [0.0, 0.0]]), gate, theta, dims
                ).reshape(-1),
                jnp.array([0.0, 1 - p, 0.0, p]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[0.0, 0.0], [1.0, 0.0]]), gate, theta, dims
                ).reshape(-1),
                jnp.array([0.0, 0.0, 1.0, 0.0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[0.0, 0.0], [0.0, 1.0]]), gate, theta, dims
                ).reshape(-1),
                jnp.array([0.0, p, 0.0, 1 - p]),
            )
        )

    def test_apply_gate_three_sites(self):
        theta = jnp.atleast_1d(1.0)
        p = 1 / (1 + jnp.exp(-1))

        gate = PMultiCNOT([0, 1, 2])
        sim = StateVectorSimulator()
        dims = (2, 2, 2)

        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([1.0, 0, 0, 0, 0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 1.0, 0, 0, 0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 1.0, 0, 0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 1.0, 0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 0, 1.0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 0, 0, 1.0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 0, 0, 0, 1 - p, p]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 0, 0, 0, p, 1 - p]),
            )
        )

        gate = PMultiCNOT([0, 2, 1])
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([1.0, 0, 0, 0, 0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 1.0, 0, 0, 0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 1.0, 0, 0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 1.0, 0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 0, 1.0, 0, 0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 0, 0, 1 - p, 0, p]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 0, 0, 0, 1.0, 0]),
            )
        )
        self.assertTrue(
            jnp.allclose(
                sim.apply_gate(
                    jnp.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]),
                    gate,
                    theta,
                    dims,
                ).reshape(-1),
                jnp.array([0, 0, 0, 0, 0, p, 0, 1 - p]),
            )
        )

    def test_density(self):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        sim = StateVectorSimulator()
        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        self.assertTrue(
            jnp.allclose(
                sim.density(circuit, jnp.array([1.0, 0, 0, 0])),
                jnp.array(
                    [
                        (1 - p[0]) * (1 - p[2]),
                        p[2] * (1 - p[0]),
                        p[0] * (1 - p[1]) * (1 - p[2]) + p[0] * p[1] * p[2],
                        p[0] * p[1] * (1 - p[2]) + p[0] * (1 - p[1]) * p[2],
                    ]
                ),
            )
        )

    def test_density_jit(self):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        sim = StateVectorSimulator()
        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        self.assertTrue(
            jnp.allclose(
                eqx.filter_jit(sim.density)(circuit, jnp.array([1.0, 0, 0, 0])),
                jnp.array(
                    [
                        (1 - p[0]) * (1 - p[2]),
                        p[2] * (1 - p[0]),
                        p[0] * (1 - p[1]) * (1 - p[2]) + p[0] * p[1] * p[2],
                        p[0] * p[1] * (1 - p[2]) + p[0] * (1 - p[1]) * p[2],
                    ]
                ),
            )
        )

    def test_expval(self):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        sim = StateVectorSimulator()
        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        self.assertTrue(
            jnp.isclose(sim.expval(circuit, jnp.array([1.0, 0, 0, 0]), 0), p[0])
        )
        self.assertTrue(
            jnp.isclose(
                sim.expval(circuit, jnp.array([1.0, 0, 0, 0]), 1),
                p[2] + p[0] * p[1] - 2 * p[0] * p[1] * p[2],
            )
        )

    def test_expval_jit(self):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        sim = StateVectorSimulator()
        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        self.assertTrue(
            jnp.isclose(
                eqx.filter_jit(sim.expval)(circuit, jnp.array([1.0, 0, 0, 0]), 0), p[0]
            )
        )
        self.assertTrue(
            jnp.isclose(
                eqx.filter_jit(sim.expval)(circuit, jnp.array([1.0, 0, 0, 0]), 1),
                p[2] + p[0] * p[1] - 2 * p[0] * p[1] * p[2],
            )
        )

    def test_expval_all(self):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        sim = StateVectorSimulator()
        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        self.assertTrue(
            jnp.allclose(
                sim.expval_all(circuit, jnp.array([1.0, 0, 0, 0])),
                jnp.array([p[0], p[2] + p[0] * p[1] - 2 * p[0] * p[1] * p[2]]),
            )
        )

    def test_pdit_expval_is_basis_value_expectation(self):
        circuit = DiscretePCircuit([PditShift(0, dims=3)])
        sim = StateVectorSimulator()
        compiled = sim.build_circuit(circuit, _thetas(0.0))

        initial = jnp.array([0.0, 0.0, 1.0])

        self.assertTrue(jnp.isclose(sim.expval(compiled, initial, 0), 1.0))
        self.assertTrue(
            jnp.allclose(sim.expval_all(compiled, initial), jnp.array([1.0]))
        )

    def test_expval_all_jit(self):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        sim = StateVectorSimulator()
        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        self.assertTrue(
            jnp.allclose(
                eqx.filter_jit(sim.expval_all)(circuit, jnp.array([1.0, 0, 0, 0])),
                jnp.array([p[0], p[2] + p[0] * p[1] - 2 * p[0] * p[1] * p[2]]),
            )
        )

    def test_gradient(self):
        def expval(params, site):
            gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
            thetas = [jnp.atleast_1d(params[i]) for i in range(3)]
            circuit = DiscretePCircuit(gates)

            sim = StateVectorSimulator()
            circuit = sim.build_circuit(circuit, thetas)

            return sim.expval(circuit, jnp.array([1.0, 0, 0, 0]), site)

        params = jnp.array([1.0, 2.0, -1.0])
        p = 1 / (1 + jnp.exp(-params))
        factor = jnp.array([p[0] * (1 - p[0]), p[1] * (1 - p[1]), p[2] * (1 - p[2])])

        self.assertTrue(
            jnp.allclose(
                jax.grad(expval)(params, 0), jnp.array([1.0, 0.0, 0.0]) * factor
            )
        )
        self.assertTrue(
            jnp.allclose(
                jax.grad(expval)(params, 1),
                jnp.array(
                    [
                        p[1] - 2 * p[1] * p[2],
                        p[0] - 2 * p[0] * p[2],
                        1 - 2 * p[0] * p[1],
                    ]
                )
                * factor,
            )
        )

    def test_gradient_jit(self):
        def loss(circuit, site):
            return sim.expval(circuit, jnp.array([1.0, 0, 0, 0]), site)

        sim = StateVectorSimulator()
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = sim.build_circuit(DiscretePCircuit(gates), thetas)

        p = 1 / (1 + jnp.exp(-params))
        factor = jnp.array([p[0] * (1 - p[0]), p[1] * (1 - p[1]), p[2] * (1 - p[2])])

        grad0 = eqx.filter_jit(eqx.filter_grad(loss))(circuit, 0)
        grad0 = jnp.stack(jax.tree.flatten(grad0)[0]).flatten()

        grad1 = eqx.filter_jit(eqx.filter_grad(loss))(circuit, 1)
        grad1 = jnp.stack(jax.tree.flatten(grad1)[0]).flatten()

        self.assertTrue(jnp.allclose(grad0, jnp.array([1.0, 0.0, 0.0]) * factor))
        self.assertTrue(
            jnp.allclose(
                grad1,
                jnp.array(
                    [
                        p[1] - 2 * p[1] * p[2],
                        p[0] - 2 * p[0] * p[2],
                        1 - 2 * p[0] * p[1],
                    ]
                )
                * factor,
            )
        )


class TestCompiledSamplePCircuit(unittest.TestCase):
    def test_from_pcircuit(self):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates, reps=3)

        built_circuit = CompiledSamplePCircuit.from_pcircuit(circuit, thetas)

        self.assertIsInstance(built_circuit, CompiledSamplePCircuit)
        self.assertEqual(built_circuit.num_pdits, 2)
        self.assertEqual(built_circuit.reps, 3)

        # branch_ops has shape (num_gates, max_branches, max_basis, max_l)
        # branch_ops[:, 0] is identity, branch_ops[:, 1] is operation
        self.assertEqual(built_circuit.branch_ops.shape[0], 3)  # 3 gates
        self.assertEqual(built_circuit.branch_ops.shape[1], 2)  # 2 branches
        self.assertEqual(built_circuit.max_branches, 2)
        self.assertTrue(
            jnp.all(built_circuit.sites == jnp.array([[0, 2], [0, 1], [1, 2]]))
        )
        self.assertTrue(jnp.all(built_circuit.thetas[:, 0] == params))

    def test_to_pcircuit(self):
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates, reps=3)

        new_circuit = CompiledSamplePCircuit.from_pcircuit(circuit, thetas).to_pcircuit(
            circuit
        )

        self.assertIsInstance(new_circuit, DiscretePCircuit)
        self.assertEqual(new_circuit.gates, gates)
        self.assertEqual(new_circuit.num_pdits, 2)
        self.assertEqual(new_circuit.reps, 3)


class TestSampleSimulator(unittest.TestCase):
    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_attributes(self, diff_method):
        sim = SampleSimulator(num_samples=1000, diff_method=diff_method)
        self.assertIs(sim.circuit_backend, CompiledSamplePCircuit)

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_build_circuit(self, diff_method):
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        sim = SampleSimulator(num_samples=1000, diff_method=diff_method)
        built_circuit = sim.build_circuit(circuit, thetas)

        self.assertIsInstance(built_circuit, CompiledSamplePCircuit)

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_sample(self, diff_method):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        num_samples = 10000
        sim = SampleSimulator(num_samples=num_samples, diff_method=diff_method)
        key = jax.random.key(100)

        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        samples = sim.sample(circuit, jnp.array([0, 0]), key)

        p00 = (
            jnp.count_nonzero(jnp.all(samples == jnp.array([0, 0]), axis=1))
            / num_samples
        )
        p01 = (
            jnp.count_nonzero(jnp.all(samples == jnp.array([0, 1]), axis=1))
            / num_samples
        )
        p10 = (
            jnp.count_nonzero(jnp.all(samples == jnp.array([1, 0]), axis=1))
            / num_samples
        )
        p11 = (
            jnp.count_nonzero(jnp.all(samples == jnp.array([1, 1]), axis=1))
            / num_samples
        )

        self.assertTrue(jnp.isclose(p00, (1 - p[0]) * (1 - p[2]), atol=0.05))
        self.assertTrue(jnp.isclose(p01, p[2] * (1 - p[0]), atol=0.05))
        self.assertTrue(
            jnp.isclose(
                p10, p[0] * (1 - p[1]) * (1 - p[2]) + p[0] * p[1] * p[2], atol=0.05
            )
        )
        self.assertTrue(
            jnp.isclose(
                p11, p[0] * p[1] * (1 - p[2]) + p[0] * (1 - p[1]) * p[2], atol=0.05
            )
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_sample_jit(self, diff_method):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        num_samples = 10000
        sim = SampleSimulator(num_samples=num_samples, diff_method=diff_method)
        key = jax.random.key(100)

        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        samples = eqx.filter_jit(sim.sample)(circuit, jnp.array([0, 0]), key)

        p00 = (
            jnp.count_nonzero(jnp.all(samples == jnp.array([0, 0]), axis=1))
            / num_samples
        )
        p01 = (
            jnp.count_nonzero(jnp.all(samples == jnp.array([0, 1]), axis=1))
            / num_samples
        )
        p10 = (
            jnp.count_nonzero(jnp.all(samples == jnp.array([1, 0]), axis=1))
            / num_samples
        )
        p11 = (
            jnp.count_nonzero(jnp.all(samples == jnp.array([1, 1]), axis=1))
            / num_samples
        )

        self.assertTrue(jnp.isclose(p00, (1 - p[0]) * (1 - p[2]), atol=0.05))
        self.assertTrue(jnp.isclose(p01, p[2] * (1 - p[0]), atol=0.05))
        self.assertTrue(
            jnp.isclose(
                p10, p[0] * (1 - p[1]) * (1 - p[2]) + p[0] * p[1] * p[2], atol=0.05
            )
        )
        self.assertTrue(
            jnp.isclose(
                p11, p[0] * p[1] * (1 - p[2]) + p[0] * (1 - p[1]) * p[2], atol=0.05
            )
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_expval(self, diff_method):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        num_samples = 10000
        sim = SampleSimulator(num_samples=num_samples, diff_method=diff_method)
        key = jax.random.key(100)

        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))

        self.assertTrue(
            jnp.isclose(
                sim.expval(circuit, jnp.array([0, 0]), 0, key),
                p[0],
                atol=0.05,
            )
        )
        self.assertTrue(
            jnp.isclose(
                sim.expval(circuit, jnp.array([0, 0]), 1, key),
                p[2] + p[0] * p[1] - 2 * p[0] * p[1] * p[2],
                atol=0.05,
            )
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_expval_jit(self, diff_method):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        num_samples = 10000
        sim = SampleSimulator(num_samples=num_samples, diff_method=diff_method)
        key = jax.random.key(100)

        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        self.assertTrue(
            jnp.isclose(
                eqx.filter_jit(sim.expval)(circuit, jnp.array([0, 0]), 0, key),
                p[0],
                atol=0.05,
            )
        )
        self.assertTrue(
            jnp.isclose(
                eqx.filter_jit(sim.expval)(circuit, jnp.array([0, 0]), 1, key),
                p[2] + p[0] * p[1] - 2 * p[0] * p[1] * p[2],
                atol=0.05,
            )
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_expval_all(self, diff_method):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        num_samples = 10000
        sim = SampleSimulator(num_samples=num_samples, diff_method=diff_method)
        key = jax.random.key(100)

        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        self.assertTrue(
            jnp.allclose(
                sim.expval_all(circuit, jnp.array([0, 0]), key),
                jnp.array([p[0], p[2] + p[0] * p[1] - 2 * p[0] * p[1] * p[2]]),
                atol=0.05,
            )
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_expval_all_jit(self, diff_method):
        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        num_samples = 10000
        sim = SampleSimulator(num_samples=num_samples, diff_method=diff_method)
        key = jax.random.key(100)

        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        self.assertTrue(
            jnp.allclose(
                eqx.filter_jit(sim.expval_all)(circuit, jnp.array([0, 0]), key),
                jnp.array([p[0], p[2] + p[0] * p[1] - 2 * p[0] * p[1] * p[2]]),
                atol=0.05,
            )
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_expval_gradient1(self, diff_method):
        def expval(params, site):
            gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
            thetas = [jnp.atleast_1d(params[i]) for i in range(3)]
            circuit = DiscretePCircuit(gates)

            num_samples = 10000
            sim = SampleSimulator(diff_method=diff_method, num_samples=num_samples)
            key = jax.random.key(100)

            circuit = sim.build_circuit(circuit, thetas)

            return sim.expval(circuit, jnp.array([0, 0]), site, key)

        params = jnp.array([1.0, 2.0, -1.0])
        p = 1 / (1 + jnp.exp(-params))
        factor = jnp.array([p[0] * (1 - p[0]), p[1] * (1 - p[1]), p[2] * (1 - p[2])])

        self.assertTrue(
            jnp.allclose(
                jax.grad(expval)(params, 0),
                jnp.array([1.0, 0.0, 0.0]) * factor,
                atol=0.05,
            )
        )
        self.assertTrue(
            jnp.allclose(
                jax.grad(expval)(params, 1),
                jnp.array(
                    [
                        p[1] - 2 * p[1] * p[2],
                        p[0] - 2 * p[0] * p[2],
                        1 - 2 * p[0] * p[1],
                    ]
                )
                * factor,
                atol=0.05,
            )
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_expval_gradient2(self, diff_method):
        def expval(params, site):
            gates = [
                PNOT(0),
                PSWAP([0, 1]),
                PCNOT([1, 0]),
            ]
            thetas = [jnp.atleast_1d(params[i]) for i in range(3)]
            circuit = DiscretePCircuit(gates)

            num_samples = 10000
            sim = SampleSimulator(diff_method=diff_method, num_samples=num_samples)
            key = jax.random.key(100)

            circuit = sim.build_circuit(circuit, thetas)

            return -sim.expval(circuit, jnp.array([0, 0]), site, key)

        params = jnp.array([1.0, 2.0, -1.0])
        p = 1 / (1 + jnp.exp(-params))
        factor = jnp.array([p[0] * (1 - p[0]), p[1] * (1 - p[1]), p[2] * (1 - p[2])])

        self.assertTrue(
            jnp.allclose(
                jax.grad(expval)(params, 0),
                -jnp.array([1 - p[1] + p[1] * p[2], p[0] * (-1 + p[2]), p[0] * p[1]])
                * factor,
                atol=0.05,
            )
        )
        self.assertTrue(
            jnp.allclose(
                jax.grad(expval)(params, 1),
                -jnp.array([p[1], p[0], 0]) * factor,
                atol=0.05,
            )
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_expval_all_gradient(self, diff_method):
        def loss(params):
            gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
            thetas = [jnp.atleast_1d(params[i]) for i in range(3)]
            circuit = DiscretePCircuit(gates)

            num_samples = 10000
            sim = SampleSimulator(diff_method=diff_method, num_samples=num_samples)
            key = jax.random.key(100)

            circuit = sim.build_circuit(circuit, thetas)

            expvals = sim.expval_all(circuit, jnp.array([0, 0]), key)
            return expvals[0] - expvals[1]

        params = jnp.array([1.0, 2.0, -1.0])
        p = 1 / (1 + jnp.exp(-params))
        factor = jnp.array([p[0] * (1 - p[0]), p[1] * (1 - p[1]), p[2] * (1 - p[2])])

        self.assertTrue(
            jnp.allclose(
                jax.grad(loss)(params),
                jnp.array(
                    [
                        1 - p[1] + 2 * p[1] * p[2],
                        -p[0] + 2 * p[0] * p[2],
                        -1 + 2 * p[0] * p[1],
                    ]
                )
                * factor,
                atol=0.05,
            )
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_gradient_jit(self, diff_method):
        def loss(circuit, site):
            return sim.expval(circuit, jnp.array([0, 0]), site, key)

        params = jnp.array([1.0, 2.0, -1.0])
        gates = [PNOT(0), PCNOT([0, 1]), PNOT(1)]
        thetas = _thetas(1.0, 2.0, -1.0)
        circuit = DiscretePCircuit(gates)

        num_samples = 10000
        sim = SampleSimulator(diff_method=diff_method, num_samples=num_samples)
        key = jax.random.key(100)

        circuit = sim.build_circuit(circuit, thetas)

        p = 1 / (1 + jnp.exp(-params))
        factor = jnp.array([p[0] * (1 - p[0]), p[1] * (1 - p[1]), p[2] * (1 - p[2])])

        grad0 = eqx.filter_jit(eqx.filter_grad(loss))(circuit, 0)
        grad0 = jnp.stack(jax.tree.flatten(grad0)[0]).flatten()

        grad1 = eqx.filter_jit(eqx.filter_grad(loss))(circuit, 1)
        grad1 = jnp.stack(jax.tree.flatten(grad1)[0]).flatten()

        self.assertTrue(
            jnp.allclose(grad0, jnp.array([1.0, 0.0, 0.0]) * factor, atol=0.05)
        )
        self.assertTrue(
            jnp.allclose(
                grad1,
                jnp.array(
                    [
                        p[1] - 2 * p[1] * p[2],
                        p[0] - 2 * p[0] * p[2],
                        1 - 2 * p[0] * p[1],
                    ]
                )
                * factor,
                atol=0.05,
            )
        )

    @parameterized.expand(["param_shift_inf", "param_shift_single"])
    def test_reps_gradient_error(self, diff_method):
        def expval(param):
            circuit = DiscretePCircuit([PNOT(0)], reps=2)
            thetas = [jnp.atleast_1d(param)]

            sim = SampleSimulator(diff_method=diff_method, num_samples=100)
            key = jax.random.key(100)

            circuit = sim.build_circuit(circuit, thetas)

            return sim.expval(circuit, jnp.array([0, 0]), 0, key)

        with self.assertRaises(NotImplementedError):
            jax.grad(expval)(jnp.array(0.0))

    def test_wrong_diff_method_error(self):
        with self.assertRaises(ValueError):
            SampleSimulator(diff_method="foo")

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_sample_input_validation(self, diff_method):
        gates = [PNOT(0)]
        thetas = _thetas(1.0)
        circuit = DiscretePCircuit(gates)

        sim = SampleSimulator(num_samples=100, diff_method=diff_method)
        key = jax.random.key(0)

        circuit = sim.build_circuit(circuit, thetas)

        x = jnp.array([0, 0])  # circuit.num_pdits == 1

        with self.assertRaises(ValueError) as context:
            sim.sample(circuit, x, key)
        self.assertIn("Malformed bitstring", str(context.exception))


class TestPditSampleSimulator(unittest.TestCase):

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_pdit_shift_deterministic(self, diff_method):
        """Test deterministic PditShift always shifts by 1."""
        circuit = DiscretePCircuit([PditShift(0, dims=3)])  # p ≈ 1
        thetas = _thetas(100.0)

        sim = SampleSimulator(num_samples=100, diff_method=diff_method)
        compiled = sim.build_circuit(circuit, thetas)
        key = jax.random.key(0)

        # Starting from 0, should always get 1
        samples = sim.sample(compiled, jnp.array([0]), key)
        self.assertTrue(jnp.all(samples == 1))

        # Starting from 2, should always get 0 (2+1 mod 3)
        samples = sim.sample(compiled, jnp.array([2]), key)
        self.assertTrue(jnp.all(samples == 0))

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_pdit_shift_probabilistic(self, diff_method):
        circuit = DiscretePCircuit([PditShift(0, dims=3)])
        thetas = _thetas(0.0)

        sim = SampleSimulator(num_samples=1000, diff_method=diff_method)
        compiled = sim.build_circuit(circuit, thetas)
        key = jax.random.key(42)

        samples = sim.sample(compiled, jnp.array([0]), key)
        mean = jnp.mean(samples)
        self.assertTrue(
            jnp.abs(mean - 0.5) < 0.1,
            f"Expected mean ≈ 0.5, got {mean}",
        )

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_pdit_swap_deterministic(self, diff_method):
        circuit = DiscretePCircuit([PditSWAP([0, 1], dims=3)])  # p ≈ 1
        thetas = _thetas(100.0)

        sim = SampleSimulator(num_samples=100, diff_method=diff_method)
        compiled = sim.build_circuit(circuit, thetas)
        key = jax.random.key(0)

        samples = sim.sample(compiled, jnp.array([1, 2]), key)
        self.assertTrue(jnp.all(samples == jnp.array([2, 1])))

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_padding_sites_do_not_read_last_pdit(self, diff_method):
        circuit = DiscretePCircuit(
            [
                PNOT(0),
                PditSWAP([1, 2], dims=3),
            ]
        )
        thetas = _thetas(-100.0, -100.0)

        sim = SampleSimulator(num_samples=100, diff_method=diff_method)
        compiled = sim.build_circuit(circuit, thetas)
        samples = sim.sample(compiled, jnp.array([1, 0, 2]), jax.random.key(0))

        self.assertTrue(jnp.all(samples == jnp.array([1, 0, 2])))

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_padding_sites_do_not_corrupt_single_pdit_gate(self, diff_method):
        circuit = DiscretePCircuit(
            [
                PditShift(0, dims=3),
                PditSWAP([0, 1], dims=3),
            ]
        )
        thetas = _thetas(100.0, -100.0)

        sim = SampleSimulator(num_samples=100, diff_method=diff_method)
        compiled = sim.build_circuit(circuit, thetas)
        samples = sim.sample(compiled, jnp.array([0, 2]), jax.random.key(0))

        self.assertTrue(jnp.all(samples == jnp.array([1, 2])))

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_mixed_binary_pdit_circuit(self, diff_method):
        circuit = DiscretePCircuit(
            [
                PNOT(0),  # flip site 0
                PditShift(1, dims=3),  # shift site 1
            ]
        )
        thetas = _thetas(100.0, 100.0)

        sim = SampleSimulator(num_samples=100, diff_method=diff_method)
        compiled = sim.build_circuit(circuit, thetas)
        key = jax.random.key(0)

        # [0, 0] -> [1, 1]
        samples = sim.sample(compiled, jnp.array([0, 0]), key)
        self.assertTrue(jnp.all(samples == jnp.array([1, 1])))

    @parameterized.expand(
        ["param_shift_inf", "param_shift_single", "param_shift_filter"]
    )
    def test_pdit_expval_matches_statevector(self, diff_method):
        circuit = DiscretePCircuit([PditShift(0, dims=3)])
        thetas = _thetas(0.5)

        sv_sim = StateVectorSimulator()
        sv_circuit = sv_sim.build_circuit(circuit, thetas)
        initial_sv = jnp.zeros(3).at[0].set(1.0)
        exact_expval = sv_sim.expval_all(sv_circuit, initial_sv)

        sample_sim = SampleSimulator(num_samples=10000, diff_method=diff_method)
        compiled = sample_sim.build_circuit(circuit, thetas)
        key = jax.random.key(123)
        sample_expval = sample_sim.expval_all(compiled, jnp.array([0]), key)

        self.assertTrue(
            jnp.allclose(sample_expval, exact_expval, atol=0.05),
            f"Sample expval {sample_expval} != exact {exact_expval}",
        )

    def test_pdit_compiled_structure(self):
        circuit = DiscretePCircuit(
            [
                PditShift(0, dims=3),
                PditSWAP([0, 1], dims=3),
            ]
        )
        thetas = _thetas(1.0, 2.0)

        compiled = CompiledSamplePCircuit.from_pcircuit(circuit, thetas)

        self.assertEqual(compiled.num_pdits, 2)
        # PditShift has basis_size 3, PditSWAP has basis_size 9
        # But with extension, both are extended to locality 2
        self.assertEqual(compiled.branch_ops.shape[0], 2)  # 2 gates
        self.assertEqual(compiled.dims.shape, (2, 2))

    def test_mixed_dims_compiled_structure(self):
        circuit = DiscretePCircuit(
            [
                PNOT(0),
                PditShift(1, dims=4),
            ]
        )
        thetas = _thetas(1.0, 2.0)

        compiled = CompiledSamplePCircuit.from_pcircuit(circuit, thetas)

        self.assertEqual(compiled.num_pdits, 2)
        self.assertEqual(compiled.branch_ops.shape[0], 2)  # 2 gates
        # PNOT: [2, 2] (padded), PditShift: [4, 2] (padded)
        self.assertEqual(compiled.dims[0, 0], 2)
        self.assertEqual(compiled.dims[1, 0], 4)
