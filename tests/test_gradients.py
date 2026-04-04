"""Test that sample gradients are unbiased estimators of analytic gradients."""

import unittest

import equinox as eqx
import jax
import jax.numpy as jnp

from torx import (
    DiscretePCircuit,
    PCNOT,
    PCSWAP,
    PDEMUX,
    PditShift,
    PditSWAP,
    PJUMP,
    PMultiCNOT,
    PNOT,
    POR,
    PReset,
    PSWAP,
    SampleSimulator,
    StateVectorSimulator,
)

# (name, class, num_sites)
ALL_BINARY_GATES = [
    ("PNOT", PNOT, 1),
    ("PReset", PReset, 1),
    ("PSWAP", PSWAP, 2),
    ("PCNOT", PCNOT, 2),
    ("PJUMP", PJUMP, 2),
    ("PDEMUX", PDEMUX, 2),
    ("POR", POR, 2),
    ("PMultiCNOT", PMultiCNOT, 3),
    ("PCSWAP", PCSWAP, 3),
]

DIFF_METHODS = ["param_shift_inf", "param_shift_single", "param_shift_filter"]


def get_sites(num_sites):
    if num_sites == 1:
        return 0
    return list(range(num_sites))


def get_initial_sv(num_sites):
    return jnp.zeros(2**num_sites).at[0].set(1.0)


def get_initial_basis(num_sites):
    return jnp.zeros(num_sites, dtype=jnp.int32)


class TestGradientUnbiasedness(unittest.TestCase):
    def test_single_gate_gradient_all_gates_all_methods(self):
        num_samples = 10000
        num_runs = 50

        sv_sim = StateVectorSimulator()

        for gate_name, gate_cls, num_sites in ALL_BINARY_GATES:
            initial_sv = get_initial_sv(num_sites)
            initial_basis = get_initial_basis(num_sites)

            @eqx.filter_jit
            def sv_loss(circ):
                return sv_sim.expval_all(circ, initial_sv).sum()

            @eqx.filter_jit
            def sample_loss(circ, key):
                return sample_sim.expval_all(circ, initial_basis, key).sum()

            for diff_method in DIFF_METHODS:
                with self.subTest(gate=gate_name, diff_method=diff_method):
                    theta = jnp.array(0.5)
                    sites = get_sites(num_sites)
                    gate = gate_cls(theta, sites)
                    circuit = DiscretePCircuit([gate])

                    # Analytic gradient
                    analytic_grad = eqx.filter_grad(sv_loss)(circuit)
                    analytic_val = jax.tree.leaves(analytic_grad)[0]

                    # Sample gradients
                    sample_sim = SampleSimulator(
                        num_samples=num_samples, diff_method=diff_method
                    )
                    compiled = sample_sim.build_circuit(circuit)

                    sample_grads = []
                    for i in range(num_runs):
                        key = jax.random.key(i)
                        grad = eqx.filter_grad(sample_loss)(compiled, key)
                        sample_grads.append(grad.thetas)

                    sample_grads = jnp.stack(sample_grads)
                    mean_grad = jnp.mean(sample_grads)

                    self.assertTrue(
                        jnp.allclose(mean_grad, analytic_val, atol=0.02),
                        f"{gate_name}/{diff_method}: mean={mean_grad:.4f}, "
                        f"analytic={analytic_val:.4f}",
                    )

    def test_gradient_at_various_theta(self):
        theta_values = [-2.0, 0.0, 2.0]
        num_samples = 10000
        num_runs = 30

        sv_sim = StateVectorSimulator()

        for gate_name, gate_cls, num_sites in ALL_BINARY_GATES:
            initial_sv = get_initial_sv(num_sites)
            initial_basis = get_initial_basis(num_sites)

            @eqx.filter_jit
            def sv_loss(circ):
                return sv_sim.expval_all(circ, initial_sv).sum()

            @eqx.filter_jit
            def sample_loss(circ, key):
                return sample_sim.expval_all(circ, initial_basis, key).sum()

            for diff_method in DIFF_METHODS:
                sample_sim = SampleSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )

                for theta_val in theta_values:
                    with self.subTest(
                        gate=gate_name, diff_method=diff_method, theta=theta_val
                    ):
                        theta = jnp.array(theta_val)
                        sites = get_sites(num_sites)
                        gate = gate_cls(theta, sites)
                        circuit = DiscretePCircuit([gate])

                        analytic_grad = eqx.filter_grad(sv_loss)(circuit)
                        analytic_val = jax.tree.leaves(analytic_grad)[0]

                        # Sample gradients
                        compiled = sample_sim.build_circuit(circuit)

                        sample_grads = []
                        for i in range(num_runs):
                            key = jax.random.key(i + 1000)
                            grad = eqx.filter_grad(sample_loss)(compiled, key)
                            sample_grads.append(grad.thetas)

                        sample_grads = jnp.stack(sample_grads)
                        mean_grad = jnp.mean(sample_grads)

                        self.assertTrue(
                            jnp.allclose(mean_grad, analytic_val, atol=0.03),
                            f"{gate_name}/{diff_method}/theta={theta_val}: "
                            f"mean={mean_grad:.4f}, analytic={analytic_val:.4f}",
                        )

    def test_multi_gate_circuit_gradient(self):
        num_samples = 10000
        num_runs = 50

        params = jnp.array([0.5, -0.3, 0.8, 0.2, -0.5])
        circuit = DiscretePCircuit(
            [
                PNOT(params[0], 0),
                PNOT(params[1], 1),
                PCNOT(params[2], [0, 1]),
                PSWAP(params[3], [0, 1]),
                PNOT(params[4], 0),
            ]
        )

        sv_sim = StateVectorSimulator()
        initial_sv = jnp.array([1.0, 0.0, 0.0, 0.0])

        @eqx.filter_jit
        def sv_loss(circ):
            return sv_sim.expval_all(circ, initial_sv).sum()

        analytic_grad = eqx.filter_grad(sv_loss)(circuit)
        analytic_vals = jnp.array(jax.tree.leaves(analytic_grad))

        for diff_method in DIFF_METHODS:
            with self.subTest(diff_method=diff_method):
                sample_sim = SampleSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )
                compiled = sample_sim.build_circuit(circuit)
                initial_basis = jnp.array([0, 0], dtype=jnp.int32)

                @eqx.filter_jit
                def sample_loss(circ, key):
                    return sample_sim.expval_all(circ, initial_basis, key).sum()

                sample_grads = []
                for i in range(num_runs):
                    key = jax.random.key(i)
                    grad = eqx.filter_grad(sample_loss)(compiled, key)
                    sample_grads.append(grad.thetas)

                sample_grads = jnp.stack(sample_grads)
                mean_grads = jnp.mean(sample_grads, axis=0).flatten()

                self.assertTrue(
                    jnp.allclose(mean_grads, analytic_vals, atol=0.03),
                    f"{diff_method}: mean={mean_grads}, analytic={analytic_vals}",
                )

    def test_three_site_circuit_gradient(self):
        num_samples = 10000
        num_runs = 50

        params = jnp.array([0.5, -0.3, 0.8])
        circuit = DiscretePCircuit(
            [
                PNOT(params[0], 0),
                PMultiCNOT(params[1], [0, 1, 2]),
                PCSWAP(params[2], [0, 1, 2]),
            ]
        )

        sv_sim = StateVectorSimulator()
        initial_sv = jnp.zeros(8).at[0].set(1.0)

        @eqx.filter_jit
        def sv_loss(circ):
            return sv_sim.expval_all(circ, initial_sv).sum()

        analytic_grad = eqx.filter_grad(sv_loss)(circuit)
        analytic_vals = jnp.array(jax.tree.leaves(analytic_grad))

        for diff_method in DIFF_METHODS:
            with self.subTest(diff_method=diff_method):
                sample_sim = SampleSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )
                compiled = sample_sim.build_circuit(circuit)
                initial_basis = jnp.array([0, 0, 0], dtype=jnp.int32)

                @eqx.filter_jit
                def sample_loss(circ, key):
                    return sample_sim.expval_all(circ, initial_basis, key).sum()

                sample_grads = []
                for i in range(num_runs):
                    key = jax.random.key(i)
                    grad = eqx.filter_grad(sample_loss)(compiled, key)
                    sample_grads.append(grad.thetas)

                sample_grads = jnp.stack(sample_grads)
                mean_grads = jnp.mean(sample_grads, axis=0).flatten()

                self.assertTrue(
                    jnp.allclose(mean_grads, analytic_vals, atol=0.03),
                    f"{diff_method}: mean={mean_grads}, analytic={analytic_vals}",
                )


class TestPditGradientUnbiasedness(unittest.TestCase):
    def test_pdit_shift_gradient(self):
        num_samples = 10000
        num_runs = 50

        sv_sim = StateVectorSimulator()

        for dim in [3, 4, 5]:
            initial_sv = jnp.zeros(dim).at[0].set(1.0)
            initial_basis = jnp.array([0], dtype=jnp.int32)

            @eqx.filter_jit
            def sv_loss(circ):
                return sv_sim.expval_all(circ, initial_sv).sum()

            @eqx.filter_jit
            def sample_loss(circ, key):
                return sample_sim.expval_all(circ, initial_basis, key).sum()

            for diff_method in DIFF_METHODS:
                with self.subTest(dim=dim, diff_method=diff_method):
                    theta = jnp.array(0.5)
                    gate = PditShift(theta, 0, dims=dim)
                    circuit = DiscretePCircuit([gate])

                    analytic_grad = eqx.filter_grad(sv_loss)(circuit)
                    analytic_val = jax.tree.leaves(analytic_grad)[0]

                    sample_sim = SampleSimulator(
                        num_samples=num_samples, diff_method=diff_method
                    )
                    compiled = sample_sim.build_circuit(circuit)

                    sample_grads = []
                    for i in range(num_runs):
                        key = jax.random.key(i)
                        grad = eqx.filter_grad(sample_loss)(compiled, key)
                        sample_grads.append(grad.thetas)

                    sample_grads = jnp.stack(sample_grads)
                    mean_grad = jnp.mean(sample_grads)

                    self.assertTrue(
                        jnp.allclose(mean_grad, analytic_val, atol=0.03),
                        f"PditShift(dim={dim})/{diff_method}: mean={mean_grad:.4f}, "
                        f"analytic={analytic_val:.4f}",
                    )

    def test_pdit_swap_gradient(self):
        """Test PditSWAP gradient matches exact for various dimensions."""
        num_samples = 10000
        num_runs = 50

        sv_sim = StateVectorSimulator()

        for dim in [3, 4]:
            d2 = dim * dim
            initial_sv = jnp.zeros(d2).at[dim].set(1.0)
            initial_basis = jnp.array([1, 0], dtype=jnp.int32)

            @eqx.filter_jit
            def sv_loss(circ):
                return sv_sim.expval_all(circ, initial_sv).sum()

            for diff_method in DIFF_METHODS:
                with self.subTest(dim=dim, diff_method=diff_method):
                    theta = jnp.array(0.5)
                    gate = PditSWAP(theta, [0, 1], dims=dim)
                    circuit = DiscretePCircuit([gate])

                    analytic_grad = eqx.filter_grad(sv_loss)(circuit)
                    analytic_val = jax.tree.leaves(analytic_grad)[0]

                    sample_sim = SampleSimulator(
                        num_samples=num_samples, diff_method=diff_method
                    )
                    compiled = sample_sim.build_circuit(circuit)

                    @eqx.filter_jit
                    def sample_loss(circ, key):
                        return sample_sim.expval_all(circ, initial_basis, key).sum()

                    sample_grads = []
                    for i in range(num_runs):
                        key = jax.random.key(i)
                        grad = eqx.filter_grad(sample_loss)(compiled, key)
                        sample_grads.append(grad.thetas)

                    sample_grads = jnp.stack(sample_grads)
                    mean_grad = jnp.mean(sample_grads)

                    self.assertTrue(
                        jnp.allclose(mean_grad, analytic_val, atol=0.03),
                        f"PditSWAP(dim={dim})/{diff_method}: mean={mean_grad:.4f}, "
                        f"analytic={analytic_val:.4f}",
                    )

    def test_mixed_binary_pdit_circuit_gradient(self):
        num_samples = 10000
        num_runs = 50

        params = jnp.array([0.5, -0.3, 0.8])
        circuit = DiscretePCircuit(
            [
                PNOT(params[0], 0),
                PditShift(params[1], 2, dims=3),
                PCNOT(params[2], [0, 1]),
            ]
        )

        sv_sim = StateVectorSimulator()
        # 2 * 2 * 3 = 12 states
        initial_sv = jnp.zeros(12).at[0].set(1.0)

        @eqx.filter_jit
        def sv_loss(circ):
            return sv_sim.expval_all(circ, initial_sv).sum()

        analytic_grad = eqx.filter_grad(sv_loss)(circuit)
        analytic_vals = jnp.array(jax.tree.leaves(analytic_grad))

        for diff_method in DIFF_METHODS:
            with self.subTest(diff_method=diff_method):
                sample_sim = SampleSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )
                compiled = sample_sim.build_circuit(circuit)
                initial_basis = jnp.array([0, 0, 0], dtype=jnp.int32)

                @eqx.filter_jit
                def sample_loss(circ, key):
                    return sample_sim.expval_all(circ, initial_basis, key).sum()

                sample_grads = []
                for i in range(num_runs):
                    key = jax.random.key(i)
                    grad = eqx.filter_grad(sample_loss)(compiled, key)
                    sample_grads.append(grad.thetas)

                sample_grads = jnp.stack(sample_grads)
                mean_grads = jnp.mean(sample_grads, axis=0).flatten()

                self.assertTrue(
                    jnp.allclose(mean_grads, analytic_vals, atol=0.03),
                    f"Mixed binary+pdit/{diff_method}: "
                    f"mean={mean_grads}, analytic={analytic_vals}",
                )

    def test_pdit_only_multi_gate_circuit(self):
        """Test gradient for circuit with multiple pdit gates only."""
        num_samples = 10000
        num_runs = 50

        params = jnp.array([0.5, -0.3])
        circuit = DiscretePCircuit(
            [
                PditShift(params[0], 0, dims=4),
                PditSWAP(params[1], [0, 1], dims=4),
            ]
        )

        sv_sim = StateVectorSimulator()
        initial_sv = jnp.zeros(16).at[0].set(1.0)

        @eqx.filter_jit
        def sv_loss(circ):
            return sv_sim.expval_all(circ, initial_sv).sum()

        analytic_grad = eqx.filter_grad(sv_loss)(circuit)
        analytic_vals = jnp.array(jax.tree.leaves(analytic_grad))

        for diff_method in DIFF_METHODS:
            with self.subTest(diff_method=diff_method):
                sample_sim = SampleSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )
                compiled = sample_sim.build_circuit(circuit)
                initial_basis = jnp.array([0, 0], dtype=jnp.int32)

                @eqx.filter_jit
                def sample_loss(circ, key):
                    return sample_sim.expval_all(circ, initial_basis, key).sum()

                sample_grads = []
                for i in range(num_runs):
                    key = jax.random.key(i)
                    grad = eqx.filter_grad(sample_loss)(compiled, key)
                    sample_grads.append(grad.thetas)

                sample_grads = jnp.stack(sample_grads)
                mean_grads = jnp.mean(sample_grads, axis=0).flatten()

                self.assertTrue(
                    jnp.allclose(mean_grads, analytic_vals, atol=0.03),
                    f"Pdit-only/{diff_method}: "
                    f"mean={mean_grads}, analytic={analytic_vals}",
                )


class TestGradientVariance(unittest.TestCase):

    def test_variance_decreases_with_samples(self):
        theta = jnp.array(0.5)
        circuit = DiscretePCircuit([PNOT(theta, 0), PCNOT(jnp.array(0.3), [0, 1])])

        sample_counts = [1000, 5000, 20000]
        num_runs = 30
        initial_basis = jnp.array([0, 0], dtype=jnp.int32)

        variances = []
        for num_samples in sample_counts:
            sample_sim = SampleSimulator(
                num_samples=num_samples, diff_method="param_shift_filter"
            )
            compiled = sample_sim.build_circuit(circuit)

            @eqx.filter_jit
            def sample_loss(circ, key):
                return sample_sim.expval_all(circ, initial_basis, key).sum()

            grads = []
            for i in range(num_runs):
                key = jax.random.key(i)
                grad = eqx.filter_grad(sample_loss)(compiled, key)
                grads.append(grad.thetas[0])

            grads = jnp.array(grads)
            variances.append(jnp.var(grads))

        self.assertLess(
            variances[-1],
            variances[0],
            f"Variance should decrease with more samples: {variances}",
        )


if __name__ == "__main__":
    unittest.main()
