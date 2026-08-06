import unittest

import jax
import jax.numpy as jnp

from torx.psc import (
    BranchingSimulator,
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
    StateVectorSimulator,
)
from torx.psc.simulation.sampled import _single_shifted_theta

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


def make_thetas(*values):
    """Build an external theta list (one ``(1,)`` array per 2-branch gate)."""
    return [jnp.atleast_1d(v) for v in values]


def test_single_shifted_theta_is_stable_for_saturated_logits():
    theta = jnp.array([-100.0, -20.0, 0.0, 20.0, 100.0], dtype=jnp.float32)

    shifted = _single_shifted_theta(theta)
    expected = theta - jnp.logaddexp(jnp.log(jnp.array(2.0)), -theta)

    assert jnp.all(jnp.isfinite(shifted))
    assert jnp.allclose(shifted, expected)


def test_param_shift_single_gradient_is_finite_for_saturated_logits():
    circuit = DiscretePCircuit([PNOT(0)])
    simulator = BranchingSimulator(num_samples=32, diff_method="param_shift_single")
    initial_basis = jnp.array([0], dtype=jnp.int32)

    def loss(theta):
        compiled = simulator.build_circuit(circuit, [theta])
        return simulator.expval_all(compiled, initial_basis, jax.random.key(0)).sum()

    for theta in [-100.0, -20.0, 0.0, 20.0, 100.0]:
        gradient = jax.grad(loss)(jnp.array([theta], dtype=jnp.float32))
        assert jnp.all(jnp.isfinite(gradient))


class TestGradientUnbiasedness(unittest.TestCase):
    def test_single_gate_gradient_all_gates_all_methods(self):
        num_samples = 10000
        num_runs = 50

        sv_sim = StateVectorSimulator()

        for gate_name, gate_cls, num_sites in ALL_BINARY_GATES:
            initial_sv = get_initial_sv(num_sites)
            initial_basis = get_initial_basis(num_sites)

            sites = get_sites(num_sites)
            gate = gate_cls(sites)
            circuit = DiscretePCircuit([gate])

            def sv_loss(thetas):
                compiled = sv_sim.build_circuit(circuit, thetas)
                return sv_sim.expval_all(compiled, initial_sv).sum()

            for diff_method in DIFF_METHODS:
                with self.subTest(gate=gate_name, diff_method=diff_method):
                    thetas = make_thetas(0.5)

                    # Analytic gradient
                    analytic_grad = jax.grad(sv_loss)(thetas)
                    analytic_val = jax.tree.leaves(analytic_grad)[0]

                    # Sample gradients
                    sample_sim = BranchingSimulator(
                        num_samples=num_samples, diff_method=diff_method
                    )

                    def sample_loss(thetas, key):
                        compiled = sample_sim.build_circuit(circuit, thetas)
                        return sample_sim.expval_all(compiled, initial_basis, key).sum()

                    sample_grad = jax.jit(jax.grad(sample_loss))

                    sample_grads = []
                    for i in range(num_runs):
                        key = jax.random.key(i)
                        grad = sample_grad(thetas, key)
                        sample_grads.append(jnp.stack(grad))

                    sample_grads = jnp.stack(sample_grads)
                    mean_grad = jnp.mean(sample_grads)

                    self.assertTrue(
                        jnp.allclose(mean_grad, analytic_val, atol=0.02),
                        f"{gate_name}/{diff_method}: mean={mean_grad:.4f}, "
                        f"analytic={jnp.squeeze(analytic_val):.4f}",
                    )

    def test_gradient_at_various_theta(self):
        theta_values = [-2.0, 0.0, 2.0]
        num_samples = 10000
        num_runs = 30

        sv_sim = StateVectorSimulator()

        for gate_name, gate_cls, num_sites in ALL_BINARY_GATES:
            initial_sv = get_initial_sv(num_sites)
            initial_basis = get_initial_basis(num_sites)

            sites = get_sites(num_sites)
            gate = gate_cls(sites)
            circuit = DiscretePCircuit([gate])

            def sv_loss(thetas):
                compiled = sv_sim.build_circuit(circuit, thetas)
                return sv_sim.expval_all(compiled, initial_sv).sum()

            for diff_method in DIFF_METHODS:
                sample_sim = BranchingSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )

                def sample_loss(thetas, key):
                    compiled = sample_sim.build_circuit(circuit, thetas)
                    return sample_sim.expval_all(compiled, initial_basis, key).sum()

                sample_grad = jax.jit(jax.grad(sample_loss))

                for theta_val in theta_values:
                    with self.subTest(
                        gate=gate_name, diff_method=diff_method, theta=theta_val
                    ):
                        thetas = make_thetas(theta_val)

                        analytic_grad = jax.grad(sv_loss)(thetas)
                        analytic_val = jax.tree.leaves(analytic_grad)[0]

                        # Sample gradients
                        sample_grads = []
                        for i in range(num_runs):
                            key = jax.random.key(i + 1000)
                            grad = sample_grad(thetas, key)
                            sample_grads.append(jnp.stack(grad))

                        sample_grads = jnp.stack(sample_grads)
                        mean_grad = jnp.mean(sample_grads)

                        self.assertTrue(
                            jnp.allclose(mean_grad, analytic_val, atol=0.03),
                            f"{gate_name}/{diff_method}/theta={theta_val}: "
                            f"mean={mean_grad:.4f}, "
                            f"analytic={jnp.squeeze(analytic_val):.4f}",
                        )

    def test_multi_gate_circuit_gradient(self):
        num_samples = 10000
        num_runs = 50

        thetas = make_thetas(0.5, -0.3, 0.8, 0.2, -0.5)
        circuit = DiscretePCircuit(
            [
                PNOT(0),
                PNOT(1),
                PCNOT([0, 1]),
                PSWAP([0, 1]),
                PNOT(0),
            ]
        )

        sv_sim = StateVectorSimulator()
        initial_sv = jnp.array([1.0, 0.0, 0.0, 0.0])

        def sv_loss(thetas):
            compiled = sv_sim.build_circuit(circuit, thetas)
            return sv_sim.expval_all(compiled, initial_sv).sum()

        analytic_grad = jax.grad(sv_loss)(thetas)
        analytic_vals = jnp.array(jax.tree.leaves(analytic_grad)).flatten()

        for diff_method in DIFF_METHODS:
            with self.subTest(diff_method=diff_method):
                sample_sim = BranchingSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )
                initial_basis = jnp.array([0, 0], dtype=jnp.int32)

                def sample_loss(thetas, key):
                    compiled = sample_sim.build_circuit(circuit, thetas)
                    return sample_sim.expval_all(compiled, initial_basis, key).sum()

                sample_grad = jax.jit(jax.grad(sample_loss))

                sample_grads = []
                for i in range(num_runs):
                    key = jax.random.key(i)
                    grad = sample_grad(thetas, key)
                    sample_grads.append(jnp.stack(grad).flatten())

                sample_grads = jnp.stack(sample_grads)
                mean_grads = jnp.mean(sample_grads, axis=0).flatten()

                self.assertTrue(
                    jnp.allclose(mean_grads, analytic_vals, atol=0.03),
                    f"{diff_method}: mean={mean_grads}, analytic={analytic_vals}",
                )

    def test_three_site_circuit_gradient(self):
        num_samples = 10000
        num_runs = 50

        thetas = make_thetas(0.5, -0.3, 0.8)
        circuit = DiscretePCircuit(
            [
                PNOT(0),
                PMultiCNOT([0, 1, 2]),
                PCSWAP([0, 1, 2]),
            ]
        )

        sv_sim = StateVectorSimulator()
        initial_sv = jnp.zeros(8).at[0].set(1.0)

        def sv_loss(thetas):
            compiled = sv_sim.build_circuit(circuit, thetas)
            return sv_sim.expval_all(compiled, initial_sv).sum()

        analytic_grad = jax.grad(sv_loss)(thetas)
        analytic_vals = jnp.array(jax.tree.leaves(analytic_grad)).flatten()

        for diff_method in DIFF_METHODS:
            with self.subTest(diff_method=diff_method):
                sample_sim = BranchingSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )
                initial_basis = jnp.array([0, 0, 0], dtype=jnp.int32)

                def sample_loss(thetas, key):
                    compiled = sample_sim.build_circuit(circuit, thetas)
                    return sample_sim.expval_all(compiled, initial_basis, key).sum()

                sample_grad = jax.jit(jax.grad(sample_loss))

                sample_grads = []
                for i in range(num_runs):
                    key = jax.random.key(i)
                    grad = sample_grad(thetas, key)
                    sample_grads.append(jnp.stack(grad).flatten())

                sample_grads = jnp.stack(sample_grads)
                mean_grads = jnp.mean(sample_grads, axis=0).flatten()

                self.assertTrue(
                    jnp.allclose(mean_grads, analytic_vals, atol=0.03),
                    f"{diff_method}: mean={mean_grads}, analytic={analytic_vals}",
                )

    def test_param_shift_filter_sums_repeated_gate_contributions(self):
        circuit = DiscretePCircuit([PReset(0)], reps=4)
        thetas = make_thetas(0.0)
        initial_sv = jnp.array([0.0, 1.0])
        initial_basis = jnp.array([1], dtype=jnp.int32)

        sv_sim = StateVectorSimulator()

        def sv_loss(thetas):
            compiled = sv_sim.build_circuit(circuit, thetas)
            return sv_sim.expval_all(compiled, initial_sv).sum()

        exact_grad = jax.grad(sv_loss)(thetas)[0]

        sample_sim = BranchingSimulator(
            num_samples=10000,
            diff_method="param_shift_filter",
        )

        def sample_loss(thetas):
            compiled = sample_sim.build_circuit(circuit, thetas)
            return sample_sim.expval_all(
                compiled, initial_basis, jax.random.key(0)
            ).sum()

        filter_grad = jax.grad(sample_loss)(thetas)[0]

        self.assertTrue(jnp.allclose(filter_grad, exact_grad, atol=0.03))


class TestPditGradientUnbiasedness(unittest.TestCase):
    def test_pdit_shift_gradient(self):
        num_samples = 10000
        num_runs = 50

        sv_sim = StateVectorSimulator()

        for dim in [3, 4, 5]:
            initial_sv = jnp.zeros(dim).at[0].set(1.0)
            initial_basis = jnp.array([0], dtype=jnp.int32)

            gate = PditShift(0, dims=dim)
            circuit = DiscretePCircuit([gate])

            def sv_loss(thetas):
                compiled = sv_sim.build_circuit(circuit, thetas)
                return sv_sim.expval_all(compiled, initial_sv).sum()

            for diff_method in DIFF_METHODS:
                with self.subTest(dim=dim, diff_method=diff_method):
                    thetas = make_thetas(0.5)

                    analytic_grad = jax.grad(sv_loss)(thetas)
                    analytic_val = jax.tree.leaves(analytic_grad)[0]

                    sample_sim = BranchingSimulator(
                        num_samples=num_samples, diff_method=diff_method
                    )

                    def sample_loss(thetas, key):
                        compiled = sample_sim.build_circuit(circuit, thetas)
                        return sample_sim.expval_all(compiled, initial_basis, key).sum()

                    sample_grad = jax.jit(jax.grad(sample_loss))

                    sample_grads = []
                    for i in range(num_runs):
                        key = jax.random.key(i)
                        grad = sample_grad(thetas, key)
                        sample_grads.append(jnp.stack(grad))

                    sample_grads = jnp.stack(sample_grads)
                    mean_grad = jnp.mean(sample_grads)

                    self.assertTrue(
                        jnp.allclose(mean_grad, analytic_val, atol=0.03),
                        f"PditShift(dim={dim})/{diff_method}: mean={mean_grad:.4f}, "
                        f"analytic={jnp.squeeze(analytic_val):.4f}",
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

            gate = PditSWAP([0, 1], dims=dim)
            circuit = DiscretePCircuit([gate])

            def sv_loss(thetas):
                compiled = sv_sim.build_circuit(circuit, thetas)
                return sv_sim.expval_all(compiled, initial_sv).sum()

            for diff_method in DIFF_METHODS:
                with self.subTest(dim=dim, diff_method=diff_method):
                    thetas = make_thetas(0.5)

                    analytic_grad = jax.grad(sv_loss)(thetas)
                    analytic_val = jax.tree.leaves(analytic_grad)[0]

                    sample_sim = BranchingSimulator(
                        num_samples=num_samples, diff_method=diff_method
                    )

                    def sample_loss(thetas, key):
                        compiled = sample_sim.build_circuit(circuit, thetas)
                        return sample_sim.expval_all(compiled, initial_basis, key).sum()

                    sample_grad = jax.jit(jax.grad(sample_loss))

                    sample_grads = []
                    for i in range(num_runs):
                        key = jax.random.key(i)
                        grad = sample_grad(thetas, key)
                        sample_grads.append(jnp.stack(grad))

                    sample_grads = jnp.stack(sample_grads)
                    mean_grad = jnp.mean(sample_grads)

                    self.assertTrue(
                        jnp.allclose(mean_grad, analytic_val, atol=0.03),
                        f"PditSWAP(dim={dim})/{diff_method}: mean={mean_grad:.4f}, "
                        f"analytic={jnp.squeeze(analytic_val):.4f}",
                    )

    def test_mixed_binary_pdit_circuit_gradient(self):
        num_samples = 10000
        num_runs = 50

        thetas = make_thetas(0.5, -0.3, 0.8)
        circuit = DiscretePCircuit(
            [
                PNOT(0),
                PditShift(2, dims=3),
                PCNOT([0, 1]),
            ]
        )

        sv_sim = StateVectorSimulator()
        # 2 * 2 * 3 = 12 states
        initial_sv = jnp.zeros(12).at[0].set(1.0)

        def sv_loss(thetas):
            compiled = sv_sim.build_circuit(circuit, thetas)
            return sv_sim.expval_all(compiled, initial_sv).sum()

        analytic_grad = jax.grad(sv_loss)(thetas)
        analytic_vals = jnp.array(jax.tree.leaves(analytic_grad)).flatten()

        for diff_method in DIFF_METHODS:
            with self.subTest(diff_method=diff_method):
                sample_sim = BranchingSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )
                initial_basis = jnp.array([0, 0, 0], dtype=jnp.int32)

                def sample_loss(thetas, key):
                    compiled = sample_sim.build_circuit(circuit, thetas)
                    return sample_sim.expval_all(compiled, initial_basis, key).sum()

                sample_grad = jax.jit(jax.grad(sample_loss))

                sample_grads = []
                for i in range(num_runs):
                    key = jax.random.key(i)
                    grad = sample_grad(thetas, key)
                    sample_grads.append(jnp.stack(grad).flatten())

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

        thetas = make_thetas(0.5, -0.3)
        circuit = DiscretePCircuit(
            [
                PditShift(0, dims=4),
                PditSWAP([0, 1], dims=4),
            ]
        )

        sv_sim = StateVectorSimulator()
        initial_sv = jnp.zeros(16).at[0].set(1.0)

        def sv_loss(thetas):
            compiled = sv_sim.build_circuit(circuit, thetas)
            return sv_sim.expval_all(compiled, initial_sv).sum()

        analytic_grad = jax.grad(sv_loss)(thetas)
        analytic_vals = jnp.array(jax.tree.leaves(analytic_grad)).flatten()

        for diff_method in DIFF_METHODS:
            with self.subTest(diff_method=diff_method):
                sample_sim = BranchingSimulator(
                    num_samples=num_samples, diff_method=diff_method
                )
                initial_basis = jnp.array([0, 0], dtype=jnp.int32)

                def sample_loss(thetas, key):
                    compiled = sample_sim.build_circuit(circuit, thetas)
                    return sample_sim.expval_all(compiled, initial_basis, key).sum()

                sample_grad = jax.jit(jax.grad(sample_loss))

                sample_grads = []
                for i in range(num_runs):
                    key = jax.random.key(i)
                    grad = sample_grad(thetas, key)
                    sample_grads.append(jnp.stack(grad).flatten())

                sample_grads = jnp.stack(sample_grads)
                mean_grads = jnp.mean(sample_grads, axis=0).flatten()

                self.assertTrue(
                    jnp.allclose(mean_grads, analytic_vals, atol=0.03),
                    f"Pdit-only/{diff_method}: "
                    f"mean={mean_grads}, analytic={analytic_vals}",
                )


class TestGradientVariance(unittest.TestCase):

    def test_variance_decreases_with_samples(self):
        thetas = make_thetas(0.5, 0.3)
        circuit = DiscretePCircuit([PNOT(0), PCNOT([0, 1])])

        sample_counts = [1000, 5000, 20000]
        num_runs = 30
        initial_basis = jnp.array([0, 0], dtype=jnp.int32)

        variances = []
        for num_samples in sample_counts:
            sample_sim = BranchingSimulator(
                num_samples=num_samples, diff_method="param_shift_filter"
            )

            def sample_loss(thetas, key):
                compiled = sample_sim.build_circuit(circuit, thetas)
                return sample_sim.expval_all(compiled, initial_basis, key).sum()

            sample_grad = jax.jit(jax.grad(sample_loss))

            grads = []
            for i in range(num_runs):
                key = jax.random.key(i)
                grad = sample_grad(thetas, key)
                grads.append(grad[0][0])

            grads = jnp.array(grads)
            variances.append(jnp.var(grads))

        self.assertLess(
            variances[-1],
            variances[0],
            f"Variance should decrease with more samples: {variances}",
        )


if __name__ == "__main__":
    unittest.main()
