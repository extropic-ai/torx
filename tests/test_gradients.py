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


class TestParamShiftFilterZeroCount(unittest.TestCase):
    """Regression tests for issue #9: param_shift_filter must remain unbiased
    when a branch receives zero samples.

    The previous estimator substituted 0 for E[O | branch] on zero-count
    branches, which biased the gradient. The current estimator uses a
    score-function identity with a leave-one-out baseline, which is unbiased
    for any fixed cotangent when samples are i.i.d. rollouts.
    """

    def test_unbiased_at_zero_count(self):
        """End-to-end unbiasedness at N=1 via forced-branch enumeration.

        PCNOT with control=0 leaves the state unchanged regardless of which
        branch fires, so the true gradient is exactly 0. We run the full
        jax.grad path at num_samples=1 for seeds covering both branch
        outcomes, then assert the p_k-weighted average equals 0.

        Testing at theta=0 alone is insufficient: at theta=0, p_k = 0.5 and
        the p_k-weighted average coincides with the plain average, which
        could mask a weighting bug. We also test at theta=0.5.
        """
        from torx.psc.simulation.sampled import sample_circuit

        circuit = DiscretePCircuit([PCNOT([0, 1])])
        initial = jnp.array([0, 1], dtype=jnp.int32)

        for theta_val in (0.0, 0.5):
            with self.subTest(theta=theta_val):
                thetas = [jnp.array([theta_val])]
                p_branch1 = float(jax.nn.sigmoid(theta_val))
                p_branch0 = 1.0 - p_branch1

                sim = BranchingSimulator(num_samples=1, diff_method="param_shift_filter")
                compiled = sim.build_circuit(circuit, thetas)

                def loss(thetas_list, key):
                    c = sim.build_circuit(circuit, thetas_list)
                    return sim.expval_all(c, initial, key).sum()

                grad_fn = jax.jit(jax.grad(loss))

                # Find one seed per branch outcome, deterministically.
                grads_by_branch = {}
                for seed in range(1000):
                    key = jax.random.key(seed)
                    _, branch_indices = sample_circuit(compiled, initial, key, 1)
                    branch = int(branch_indices[0, 0, 0])
                    if branch not in grads_by_branch:
                        grads_by_branch[branch] = float(grad_fn(thetas, key)[0][0])
                    if len(grads_by_branch) == 2:
                        break

                self.assertEqual(
                    len(grads_by_branch), 2,
                    f"Could not find seeds covering both branches at theta={theta_val}"
                )

                expected = (
                    p_branch0 * grads_by_branch[0] + p_branch1 * grads_by_branch[1]
                )
                self.assertAlmostEqual(
                    expected, 0.0, places=6,
                    msg=(
                        f"theta={theta_val}: p_k-weighted mean gradient "
                        f"{expected:.6f} != 0; branch grads = {grads_by_branch}"
                    ),
                )

    def test_variance_pin_at_1k(self):
        """Score+LOO gradient SD at N=1000 must be below the analytic bound.

        Circuit: PNOT(0) on a 5-pbit state where pbits 1..4 are pinned at 1
        by the initial basis; loss = weighted sum over all pbits.

        Bound derivation (theta=0, p=0.5):
          E[score^2] = p(1-p) = 0.25
          With random per-pbit weights w_i in [0.5, 2]:
            Var(O * w) under LOO = Var(O_pbit0) * w_0^2 <= (1/4) * 4 = 1
          Leading-order estimator SD <= sqrt(E[score^2] * 1 / N)
                                     = sqrt(0.25 / 1000) = 0.0158
          With 1.5x slack for baseline cross-terms (the N summands are
          pairwise-uncorrelated with the score but not mutually independent
          through the shared baseline):
            bound = 0.0158 * 1.5 = 0.0237, rounded up to 0.025.

        Discrimination: the raw-score estimator (no baseline) scales with
        E[O^2], not Var(O). With weights in [0.5, 2] and four pbits pinned
        at 1, E[O^2] ~= (sum w_i)^2 + O(1) ~= 25+, giving raw-score SD
        ~= sqrt(0.25 * 25 / 1000) = 0.079 >> 0.025. The bound therefore
        separates score+LOO from raw score on this circuit.
        """
        # Random per-pbit weights ensure the constant offset from pinned
        # pbits has non-trivial magnitude across coordinates.
        rng_key = jax.random.key(0)
        weights = jax.random.uniform(rng_key, (5,), minval=0.5, maxval=2.0)

        thetas = [jnp.array([0.0])]
        circuit = DiscretePCircuit([PNOT(0)])
        initial = jnp.array([0, 1, 1, 1, 1], dtype=jnp.int32)

        N = 1000
        num_runs = 50

        sim = BranchingSimulator(num_samples=N, diff_method="param_shift_filter")
        compiled = sim.build_circuit(circuit, thetas)

        def weighted_loss(thetas_list, key):
            c = sim.build_circuit(circuit, thetas_list)
            return (sim.expval_all(c, initial, key) * weights).sum()

        grad_fn = jax.jit(jax.grad(weighted_loss))

        grads = jnp.stack(
            [grad_fn(thetas, jax.random.key(i))[0][0] for i in range(num_runs)]
        )
        sd = float(jnp.std(grads))

        # Analytic: E[score^2] = 0.25. Var(O * w) under LOO = Var(O_pbit0 * w_0)
        # = w_0^2 * p(1-p) * 1 <= 4 * 0.25 = 1. Leading-order SD <= sqrt(0.25 * 1 / N)
        # = sqrt(0.25/1000) = 0.0158. With 1.5x slack: 0.0237.
        bound = 0.025
        self.assertLess(
            sd, bound,
            f"param_shift_filter SD {sd:.4f} exceeds analytic bound {bound} "
            f"at N={N}; LOO baseline may be missing or misimplemented",
        )

    def test_mixed_arity_circuit(self):
        """K=2 gate + K=4 pdit gate in the same circuit.

        Padded branch slots must contribute exactly zero gradient. Verify by
        comparing the K=2 gate's gradient to the same circuit without the
        K=4 gate (the K=4 gate's presence should not change the K=2 gate's
        gradient up to sampling noise).
        """
        # PditShift is a K=d gate on a d-state pdit (here d=4)
        thetas_iso = make_thetas(0.3)
        circuit_iso = DiscretePCircuit([PNOT(0)])

        thetas_mixed = [jnp.array([0.3]), jnp.zeros(3)]  # K=4 → 3 logits
        circuit_mixed = DiscretePCircuit([PNOT(0), PditShift(1, 4)])

        sv_sim = StateVectorSimulator()

        def sv_loss_iso(thetas):
            c = sv_sim.build_circuit(circuit_iso, thetas)
            return sv_sim.expval_all(c, jnp.array([1.0, 0.0])).sum()

        analytic_iso = float(jax.grad(sv_loss_iso)(thetas_iso)[0][0])

        # Sampled mixed circuit at high N
        sim = BranchingSimulator(num_samples=20000, diff_method="param_shift_filter")
        initial = jnp.array([0, 0], dtype=jnp.int32)

        def sample_loss_mixed(thetas_list, key):
            c = sim.build_circuit(circuit_mixed, thetas_list)
            return sim.expval_all(c, initial, key)[0]  # just pbit 0

        sample_grad_mixed = float(
            jax.grad(sample_loss_mixed)(thetas_mixed, jax.random.key(0))[0][0]
        )

        # If padded branches leaked, the K=2 gate's gradient would be off.
        # Compare to the isolated analytic value with generous tolerance.
        self.assertAlmostEqual(
            sample_grad_mixed, analytic_iso, delta=0.02,
            msg=(
                f"Mixed-arity: K=2 gate grad {sample_grad_mixed:.4f} "
                f"vs isolated analytic {analytic_iso:.4f}"
            ),
        )

    def test_reps_at_one_sample(self):
        """reps=4 with num_samples=1: estimator must remain unbiased."""
        circuit = DiscretePCircuit([PReset(0)], reps=4)
        initial = jnp.array([1], dtype=jnp.int32)

        thetas = make_thetas(0.0)
        sim = BranchingSimulator(num_samples=1, diff_method="param_shift_filter")

        def loss(thetas_list, key):
            c = sim.build_circuit(circuit, thetas_list)
            return sim.expval_all(c, initial, key).sum()

        grad_fn = jax.jit(jax.grad(loss))

        # PReset(0) sets pbit 0 to 0 with probability sigmoid(theta); at
        # reps=4 and theta=0, the true gradient can be computed by
        # StateVector. Compare the mean over forced-branch seeds to it.
        sv_sim = StateVectorSimulator()

        def sv_loss(thetas):
            c = sv_sim.build_circuit(circuit, thetas)
            return sv_sim.expval_all(c, jnp.array([0.0, 1.0])).sum()

        true_grad = float(jax.grad(sv_loss)(thetas)[0][0])

        # Average over many seeds at N=1 — should converge to true_grad.
        num_seeds = 2000
        grads = jnp.stack(
            [grad_fn(thetas, jax.random.key(s))[0][0] for s in range(num_seeds)]
        )
        mean_grad = float(jnp.mean(grads))
        # Each seed's grad has magnitude ~O(1); SD of the mean over 2000
        # seeds is ~O(0.02). Use a 5-sigma tolerance to avoid flakes.
        self.assertAlmostEqual(
            mean_grad, true_grad, delta=0.1,
            msg=f"reps=4 N=1: mean grad {mean_grad:.4f} vs true {true_grad:.4f}",
        )

    def test_degenerate_identical_observables(self):
        """When all samples produce identical O, gradient must be 0 (no NaN)."""
        # PNOT with theta very large => p ~= 1 => all samples take the same
        # branch, all observables identical.
        thetas = [jnp.array([20.0])]
        circuit = DiscretePCircuit([PNOT(0)])
        initial = jnp.array([0], dtype=jnp.int32)

        sim = BranchingSimulator(num_samples=100, diff_method="param_shift_filter")

        def loss(thetas_list):
            c = sim.build_circuit(circuit, thetas_list)
            return sim.expval_all(c, initial, jax.random.key(0)).sum()

        grad = jax.grad(loss)(thetas)[0][0]
        self.assertTrue(jnp.isfinite(grad), f"non-finite gradient: {grad}")
        self.assertAlmostEqual(
            float(grad), 0.0, places=5,
            msg=f"degenerate-O gradient should be ~0, got {grad}",
        )


if __name__ == "__main__":
    unittest.main()
