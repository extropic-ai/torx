"""Tests for the `PAsymSwap` gate."""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from torx.psc import (
    BranchingSimulator,
    DiscretePCircuit,
    PAsymSwap,
    PJUMP,
    PSWAP,
    StateVectorSimulator,
)


def _hamming_weight(states):
    return jnp.sum(states, axis=-1)


class TestPAsymSwap(unittest.TestCase):
    def test_properties(self):
        sites = [0, 1]

        gate = PAsymSwap(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 3)
        self.assertTrue(
            jnp.all(gate.branches[0] == jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        )
        self.assertTrue(
            jnp.all(gate.branches[1] == jnp.array([[0, 0], [0, 1], [0, 1], [1, 1]]))
        )
        self.assertTrue(
            jnp.all(gate.branches[2] == jnp.array([[0, 0], [1, 0], [1, 0], [1, 1]]))
        )
        self.assertEqual(gate.init_params(jax.random.key(0)).shape, (2,))

    def test_matrix(self):
        gate = PAsymSwap(sites=[0, 1])
        for p, q in [(0.1, 0.2), (0.5, 0.25), (0.098, 0.002), (0.3, 0.3)]:
            theta = gate.theta_from_probs(jnp.array(p), jnp.array(q))
            expected_matrix = jnp.array(
                [[1, 0, 0, 0], [0, 1 - q, p, 0], [0, q, 1 - p, 0], [0, 0, 0, 1]]
            )
            self.assertTrue(
                jnp.allclose(expected_matrix, gate.get_matrix(theta), atol=1e-6)
            )
            self.assertTrue(
                jnp.allclose(gate.probs(theta), jnp.array([1 - p - q, p, q]), atol=1e-6)
            )

    def test_reduces_to_pjump_and_pswap(self):
        gate = PAsymSwap(sites=[0, 1])
        for p in [0.05, 0.3, 0.7]:
            # q -> 0 (theta_1 -> -inf) is PJUMP with the same p
            theta = jnp.array([jnp.log(p / (1 - p)), -jnp.inf])
            pjump_theta = jnp.atleast_1d(jnp.log(p / (1 - p)))
            self.assertTrue(
                jnp.allclose(
                    gate.get_matrix(theta), PJUMP([0, 1]).get_matrix(pjump_theta)
                )
            )
            # p = q is PSWAP with that p (only reachable for p < 1/2)
            if p < 0.5:
                theta = gate.theta_from_probs(jnp.array(p), jnp.array(p))
                pswap_theta = jnp.atleast_1d(jnp.log(p / (1 - p)))
                self.assertTrue(
                    jnp.allclose(
                        gate.get_matrix(theta),
                        PSWAP([0, 1]).get_matrix(pswap_theta),
                        atol=1e-6,
                    )
                )

    def test_sector_preservation_random_theta(self):
        """Occupation number is conserved for every parameter value.

        Checked three ways for 64 random parameter draws: every branch table
        maps each basis state to one of the same Hamming weight; the transition
        matrix is column-stochastic and block-diagonal in Hamming weight; and
        the matrix is exactly the branch-probability mixture of the branch
        permutations (so the sampled and matrix descriptions agree).
        """
        gate = PAsymSwap(sites=[0, 1])
        basis = jnp.array(list(itertools.product([0, 1], repeat=2)))
        weight = _hamming_weight(basis)

        # (i) branch tables are permutations within each sector
        for k in range(gate.num_branches):
            self.assertTrue(jnp.all(_hamming_weight(gate.branches[k]) == weight))

        keys = jax.random.split(jax.random.key(42), 64)
        for key in keys:
            theta = 6.0 * jax.random.normal(key, (2,))
            matrix = gate.get_matrix(theta)

            # (ii) column-stochastic and block-diagonal in Hamming weight
            self.assertTrue(jnp.allclose(matrix.sum(axis=0), 1.0, atol=1e-6))
            cross_sector = weight[:, None] != weight[None, :]
            self.assertTrue(jnp.all(jnp.where(cross_sector, matrix, 0.0) == 0.0))

            # (iii) matrix == sum_k probs[k] * permutation(branches[k])
            probs = gate.probs(theta)
            mixture = jnp.zeros((4, 4))
            for k in range(gate.num_branches):
                out_index = gate.branches[k] @ jnp.array([2, 1])
                perm = jax.nn.one_hot(out_index, 4).T  # perm[out, in]
                mixture = mixture + probs[k] * perm
            self.assertTrue(jnp.allclose(matrix, mixture, atol=1e-6))

    def test_sampled_circuit_conserves_particle_number(self):
        """Every sampled trajectory of a random PAsymSwap circuit keeps the
        number of occupied pbits fixed, in integer arithmetic (no tolerance)."""
        num_nodes = 6
        edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        keys = jax.random.split(jax.random.key(7), len(edges))
        thetas = [3.0 * jax.random.normal(k, (2,)) for k in keys]
        circuit = DiscretePCircuit(
            [PAsymSwap([int(i), int(j)]) for i, j in edges], reps=25
        )
        sim = BranchingSimulator(num_samples=2_000)
        compiled = sim.build_circuit(circuit, thetas)
        for n_particles in [1, 2, 3]:
            x = jnp.zeros(num_nodes, dtype=jnp.int32).at[:n_particles].set(1)
            samples = sim.sample(compiled, x, jax.random.key(n_particles))
            self.assertEqual(samples.shape, (2_000, num_nodes))
            self.assertTrue(bool(jnp.all(_hamming_weight(samples) == n_particles)))

    def test_single_walker_matches_matrix(self):
        """One walker on one edge: sampled occupancies match the 4x4 matrix
        column for |10) within six binomial standard errors."""
        p, q = 0.3, 0.1
        gate = PAsymSwap([0, 1])
        theta = gate.theta_from_probs(jnp.array(p), jnp.array(q))
        sim = BranchingSimulator(num_samples=20_000)
        compiled = sim.build_circuit(DiscretePCircuit([gate]), [theta])
        occ = np.asarray(sim.expval_all(compiled, jnp.array([1, 0]), jax.random.key(0)))
        self.assertTrue(np.allclose(occ, [1 - p, p], atol=6 * 0.5 / np.sqrt(20_000)))
        occ = np.asarray(sim.expval_all(compiled, jnp.array([0, 1]), jax.random.key(1)))
        self.assertTrue(np.allclose(occ, [q, 1 - q], atol=6 * 0.5 / np.sqrt(20_000)))

    def test_theta_from_probs_domain(self):
        gate = PAsymSwap([0, 1])
        # q = 0 gives a -inf logit and reproduces PJUMP exactly
        theta = gate.theta_from_probs(jnp.array(0.3), jnp.array(0.0))
        self.assertTrue(bool(jnp.isneginf(theta[1])))
        self.assertTrue(
            jnp.allclose(gate.get_matrix(theta), PJUMP([0, 1]).get_matrix(theta[:1]))
        )
        # p + q = 1 is outside the domain: the stay probability is floored,
        # the logits stay finite and the hop probabilities keep their ratio
        theta = gate.theta_from_probs(jnp.array(0.75), jnp.array(0.25))
        self.assertTrue(bool(jnp.all(jnp.isfinite(theta))))
        probs = gate.probs(theta)
        self.assertLess(float(probs[0]), 1e-6)
        self.assertTrue(jnp.allclose(probs[1] / probs[2], 3.0, rtol=1e-4))


def _weighted_expval_loss(sim, circuit, x0, weights):
    def loss(thetas, key):
        compiled = sim.build_circuit(circuit, thetas)
        return (sim.expval_all(compiled, x0, key) * weights).sum()

    return loss


def _exact_and_finite_difference_grads(circuit, thetas, sv0, weights, eps=1e-3):
    sv_sim = StateVectorSimulator()

    def sv_loss(ths):
        compiled = sv_sim.build_circuit(circuit, ths)
        return (sv_sim.expval_all(compiled, sv0) * weights).sum()

    exact = jax.grad(sv_loss)(thetas)
    finite_diff = []
    for g, theta in enumerate(thetas):
        grad = np.zeros(theta.shape)
        for j in range(theta.shape[0]):
            plus = [t.at[j].add(eps) if i == g else t for i, t in enumerate(thetas)]
            minus = [t.at[j].add(-eps) if i == g else t for i, t in enumerate(thetas)]
            grad[j] = (sv_loss(plus) - sv_loss(minus)) / (2 * eps)
        finite_diff.append(grad)
    return exact, finite_diff


class TestPAsymSwapParamShiftSingle(unittest.TestCase):
    """`param_shift_single` for K = 3 gates: the shifted logit
    theta_j + ln p_{j+1} - ln(1 + p_{j+1}) reproduces the K = 2 rule and gives
    an unbiased gradient for PAsymSwap circuits, alone and mixed with K = 2 gates.
    """

    num_samples = 20_000
    num_runs = 40

    def _check_sampled_gradient(self, circuit, thetas, x0, sv0, weights, label):
        exact, finite_diff = _exact_and_finite_difference_grads(
            circuit, thetas, sv0, weights
        )
        for g in range(len(thetas)):
            self.assertTrue(np.allclose(exact[g], finite_diff[g], atol=2e-3))

        sim = BranchingSimulator(
            num_samples=self.num_samples, diff_method="param_shift_single"
        )
        sample_grad = jax.jit(
            jax.grad(_weighted_expval_loss(sim, circuit, x0, weights))
        )
        grads = [sample_grad(thetas, jax.random.key(i)) for i in range(self.num_runs)]
        for g in range(len(thetas)):
            stacked = np.stack([np.asarray(grad[g]) for grad in grads])
            mean = stacked.mean(0)
            stderr = stacked.std(0) / np.sqrt(self.num_runs)
            self.assertTrue(
                np.all(np.abs(mean - np.asarray(exact[g])) < 5 * stderr + 2e-3),
                f"{label} gate {g}: sampled {mean}, exact {np.asarray(exact[g])}",
            )

    def test_shift_reduces_to_k2_rule(self):
        theta = jnp.linspace(-6.0, 6.0, 25)
        k2_rule = -jnp.log((1 + jnp.exp(-theta)) ** 2 - 1)
        log_p = jax.nn.log_softmax(jnp.stack([jnp.zeros_like(theta), theta], -1))[:, 1]
        general_rule = theta + log_p - jax.nn.softplus(log_p)
        self.assertTrue(jnp.allclose(general_rule, k2_rule, atol=1e-4))

    def test_pasymswap_matches_finite_differences(self):
        gate = PAsymSwap([0, 1])
        thetas = [gate.theta_from_probs(jnp.array(0.3), jnp.array(0.2))]
        self._check_sampled_gradient(
            DiscretePCircuit([gate]),
            thetas,
            jnp.array([1, 0], dtype=jnp.int32),
            jnp.zeros(4).at[2].set(1.0),
            jnp.array([1.0, 2.0]),
            "PAsymSwap",
        )

    def test_pjump_k2_unchanged(self):
        self._check_sampled_gradient(
            DiscretePCircuit([PJUMP([0, 1])]),
            [jnp.array([0.5])],
            jnp.array([1, 0], dtype=jnp.int32),
            jnp.zeros(4).at[2].set(1.0),
            jnp.array([1.0, 2.0]),
            "PJUMP",
        )

    def test_mixed_k_circuit(self):
        circuit = DiscretePCircuit(
            [PAsymSwap([0, 1]), PJUMP([1, 2]), PAsymSwap([2, 0])]
        )
        thetas = [jnp.array([0.3, -0.4]), jnp.array([0.7]), jnp.array([-0.2, 0.9])]
        self._check_sampled_gradient(
            circuit,
            thetas,
            jnp.array([1, 0, 1], dtype=jnp.int32),
            jnp.zeros(8).at[5].set(1.0),
            jnp.array([1.0, 2.0, 3.0]),
            "mixed",
        )


if __name__ == "__main__":
    unittest.main()
