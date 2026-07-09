"""Test the probabilistic gates in `gates.py`"""

import unittest

import jax.numpy as jnp

from torx.psc import (
    PCNOT,
    PCopy,
    PCSWAP,
    PDEMUX,
    PditCycle,
    PJUMP,
    PMultiCNOT,
    PNOT,
    POR,
    PReset,
    PSWAP,
)


class TestPNOT(unittest.TestCase):
    def test_properties(self):
        sites = 0

        gate = PNOT(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(jnp.all(gate.branches[1] == jnp.array([[1], [0]])))
        self.assertTrue(jnp.all(gate.branches[0] == jnp.array([[0], [1]])))

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array([[1 - p, p], [p, 1 - p]])

            gate = PNOT(sites=0)
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))


class TestPSWAP(unittest.TestCase):
    def test_properties(self):
        sites = [0, 1]

        gate = PSWAP(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(
            jnp.all(gate.branches[1] == jnp.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
        )
        self.assertTrue(
            jnp.all(gate.branches[0] == jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        )

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array(
                [[1, 0, 0, 0], [0, 1 - p, p, 0], [0, p, 1 - p, 0], [0, 0, 0, 1]]
            )

            gate = PSWAP(sites=[0, 1])
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))


class TestPCNOT(unittest.TestCase):
    def test_properties(self):
        sites = [0, 1]

        gate = PCNOT(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(
            jnp.all(gate.branches[1] == jnp.array([[0, 0], [0, 1], [1, 1], [1, 0]]))
        )
        self.assertTrue(
            jnp.all(gate.branches[0] == jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        )

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1 - p, p], [0, 0, p, 1 - p]]
            )

            gate = PCNOT(sites=[0, 1])
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))


class TestPJUMP(unittest.TestCase):
    def test_properties(self):
        sites = [0, 1]

        gate = PJUMP(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(
            jnp.all(gate.branches[1] == jnp.array([[0, 0], [0, 1], [0, 1], [1, 1]]))
        )
        self.assertTrue(
            jnp.all(gate.branches[0] == jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        )

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array(
                [[1, 0, 0, 0], [0, 1, p, 0], [0, 0, 1 - p, 0], [0, 0, 0, 1]]
            )

            gate = PJUMP(sites=[0, 1])
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))


class TestPMultiCNOT(unittest.TestCase):
    def test_properties(self):
        sites = [0, 1, 2]

        gate = PMultiCNOT(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(
            jnp.all(
                gate.branches[1]
                == jnp.array(
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 0],
                    ]
                )
            )
        )
        self.assertTrue(
            jnp.all(
                gate.branches[0]
                == jnp.array(
                    [
                        [0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [0, 1, 1],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 0],
                        [1, 1, 1],
                    ]
                )
            )
        )

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1 - p, p],
                    [0, 0, 0, 0, 0, 0, p, 1 - p],
                ]
            )

            gate = PMultiCNOT(sites=[0, 1, 2])
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))


class TestPDEMUX(unittest.TestCase):
    def test_properties(self):
        sites = [0, 1]

        gate = PDEMUX(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(
            jnp.all(gate.branches[1] == jnp.array([[0, 0], [0, 0], [0, 1], [0, 1]]))
        )
        self.assertTrue(
            jnp.all(gate.branches[0] == jnp.array([[0, 0], [0, 0], [1, 0], [1, 0]]))
        )

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array(
                [[1, 1, 0, 0], [0, 0, p, p], [0, 0, 1 - p, 1 - p], [0, 0, 0, 0]]
            )

            gate = PDEMUX(sites=[0, 1])
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))


class TestPOR(unittest.TestCase):
    def test_properties(self):
        sites = [0, 1]

        gate = POR(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(
            jnp.all(gate.branches[1] == jnp.array([[0, 0], [1, 0], [1, 0], [1, 0]]))
        )
        self.assertTrue(
            jnp.all(gate.branches[0] == jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        )

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array(
                [[1, 0, 0, 0], [0, 1 - p, 0, 0], [0, p, 1, p], [0, 0, 0, 1 - p]]
            )

            gate = POR(sites=[0, 1])
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))


class TestPReset(unittest.TestCase):
    def test_properties(self):
        sites = 0

        gate = PReset(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(jnp.all(gate.branches[1] == jnp.array([[0], [0]])))
        self.assertTrue(jnp.all(gate.branches[0] == jnp.array([[0], [1]])))

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array([[1, p], [0, 1 - p]])

            gate = PReset(sites=0)
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))


class TestPCopy(unittest.TestCase):
    def test_properties(self):
        sites = [0, 1]

        gate = PCopy(sites)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(
            jnp.all(gate.branches[1] == jnp.array([[0, 0], [0, 0], [1, 1], [1, 1]]))
        )
        self.assertTrue(
            jnp.all(gate.branches[0] == jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        )

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array(
                [
                    [1, p, 0, 0],
                    [0, 1 - p, 0, 0],
                    [0, 0, 1 - p, 0],
                    [0, 0, p, 1],
                ]
            )

            gate = PCopy(sites=[0, 1])
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))


class TestPCSWAP(unittest.TestCase):
    def test_properties(self):
        sites = [0, 1, 2]

        gate = PCSWAP(sites)

        self.assertEqual(gate.sites, sites)

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            # Only states 101 and 110 swap with probability p
            expected_matrix = jnp.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1 - p, p, 0],
                    [0, 0, 0, 0, 0, p, 1 - p, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )

            gate = PCSWAP(sites=[0, 1, 2])
            theta_arr = jnp.atleast_1d(theta)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix(theta_arr)))

    def test_control_zero_no_swap(self):
        gate = PCSWAP(sites=[0, 1, 2])
        theta_arr = jnp.atleast_1d(100.0)
        mat = gate.get_matrix(theta_arr)

        for i in range(4):
            self.assertAlmostEqual(mat[i, i], 1.0)
            for j in range(8):
                if j != i:
                    self.assertAlmostEqual(mat[i, j], 0.0)


class TestPditCycle(unittest.TestCase):
    def test_properties(self):
        sites = 0

        gate = PditCycle(sites, dims=3)

        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 3)
        self.assertEqual(gate.dims, (3,))

    def test_properties_custom_dim(self):
        gate = PditCycle(sites=0, dims=(5,))

        self.assertEqual(gate.dims, (5,))
        self.assertEqual(gate.branches.shape, (3, 5, 1))

    def test_branches_dim3(self):
        gate = PditCycle(sites=0, dims=(3,))

        # Branch 0: identity
        self.assertTrue(jnp.all(gate.branches[0] == jnp.array([[0], [1], [2]])))
        # Branch 1: forward cycle (0->1->2->0)
        self.assertTrue(jnp.all(gate.branches[1] == jnp.array([[1], [2], [0]])))
        # Branch 2: backward cycle (0->2, 1->0, 2->1)
        self.assertTrue(jnp.all(gate.branches[2] == jnp.array([[2], [0], [1]])))

    def test_branches_dim5(self):
        gate = PditCycle(sites=0, dims=(5,))

        # Branch 0: identity
        expected_identity = jnp.array([[0], [1], [2], [3], [4]])
        self.assertTrue(jnp.all(gate.branches[0] == expected_identity))
        # Branch 1: forward cycle
        expected_forward = jnp.array([[1], [2], [3], [4], [0]])
        self.assertTrue(jnp.all(gate.branches[1] == expected_forward))
        # Branch 2: backward cycle
        expected_backward = jnp.array([[4], [0], [1], [2], [3]])
        self.assertTrue(jnp.all(gate.branches[2] == expected_backward))

    def test_probs_sum_to_one(self):
        gate = PditCycle(sites=0, dims=3)
        theta = jnp.array([0.5, -0.3], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(jnp.sum(gate.probs(theta)), 1.0))

    def test_deterministic_identity(self):
        gate = PditCycle(sites=0, dims=(4,))
        theta = jnp.array([-100.0, -100.0], dtype=jnp.float32)
        matrix = gate.get_matrix(theta)
        self.assertTrue(jnp.allclose(matrix, jnp.eye(4)))

    def test_deterministic_forward(self):
        gate = PditCycle(sites=0, dims=(3,))
        theta = jnp.array([100.0, -100.0], dtype=jnp.float32)
        matrix = gate.get_matrix(theta)
        # Forward cycle: 0->1, 1->2, 2->0
        expected = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(matrix, expected))

    def test_deterministic_backward(self):
        gate = PditCycle(sites=0, dims=(3,))
        theta = jnp.array([-100.0, 100.0], dtype=jnp.float32)
        matrix = gate.get_matrix(theta)
        # Backward cycle: 0->2, 1->0, 2->1
        expected = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(matrix, expected))
