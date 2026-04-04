"""Test the probabilistic gates in `gates.py`"""

import unittest

import jax.numpy as jnp

from torx import (
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
        theta = 1.5
        sites = 0

        gate = PNOT(theta, sites)

        self.assertEqual(gate.theta, theta)
        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(jnp.all(gate.branches[1] == jnp.array([[1], [0]])))
        self.assertTrue(jnp.all(gate.branches[0] == jnp.array([[0], [1]])))

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array([[1 - p, p], [p, 1 - p]])

            gate = PNOT(theta, sites=0)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))


class TestPSWAP(unittest.TestCase):
    def test_properties(self):
        theta = 1.5
        sites = [0, 1]

        gate = PSWAP(theta, sites)

        self.assertEqual(gate.theta, theta)
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

            gate = PSWAP(theta, sites=[0, 1])
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))


class TestPCNOT(unittest.TestCase):
    def test_properties(self):
        theta = 1.5
        sites = [0, 1]

        gate = PCNOT(theta, sites)

        self.assertEqual(gate.theta, theta)
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

            gate = PCNOT(theta, sites=[0, 1])
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))


class TestPJUMP(unittest.TestCase):
    def test_properties(self):
        theta = 1.5
        sites = [0, 1]

        gate = PJUMP(theta, sites)

        self.assertEqual(gate.theta, theta)
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

            gate = PJUMP(theta, sites=[0, 1])
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))


class TestPMultiCNOT(unittest.TestCase):
    def test_properties(self):
        theta = 1.5
        sites = [0, 1, 2]

        gate = PMultiCNOT(theta, sites)

        self.assertEqual(gate.theta, theta)
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

            gate = PMultiCNOT(theta, sites=[0, 1, 2])
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))


class TestPDEMUX(unittest.TestCase):
    def test_properties(self):
        theta = 1.5
        sites = [0, 1]

        gate = PDEMUX(theta, sites)

        self.assertEqual(gate.theta, theta)
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

            gate = PDEMUX(theta, sites=[0, 1])
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))


class TestPOR(unittest.TestCase):
    def test_properties(self):
        theta = 1.5
        sites = [0, 1]

        gate = POR(theta, sites)

        self.assertEqual(gate.theta, theta)
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

            gate = POR(theta, sites=[0, 1])
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))


class TestPReset(unittest.TestCase):
    def test_properties(self):
        theta = 1.5
        sites = 0

        gate = PReset(theta, sites)

        self.assertEqual(gate.theta, theta)
        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 2)
        self.assertTrue(jnp.all(gate.branches[1] == jnp.array([[0], [0]])))
        self.assertTrue(jnp.all(gate.branches[0] == jnp.array([[0], [1]])))

    def test_matrix(self):
        for theta in [-jnp.array(1.0), jnp.array(0.0), jnp.array(1.0)]:
            p = 1 / (1 + jnp.exp(-theta))
            expected_matrix = jnp.array([[1, p], [0, 1 - p]])

            gate = PReset(theta, sites=0)
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))


class TestPCopy(unittest.TestCase):
    def test_properties(self):
        theta = 1.5
        sites = [0, 1]

        gate = PCopy(theta, sites)

        self.assertEqual(gate.theta, theta)
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

            gate = PCopy(theta, sites=[0, 1])
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))


class TestPCSWAP(unittest.TestCase):
    def test_properties(self):
        theta = 1.5
        sites = [0, 1, 2]

        gate = PCSWAP(theta, sites)

        self.assertEqual(gate.theta, theta)
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

            gate = PCSWAP(theta, sites=[0, 1, 2])
            self.assertTrue(jnp.allclose(expected_matrix, gate.get_matrix()))

    def test_control_zero_no_swap(self):
        gate = PCSWAP(jnp.array(100.0), sites=[0, 1, 2])
        mat = gate.get_matrix()

        for i in range(4):
            self.assertAlmostEqual(float(mat[i, i]), 1.0)
            for j in range(8):
                if j != i:
                    self.assertAlmostEqual(float(mat[i, j]), 0.0)


class TestPditCycle(unittest.TestCase):
    def test_properties(self):
        theta = jnp.array([0.5, -0.3])
        sites = 0

        gate = PditCycle(theta, sites, dims=3)

        self.assertTrue(jnp.allclose(gate.theta, theta))
        self.assertEqual(gate.sites, sites)
        self.assertEqual(gate.num_branches, 3)
        self.assertEqual(gate.dims, (3,))

    def test_properties_custom_dim(self):
        theta = jnp.array([0.5, -0.3])
        gate = PditCycle(theta, sites=0, dims=(5,))

        self.assertEqual(gate.dims, (5,))
        self.assertEqual(gate.branches.shape, (3, 5, 1))

    def test_branches_dim3(self):
        gate = PditCycle(jnp.array([0.0, 0.0]), sites=0, dims=(3,))

        # Branch 0: identity
        self.assertTrue(jnp.all(gate.branches[0] == jnp.array([[0], [1], [2]])))
        # Branch 1: forward cycle (0->1->2->0)
        self.assertTrue(jnp.all(gate.branches[1] == jnp.array([[1], [2], [0]])))
        # Branch 2: backward cycle (0->2, 1->0, 2->1)
        self.assertTrue(jnp.all(gate.branches[2] == jnp.array([[2], [0], [1]])))

    def test_branches_dim5(self):
        gate = PditCycle(jnp.array([0.0, 0.0]), sites=0, dims=(5,))

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
        gate = PditCycle(jnp.array([0.5, -0.3]), sites=0, dims=3)
        self.assertAlmostEqual(float(jnp.sum(gate.probs)), 1.0)

    def test_deterministic_identity(self):
        gate = PditCycle(jnp.array([-100.0, -100.0]), sites=0, dims=(4,))
        matrix = gate.get_matrix()
        self.assertTrue(jnp.allclose(matrix, jnp.eye(4)))

    def test_deterministic_forward(self):
        gate = PditCycle(jnp.array([100.0, -100.0]), sites=0, dims=(3,))
        matrix = gate.get_matrix()
        # Forward cycle: 0->1, 1->2, 2->0
        expected = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(matrix, expected))

    def test_deterministic_backward(self):
        gate = PditCycle(jnp.array([-100.0, 100.0]), sites=0, dims=(3,))
        matrix = gate.get_matrix()
        # Backward cycle: 0->2, 1->0, 2->1
        expected = jnp.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=jnp.float32)
        self.assertTrue(jnp.allclose(matrix, expected))
