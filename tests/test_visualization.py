"""Tests for circuit visualization."""

import unittest

from torx.psc import draw_circuit, HybridPCircuit, MixtureGaussianGate, PNOT


class TestDrawCircuit(unittest.TestCase):
    def test_controlled_continuous_gate_draws_all_target_wires(self):
        circuit = HybridPCircuit(
            [
                PNOT(sites=0),
                MixtureGaussianGate(
                    sites=([0], [0, 1]),
                    dims=(1, 1),
                    num_components=2,
                ),
            ]
        )

        drawing = draw_circuit(circuit, output=False)
        self.assertIsNotNone(drawing)

        continuous_lines = [
            line for line in str(drawing).splitlines() if line.lstrip().startswith("c_")
        ]
        self.assertEqual(len(continuous_lines), 2)
        self.assertIn("╡ MoG ╞", continuous_lines[0])
        self.assertIn("╡", continuous_lines[1])
        self.assertIn("╞", continuous_lines[1])
