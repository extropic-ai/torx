"""Test the probabilistic circuit in `circuit.py`"""

import unittest

from torx import DiscretePCircuit, PCNOT, PNOT


class TestDiscretePCircuit(unittest.TestCase):
    def test_properties(self):
        gates = [PNOT(1.0, 0), PCNOT(2.0, [0, 1]), PNOT(0.0, 1)]
        circuit = DiscretePCircuit(gates)

        self.assertEqual(circuit.gates, gates)
        self.assertEqual(circuit.num_pdits, 2)
        self.assertEqual(circuit.reps, 1)

    def test_iteration(self):
        gates = [PNOT(1.0, 0), PCNOT(2.0, [0, 1]), PNOT(0.0, 1)]
        circuit = DiscretePCircuit(gates)

        for gate, cgate in zip(gates, circuit):
            self.assertEqual(gate, cgate)

    def test_len(self):
        gates1 = [PNOT(1.0, 0)]
        circuit1 = DiscretePCircuit(gates1)
        self.assertEqual(len(circuit1), 1)

        gates2 = [PNOT(1.0, 0), PCNOT(2.0, [0, 1]), PNOT(0.0, 1)]
        circuit2 = DiscretePCircuit(gates2)
        self.assertEqual(len(circuit2), 3)

    def test_indexing(self):
        gates = [PNOT(1.0, 0), PCNOT(2.0, [0, 1]), PNOT(0.0, 1)]
        circuit = DiscretePCircuit(gates)

        for i, gate in enumerate(gates):
            self.assertEqual(circuit[i], gate)

    def test_add(self):
        gates1 = [PNOT(1.0, 0)]
        gates2 = [PCNOT(2.0, [0, 1]), PNOT(0.0, 1)]

        circuit1 = DiscretePCircuit(gates1)
        circuit2 = DiscretePCircuit(gates2)

        circuit_sum = circuit1 + circuit2
        self.assertEqual(circuit_sum.gates, gates1 + gates2)

        circuit_sum: DiscretePCircuit = sum(  # pyright: ignore[reportAssignmentType]
            [circuit1, circuit2], start=DiscretePCircuit([])
        )
        self.assertEqual(circuit_sum.gates, gates1 + gates2)
