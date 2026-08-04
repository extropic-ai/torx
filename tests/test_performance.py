"""Test the performance"""

import time
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp

from torx.psc import BranchingSimulator, DiscretePCircuit, PCNOT


class TestPerformance(unittest.TestCase):
    def test_compile_and_execute_time(self):
        num_sites = 100
        depth = 10000
        num_samples = 1000

        gates = [PCNOT([num_sites - 2, num_sites - 1])]
        circuit = DiscretePCircuit(gates * depth)

        sim = BranchingSimulator(num_samples=num_samples)
        thetas = circuit.init_params(jax.random.PRNGKey(0))
        circuit = sim.build_circuit(circuit, thetas)

        key = jax.random.key(100)
        init_state = jnp.ones(num_sites, dtype=jnp.int32)

        expval_jit = eqx.filter_jit(sim.expval_all)

        start = time.time()
        _ = expval_jit(circuit, init_state, key).block_until_ready()

        compile_time_dev = time.time() - start

        num_iters = 100

        start = time.time()
        for _ in range(num_iters):
            _ = expval_jit(circuit, init_state, key).block_until_ready()

        runtime = time.time() - start

        execute_time_dev = runtime / num_iters

        # Expected runtimes (hardware-dependent)
        expected_compile_time, expected_execute_time = 0.40, 0.14

        tol = 0.15
        self.assertLess(compile_time_dev, expected_compile_time * (1 + tol))
        self.assertLess(execute_time_dev, expected_execute_time * (1 + tol))
