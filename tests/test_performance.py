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

    def test_compile_once_latency_invariant(self):
        """Compile-once path must remain substantially faster than rebuild-per-call.

        Rebuilding circuit structure inside the region that JAX traces forces a
        full re-trace on every invocation and has been measured at ~100-900x
        higher latency (issue #24). This test keeps the compile-once path the
        fast path under a relative ordering constraint.
        """
        gates = [PCNOT([0, 1])]
        structure = DiscretePCircuit(gates * 32)
        sim = BranchingSimulator(num_samples=32)
        thetas = structure.init_params(jax.random.PRNGKey(0))
        key = jax.random.key(0)
        init_state = jnp.zeros(2, dtype=jnp.int32)

        # Correct path: compile structure once, then JIT only the execution.
        compiled = sim.build_circuit(structure, thetas)
        expval_jit = eqx.filter_jit(sim.expval_all)

        # Warm-up
        _ = expval_jit(compiled, init_state, key).block_until_ready()

        start = time.time()
        for _ in range(20):
            _ = expval_jit(compiled, init_state, key).block_until_ready()
        compiled_time = (time.time() - start) / 20

        # Anti-pattern approximation: re-build structure on every call.
        def rebuild_and_run(_):
            c = sim.build_circuit(structure, thetas)
            return sim.expval_all(c, init_state, key)

        # Intentionally *not* JITting the rebuild so the cost of structure
        # construction + re-trace is visible.
        start = time.time()
        for _ in range(5):
            _ = rebuild_and_run(None).block_until_ready()
        rebuild_time = (time.time() - start) / 5

        # Structural invariant: compiled path must stay clearly faster.
        self.assertLess(
            compiled_time * 3,
            rebuild_time,
            msg=(
                "compile-once path lost its latency advantage "
                f"(compiled={compiled_time:.4f}s, rebuild={rebuild_time:.4f}s)"
            ),
        )
