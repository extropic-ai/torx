"""Graph-diffusion helpers for Torx example notebooks."""

from collections.abc import Sequence
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from torx.psc import DiscretePCircuit, PSWAP, SampleSimulator

_Edge: TypeAlias = tuple[int, int]


def build_graph_generator(
    num_nodes: int,
    edges: Sequence[_Edge],
    *,
    rate: float = 1.0,
) -> np.ndarray:
    """Build a symmetric continuous-time edge-diffusion generator."""
    generator = np.zeros((num_nodes, num_nodes), dtype=float)
    for i, j in edges:
        generator[i, j] += rate
        generator[j, i] += rate
        generator[i, i] -= rate
        generator[j, j] -= rate
    return generator


def reference_heat_flow(
    initial: np.ndarray, generator: np.ndarray, time: float
) -> np.ndarray:
    """Exact heat flow ``exp(time * generator) @ initial`` for a 1-D node vector.

    The eigendecomposition path assumes a symmetric ``generator`` (as produced by
    ``build_graph_generator``); a non-symmetric generator would silently give the
    wrong reference.
    """
    generator = np.asarray(generator, dtype=float)
    if not np.allclose(generator, generator.T):
        raise ValueError("reference_heat_flow requires a symmetric generator")
    vals, vecs = np.linalg.eigh(generator)
    return vecs @ (np.exp(time * vals) * (vecs.T @ initial))


def _edge_swap_probability(rate: float, total_time: float, steps: int) -> float:
    decay = np.exp(-2.0 * rate * total_time / steps)
    return float(0.5 * (1.0 - decay))


def ordered_pswap_product_formula_mean(
    initial: np.ndarray,
    edges: Sequence[_Edge],
    *,
    reps: int,
    swap_probability: float,
) -> np.ndarray:
    """Return the exact mean of repeated, ordered PSWAP edge sweeps."""
    mean = np.asarray(initial, dtype=float).copy()
    for _ in range(reps):
        for i, j in edges:
            left, right = mean[i], mean[j]
            mean[i] = (1.0 - swap_probability) * left + swap_probability * right
            mean[j] = swap_probability * left + (1.0 - swap_probability) * right
    return mean


def sample_pswap_product_formula(
    initial: np.ndarray,
    edges: Sequence[_Edge],
    *,
    reps: int,
    swap_probability: float,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Estimate a PSWAP product-formula layer with Torx samples."""
    # torx gates are structure-only; swap probability lives in `thetas`
    # as the logit theta = sigma^{-1}(p), one (1,)-vector per gate
    theta = jnp.asarray(np.log(swap_probability / (1.0 - swap_probability)))
    layer = [PSWAP([int(i), int(j)]) for i, j in edges]
    thetas = [jnp.reshape(theta, (1,)) for _ in layer]
    circuit = DiscretePCircuit(layer, reps=reps)
    simulator = SampleSimulator(num_samples=num_samples)
    compiled = simulator.build_circuit(circuit, thetas)

    initial = np.asarray(initial, dtype=float).ravel()
    active_sources = np.flatnonzero(initial)
    keys = jax.random.split(jax.random.key(seed), len(active_sources))
    one_hots = jnp.eye(initial.size, dtype=jnp.int32)[active_sources]
    # one batched, vmapped dispatch instead of a per-source jitted call + host sync
    batched_expval = eqx.filter_jit(
        lambda compiled, one_hots, keys: jax.vmap(
            simulator.expval_all, in_axes=(None, 0, 0)
        )(compiled, one_hots, keys)
    )
    expectations = batched_expval(compiled, one_hots, keys)
    weights = jnp.asarray(initial[active_sources])
    heat = jnp.einsum("s,sn->n", weights, expectations)
    return np.asarray(heat)


def sample_pswap_product_formula_multiseed(
    initial: np.ndarray,
    edges: Sequence[_Edge],
    *,
    reps: int,
    swap_probability: float,
    num_samples: int,
    seeds: Sequence[int],
) -> np.ndarray:
    """Estimate the PSWAP product formula for several seeds from one compile.

    Same per-seed estimate as ``sample_pswap_product_formula`` but the circuit is
    compiled once and reused across all seeds, which is the right pattern when a
    convergence sweep draws many seeds at a fixed resolution. Returns one heat
    vector per seed with shape ``(len(seeds), nodes)``.
    """
    initial = np.asarray(initial, dtype=float).ravel()
    theta = jnp.asarray(np.log(swap_probability / (1.0 - swap_probability)))
    layer = [PSWAP([int(i), int(j)]) for i, j in edges]
    thetas = [jnp.reshape(theta, (1,)) for _ in layer]
    circuit = DiscretePCircuit(layer, reps=reps)
    simulator = SampleSimulator(num_samples=num_samples)
    compiled = simulator.build_circuit(circuit, thetas)

    active_sources = np.flatnonzero(initial)
    one_hots = jnp.eye(initial.size, dtype=jnp.int32)[active_sources]
    weights = jnp.asarray(initial[active_sources])
    batched_expval = eqx.filter_jit(
        lambda compiled, one_hots, keys: jax.vmap(
            simulator.expval_all, in_axes=(None, 0, 0)
        )(compiled, one_hots, keys)
    )

    heats = []
    for seed in seeds:
        keys = jax.random.split(jax.random.key(int(seed)), len(active_sources))
        expectations = batched_expval(compiled, one_hots, keys)
        heats.append(np.asarray(jnp.einsum("s,sn->n", weights, expectations)))
    return np.stack(heats)


def apply_edge_product_formula(
    initial: np.ndarray,
    edges: Sequence[_Edge],
    *,
    steps: int,
    total_time: float,
    rate: float = 1.0,
    num_samples: int = 12_000,
    seed: int = 0,
) -> np.ndarray:
    """Approximate graph heat flow with a Torx PSWAP product formula."""
    probability = _edge_swap_probability(rate, total_time, steps)
    return sample_pswap_product_formula(
        initial,
        edges,
        reps=steps,
        swap_probability=probability,
        num_samples=num_samples,
        seed=seed,
    )
