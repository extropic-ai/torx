"""MaxCut stochastic graph network for nb09.

The circuit applies one ``PISING`` gate per edge in sequence, then repeats that
sweep. ``StateVectorSimulator`` supplies the exact differentiable probability
vector used by the notebook's REINFORCE score terms. The exact expected cut is
recorded only as a diagnostic. This module also supplies graph references and
figures.
"""

from collections.abc import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from _notebook_style import (
    COARSE_COLOR,
    EXACT_COLOR,
    FIGURE_BG,
    NEUTRAL_GRAY,
    TORX_COLOR,
)
from matplotlib.lines import Line2D

from torx.psc import DiscretePCircuit, PISING, StateVectorSimulator

_Edges = Sequence[tuple[int, int]]


# --- Graph and exact reference ----------------------------------------------


def random_regular_maxcut(num_nodes: int, degree: int, *, seed: int):
    """A connected random regular graph and its edge list.

    Connectivity is not required by the cut objective; it only keeps the drawn
    partition figure readable, so we resample the seed rather than fail on one.
    """
    trial = seed
    graph = nx.random_regular_graph(degree, num_nodes, seed=trial)
    while not nx.is_connected(graph):
        trial += 1
        graph = nx.random_regular_graph(degree, num_nodes, seed=trial)
    edges = [(int(i), int(j)) for i, j in graph.edges()]
    return graph, edges


def spin_table(num_nodes: int) -> np.ndarray:
    """Every basis state as a row of spins in ``{-1, +1}`` (big-endian)."""
    bits = (np.arange(2**num_nodes)[:, None] >> np.arange(num_nodes - 1, -1, -1)) & 1
    return (2 * bits - 1).astype(np.float64)


def brute_force_cuts(edges: _Edges, num_nodes: int):
    """Cut value of every bitstring, the optimum, and how many achieve it."""
    spins = spin_table(num_nodes)
    cut_values = np.zeros(2**num_nodes)
    for i, j in edges:
        cut_values += (1.0 - spins[:, i] * spins[:, j]) / 2.0
    opt = int(round(cut_values.max()))
    n_optimal = int((np.rint(cut_values) == opt).sum())
    return cut_values, opt, n_optimal


def edge_pising_matrix(J: float, *, beta: float, dt: float) -> np.ndarray:
    """4x4 column-stochastic transition matrix for one PISING edge."""
    # gate structure is static; trainable values enter through the theta pytree
    gate = PISING([0, 1])
    theta = jnp.array([J, 0.0, 0.0, beta, dt])
    return np.asarray(gate.get_matrix(theta))


# --- Exact classical probability vector and expected-cut diagnostic --------


def make_expected_cut(
    edges: _Edges, num_nodes: int, *, beta: float, dt: float, reps: int
):
    """Return ``(expected_cut, density)`` as differentiable functions of ``J``.

    ``density(J)`` propagates the uniform start through the ``PISING`` gates in
    edge-list order, then repeats that sequential sweep ``reps`` times. It
    returns the exact output distribution. ``expected_cut(J)`` reads and sums
    the per-edge cut from that distribution. Both functions are pure JAX.
    """
    sim = StateVectorSimulator()
    spins = jnp.asarray(spin_table(num_nodes))
    x0 = jnp.ones(2**num_nodes) / float(2**num_nodes)
    edge_sites = [[int(i), int(j)] for i, j in edges]
    # gate structure is static; trainable couplings J enter through theta pytrees
    circuit = DiscretePCircuit([PISING(sites) for sites in edge_sites], reps=reps)
    edge_i = jnp.asarray([i for i, _ in edge_sites])
    edge_j = jnp.asarray([j for _, j in edge_sites])

    def density(J):
        thetas = [
            jnp.stack([J[e], 0.0, 0.0, jnp.asarray(beta), jnp.asarray(dt)])
            for e in range(len(edge_sites))
        ]
        compiled = sim.build_circuit(circuit, thetas)
        return sim.density(compiled, x0)

    def expected_cut(J):
        rho = density(J)
        edge_products = spins[:, edge_i] * spins[:, edge_j]
        correlations = rho @ edge_products
        return jnp.sum((1.0 - correlations) / 2.0)

    return expected_cut, density


def cut_distribution(
    rho: np.ndarray, cut_values: np.ndarray, num_edges: int
) -> np.ndarray:
    """Probability mass by integer cut value under a state distribution ``rho``."""
    return np.bincount(
        np.rint(cut_values).astype(int),
        weights=np.asarray(rho),
        minlength=num_edges + 1,
    )


# --- Plots ------------------------------------------------------------------


def plot_training_trajectory(
    exact_history, sampled_history, *, opt, uniform_mean, num_samples
):
    """Plot the sampled objective and exact expected-cut diagnostic."""
    exact_history = np.asarray(exact_history, dtype=float)
    sampled_history = np.asarray(sampled_history, dtype=float)
    exact_steps = np.arange(len(exact_history))
    sampled_steps = np.arange(len(sampled_history))
    fig, ax = plt.subplots(figsize=(6.2, 4.2), layout="constrained")

    ax.axhline(
        opt, color=NEUTRAL_GRAY, linestyle=":", linewidth=1.1, label=f"optimum ({opt})"
    )
    ax.axhline(
        uniform_mean,
        color=COARSE_COLOR,
        linestyle="--",
        linewidth=1.0,
        label=f"uniform ({uniform_mean:.1f})",
    )
    ax.plot(
        exact_steps,
        exact_history,
        color=TORX_COLOR,
        linewidth=2.0,
        label="exact simulator diagnostic",
    )
    ax.plot(
        sampled_steps,
        sampled_history,
        color=EXACT_COLOR,
        linewidth=0.8,
        alpha=0.55,
        label=f"sampled batch objective (n={num_samples})",
    )

    ax.set_xlim(0, len(exact_history) - 1)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("cut value")
    ax.set_title("REINFORCE training diagnostics")
    # opaque frame keeps the legend legible over the reference lines
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor=FIGURE_BG,
        edgecolor="#CFC8B8",
        framealpha=0.96,
        borderpad=0.7,
        fontsize=8,
    )
    ax.grid(
        True, axis="y", color=NEUTRAL_GRAY, linestyle=":", linewidth=0.6, alpha=0.28
    )
    return fig


def plot_cut_distribution(cut_support, uniform_dist, learned_dist, *, opt):
    """Probability mass by cut value for the uniform baseline and the learned circuit."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0), layout="constrained")
    width = 0.4
    ax.bar(
        cut_support - width / 2,
        uniform_dist,
        width,
        color=COARSE_COLOR,
        label="uniform",
        alpha=0.9,
    )
    ax.bar(
        cut_support + width / 2,
        learned_dist,
        width,
        color=TORX_COLOR,
        label="learned",
    )
    ax.axvline(opt, color=NEUTRAL_GRAY, linestyle=":", linewidth=1.1)
    ax.annotate(
        "optimum",
        xy=(opt, ax.get_ylim()[1] * 0.92),
        ha="right",
        va="top",
        fontsize=8,
        color="0.3",
        rotation=90,
        xytext=(opt - 0.12, ax.get_ylim()[1] * 0.92),
    )
    ax.set_xlabel("cut value")
    ax.set_ylabel("probability")
    ax.set_title("Where each sampler puts its probability mass")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.margins(x=0.02)
    return fig


def plot_partition(
    G, partition, *, learned_cut, num_edges, layout_seed, figsize=(6.2, 5.6)
):
    """Draw the learned MaxCut partition.

    Two node colors (``EXACT_COLOR`` / ``TORX_COLOR``) for the two sides; cut
    edges in ``COARSE_COLOR`` (gold), within-side edges in neutral gray; node
    labels in white on the markers.
    """
    pos = nx.spring_layout(G, seed=layout_seed, k=0.35, iterations=150)
    edges = list(G.edges())
    cut_edges = [(i, j) for i, j in edges if partition[i] != partition[j]]
    noncut_edges = [(i, j) for i, j in edges if partition[i] == partition[j]]
    node_colors = [TORX_COLOR if partition[n] else EXACT_COLOR for n in G.nodes()]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=noncut_edges,
        edge_color=NEUTRAL_GRAY,
        width=1.1,
        alpha=0.7,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=cut_edges,
        edge_color=COARSE_COLOR,
        width=2.0,
        alpha=0.85,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.9,
        node_size=520,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=10, font_color="white", ax=ax)
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="white",
            markerfacecolor=EXACT_COLOR,
            markeredgecolor="white",
            markersize=8,
            label="side A",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color="white",
            markerfacecolor=TORX_COLOR,
            markeredgecolor="white",
            markersize=8,
            label="side B",
        ),
        Line2D([], [], color=COARSE_COLOR, lw=2.0, label="cut edge"),
        Line2D([], [], color=NEUTRAL_GRAY, lw=1.1, label="uncut edge"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.4,
    )
    ax.set_title(f"Learned partition, cut {learned_cut}/{num_edges}", pad=4)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.grid(False)
    ax.margins(0.06)
    return fig
