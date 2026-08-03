"""Figures for the basic_usage quickstart notebook."""

import matplotlib.pyplot as plt
import numpy as np
from _notebook_style import (
    EXACT_COLOR,
    FIGSIZE_CONVERGENCE,
    FIGSIZE_MULTIPANEL,
    NEUTRAL_GRAY,
    TORX_COLOR,
)


def _style_axes(ax):
    ax.grid(axis="y", color=NEUTRAL_GRAY, alpha=0.18, linestyle=":", linewidth=0.6)
    ax.tick_params(axis="both", length=3, width=0.6)


def training_loss(losses):
    """Log-scale MSE training loss against optimisation step."""
    loss = np.asarray(losses, dtype=float)
    steps = np.arange(len(loss))

    fig, ax = plt.subplots(figsize=FIGSIZE_CONVERGENCE, constrained_layout=True)
    ax.plot(steps, loss, color=TORX_COLOR, linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("MSE loss")
    ax.set_title("State-vector training loss")
    ax.set_xlim(float(steps[0]), float(steps[-1]))
    _style_axes(ax)
    return fig


def pising_relaxation(dists, dts, *, labels, coupling, beta):
    """Small multiples of the PISING output distribution as the step size grows.

    ``dists[k]`` is the probability over the four two-pbit configurations after
    running PISING for ``dts[k]``. Larger steps push the ferromagnet toward the
    aligned |00) and |11) states.
    """
    fig, axes = plt.subplots(
        1,
        len(dts),
        figsize=FIGSIZE_MULTIPANEL,
        sharey=True,
        constrained_layout=True,
    )
    for ax, dt, dist in zip(axes, dts, dists):
        ax.bar(labels, np.asarray(dist, dtype=float), color=TORX_COLOR, width=0.72)
        ax.set_title(f"dt = {dt:g}")
        ax.set_ylim(0, 0.6)
        _style_axes(ax)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("probability")
    fig.suptitle(
        f"PISING relaxation toward the ferromagnet (J={coupling:g}, \u03b2={beta:g})"
    )
    return fig


def gradient_comparison(sampled, exact):
    """Overlay the sampled param-shift loss on the exact state-vector loss.

    The exact curve is the reference; the sampled curve is the Monte Carlo
    estimate that should track it up to sampling noise.
    """
    sampled = np.asarray(sampled, dtype=float)
    exact = np.asarray(exact, dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE_CONVERGENCE, constrained_layout=True)
    ax.plot(
        np.arange(len(sampled)),
        sampled,
        color=TORX_COLOR,
        linewidth=0.8,
        alpha=0.6,
        label="sampled (param-shift)",
    )
    ax.plot(
        np.arange(len(exact)),
        exact,
        color=EXACT_COLOR,
        linewidth=1.8,
        label="exact (state vector)",
    )
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("expectation loss")
    ax.set_title("Sampled gradients track the exact ones")
    ax.set_xlim(0.0, float(max(len(sampled), len(exact)) - 1))
    ax.legend(loc="upper right")
    _style_axes(ax)
    return fig
