"""Figures for notebook 05's exact, Gillespie, and Torx count comparisons."""

import matplotlib.pyplot as plt
import numpy as np
from _notebook_style import FIGURE_BG, NEUTRAL_GRAY
from matplotlib.lines import Line2D


def plot_species_dynamics(
    title,
    times,
    ref_counts,
    gil_counts,
    snap_times,
    torx_counts,
    series,
):
    """Plot exact, Gillespie, and Torx count means with residuals."""
    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(7.4, 5.5),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    species_handles = []
    for label, column, colour, marker in series:
        ax.plot(times, ref_counts[:, column], color=colour, lw=2.0)
        ax.plot(
            times,
            gil_counts[:, column],
            color=colour,
            lw=1.0,
            ls="--",
            alpha=0.8,
        )
        ax.scatter(
            snap_times,
            torx_counts[:, column],
            color=colour,
            marker=marker,
            s=36,
            edgecolor=FIGURE_BG,
            linewidth=0.8,
            zorder=5,
        )

        ref_at_snapshots = np.interp(snap_times, times, ref_counts[:, column])
        ax_res.plot(
            times,
            gil_counts[:, column] - ref_counts[:, column],
            color=colour,
            lw=1.0,
            ls="--",
            alpha=0.8,
        )
        ax_res.scatter(
            snap_times,
            torx_counts[:, column] - ref_at_snapshots,
            color=colour,
            marker=marker,
            s=30,
            edgecolor=FIGURE_BG,
            linewidth=0.8,
            zorder=5,
        )
        species_handles.append(
            Line2D(
                [],
                [],
                color=colour,
                marker=marker,
                lw=2.4,
                ms=5,
                markeredgecolor=FIGURE_BG,
            )
        )

    ax.set_ylabel("expected molecule count")
    ax.set_title(title)
    ax_res.axhline(0.0, color=NEUTRAL_GRAY, lw=0.8, alpha=0.7)
    ax_res.set_xlabel("time")
    ax_res.set_ylabel("residual\nvs exact")

    method_handles = [
        Line2D([], [], color=NEUTRAL_GRAY, lw=2.0, ls="-"),
        Line2D([], [], color=NEUTRAL_GRAY, lw=1.0, ls="--"),
        Line2D(
            [],
            [],
            color=NEUTRAL_GRAY,
            marker="o",
            ls="none",
            ms=6,
            markeredgecolor=FIGURE_BG,
        ),
    ]
    method_labels = [r"exact $e^{Qt}$", "Gillespie mean", "Torx sample mean"]

    species_legend = ax.legend(
        species_handles,
        [label for label, _, _, _ in series],
        title="species",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        alignment="left",
    )
    ax.add_artist(species_legend)
    ax.legend(
        method_handles,
        method_labels,
        title="method",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.48),
        frameon=False,
        alignment="left",
    )
    return fig
