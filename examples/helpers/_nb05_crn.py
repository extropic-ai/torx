"""Figure helper for the chemical-reaction-network notebook (05).

One function draws expected molecule counts over time for a reaction network,
overlaying the exact CTMC solution, a Gillespie average, and the Torx sample
mean. A species legend (colour) and a method legend (line style) sit outside the
axes so the encoding is self-documenting.
"""

import matplotlib.pyplot as plt
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
    """Overlay exact, Gillespie, and Torx trajectories of molecule counts.

    ``series`` is a list of ``(label, column, colour)`` triples, one per plotted
    curve. ``column`` indexes the species column in the count arrays, so a pair
    of species that coincide by a conservation law can share a single curve.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    species_handles = []
    for label, column, colour in series:
        ax.plot(times, ref_counts[:, column], color=colour, lw=2.0)
        ax.plot(times, gil_counts[:, column], color=colour, lw=1.0, ls="--", alpha=0.7)
        ax.scatter(
            snap_times,
            torx_counts[:, column],
            color=colour,
            s=34,
            edgecolor=FIGURE_BG,
            linewidth=0.8,
            zorder=5,
        )
        species_handles.append(Line2D([], [], color=colour, lw=2.4))

    ax.set_xlabel("time")
    ax.set_ylabel("expected molecule count")
    ax.set_title(title)

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
    method_labels = [r"exact $e^{Qt}$", "Gillespie", "Torx sample"]

    species_legend = ax.legend(
        species_handles,
        [label for label, _, _ in series],
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
        bbox_to_anchor=(1.02, 0.52),
        frameon=False,
        alignment="left",
    )
    fig.tight_layout()
    return fig
