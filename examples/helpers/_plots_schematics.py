"""Schematic figures: circuit, factor, graph, lattice and gate-panel diagrams."""

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from _notebook_style import (
    COARSE_COLOR,
    EXACT_COLOR,
    EXTROPIC_COPPER,
    EXTROPIC_DIVERGING,
    EXTROPIC_FUCHSIA,
    EXTROPIC_GOLD,
    EXTROPIC_ORANGE,
    format_mu_label,
    HEAT_CMAP,
    NEUTRAL_GRAY,
    REGIME_COLORS,
    TORX_COLOR,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
)

RING_SITE_B_COLOR = TORX_COLOR

RING_FOCUS_COLOR = "0.35"

FIGSIZE_SCHEMATIC = (6.2, 4.6)

_COLPOOL_COLOR = EXTROPIC_GOLD

_ROWPOOL_COLOR = EXTROPIC_COPPER

_READOUT_COLOR = EXACT_COLOR


CLUSTER_COLORS = [EXACT_COLOR, TORX_COLOR, COARSE_COLOR]

GRID = 4


# circuit-diagram drawing:
# labeled gate boxes on horizontal wires, in the style of the Torx notebooks.

# map a gate class name (lower-cased) to (display label, role color)
_GATE_STYLE = {
    "pising": ("PISING", EXTROPIC_ORANGE),
    "pswap": ("PSWAP", EXTROPIC_ORANGE),
    "pcnot": ("PCNOT", EXTROPIC_ORANGE),
    "pcswap": ("PCSWAP", EXTROPIC_ORANGE),
    "pjump": ("PJUMP", EXTROPIC_ORANGE),
    "pmulticnot": ("PMCNOT", EXTROPIC_ORANGE),
    "por": ("POR", EXTROPIC_ORANGE),
    "pnot": ("PNOT", EXACT_COLOR),
    "pcopy": ("PCopy", EXACT_COLOR),
    "pdemux": ("PDEMUX", EXACT_COLOR),
    "preset": ("PReset", EXACT_COLOR),
    "pditcycle": ("PditCycle", EXTROPIC_COPPER),
    "pditshift": ("PditShift", EXTROPIC_COPPER),
    "pditswap": ("PditSWAP", EXTROPIC_COPPER),
    "affinegaussiangate": ("AffineGauss", EXTROPIC_FUCHSIA),
    "mixturegaussiangate": ("MoG", EXTROPIC_FUCHSIA),
    "gaussiannoisegate": ("Gauss", EXTROPIC_FUCHSIA),
    "jumpdiffusiongate": ("JumpDiff", EXTROPIC_FUCHSIA),
    # public affine-Gaussian gates, keyed by class name
    "displace": ("Displace", EXTROPIC_FUCHSIA),
    "scale": ("Scale", EXTROPIC_FUCHSIA),
    "mix": ("Mix", EXTROPIC_FUCHSIA),
    "diffuse": ("Diffuse", EXTROPIC_FUCHSIA),
}

# display label (lower-cased) -> role color, derived from the class-name styles
# above so hand-written specs like ("MoG", [...]) resolve to the same color.
_LABEL_COLOR = {label.lower(): color for label, color in _GATE_STYLE.values()}

# fixed diagram geometry (no call site overrides these)
_COL_SPACING = 1.45
_BOX_W = 1.02
_BOX_H = 0.62
_FONTSIZE = 9.5


def _gate_box(ax, x, cy, bw, height, label, color):
    """Draw one rounded gate box with its centered label."""
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x - bw / 2, cy - height / 2),
            bw,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.7,
            edgecolor=color,
            facecolor="white",
            zorder=3,
        )
    )
    ax.text(
        x,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=_FONTSIZE,
        color=color,
        fontweight="bold",
        zorder=4,
    )


def _normalize_sites(sites) -> list[int]:
    if isinstance(sites, (int, np.integer)):
        return [int(sites)]
    if isinstance(sites, dict):
        # HybridSites mixes separate discrete/continuous index spaces; the diagram
        # cannot place those unambiguously, so the caller must pass explicit specs
        raise TypeError(
            "hybrid (continuous) gates have ambiguous wire indices; pass "
            "draw_pcircuit an explicit list of (label, sites) specs instead"
        )
    return [int(s) for s in sites]


def _circuit_specs(circuit, *, label_map=None) -> list[tuple[str, list[int]]]:
    """Extract ``(label, sites)`` specs from a Torx circuit."""
    label_map = {k.lower(): v for k, v in (label_map or {}).items()}
    specs = []
    for gate in circuit.gates:
        name = type(gate).__name__
        key = name.lower()
        # prefer the gate's own draw label so diagrams track the library
        draw_label = getattr(gate, "_draw_label", None)
        label = (
            label_map.get(key) or draw_label or _GATE_STYLE.get(key, (name, None))[0]
        )
        specs.append((label, _normalize_sites(gate.sites)))
    return specs


def _spec_color(label: str, sites: list[int]) -> str:
    color = _LABEL_COLOR.get(label.lower())
    if color is not None:
        return color
    # unknown label: color by arity (two-body = coupling = orange)
    return EXTROPIC_ORANGE if len(sites) > 1 else NEUTRAL_GRAY


def _pack_columns(specs):
    """Greedy left-to-right packing: disjoint adjacent gates share a column."""
    columns: list[set[int]] = []
    col_of: list[int] = []
    for _, sites in specs:
        s = set(sites)
        if columns and not (columns[-1] & s):
            columns[-1] |= s
            col_of.append(len(columns) - 1)
        else:
            columns.append(set(s))
            col_of.append(len(columns) - 1)
    return col_of, len(columns)


def draw_pcircuit(
    circuit_or_specs,
    *,
    ax=None,
    wire_labels=None,
    title=None,
    reps=None,
    label_map=None,
    figsize=None,
):
    """Draw a Torx circuit as a clean labeled gate diagram.

    Parameters
    ----------
    circuit_or_specs : Torx circuit or list of (label, sites)
        A circuit exposing ``.gates`` (each gate with ``.sites``), or an
        explicit list of specs for full control (useful for hybrid circuits
        whose discrete/continuous registers share indices).
    wire_labels : list of str, optional
        Left-margin labels, top wire first. Defaults to ``$p_0, p_1, ...$``.
    reps : int, optional
        If given, wrap the gates in a dashed block annotated ``x reps``.
    """
    if hasattr(circuit_or_specs, "gates"):
        specs = _circuit_specs(circuit_or_specs, label_map=label_map)
    else:
        specs = [(lbl, _normalize_sites(s)) for lbl, s in circuit_or_specs]

    n_wires = 1 + max(w for _, sites in specs for w in sites)
    col_of, n_cols = _pack_columns(specs)

    def _box_width(label: str) -> float:
        return max(_BOX_W, 0.30 + 0.145 * len(label))

    widest = max(_box_width(lbl) for lbl, _ in specs)
    col_spacing = max(_COL_SPACING, widest + 0.45)

    def y_of(wire: int) -> float:
        return n_wires - 1 - wire

    if figsize is None:
        figsize = (1.2 + col_spacing * n_cols, 0.5 + 0.85 * n_wires)
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    half = widest / 2
    x_left = -half - 0.35
    x_right = (n_cols - 1) * col_spacing + half + 0.35
    for wire in range(n_wires):
        y = y_of(wire)
        ax.plot([x_left, x_right], [y, y], color="black", lw=1.1, zorder=1)
        label = (
            wire_labels[wire]
            if wire_labels and wire < len(wire_labels)
            else f"$p_{{{wire}}}$"
        )
        ax.text(
            x_left - 0.18,
            y,
            label,
            ha="right",
            va="center",
            fontsize=_FONTSIZE,
            clip_on=False,
        )

    for (label, sites), col in zip(specs, col_of, strict=True):
        x = col * col_spacing
        color = _spec_color(label, sites)
        bw = _box_width(label)
        wmin, wmax = min(sites), max(sites)
        contiguous = (wmax - wmin) == (len(sites) - 1)
        if len(sites) == 1 or contiguous:
            ylo, yhi = y_of(wmax), y_of(wmin)
            cy = 0.5 * (ylo + yhi)
            height = (yhi - ylo) + _BOX_H
            _gate_box(ax, x, cy, bw, height, label, color)
        else:
            ax.plot(
                [x, x],
                [y_of(wmax) + _BOX_H / 2, y_of(wmin) - _BOX_H / 2],
                color=color,
                lw=1.7,
                zorder=2,
            )
            for w in sites:
                _gate_box(ax, x, y_of(w), bw, _BOX_H, label, color)

    if reps:
        pad = 0.42
        rect = mpatches.FancyBboxPatch(
            (-widest / 2 - pad, -0.5 - pad),
            (n_cols - 1) * col_spacing + widest + 2 * pad,
            (n_wires - 1) + _BOX_H + 2 * pad,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            linewidth=1.1,
            linestyle=(0, (4, 3)),
            edgecolor=NEUTRAL_GRAY,
            facecolor="none",
            zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            x_right + 0.15,
            -0.5 - pad - 0.05,
            rf"$\times\,{int(reps)}$",
            ha="left",
            va="top",
            fontsize=_FONTSIZE,
            color=NEUTRAL_GRAY,
        )

    longest_label = max(
        (len(wire_labels[w]) if wire_labels and w < len(wire_labels) else 4)
        for w in range(n_wires)
    )
    left_margin = 0.5 + 0.13 * longest_label
    xlim_lo = x_left - left_margin
    xlim_hi = x_right + (1.0 if reps else 0.4)
    ax.set_xlim(xlim_lo, xlim_hi)
    if title:
        # center the title over the circuit content, not the label-padded axes
        content_mid = 0.5 * (x_left + x_right)
        title_x = (content_mid - xlim_lo) / (xlim_hi - xlim_lo)
        ax.set_title(title, fontsize=_FONTSIZE + 1.5, x=title_x)
    ax.set_ylim(-0.5 - (0.7 if reps else 0.45), (n_wires - 1) + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    if created:
        fig.tight_layout()
    return fig


def bar_chart(ax, dist, labels, title, *, ylabel=False):
    """Draw one output-distribution bar chart in the gate-demo style."""
    ax.bar(labels, dist, width=0.72, color=TORX_COLOR)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("output state")
    if ylabel:
        ax.set_ylabel("probability")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color=NEUTRAL_GRAY, linestyle=":", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.spines["left"].set_bounds(0.0, 1.0)


def transition_bars(dists, *, gate, p, figsize=None):
    """Grid of output-distribution bar charts, one per labeled input.

    ``dists`` is a list of ``(panel_title, output_labels, distribution)``. The
    suptitle is derived from the gate name and switching probability ``p``, and
    the figure width scales with the number of panels.
    """
    n = len(dists)
    if figsize is None:
        figsize = (min(6.2, 2.0 + 1.6 * n), 2.6)
    fig, axes = plt.subplots(
        1, n, figsize=figsize, sharey=True, constrained_layout=True, squeeze=False
    )
    axes = axes[0]
    for i, (ax, (panel_title, labels, dist)) in enumerate(
        zip(axes, dists, strict=True)
    ):
        bar_chart(ax, dist, labels, panel_title, ylabel=(i == 0))
    fig.suptitle(f"{gate} transition probabilities (p = {p})")
    return fig


def heatmap_matrix(ax, M, *, title, labels=None, annotate=False):
    """Draw one column-stochastic transition matrix as a 0..1 heatmap."""
    im = ax.imshow(M, cmap=HEAT_CMAP, vmin=0.0, vmax=1.0)
    n = M.shape[0]
    if labels is not None:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    else:
        ticks = list(range(n)) if n <= 4 else [0, n // 2, n - 1]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(axis="both", length=0)
    ax.set_aspect("equal")
    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if M[i, j] < 0.5 else "0.1",
                    fontsize=8,
                )
    return im


def pdit_matrices(matrices, *, dims, figsize=None):
    """Row of pdit transition heatmaps sharing one 0..1 colorbar.

    ``matrices`` is a list of ``(title, M)``. Each matrix is shown with the same
    ``HEAT_CMAP`` and ``vmin=0, vmax=1`` so the panels are directly comparable.
    A single shared colorbar labels probability. The suptitle reports ``dims``.
    """
    n = len(matrices)
    if figsize is None:
        figsize = (4.2, 3.6) if n == 1 else (6.2, 2.6)
    fig, axes = plt.subplots(1, n, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    last_im = None
    for ax, (title, M) in zip(axes, matrices, strict=True):
        last_im = heatmap_matrix(
            ax, M, title=None if n == 1 else title, annotate=(n == 1)
        )
    if n == 1:
        fig.suptitle(f"{matrices[0][0]} transition matrix (d = {dims})")
    else:
        fig.suptitle(f"Pdit transition matrices (d = {dims})")
    fig.supxlabel("input state")
    fig.supylabel("output state")
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.68, pad=0.02)
    cbar.set_label("probability")
    cbar.outline.set_visible(False)
    return fig


def ising_matrix(M, *, J, beta, dt, figsize=(3.8, 3.6)):
    """Single annotated 4x4 PISING transition heatmap with a colorbar.

    The title is derived from the bond coupling ``J``, inverse temperature
    ``beta``, and time step ``dt``.
    """
    labels = ["00", "01", "10", "11"]
    title = rf"PISING transition matrix ($J$={J}, $\beta$={beta}, $\Delta t$={dt})"
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = heatmap_matrix(ax, M, title=title, labels=labels, annotate=True)
    ax.set_xlabel("input state")
    ax.set_ylabel("output state")
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
    cbar.set_label("probability")
    cbar.outline.set_visible(False)
    return fig


def draw_graph(
    ax,
    graph,
    pos,
    mass,
    *,
    cycle_edges,
    chord_edges,
    vmax=None,
    title="",
    node_scale=1.0,
):
    """Render one graph snapshot: node color and size encode occupancy mass."""
    vmax = float(mass.max()) if vmax is None else vmax
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=cycle_edges,
        edge_color="0.80",
        width=1.4 * node_scale,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=chord_edges,
        edge_color="0.62",
        width=1.1 * node_scale,
        style="--",
        ax=ax,
    )
    sizes = (480 + 340 * np.clip(mass / max(vmax, 1e-6), 0.0, 1.0)) * node_scale
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=mass,
        node_size=sizes,
        cmap=HEAT_CMAP,
        vmin=0.0,
        vmax=vmax,
        edgecolors="0.78",
        linewidths=1.3,
        ax=ax,
    )
    labels = nx.draw_networkx_labels(
        graph,
        pos,
        font_color="white",
        font_size=max(6, round(4.5 + 4.5 * node_scale)),
        font_weight="bold",
        ax=ax,
    )
    for text in labels.values():
        text.set_path_effects([pe.withStroke(linewidth=1.8, foreground="0.15")])
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=4)


def diffusion_progression_figure(
    graph,
    pos,
    snapshots,
    snapshot_times,
    *,
    cycle_edges,
    chord_edges,
):
    """Small-multiples of heat flow spreading across the graph over time.

    One panel per snapshot, a shared color scale (clipped just below the
    initial peak so the spread stays visible), and a single colorbar.
    """
    n = len(snapshot_times)
    fig, axes_grid = plt.subplots(1, n, figsize=(6.2, 1.9))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.31, top=0.80, wspace=0.13)
    fig.suptitle("NumPy exact graph-diffusion reference", y=0.97, fontsize=9)
    axes = np.asarray(axes_grid).ravel()
    # clip the shared scale below the initial peak so later spread is visible
    vmax = float(np.quantile(snapshots[1:], 0.98))
    for i, (t, mass) in enumerate(zip(snapshot_times, snapshots, strict=True)):
        draw_graph(
            axes[i],
            graph,
            pos,
            mass,
            cycle_edges=cycle_edges,
            chord_edges=chord_edges,
            vmax=vmax,
            title=f"$t = {t:.2f}$",
            node_scale=0.46,
        )

    sm = plt.cm.ScalarMappable(cmap=HEAT_CMAP, norm=plt.Normalize(vmin=0.0, vmax=vmax))
    cax = fig.add_axes([0.33, 0.13, 0.34, 0.045])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(f"node occupancy, clipped above {vmax:.2f}", labelpad=1)
    cbar.ax.tick_params(labelsize=7, length=2, pad=1)
    cbar.outline.set_edgecolor(NEUTRAL_GRAY)
    return fig


def decomposition_schematic(
    *,
    toy_edges,
    toy_pos,
    highlight_edge,
    figsize=(6.2, 2.75),
):
    """Three-panel schematic: edge -> PSWAP -> layer."""
    edges = list(toy_edges)
    pos = dict(toy_pos)
    hi, hj = highlight_edge[0], highlight_edge[1]
    g = nx.Graph()
    g.add_edges_from(edges)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 45.0)
    ax.set_aspect("equal")
    ax.axis("off")

    centers = [18.0, 50.0, 82.0]
    cy = 23.5
    g_scale = 6.6

    def stage_arrow(x0, x1):
        ax.annotate(
            "",
            xy=(x1, cy),
            xytext=(x0, cy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=NEUTRAL_GRAY,
                lw=1.6,
                shrinkA=0,
                shrinkB=0,
                mutation_scale=14,
            ),
        )

    def stage_title(x, text):
        ax.text(
            x,
            39.0,
            text,
            ha="center",
            va="center",
            fontsize=9.5,
            color="0.25",
            fontweight="normal",
        )

    def stage_caption(x, text):
        ax.text(x, 5.0, text, ha="center", va="center", fontsize=9, color="0.30")

    def _draw_toy_graph(cx, all_edges):
        gp = {n: (cx + g_scale * pos[n][0], cy + g_scale * pos[n][1]) for n in g.nodes}
        node_r = 1.5  # stop edges just inside the node disk so labels keep clear space
        for a, b in edges:
            hot = all_edges or ({a, b} == {hi, hj})
            (ax0, ay0), (bx0, by0) = gp[a], gp[b]
            dx, dy = bx0 - ax0, by0 - ay0
            norm = np.hypot(dx, dy) or 1.0
            ux, uy = node_r * dx / norm, node_r * dy / norm
            ax.plot(
                [ax0 + ux, bx0 - ux],
                [ay0 + uy, by0 - uy],
                color=(EXTROPIC_ORANGE if hot else "0.78"),
                lw=(3.0 if hot else 1.4),
                zorder=1,
                solid_capstyle="round",
            )
        if all_edges:
            # a gate marker on every edge: each edge is now one PSWAP
            for a, b in edges:
                mx, my = (gp[a][0] + gp[b][0]) / 2, (gp[a][1] + gp[b][1]) / 2
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (mx - 1.15, my - 1.15),
                        2.3,
                        2.3,
                        boxstyle="round,pad=0.04,rounding_size=0.4",
                        linewidth=1.3,
                        edgecolor=EXTROPIC_ORANGE,
                        facecolor="white",
                        zorder=2.5,
                    )
                )
        for n in g.nodes:
            hot = all_edges or (n in (hi, hj))
            ax.scatter(
                [gp[n][0]],
                [gp[n][1]],
                s=150,
                color=(EXTROPIC_ORANGE if hot else "white"),
                edgecolors=(EXTROPIC_ORANGE if hot else "0.55"),
                linewidths=1.4,
                zorder=2,
            )
            ax.text(
                gp[n][0],
                gp[n][1],
                f"{n}",
                ha="center",
                va="center",
                fontsize=7.5,
                color=("white" if hot else "0.35"),
                fontweight="bold",
                zorder=3,
            )

    # ---- panel 1: the graph, with one edge picked out ----
    _draw_toy_graph(centers[0], all_edges=False)
    stage_title(centers[0], "one edge")
    stage_caption(centers[0], r"edge $\{i,j\}$")
    stage_arrow(centers[0] + 8.2, centers[1] - 12.6)

    # ---- panel 2: that edge as one PSWAP gate ----
    c2 = centers[1]
    wire_dx, wire_dy = 10.2, 3.8
    gbw, gbh = 12.2, 9.0
    for k, dy in enumerate((wire_dy, -wire_dy)):
        y = cy + dy
        ax.plot([c2 - wire_dx, c2 + wire_dx], [y, y], color="black", lw=1.1, zorder=1)
        ax.text(
            c2 - wire_dx - 0.6,
            y,
            r"$p_i$" if k == 0 else r"$p_j$",
            ha="right",
            va="center",
            fontsize=7.5,
            color="0.3",
        )
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (c2 - gbw / 2, cy - gbh / 2),
            gbw,
            gbh,
            boxstyle="round,pad=0.1,rounding_size=0.6",
            linewidth=1.7,
            edgecolor=EXTROPIC_ORANGE,
            facecolor="white",
            zorder=3,
        )
    )
    ax.text(
        c2,
        cy,
        "PSWAP",
        ha="center",
        va="center",
        fontsize=7.5,
        color=EXTROPIC_ORANGE,
        fontweight="bold",
        zorder=4,
    )
    stage_title(c2, "one PSWAP")
    stage_caption(c2, r"on $p_i, p_j$")
    stage_arrow(c2 + wire_dx + 1.0, centers[2] - 8.2)

    # ---- panel 3: every edge becomes a PSWAP, and together they form layer T ----
    _draw_toy_graph(centers[2], all_edges=True)
    stage_title(centers[2], r"one layer $T$")
    stage_caption(centers[2], r"$T = \prod_{\{i,j\}} \mathrm{PSWAP}_{ij}$")

    ax.set_title("Torx construction: one edge becomes one PSWAP", fontsize=10, pad=8)
    fig.tight_layout()
    return fig


def draw_edge_pswap(circuit) -> plt.Figure:
    """Draw one mesh edge as a single symmetric `PSWAP` on two adjacent pbits."""
    return draw_pcircuit(
        circuit,
        wire_labels=[r"$x_i$", r"$x_j$"],
        title="Torx PSWAP\n" r"$10 \leftrightarrow 01$ with $p=\sigma(\theta)$",
    )


def plot_chromatic_ring(*, N, ring_edges, colors, annotate_site=2):
    """Schematic of the ``N``-site ring and its even/odd two-coloring.

    Sites colored by parity class, with one focus site and its two incident
    edges and local field ``l_i = h_i + J_{left,i} s_left + J_{i,right} s_right``
    annotated with edge-specific couplings.
    """
    color_a, color_b = np.asarray(colors[0]), np.asarray(colors[1])
    class_of = {int(i): 0 for i in color_a}
    class_of.update({int(i): 1 for i in color_b})
    site_colors = (EXACT_COLOR, RING_SITE_B_COLOR)
    class_names = ("color A (even sites)", "color B (odd sites)")

    # site positions on the unit circle: site 0 at top, going clockwise
    theta = np.pi / 2 - 2 * np.pi * np.arange(N) / N
    pos = np.column_stack([np.cos(theta), np.sin(theta)])

    fig, ax = plt.subplots(figsize=(4.6, 4.6), constrained_layout=True)

    i0 = int(annotate_site)
    left = (i0 - 1) % N
    right = (i0 + 1) % N
    hot_edges = {tuple(sorted((i0, left))), tuple(sorted((i0, right)))}

    r = 0.13  # node radius
    for i, j in ring_edges:
        is_hot = tuple(sorted((i, j))) in hot_edges
        # trim each edge to the node boundary so a site label never sits on an edge
        u = (pos[j] - pos[i]) / np.linalg.norm(pos[j] - pos[i])
        p_i, p_j = pos[i] + u * r, pos[j] - u * r
        ax.plot(
            [p_i[0], p_j[0]],
            [p_i[1], p_j[1]],
            color=(RING_FOCUS_COLOR if is_hot else NEUTRAL_GRAY),
            lw=(2.4 if is_hot else 1.3),
            zorder=(2 if is_hot else 1),
            solid_capstyle="round",
        )

    for i in range(N):
        c = site_colors[class_of[i]]
        ax.add_patch(
            mpatches.Circle(
                pos[i],
                r,
                facecolor="white",
                edgecolor=c,
                linewidth=(2.6 if i == i0 else 1.6),
                zorder=4,
            )
        )
        ax.add_patch(
            mpatches.Circle(
                pos[i],
                r * 0.62,
                facecolor=c,
                edgecolor="none",
                alpha=0.22,
                zorder=4,
            )
        )
        ax.text(
            pos[i, 0],
            pos[i, 1],
            f"$s_{{{i}}}$",
            ha="center",
            va="center",
            fontsize=9.5,
            color=c,
            fontweight="bold",
            zorder=5,
        )

    out = pos[i0] / np.linalg.norm(pos[i0])
    tx, ty = pos[i0] + out * 0.62
    ha = "left" if out[0] >= 0 else "right"
    ax.annotate(
        rf"$\ell_{{{i0}}} = h_{{{i0}}} + J_{{{left},{i0}}}\,s_{{{left}}}"
        rf" + J_{{{i0},{right}}}\,s_{{{right}}}$",
        xy=(pos[i0, 0] + out[0] * (r + 0.02), pos[i0, 1] + out[1] * (r + 0.02)),
        xytext=(tx, ty),
        ha=ha,
        va="center",
        fontsize=9,
        color=RING_FOCUS_COLOR,
        arrowprops=dict(
            arrowstyle="-", color=RING_FOCUS_COLOR, lw=1.1, shrinkA=2, shrinkB=2
        ),
        zorder=6,
    )
    for nb_i in (left, right):
        mx, my = 0.5 * (pos[i0] + pos[nb_i])
        # ordered indices so the label matches the annotation's J_{left,i}/J_{i,right}
        lo, hi = sorted((i0, nb_i))
        ax.text(
            mx,
            my,
            rf"$J_{{{lo},{hi}}}$",
            ha="center",
            va="center",
            fontsize=8,
            color=RING_FOCUS_COLOR,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"),
            zorder=3,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=site_colors[k],
            markeredgewidth=1.8,
            markersize=9,
            label=class_names[k],
        )
        for k in (0, 1)
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    ax.set_title(f"{N}-site ring, two-coloring")
    ax.set_aspect("equal")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    return fig


def plot_kernel_circuit(kernel):
    """Draw the four-gate shared kernel (PJUMP -> PReset -> PNOT -> PJUMP).

    ``kernel`` is the ``DiscretePCircuit`` holding the first four gates of the
    full SCNN, the local stochastic convolution applied to one pixel pair.
    """
    return draw_pcircuit(
        kernel,
        wire_labels=[r"$p_i$", r"$p_j$"],
        title="Shared stochastic convolutional kernel",
        figsize=FIGSIZE_SCHEMATIC,
    )


def _grid_xy(site, n_side):
    """Map a flat pixel index ``r * n_side + c`` to (x, y), row 0 at the top."""
    r, c = divmod(int(site), n_side)
    return float(c), float(n_side - 1 - r)


def plot_weight_sharing_schematic(
    n_side,
    conv_pairs,
    col_pool_edges,
    row_pool_edges,
    readout_sites,
    n_gates,
    n_logits,
    *,
    title=None,
):
    """Spatial weight-sharing + pooling schematic over the pbit grid.

    Parameters
    ----------
    n_side : int
        Grid side length; sites are indexed ``r * n_side + c``.
    conv_pairs : list of (i, j)
        Horizontal adjacent pixel pairs the shared kernel acts on.
    col_pool_edges, row_pool_edges : list of (src, sink)
        Stage-1 and stage-2 ``PCNOT`` pooling edges, source into sink.
    readout_sites : sequence of int
        The pooled sites read out by the classifier head.
    n_gates, n_logits : int
        Total gate positions and shared logits, for the title.
    """
    readout = {int(s) for s in readout_sites}
    if title is None:
        title = f"Weight sharing: {n_gates} gate positions, {n_logits} shared logits"

    fig, ax = plt.subplots(figsize=FIGSIZE_SCHEMATIC, constrained_layout=True)

    # shared-kernel links on horizontal adjacent pairs, trimmed to the node
    # boundary so an endpoint never lands under a pbit-index label
    node_r = 0.24
    for i, j in conv_pairs:
        x0, y0 = _grid_xy(i, n_side)
        x1, y1 = _grid_xy(j, n_side)
        dx, dy = x1 - x0, y1 - y0
        d = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / d, dy / d
        ax.plot(
            [x0 + ux * node_r, x1 - ux * node_r],
            [y0 + uy * node_r, y1 - uy * node_r],
            color=EXTROPIC_ORANGE,
            lw=2.8,
            solid_capstyle="round",
            zorder=3,
            alpha=0.85,
        )

    # stage-1 column pooling
    for src, sink in col_pool_edges:
        x0, y0 = _grid_xy(src, n_side)
        x1, y1 = _grid_xy(sink, n_side)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": _COLPOOL_COLOR,
                "lw": 1.7,
                "shrinkA": 10,
                "shrinkB": 10,
                "connectionstyle": "arc3,rad=-0.45",
            },
            zorder=4,
        )

    # stage-2 row pooling
    for src, sink in row_pool_edges:
        x0, y0 = _grid_xy(src, n_side)
        x1, y1 = _grid_xy(sink, n_side)
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": _ROWPOOL_COLOR,
                "lw": 2.0,
                "shrinkA": 11,
                "shrinkB": 13,
                "connectionstyle": "arc3,rad=0.22",
            },
            zorder=4,
        )

    # pixel sites: readout filled, rest open circles
    for site in range(n_side * n_side):
        x, y = _grid_xy(site, n_side)
        is_readout = site in readout
        ax.add_patch(
            mpatches.Circle(
                (x, y),
                0.20,
                facecolor=_READOUT_COLOR if is_readout else "white",
                edgecolor=_READOUT_COLOR if is_readout else NEUTRAL_GRAY,
                linewidth=1.8 if is_readout else 1.1,
                zorder=5,
            )
        )
        ax.text(
            x,
            y,
            str(site),
            ha="center",
            va="center",
            fontsize=7.5,
            zorder=6,
            color="white" if is_readout else "0.30",
            fontweight="bold" if is_readout else "normal",
        )

    ax.set_xlim(-1.05, n_side - 1 + 0.85)
    ax.set_ylim(-1.0, n_side - 1 + 0.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # spatial hints: pixel-grid rows and columns (no numeric units)
    ax.text(
        -0.85, n_side - 1, "row 0", ha="right", va="center", fontsize=7, color="0.45"
    )
    ax.text(
        -0.85,
        0.0,
        f"row {n_side - 1}",
        ha="right",
        va="center",
        fontsize=7,
        color="0.45",
    )
    ax.text(0.0, -0.62, "col 0", ha="center", va="top", fontsize=7, color="0.45")
    ax.text(
        n_side - 1,
        -0.62,
        f"col {n_side - 1}",
        ha="center",
        va="top",
        fontsize=7,
        color="0.45",
    )

    handles = [
        Line2D(
            [0],
            [0],
            color=EXTROPIC_ORANGE,
            lw=2.8,
            label="shared 4-gate kernel: PJUMP, PReset, PNOT, PJUMP (logits 0-2)",
        ),
        Line2D(
            [0],
            [0],
            color=_COLPOOL_COLOR,
            lw=1.8,
            marker=">",
            markersize=5,
            label="column pool PCNOT (logit 3)",
        ),
        Line2D(
            [0],
            [0],
            color=_ROWPOOL_COLOR,
            lw=1.8,
            marker=">",
            markersize=5,
            label="row pool PCNOT (logit 4)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_READOUT_COLOR,
            markeredgecolor=_READOUT_COLOR,
            markersize=8,
            label=f"readout sites {sorted(readout)}",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        columnspacing=1.4,
        fontsize=7.0,
        handlelength=1.7,
        borderaxespad=0.0,
    )

    ax.set_title(title, fontsize=10)
    return fig


def plot_ssm_slice_circuit():
    """Draw directed transition and emission semantics for one SSM slice."""
    fig, ax = plt.subplots(figsize=(7.0, 2.8), layout="constrained")
    wire_y = [2.0, 1.0, 0.0]
    wire_labels = [
        r"$z_{t-1}$, passed through",
        r"$z_t$, written then passed",
        r"$x_t$, written",
    ]
    for y, label in zip(wire_y, wire_labels, strict=True):
        ax.annotate(
            "",
            xy=(9.4, y),
            xytext=(0.2, y),
            arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.1},
            zorder=1,
        )
        ax.text(0.0, y, label, ha="right", va="center", fontsize=_FONTSIZE)

    boxes = [
        (3.0, 0.75, 1.8, 1.5, "transition\n$A, Q$\nwrites $z_t$"),
        (7.0, -0.25, 1.8, 1.5, "emission\n$C, R$\nwrites $x_t$"),
    ]
    for x, y, width, height, label in boxes:
        patch = mpatches.FancyBboxPatch(
            (x - width / 2, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="white",
            edgecolor=TORX_COLOR,
            linewidth=1.8,
            zorder=2,
        )
        ax.add_patch(patch)
        ax.text(
            x, y + height / 2, label, ha="center", va="center", fontsize=8, zorder=3
        )

    ax.set_xlim(-2.2, 9.7)
    ax.set_ylim(-0.55, 2.45)
    ax.set_title("One time-homogeneous state-space slice", fontsize=10)
    ax.axis("off")
    return fig


def plot_terminal_field(graph, pos, mean_soft_spin):
    """Draw the terminal mean soft spin on the graph."""
    fig, ax_graph = plt.subplots(figsize=(5.0, 4.4))
    nx.draw_networkx_edges(graph, pos, ax=ax_graph, edge_color="0.78", width=1.2)
    nodes_artist = nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=mean_soft_spin,
        cmap=EXTROPIC_DIVERGING,
        vmin=-1,
        vmax=1,
        node_size=900,
        edgecolors="0.25",
        linewidths=1.2,
        ax=ax_graph,
    )
    for idx, node in enumerate(graph.nodes):
        spin = float(mean_soft_spin[idx])
        txt = ax_graph.text(
            pos[node][0],
            pos[node][1],
            f"{node}\n{spin:+.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color="0.15",
            zorder=5,
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    xs = [pos[node][0] for node in graph.nodes]
    ys = [pos[node][1] for node in graph.nodes]
    margin = 0.25
    ax_graph.set_xlim(min(xs) - margin, max(xs) + margin)
    ax_graph.set_ylim(min(ys) - margin, max(ys) + margin)
    ax_graph.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(nodes_artist, ax=ax_graph, shrink=0.82, pad=0.03)
    cbar.set_label(r"mean $\tanh(x_i)$")
    ax_graph.set_title("Terminal soft-spin field, node ID and mean")
    ax_graph.axis("off")
    fig.tight_layout()
    return fig


def draw_regime_chain(
    *,
    mu,
    lambda_plus: float,
    lambda_minus: float,
    node_colors=REGIME_COLORS,
    title: str = "Regime transition graph",
):
    """Draw the cyclic 3-regime transition graph as a clean on-brand schematic."""
    K = len(node_colors)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_aspect("equal")
    ax.axis("off")

    angles = [np.pi / 2 + 2 * np.pi * k / K for k in range(K)]
    pos = np.array([(np.cos(a), np.sin(a)) for a in angles])

    node_r = 0.30  # node radius in data units, edges stop at the rim

    def _rim(p_from, p_to, rad):
        """Endpoint on the node rim along the (curved) edge direction."""
        d = p_to - p_from
        # rotate toward the arc to meet the rim cleanly
        theta = rad
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        d = rot @ (d / np.linalg.norm(d))
        return p_from + d * node_r

    rad = 0.22

    for k in range(K):
        j = (k + 1) % K
        start = _rim(pos[k], pos[j], -rad)
        end = _rim(pos[j], pos[k], rad)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=18,
                lw=2.2,
                color=node_colors[k],
                alpha=0.85,
                zorder=2,
            )
        )
        start_b = _rim(pos[j], pos[k], -rad)
        end_b = _rim(pos[k], pos[j], rad)
        ax.add_patch(
            FancyArrowPatch(
                start_b,
                end_b,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=13,
                lw=1.3,
                linestyle="--",
                color="0.45",
                alpha=0.9,
                zorder=2,
            )
        )

    for k in range(K):
        mu_label = format_mu_label(mu[k])
        ax.add_patch(
            Circle(
                pos[k],
                node_r,
                facecolor=node_colors[k],
                edgecolor="#1C0101",
                linewidth=1.6,
                zorder=3,
            )
        )
        txt = ax.text(
            pos[k, 0],
            pos[k, 1],
            f"{k}\n$\\mu={mu_label}$",
            ha="center",
            va="center",
            color="white",
            fontsize=10.5,
            fontweight="bold",
            zorder=4,
        )
        txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground="#1C0101")])

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="0.3",
            lw=2.2,
            label=rf"$\lambda_+ = {lambda_plus}$  forward",
        ),
        Line2D(
            [0],
            [0],
            color="0.45",
            lw=1.3,
            ls="--",
            label=rf"$\lambda_- = {lambda_minus}$  backward",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=2,
        fontsize=9,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.6,
    )
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.45)
    fig.tight_layout()
    return fig


def cluster_gate_circuit():
    """Draw the registration shift followed by the conditional mixture gate."""
    return draw_pcircuit(
        [("PditShift\nnear-no-op", [0]), ("MoG", [0, 1])],
        wire_labels=[r"$|h)$ one-hot cluster", r"$|v)$ visible"],
        title="Gaussian-categorical conditional sampling circuit",
    )


def energy_factor_graph(K, *, active_cluster=1, figsize=(6.2, 4.4)):
    """Schematic of the Gaussian-categorical energy and its two conditional readings."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    v_xy = (0.0, 0.0)
    E_x = 3.1
    h_x = 6.2

    # central shared-energy factor (square, per factor-graph convention)
    fac = 1.05
    ax.add_patch(
        FancyBboxPatch(
            (E_x - fac / 2, -fac / 2),
            fac,
            fac,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.8,
            edgecolor=EXTROPIC_COPPER,
            facecolor="white",
            zorder=4,
        )
    )
    ax.text(
        E_x,
        0.0,
        r"$E(v,h)$",
        ha="center",
        va="center",
        fontsize=12,
        color=EXTROPIC_COPPER,
        fontweight="bold",
        zorder=5,
    )

    # visible node: continuous, a disk holding a small Gaussian glyph
    v_r = 0.62
    ax.add_patch(
        Circle(
            v_xy,
            v_r,
            facecolor="white",
            edgecolor=EXTROPIC_FUCHSIA,
            linewidth=1.8,
            zorder=4,
        )
    )
    gx = np.linspace(-0.40, 0.40, 60)
    gy = 0.34 * np.exp(-((gx / 0.18) ** 2)) - 0.20
    ax.plot(v_xy[0] + gx, v_xy[1] + gy, color=EXTROPIC_FUCHSIA, lw=1.5, zorder=5)
    ax.text(
        v_xy[0],
        1.10,
        r"visible $v\in\mathbb{R}^D$",
        ha="center",
        va="bottom",
        fontsize=10,
        color=EXTROPIC_FUCHSIA,
        fontweight="medium",
    )

    # hidden node: K stacked one-hot cells, exactly one filled
    cell = 0.42
    h_top = (K * cell) / 2
    for k in range(K):
        cy = h_top - (k + 0.5) * cell
        active = k == active_cluster
        color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.add_patch(
            FancyBboxPatch(
                (h_x - cell / 2, cy - cell / 2),
                cell,
                cell,
                boxstyle="square,pad=0.0",
                linewidth=1.5,
                edgecolor=color,
                facecolor=color if active else "white",
                zorder=4,
            )
        )
        ax.text(
            h_x,
            cy,
            "1" if active else "0",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=5,
            color="white" if active else color,
        )
        ax.text(
            h_x + cell / 2 + 0.14,
            cy,
            rf"$h_{k}$",
            ha="left",
            va="center",
            fontsize=8.5,
            color=NEUTRAL_GRAY,
        )
    ax.text(
        h_x,
        h_top + 0.47,
        r"one-hot hidden $h=e_k$",
        ha="center",
        va="bottom",
        fontsize=10,
        color=NEUTRAL_GRAY,
        fontweight="medium",
    )

    # factor-graph edges, with the coupling matrix W living on the h-side edge
    ax.plot([v_xy[0] + v_r, E_x - fac / 2], [0, 0], color="black", lw=1.2, zorder=1)
    ax.plot([E_x + fac / 2, h_x - cell / 2], [0, 0], color="black", lw=1.2, zorder=1)
    ax.text(
        (E_x + fac / 2 + h_x - cell / 2) / 2,
        0.18,
        r"$W$",
        ha="center",
        va="bottom",
        fontsize=10,
        color=EXTROPIC_COPPER,
    )

    # the one energy, composed as one line anchoring the figure; roles under each term
    eq_y, role_y = -2.98, -3.44
    ax.text(
        1.75,
        eq_y,
        r"$E(v,h)\;=$",
        ha="right",
        va="center",
        fontsize=10.5,
        color="0.2",
    )
    eq_terms = [
        (2.45, r"$\dfrac{\|v-a\|^2}{2\sigma^2}$", "visible", EXTROPIC_FUCHSIA),
        (3.10, r"$-$", None, None),
        (3.55, r"$c_k\,h_k$", "corrected bias", NEUTRAL_GRAY),
        (4.13, r"$-$", None, None),
        (4.80, r"$\dfrac{v}{\sigma}\!\cdot\! W h$", "coupling", EXTROPIC_COPPER),
    ]
    for tx, expr, role, col in eq_terms:
        ax.text(tx, eq_y, expr, ha="center", va="center", fontsize=10.5, color="0.2")
        if role:
            ax.text(tx, role_y, role, ha="center", va="top", fontsize=7.5, color=col)

    # reading 1 (top): clamp v, the hidden conditional is a softmax
    ax.add_patch(
        FancyArrowPatch(
            (v_xy[0] + 0.46, 0.46),
            (h_x - cell / 2 - 0.12, h_top - 0.13),
            connectionstyle="arc3,rad=-0.28",
            arrowstyle="-|>",
            mutation_scale=15,
            lw=1.8,
            color=TORX_COLOR,
            zorder=6,
        )
    )
    ax.text(
        E_x,
        1.98,
        r"fix $v$:  $p(h\mid v)=\mathrm{softmax}_k\,\theta_k(v)$",
        ha="center",
        va="bottom",
        fontsize=10,
        color=TORX_COLOR,
        fontweight="medium",
    )

    # reading 2 (bottom): clamp h=e_k, the visible conditional is that cluster's Gaussian
    ax.add_patch(
        FancyArrowPatch(
            (h_x - cell / 2 - 0.12, -h_top - 0.14),
            (v_xy[0] + 0.46, -0.46),
            connectionstyle="arc3,rad=-0.45",
            arrowstyle="-|>",
            mutation_scale=15,
            lw=1.8,
            color=EXACT_COLOR,
            zorder=6,
        )
    )
    ax.text(
        E_x,
        -2.10,
        r"fix $h=e_k$:  $p(v\mid h)=\mathcal{N}(\mu_k,\Sigma)$",
        ha="center",
        va="top",
        fontsize=10,
        color=EXACT_COLOR,
        fontweight="medium",
    )

    ax.set_xlim(-1.35, 7.7)
    ax.set_ylim(-3.85, 2.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "One energy, two readings: Gaussian-categorical conditionals", fontsize=12
    )
    return fig


def plot_factor_anatomy() -> Figure:
    """Draw a factor as a sampler with named ports, external parameters, and one output."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box = mpatches.Rectangle(
        (0.34, 0.32),
        0.30,
        0.36,
        facecolor="white",
        edgecolor=EXACT_COLOR,
        linewidth=1.8,
    )
    ax.add_patch(box)
    ax.text(
        0.49, 0.53, "factor", ha="center", va="center", fontsize=13, color=EXACT_COLOR
    )
    ax.text(
        0.49,
        0.43,
        r"$P(\mathrm{out}\mid\mathrm{inputs})$",
        ha="center",
        va="center",
        fontsize=10,
    )

    for y, label in ((0.58, "port a"), (0.42, "port b")):
        ax.annotate(
            "",
            xy=(0.34, y),
            xytext=(0.14, y),
            arrowprops=dict(arrowstyle="->", color=TORX_COLOR, lw=1.6),
        )
        ax.text(0.12, y, label, ha="right", va="center", color=TORX_COLOR, fontsize=9)

    ax.annotate(
        "",
        xy=(0.49, 0.68),
        xytext=(0.49, 0.88),
        arrowprops=dict(arrowstyle="->", color=NEUTRAL_GRAY, lw=1.4),
    )
    ax.text(
        0.49,
        0.91,
        "parameters",
        ha="center",
        va="bottom",
        color=NEUTRAL_GRAY,
        fontsize=9,
    )

    ax.annotate(
        "",
        xy=(0.84, 0.50),
        xytext=(0.64, 0.50),
        arrowprops=dict(arrowstyle="->", color=EXACT_COLOR, lw=1.8),
    )
    ax.text(
        0.86, 0.50, "one sample", ha="left", va="center", color=EXACT_COLOR, fontsize=9
    )

    ax.text(
        0.08,
        0.16,
        "no input ports → plain distribution\nA factor as a sampler",
        va="center",
        fontsize=9,
        color=NEUTRAL_GRAY,
    )
    ax.text(
        0.57,
        0.16,
        "with ports → conditional\nA factor with inputs",
        va="center",
        fontsize=9,
        color=NEUTRAL_GRAY,
    )
    ax.set_title("A factor reads inputs and parameters, then emits one sample")
    return fig


def plot_two_node_dfg() -> Figure:
    """Draw the two-site directed factor graph used in notebook 15."""
    fig, ax = plt.subplots(figsize=(6.2, 2.9), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    coin = mpatches.FancyBboxPatch(
        (0.18, 0.38),
        0.20,
        0.22,
        boxstyle="round,pad=0.04,rounding_size=0.035",
        facecolor="white",
        edgecolor=TORX_COLOR,
        linewidth=1.8,
    )
    out = mpatches.FancyBboxPatch(
        (0.62, 0.38),
        0.20,
        0.22,
        boxstyle="round,pad=0.04,rounding_size=0.035",
        facecolor="white",
        edgecolor=EXACT_COLOR,
        linewidth=1.8,
    )
    ax.add_patch(coin)
    ax.add_patch(out)
    ax.text(0.28, 0.51, "coin", ha="center", va="center", fontsize=12, color=TORX_COLOR)
    ax.text(
        0.28,
        0.43,
        "no parents",
        ha="center",
        va="center",
        fontsize=8,
        color=NEUTRAL_GRAY,
    )
    ax.text(0.72, 0.51, "out", ha="center", va="center", fontsize=12, color=EXACT_COLOR)
    ax.text(
        0.72,
        0.43,
        "parent coin",
        ha="center",
        va="center",
        fontsize=8,
        color=NEUTRAL_GRAY,
    )

    ax.annotate(
        "",
        xy=(0.62, 0.49),
        xytext=(0.38, 0.49),
        arrowprops=dict(arrowstyle="->", color=NEUTRAL_GRAY, lw=1.8),
    )
    ax.text(
        0.50, 0.56, "drive", ha="center", va="bottom", color=NEUTRAL_GRAY, fontsize=9
    )
    ax.text(
        0.50,
        0.18,
        "draws in topological order: coin first, out second",
        ha="center",
        va="center",
        fontsize=9,
        color=NEUTRAL_GRAY,
    )
    ax.set_title("A Torx DFG routes parent outputs onto child ports")
    return fig


def lattice_index(r, c, grid=GRID):
    """Wrap (row, col) onto the periodic grid and return the flat site index."""
    return (r % grid) * grid + (c % grid)


def lattice_neighbors(i, grid=GRID):
    """The four periodic neighbors (up, down, left, right) of site ``i``."""
    r, c = divmod(i, grid)
    return [
        lattice_index(r - 1, c, grid),
        lattice_index(r + 1, c, grid),
        lattice_index(r, c - 1, grid),
        lattice_index(r, c + 1, grid),
    ]


def lattice_edges(grid=GRID):
    """The ``2 * grid**2`` unique periodic neighbor connections of the grid."""
    edges = set()
    for i in range(grid * grid):
        for j in lattice_neighbors(i, grid):
            edges.add(tuple(sorted((i, j))))
    return sorted(edges)


def checkerboard_colors(grid=GRID):
    """The two checkerboard update groups A and B as lists of site indices."""
    n = grid * grid
    color_a = [i for i in range(n) if (i // grid + i % grid) % 2 == 0]
    color_b = [i for i in range(n) if (i // grid + i % grid) % 2 == 1]
    return color_a, color_b


def plot_factor_graph(grid=GRID):
    """The lattice the two tiled blocks act on: grid**2 spins, neighbor links, two colors."""
    n = grid * grid
    color_a, color_b = checkerboard_colors(grid)
    class_of = {i: 0 for i in color_a}
    class_of.update({i: 1 for i in color_b})
    site_colors = (EXACT_COLOR, EXTROPIC_ORANGE)
    class_names = ("color A (reads incoming state)", "color B (reads updated A)")

    pos = {i: np.array([i % grid, grid - 1 - i // grid], dtype=float) for i in range(n)}
    node_r = 0.32
    stub_len = 0.34

    fig, ax = plt.subplots(figsize=(5.4, 5.3), constrained_layout=True)

    # Interior neighbor links are trimmed to the node rim so they meet the disks
    # cleanly. Periodic (torus) links wrap to the opposite edge; drawing them
    # straight across the grid would run a line over the sites in between, so
    # each is shown as a short dashed stub off the boundary node it belongs to.
    for i, j in lattice_edges(grid):
        ri, ci = divmod(i, grid)
        rj, cj = divmod(j, grid)
        if abs(ri - rj) <= 1 and abs(ci - cj) <= 1:
            u = (pos[j] - pos[i]) / np.linalg.norm(pos[j] - pos[i])
            a, b = pos[i] + u * node_r, pos[j] - u * node_r
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color="0.68",
                lw=1.6,
                zorder=1,
                solid_capstyle="round",
            )
            continue
        for site, other in ((i, j), (j, i)):
            rs, _ = divmod(site, grid)
            ro, _ = divmod(other, grid)
            if rs == ro:  # horizontal wrap: point out through the left/right edge
                out = (
                    np.array([1.0, 0.0])
                    if pos[site][0] > pos[other][0]
                    else np.array([-1.0, 0.0])
                )
            else:  # vertical wrap: point out through the top/bottom edge
                out = (
                    np.array([0.0, 1.0])
                    if pos[site][1] > pos[other][1]
                    else np.array([0.0, -1.0])
                )
            start, end = pos[site] + out * node_r, pos[site] + out * (node_r + stub_len)
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="0.8",
                lw=1.4,
                ls=(0, (3, 3)),
                zorder=1,
                solid_capstyle="round",
            )

    for i in range(n):
        ax.add_patch(
            Circle(
                pos[i],
                node_r,
                facecolor=site_colors[class_of[i]],
                edgecolor="white",
                linewidth=2.0,
                zorder=4,
            )
        )
        ax.text(
            pos[i][0],
            pos[i][1],
            f"$s_{{{i}}}$",
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            zorder=5,
        )

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=site_colors[k],
            markeredgecolor="white",
            markeredgewidth=1.4,
            markersize=11,
            label=class_names[k],
        )
        for k in (0, 1)
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    ax.set_title(f"the {grid}x{grid} spin lattice")
    ax.set_aspect("equal")
    lim = grid - 1 + 0.95
    ax.set_xlim(-0.95, lim)
    ax.set_ylim(-0.95, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    return fig
