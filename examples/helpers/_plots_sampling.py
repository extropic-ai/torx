"""Sampler-quality diagnostics: convergence curves, parities, histograms, traces."""

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from _notebook_style import (
    add_reference_slope,
    COARSE_COLOR,
    EXACT_COLOR,
    EXTROPIC_DIVERGING,
    EXTROPIC_ORANGE,
    FIGSIZE_CONVERGENCE,
    FIGSIZE_MULTIPANEL,
    FIGSIZE_PARITY,
    FIGSIZE_SINGLE,
    NEUTRAL_GRAY,
    TORX_COLOR,
)
from _plot_utils import (
    _clean_axes,
    _padded_range,
    _parity_panel,
)
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedFormatter,
    NullFormatter,
)

from torx.psc import DiscretePCircuit, PNOT, SampleSimulator
from torx.psc.simulation.sampled import sample_circuit

_STATE_LABELS = [r"$|00)$", r"$|01)$", r"$|10)$", r"$|11)$"]

GIBBS_COLOR = "#2A9D8F"

MARGINAL_COLORS = {
    "exact": EXACT_COLOR,
    "chromatic Gibbs": GIBBS_COLOR,
    "per-edge PISING": EXTROPIC_ORANGE,
}

CLUSTER_COLORS = [EXACT_COLOR, TORX_COLOR, COARSE_COLOR]

GRID = 4

N_SPINS = GRID * GRID


def trotter_error_figure(
    steps_arr,
    det_errors,
    sam_mean,
    sam_std,
    *,
    num_samples: int = 12_000,
    num_sources: int = 1,
    num_seeds: int = 1,
):
    """Log-log plot of Trotter bias vs Monte Carlo noise over a step sweep.

    The deterministic product formula (slate) is pure Trotter bias and tracks the
    $1/m$ reference slope. The sampled Torx curve (orange) carries a +/-1 sigma
    band across seeds and plateaus at the Monte Carlo floor.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.3))

    ax.fill_between(
        steps_arr,
        np.maximum(sam_mean - sam_std, 1e-6),
        sam_mean + sam_std,
        color=TORX_COLOR,
        alpha=0.18,
    )
    ax.loglog(
        steps_arr,
        sam_mean,
        marker="o",
        linewidth=1.9,
        markersize=7,
        color=TORX_COLOR,
        label=(
            f"Torx mean +/- 1 SD, {num_seeds} seeds\n"
            f"{num_samples // 1000}k/source, "
            f"{num_samples * num_sources // 1000}k total/seed"
        ),
    )
    ax.loglog(
        steps_arr,
        det_errors,
        marker="s",
        linewidth=1.9,
        markersize=6,
        color=EXACT_COLOR,
        label="NumPy deterministic ordered mean",
    )

    add_reference_slope(
        ax,
        slope=-1.0,
        x_range=(float(steps_arr[0]), float(steps_arr[-1])),
        label=r"$1/m$ slope",
        anchor_y=float(det_errors[0]) / float(steps_arr[0]),
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(steps_arr)
    ax.set_xticklabels([str(int(s)) for s in steps_arr])
    ax.set_xlabel("Trotter steps $m$")
    ax.set_ylabel("$L_2$ error vs exact")
    ax.set_title("Torx sampling and NumPy Trotter bias")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.20)
    fig.tight_layout()
    return fig


def splitting_order_figure(
    steps_arr,
    lie_errors,
    strang_errors,
    *,
    torx_steps: int | None = None,
    torx_error: float | None = None,
    num_samples: int | None = None,
    num_sources: int = 1,
):
    """Log-log bias curves for first-order Lie-Trotter vs second-order Strang.

    Both use the same exact per-edge gates; only the ordering differs. The
    first-order curve (slate) tracks the $1/m$ slope, the Strang curve (orange)
    tracks the steeper $1/m^2$ slope, so reordering the same gates squares the
    convergence rate.
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.3))

    ax.loglog(
        steps_arr,
        lie_errors,
        marker="s",
        linewidth=1.9,
        markersize=6,
        color=EXACT_COLOR,
        label="NumPy first-order (Lie-Trotter)",
    )
    ax.loglog(
        steps_arr,
        strang_errors,
        marker="o",
        linewidth=1.9,
        markersize=7,
        color=TORX_COLOR,
        label="NumPy second-order (Strang)",
    )
    if torx_steps is not None and torx_error is not None:
        total_samples = (
            f", {num_samples // 1000}k/source, "
            f"{num_samples * num_sources // 1000}k total"
            if num_samples is not None
            else ""
        )
        ax.scatter(
            [torx_steps],
            [torx_error],
            marker="*",
            s=120,
            color=TORX_COLOR,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
            label=f"Torx sampled Strang{total_samples}",
        )

    x_range = (float(steps_arr[0]), float(steps_arr[-1]))
    add_reference_slope(
        ax,
        slope=-1.0,
        x_range=x_range,
        label=r"$1/m$ slope",
        anchor_y=float(lie_errors[0]),
    )
    add_reference_slope(
        ax,
        slope=-2.0,
        x_range=x_range,
        label=r"$1/m^2$ slope",
        anchor_y=float(strang_errors[0]),
        linestyle=":",
        linewidth=1.1,
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(steps_arr)
    ax.set_xticklabels([str(int(s)) for s in steps_arr])
    ax.set_xlabel("splitting steps $m$")
    ax.set_ylabel("$L_2$ bias vs exact")
    ax.set_title("NumPy splitting order with Torx check")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.20)
    fig.tight_layout()
    return fig


def patch_parity_figure(
    deterministic: np.ndarray,
    sampled: np.ndarray,
    *,
    source_index: int | None = None,
    l2_error: float,
    num_samples: int | None = None,
    seed: int | None = None,
) -> plt.Figure:
    """Finished patch-check parity figure (sampled vs deterministic occupancy).

    Points share one neutral fill; the source vertex is ringed in Torx orange.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_PARITY)
    _parity_panel(
        ax,
        deterministic,
        sampled,
        pad=0.05,
        color="0.78",
        marker_size=40,
        alpha=1.0,
        edgecolor="k",
        scatter_zorder=3,
        line_width=0.9,
        line_label=r"$y = x$",
        clean=False,
    )
    if source_index is not None:
        ax.scatter(
            [deterministic[source_index]],
            [sampled[source_index]],
            s=110,
            facecolors="none",
            edgecolors=TORX_COLOR,
            linewidths=1.8,
            label="source vertex",
            zorder=4,
        )
    ax.set_xlabel("NumPy deterministic ordered mean")
    ax.set_ylabel("Torx sampled occupancy")
    run_label = ""
    if num_samples is not None:
        run_label = f", {num_samples:,} samples"
    if seed is not None:
        run_label += f", seed {seed}"
    ax.set_title(rf"Torx PSWAP patch check{run_label}" "\n" rf"$L_2 = {l2_error:.3f}$")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def _exact_marker(ax, x, y, *, size, marker="_"):
    """Dark, thick tick marks for an exact / reference quantity."""
    return ax.scatter(
        x,
        y,
        marker=marker,
        s=size,
        color=EXACT_COLOR,
        linewidths=3.0,
        zorder=4,
    )


def readout_histogram_expectation(
    sample_probs,
    exact_density,
    exact_expval,
    sample_mean,
    api_expval,
):
    """Two panels: sample histogram vs exact density, and per-pbit expectation."""
    x_states = np.arange(4)
    x_pbits = np.arange(2)

    fig, (ax_hist, ax_exp) = plt.subplots(
        1,
        2,
        figsize=(11, 6.2),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    fig.subplots_adjust(bottom=0.24, wspace=0.36)

    hist_top = max(sample_probs.max(), exact_density.max()) * 1.14
    bars = ax_hist.bar(
        x_states, sample_probs, color=TORX_COLOR, width=0.62, label="samples"
    )
    ref = _exact_marker(ax_hist, x_states, exact_density, size=460)
    ax_hist.set_xticks(x_states)
    ax_hist.set_xticklabels(_STATE_LABELS)
    ax_hist.set_xlim(x_states[0] - 0.5, x_states[-1] + 0.5)
    ax_hist.set_ylim(0.0, hist_top)
    ax_hist.set_xlabel("basis state")
    ax_hist.set_ylabel("probability")
    ax_hist.tick_params(axis="x", length=0)

    bar_width = 0.32
    exp_top = min(1.0, max(exact_expval.max(), sample_mean.max()) * 1.18)
    exact_bars = ax_exp.bar(
        x_pbits - bar_width / 2,
        exact_expval,
        bar_width,
        color=EXACT_COLOR,
        label="exact",
    )
    sample_bars = ax_exp.bar(
        x_pbits + bar_width / 2,
        sample_mean,
        bar_width,
        color=TORX_COLOR,
        label="sample mean",
    )
    api = _exact_marker(
        ax_exp, x_pbits + bar_width / 2, api_expval, size=110, marker="x"
    )
    ax_exp.set_xticks(x_pbits)
    ax_exp.set_xticklabels([f"pbit {i}" for i in x_pbits])
    ax_exp.set_xlim(x_pbits[0] - 0.5, x_pbits[-1] + 0.5)
    ax_exp.set_ylim(0.0, exp_top)
    ax_exp.set_ylabel(r"expectation $\langle s_i \rangle$")
    ax_exp.tick_params(axis="x", length=0)

    handles = [bars, ref, sample_bars, exact_bars, api]
    labels = [
        "empirical density",
        "exact density",
        "sample mean",
        "exact expectation",
        "independent expval draw",
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncols=3,
        frameon=False,
        fontsize=8,
    )
    return fig


def histogram_convergence(panel_ns, panel_probs, exact_density, tv_means, n_blocks):
    """One histogram panel per N, with block-mean TV diagnostics."""
    x_states = np.arange(4)
    fig, axes = plt.subplots(
        1,
        len(panel_ns),
        figsize=(6.2, 3.4),
        sharey=True,
        layout="constrained",
    )
    ymax = max(max(p.max() for p in panel_probs), exact_density.max()) * 1.18

    bars = ref = None
    for ax, n, probs, tv in zip(axes, panel_ns, panel_probs, tv_means, strict=True):
        bars = ax.bar(x_states, probs, color=TORX_COLOR, width=0.62)
        ref = _exact_marker(ax, x_states, exact_density, size=320)
        ax.set_xticks(x_states)
        ax.set_xticklabels(_STATE_LABELS)
        ax.set_xlim(x_states[0] - 0.5, x_states[-1] + 0.5)
        ax.set_ylim(0.0, ymax)
        ax.set_title(f"$N = {n}$")
        ax.tick_params(axis="x", length=0)
        ax.text(
            0.04,
            0.96,
            f"{n_blocks}-block mean\n" rf"$\mathrm{{TV}} = {tv:.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.3",
        )

    axes[0].set_ylabel("probability")
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)

    fig.supxlabel("basis state")
    fig.legend(
        [bars, ref],
        ["empirical density", "exact density"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=2,
        frameon=False,
        fontsize=8,
    )
    return fig


def sample_mean_error(ns, err_mean, err_std, n_blocks):
    """Log-log plot of sample-mean error vs block size, with the CLT slope."""
    fig, ax = plt.subplots(figsize=FIGSIZE_CONVERGENCE, layout="constrained")
    # band is the standard error of the mean across blocks, not the raw block
    # spread, so it stays tight around the mean line instead of spiking
    err_sem = np.asarray(err_std) / np.sqrt(n_blocks)
    err_low = np.maximum(err_mean - err_sem, 1e-6)
    err_high = err_mean + err_sem
    ax.fill_between(
        ns,
        err_low,
        err_high,
        color=TORX_COLOR,
        alpha=0.16,
        linewidth=0,
        label="standard error",
    )
    ax.loglog(
        ns,
        err_mean,
        marker="o",
        linewidth=1.8,
        markersize=6,
        color=TORX_COLOR,
        label="mean absolute error",
    )

    slope_y = add_reference_slope(
        ax,
        slope=-0.5,
        x_range=(float(ns[0]), float(ns[-1])),
        label=r"$N^{-1/2}$ reference",
        anchor_y=float(err_mean[0]),
    )

    ymin = min(float(err_low.min()), float(err_mean.min()), float(slope_y.min()))
    ymax = max(float(err_high.max()), float(err_mean.max()), float(slope_y.max()))
    ax.set_xlim(float(ns[0]), float(ns[-1]))
    ax.set_ylim(ymin * 0.95, ymax * 1.05)
    ax.set_xlabel("samples per block, $N$")
    ax.set_ylabel(r"$|\widehat{\langle s_1 \rangle}_N - \langle s_1 \rangle|$")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(int(n)) for n in ns])
    # only the chosen ns labels should show; silence matplotlib's default minor
    # decade labels (3x10^1, 4x10^1, ...) that otherwise overlap them
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(n)) for n in ns]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(
        True,
        which="major",
        color=NEUTRAL_GRAY,
        linestyle=":",
        linewidth=0.6,
        alpha=0.28,
    )
    ax.grid(False, which="minor")
    ax.tick_params(which="minor", length=0)
    return fig


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def moments(bits, edges):
    """Per-site magnetizations and per-edge correlations of a bit batch."""
    s = 2 * bits - 1
    mag = s.mean(axis=0)
    corr = np.array([(s[:, i] * s[:, j]).mean() for i, j in edges])
    return mag, corr


def chromatic_gibbs(samples, J, h, *, N, beta, colors, sweeps, rng):
    """Host-NumPy two-color single-site Gibbs sweep on a ring (reference only).

    Each color class (even sites, then odd sites) resamples in parallel from the
    exact conditional sigmoid(2 beta (h_i + J_left s_left + J_right s_right)).
    This is the host mirror of the per-site PNOT-from-zero kernel; the headline
    moment comparison uses the genuine Torx sampler ``torx_chromatic_gibbs``,
    while this fast host loop drives the persistent-contrastive-divergence fit.
    """
    samples = samples.copy()
    for _ in range(sweeps):
        for color in colors:
            spin = 2 * samples - 1
            for i in color:
                left = (i - 1) % N
                right = (i + 1) % N
                field = h[i] + J[left] * spin[:, left] + J[i] * spin[:, right]
                p = sigmoid(2.0 * beta * field)
                samples[:, i] = rng.random(len(samples)) < p
    return samples.astype(int)


def torx_chromatic_gibbs(init_bits, J, h, *, N, beta, colors, sweeps, key):
    """Chromatic single-site Gibbs on a ring, sampled through the Torx kernel.

    Every site update zeroes its pbit and draws it through the one-pbit circuit
    PNOT(2 beta ell_i) on ``SampleSimulator``: from the seeded 0, PNOT sets the
    bit to 1 with probability sigmoid(2 beta ell_i), which is exactly the Gibbs
    conditional (a deterministic reset to 0 followed by PNOT, with no reset leak).
    The local field ell_i is read from the live neighbor spins each sweep, so this
    is genuine Gibbs driven by the validated Torx gate rather than a host sigmoid
    draw. The per-chain logit is folded into the gate theta and the kernel is
    vmapped over every chain and color site at once.
    """
    sim = SampleSimulator(num_samples=1)
    base = sim.build_circuit(
        DiscretePCircuit([PNOT(0)]),
        [jnp.array([0.0])],
    )

    def sample_bit(logit, bit_key):
        # fold the per-chain conditional logit into the PNOT gate theta
        thetas = base.thetas.at[0, 0].set(logit)
        circ = eqx.tree_at(lambda c: c.thetas, base, thetas)
        # seed 0 so PNOT(p) draws Bernoulli(p), the exact one-site conditional
        out = sample_circuit(
            circ, jnp.zeros(1, dtype=jnp.int32), bit_key, num_samples=1
        )[0]
        return out[0, 0]

    sample_bits = jax.vmap(sample_bit)

    Jj = jnp.asarray(J)
    hj = jnp.asarray(h)
    colors_j = [jnp.asarray(c) for c in colors]

    def do_color(s, color, color_key):
        spin = 2 * s - 1
        left = (color - 1) % N
        right = (color + 1) % N
        field = (
            hj[color][None, :]
            + Jj[left][None, :] * spin[:, left]
            + Jj[color][None, :] * spin[:, right]
        )
        logits = (2.0 * beta * field).reshape(-1)
        keys = jax.random.split(color_key, logits.shape[0])
        bits = sample_bits(logits, keys).reshape(s.shape[0], color.shape[0])
        return s.at[:, color].set(bits.astype(s.dtype))

    @eqx.filter_jit
    def run(init, sweep_keys):
        def body(s, k):
            for color in colors_j:
                k, sub = jax.random.split(k)
                s = do_color(s, color, sub)
            return s, None

        out, _ = jax.lax.scan(body, init, sweep_keys)
        return out

    init = jnp.asarray(init_bits, dtype=jnp.int32)
    sweep_keys = jax.random.split(key, sweeps)
    return np.asarray(run(init, sweep_keys)).astype(int)


def pising_ring_samples(ring_mats, *, N, ring_edges, num_samples, sweeps, seed):
    """Walk caller-supplied PISING matrices with a host NumPy sampler.

    The notebook constructs each matrix with Torx ``PISING.get_matrix``. This
    helper reads the column for the current pair state, draws the next state on
    the host, and walks the ring one edge at a time.
    """
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, size=(num_samples, N), dtype=np.int8)
    for _ in range(sweeps):
        for idx, (i, j) in enumerate(ring_edges):
            current = (s[:, i] << 1) | s[:, j]
            probs = ring_mats[idx][:, current].T
            cdf = np.cumsum(probs, axis=1)
            draw = rng.random(num_samples)
            next_state = (draw[:, None] >= cdf).sum(axis=1).astype(np.int8)
            s[:, i] = next_state >> 1
            s[:, j] = next_state & 1
    return s


def _bar_ylim(*arrays):
    return _padded_range(*arrays, pad=0.08, include_zero=True)


def plot_ring_marginals(
    *,
    N,
    ring_edges,
    exact_mag,
    gibbs_mag,
    pising_mag,
    exact_corr,
    gibbs_corr,
    pising_corr,
):
    """Two panels: site magnetizations and edge correlations, three samplers each."""
    fig, (ax_mag, ax_corr) = plt.subplots(
        1, 2, figsize=FIGSIZE_MULTIPANEL, constrained_layout=True
    )
    width = 0.27

    site_x = np.arange(N)
    ax_mag.bar(
        site_x - width, exact_mag, width, label="exact", color=MARGINAL_COLORS["exact"]
    )
    ax_mag.bar(
        site_x,
        gibbs_mag,
        width,
        label="chromatic Gibbs",
        color=MARGINAL_COLORS["chromatic Gibbs"],
    )
    ax_mag.bar(
        site_x + width,
        pising_mag,
        width,
        label="host, Torx PISING matrices",
        color=MARGINAL_COLORS["per-edge PISING"],
    )
    ax_mag.axhline(0.0, color=NEUTRAL_GRAY, linewidth=0.6, linestyle=":", zorder=0)
    ax_mag.set_xticks(site_x)
    ax_mag.set_xticklabels([f"$s_{{{i}}}$" for i in range(N)])
    ax_mag.set_xlim(float(site_x[0] - 1.6 * width), float(site_x[-1] + 1.6 * width))
    ax_mag.set_ylim(*_bar_ylim(exact_mag, gibbs_mag, pising_mag))
    ax_mag.set_xlabel("site")
    ax_mag.set_ylabel(r"magnetization $\langle s_i \rangle$")
    _clean_axes(ax_mag)

    edge_x = np.arange(len(ring_edges))
    ax_corr.bar(edge_x - width, exact_corr, width, color=MARGINAL_COLORS["exact"])
    ax_corr.bar(edge_x, gibbs_corr, width, color=MARGINAL_COLORS["chromatic Gibbs"])
    ax_corr.bar(
        edge_x + width, pising_corr, width, color=MARGINAL_COLORS["per-edge PISING"]
    )
    ax_corr.axhline(0.0, color=NEUTRAL_GRAY, linewidth=0.6, linestyle=":", zorder=0)
    ax_corr.set_xticks(edge_x)
    ax_corr.set_xticklabels([f"{i}-{j}" for (i, j) in ring_edges], fontsize=8)
    ax_corr.set_xlim(float(edge_x[0] - 1.6 * width), float(edge_x[-1] + 1.6 * width))
    ax_corr.set_ylim(*_bar_ylim(exact_corr, gibbs_corr, pising_corr))
    ax_corr.set_xlabel("ring edge")
    ax_corr.set_ylabel(r"correlation $\langle s_i s_j \rangle$")
    _clean_axes(ax_corr)

    fig.legend(
        *ax_mag.get_legend_handles_labels(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
    )
    return fig


def plot_pcd_recovery(*, J_true, J, h_true, h):
    """Two parity panels: learned vs true couplings J and fields h."""
    fig, (ax_j, ax_h) = plt.subplots(
        1, 2, figsize=FIGSIZE_MULTIPANEL, constrained_layout=True
    )
    panels = [
        (ax_j, J_true, J, r"true coupling $J_e$", r"learned coupling $J_e$"),
        (ax_h, h_true, h, r"true field $h_i$", r"learned field $h_i$"),
    ]
    for ax, true_vals, learned_vals, xlabel, ylabel in panels:
        _parity_panel(
            ax,
            true_vals,
            learned_vals,
            color=TORX_COLOR,
            line_style=":",
            line_width=0.8,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax_j.set_title(f"couplings  (MAE = {np.mean(np.abs(J_true - J)):.3f})")
    ax_h.set_title(f"fields  (MAE = {np.mean(np.abs(h_true - h)):.3f})")
    return fig


def moment_parity(
    exact_vals,
    sample_vals,
    labels,
    tolerances,
    num_samples,
    *,
    figsize,
):
    """Parity and normalized residual evidence for exact and sampled moments."""
    exact_vals = np.asarray(exact_vals)
    sample_vals = np.asarray(sample_vals)
    tolerances = np.asarray(tolerances)
    residual_ratio = (sample_vals - exact_vals) / tolerances

    fig, (ax, ax_resid) = plt.subplots(
        2,
        1,
        figsize=figsize,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.25]},
    )
    _parity_panel(
        ax,
        exact_vals,
        sample_vals,
        pad=0.07,
        color=TORX_COLOR,
        marker_size=70,
        alpha=0.95,
        edge_width=0.6,
        scatter_zorder=3,
        line_label="exact = sampled",
        clean=False,
    )
    fig.canvas.draw()
    disp = ax.transData.transform(np.column_stack([exact_vals, sample_vals]))
    for i, (xv, yv, lab) in enumerate(
        zip(exact_vals, sample_vals, labels, strict=True)
    ):
        dy = -11
        for j in range(len(exact_vals)):
            if (
                j != i
                and abs(disp[j, 0] - disp[i, 0]) < 25
                and abs(disp[j, 1] - disp[i, 1]) < 45
            ):
                dy = 11 if disp[i, 1] >= disp[j, 1] else -14
                break
        ax.annotate(
            lab,
            (xv, yv),
            xytext=(8, dy),
            textcoords="offset points",
            fontsize=10,
            color="0.15",
        )
    ax.set_xlabel("exact moment")
    ax.set_ylabel(f"sampled moment ({num_samples:,} samples)")
    ax.set_title("Sample estimates agree within six-standard-error tolerances")
    ax.legend(loc="upper left")

    positions = np.arange(len(labels))
    ax_resid.axhspan(-1.0, 1.0, color=NEUTRAL_GRAY, alpha=0.16, label="tolerance")
    ax_resid.axhline(0.0, color="0.3", linewidth=0.8)
    ax_resid.scatter(positions, residual_ratio, s=42, color=TORX_COLOR, zorder=3)
    ax_resid.set_xticks(positions, labels)
    ax_resid.set_ylabel("residual / tolerance")
    bound = max(1.15, 1.15 * float(np.max(np.abs(residual_ratio))))
    ax_resid.set_ylim(-bound, bound)
    ax_resid.legend(loc="upper right", fontsize=8)
    return fig


def plot_magnetization_histogram_langevin(
    terminal_magnetization,
    *,
    title_hist="Magnetization distribution",
):
    """Plot the per-path terminal magnetization distribution."""
    fig, ax_hist = plt.subplots(figsize=(6.2, 3.6))
    mean_mag = float(terminal_magnetization.mean())
    ax_hist.hist(
        terminal_magnetization,
        bins=24,
        color=TORX_COLOR,
        edgecolor="white",
        linewidth=0.5,
    )
    ax_hist.axvline(
        mean_mag,
        color=NEUTRAL_GRAY,
        lw=1.4,
        ls="--",
        label=f"mean = {mean_mag:.3f}",
    )
    ax_hist.set_xlabel(r"terminal magnetization  $\frac{1}{n}\sum_i \tanh(x_i)$")
    ax_hist.set_ylabel("sample count")
    ax_hist.set_title(title_hist)
    ax_hist.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_energy_trace(energies, last_window_mean):
    """Plot a single path's energy-relaxation diagnostic."""
    fig, ax = plt.subplots(figsize=FIGSIZE_CONVERGENCE)
    ax.plot(energies, color=TORX_COLOR, lw=1.8, label="energy $V(x_t)$")
    ax.axhline(
        last_window_mean,
        color=NEUTRAL_GRAY,
        lw=1.0,
        ls="--",
        label=f"last-100-step mean = {last_window_mean:.3f}",
    )
    ax.set_xlabel("Langevin step")
    ax.set_ylabel("Ising energy")
    ax.set_title(f"Energy relaxation diagnostic  ({len(energies)} steps)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def gibbs_convergence(
    sweeps,
    occupancy,
    target,
    *,
    figsize=FIGSIZE_CONVERGENCE,
):
    """Plot Gibbs cluster occupancy against the target mixture weights."""
    sweeps = np.asarray(sweeps)
    values = np.asarray(occupancy)
    target = np.asarray(target)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if values.ndim == 1:
        (line,) = ax.plot(sweeps, values, marker="o", lw=1.8, label="empirical")
        ax.axhline(
            float(target), color=line.get_color(), lw=1.0, ls="--", label="target"
        )
    else:
        for k in range(values.shape[1]):
            color = CLUSTER_COLORS[k]
            ax.plot(
                sweeps,
                values[:, k],
                marker="o",
                lw=1.7,
                color=color,
                label=f"cluster {k}",
            )
            ax.axhline(target[k], color=color, lw=1.0, ls="--", alpha=0.65)
    ax.set_xlabel("Gibbs sweep")
    ax.set_ylabel("cluster occupancy")
    ax.set_ylim(0, max(0.55, float(values.max()) + 0.08))
    ax.set_title(r"Block Gibbs occupancy approaches $\pi$")
    ax.legend(loc="upper right", frameon=False, ncol=2)
    return fig


def winner_take_all(
    configs_labels,
    probs,
    is_onehot,
    lambdas,
    p_onehot_sweep,
    *,
    figsize=(6.2, 3.4),
):
    """Plot exact WTA Boltzmann probabilities and total one-hot mass."""
    labels = np.asarray(configs_labels)
    probs = np.asarray(probs, dtype=float)
    is_onehot = np.asarray(is_onehot, dtype=bool)
    lambdas = np.asarray(lambdas, dtype=float)
    p_onehot_sweep = np.asarray(p_onehot_sweep, dtype=float)

    bar_colors = [NEUTRAL_GRAY] * len(labels)
    for idx in np.flatnonzero(is_onehot):
        label = str(labels[idx])
        active = label.index("1") if "1" in label else 0
        bar_colors[idx] = CLUSTER_COLORS[active % len(CLUSTER_COLORS)]

    fig, (ax_dist, ax_sweep) = plt.subplots(
        1, 2, figsize=figsize, constrained_layout=True
    )

    x = np.arange(len(labels))
    ax_dist.bar(x, probs, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_dist.set_xticks(x)
    ax_dist.set_xticklabels(labels, rotation=45, ha="center")
    ax_dist.set_ylabel("probability")
    ax_dist.set_title("Boltzmann probability of each configuration")
    ax_dist.set_ylim(0.0, min(1.0, max(0.05, float(probs.max()) * 1.25)))
    handles = []
    for idx in np.flatnonzero(is_onehot):
        active = str(labels[idx]).index("1")
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                linestyle="none",
                markerfacecolor=CLUSTER_COLORS[active % len(CLUSTER_COLORS)],
                markeredgecolor="none",
                label=rf"$e_{active}$",
            )
        )
    ax_dist.legend(
        handles=handles, title="one-hot states", loc="upper right", frameon=False
    )

    ax_sweep.plot(lambdas, p_onehot_sweep, color=TORX_COLOR, lw=2.0)
    ax_sweep.axhline(1.0, color=NEUTRAL_GRAY, ls="--", lw=1.0)
    chosen_mass = float(probs[is_onehot].sum())
    chosen_idx = int(np.argmin(np.abs(p_onehot_sweep - chosen_mass)))
    chosen_lambda = float(lambdas[chosen_idx])
    chosen_y = float(p_onehot_sweep[chosen_idx])
    ax_sweep.scatter(
        [chosen_lambda],
        [chosen_y],
        color=TORX_COLOR,
        edgecolors="white",
        linewidths=0.6,
        zorder=3,
        label=rf"chosen $\lambda={chosen_lambda:.1f}$",
    )
    ax_sweep.set_xlabel(r"$\lambda$")
    ax_sweep.set_ylabel(r"$P(\sum_k z_k = 1)$")
    ax_sweep.set_ylim(0.0, 1.05)
    ax_sweep.set_title("One-hot mass vs penalty strength")
    ax_sweep.legend(loc="lower right", frameon=False)
    return fig


def plot_categorical_parity(exact_p, emp_p) -> Figure:
    """Compare exact probabilities with sampled frequencies for a 3-state pdit."""
    exact_p = np.asarray(exact_p)
    emp_p = np.asarray(emp_p)
    states = np.arange(3)
    width = 0.34

    fig, ax = plt.subplots(figsize=FIGSIZE_PARITY, constrained_layout=True)
    ax.bar(
        states - width / 2,
        exact_p,
        width=width,
        color=EXACT_COLOR,
        alpha=0.78,
        label="exact",
    )
    ax.bar(
        states + width / 2,
        emp_p,
        width=width,
        color=TORX_COLOR,
        alpha=0.72,
        label="sampled",
    )
    ax.set_xticks(states)
    ax.set_xlabel("state x")
    ax.set_ylabel("probability")
    ax.set_ylim(0, max(float(exact_p.max()), float(emp_p.max())) * 1.18)
    ax.set_title("Sampled frequencies match the exact 3-state probabilities")
    ax.legend(loc="upper right")
    _clean_axes(ax)
    return fig


def plot_conditional_sweep(xs, sampled, exact_curve) -> Figure:
    """Plot sampled conditional means against the sigmoid curve."""
    xs = np.asarray(xs)
    sampled = np.asarray(sampled)
    exact_curve = np.asarray(exact_curve)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    ax.plot(xs, exact_curve, color=EXACT_COLOR, lw=2.0, label=r"$\sigma(wx + b)$")
    ax.plot(
        xs, sampled, color=TORX_COLOR, marker="o", ms=3.5, lw=1.2, label="sampled mean"
    )
    ax.set_xlabel("input x")
    ax.set_ylabel("P(out = 1)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("The conditional factor tracks the sigmoid bias")
    ax.legend(loc="upper left")
    _clean_axes(ax)
    return fig


def total_magnetization(samples):
    """Total magnetization (sum of all spins) for each sample in a batch."""
    return np.asarray(samples).sum(axis=-1)


def order_parameter(samples):
    """Mean of |average spin| over a batch, on a 0-to-1 scale."""
    s = np.asarray(samples, dtype=float)
    return float(np.mean(np.abs(s.mean(axis=-1))))


def sweep_temperatures(run, betas, *, n_chains, seed, n_spins=N_SPINS):
    """Sampler order parameter at each inverse temperature, at zero field.

    ``run`` is the notebook's ``run(key, beta, init, field)`` closure that draws
    one equilibrium sample per chain.
    """
    key = jax.random.key(seed)
    zero_field = jnp.zeros(n_spins, dtype=jnp.float32)
    # beta enters as a traced scalar so the sampler compiles once and is reused
    # across temperatures instead of recompiling per beta.
    runner = eqx.filter_jit(
        lambda keys, beta, inits: jax.vmap(
            lambda k, x: run(k, beta, x, zero_field), in_axes=(0, 0)
        )(keys, inits)
    )
    orders = []
    for beta in betas:
        key, sub = jax.random.split(key)
        chain_keys = jax.random.split(sub, n_chains)
        inits = (
            jax.random.bernoulli(
                jax.random.fold_in(sub, 1), 0.5, (n_chains, n_spins)
            ).astype(jnp.int32)
            * 2
            - 1
        )
        beta_j = jnp.asarray(float(beta), jnp.float32)
        samples = runner(chain_keys, beta_j, inits)
        orders.append(order_parameter(np.asarray(samples)))
    return np.asarray(orders)


def plot_magnetization_histogram_lattice(
    sampler_mag, exact_levels, exact_probs, *, gap, n_spins=N_SPINS
):
    """Sampler magnetization counts as orange bars over the exact slate line.

    ``sampler_mag`` is the total magnetization of every sampled configuration;
    ``exact_levels`` and ``exact_probs`` are the achievable values and their
    exact Boltzmann probabilities; ``gap`` is the reported distance to exact.
    """
    sampler_mag = np.asarray(sampler_mag)
    exact_levels = np.asarray(exact_levels)
    exact_probs = np.asarray(exact_probs)

    counts = np.array([np.mean(sampler_mag == m) for m in exact_levels])

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    ax.bar(
        exact_levels,
        counts,
        width=1.6,
        color=EXTROPIC_ORANGE,
        alpha=0.55,
        label="sampler",
    )
    ax.plot(
        exact_levels,
        exact_probs,
        color=EXACT_COLOR,
        lw=2.2,
        marker="o",
        markersize=3.2,
        label="exact",
    )
    ax.set_xlabel("total magnetization  (sum of all spins)")
    ax.set_ylabel("probability")
    ax.set_title("sampler vs exact")
    ax.text(
        0.5,
        0.46,
        f"gap to exact = {gap:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.set_xticks(np.arange(-n_spins, n_spins + 1, 8))
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.62), frameon=False)
    _clean_axes(ax)
    return fig


def plot_settling_trace(order_traces, *, settling_step):
    """Order parameter across chains with a heuristic settling marker.

    ``order_traces`` is a (n_chains, n_steps + 1) array of the per-chain order
    parameter. The bold line is the mean across chains, the band is the 16th to
    84th percentile range, and the dashed line is a heuristic marker.
    """
    order_traces = np.asarray(order_traces)
    steps = np.arange(order_traces.shape[1])
    mean = order_traces.mean(axis=0)
    lo = np.percentile(order_traces, 16, axis=0)
    hi = np.percentile(order_traces, 84, axis=0)

    fig, ax = plt.subplots(figsize=FIGSIZE_CONVERGENCE, constrained_layout=True)
    ax.fill_between(steps, lo, hi, color=NEUTRAL_GRAY, alpha=0.18, lw=0)
    ax.plot(
        steps,
        mean,
        color=EXTROPIC_ORANGE,
        lw=2.0,
        label="order parameter (mean over chains)",
    )
    ax.axvline(
        settling_step,
        color=NEUTRAL_GRAY,
        lw=1.3,
        ls="--",
        label="heuristic settling marker",
        zorder=3,
    )
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Gibbs step")
    ax.set_ylabel("order parameter")
    ax.set_title("heuristic settling from disordered starts")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    _clean_axes(ax)
    return fig


def plot_temperature_sweep(betas, exact_order, sampler_order, mean_field_order):
    """Order parameter vs inverse temperature for exact, sampler, mean-field."""
    betas = np.asarray(betas)
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, constrained_layout=True)
    ax.plot(
        betas,
        mean_field_order,
        color=COARSE_COLOR,
        lw=2.2,
        ls="--",
        label="mean-field",
    )
    ax.plot(betas, exact_order, color=EXACT_COLOR, lw=2.2, label="exact")
    ax.plot(
        betas,
        sampler_order,
        color=EXTROPIC_ORANGE,
        lw=0,
        marker="o",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="sampler",
    )
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("inverse temperature  (colder to the right)")
    ax.set_ylabel("order parameter  (0 to 1)")
    ax.set_title("order rising as the model cools")
    ax.legend(loc="lower right")
    _clean_axes(ax)
    return fig


def plot_parity(exact_mag, sampler_mag, exact_corr, sampler_corr):
    """Two parity panels: per-spin average (left) and per-connection agreement (right)."""
    fig, (ax_spin, ax_conn) = plt.subplots(
        1, 2, figsize=FIGSIZE_MULTIPANEL, constrained_layout=True
    )
    panels = [
        (ax_spin, exact_mag, sampler_mag, "average of each spin"),
        (ax_conn, exact_corr, sampler_corr, "agreement on each connection"),
    ]
    for ax, exact_vals, sampler_vals, title in panels:
        _parity_panel(ax, exact_vals, sampler_vals)
        ax.set_xlabel("exact")
        ax.set_ylabel("sampler")
        ax.set_title(title)
    return fig


def _completion_grid(ax, values, *, free_mask=None, title, grid=GRID):
    # sample the diverging colormap's endpoints so +1/-1 read the same in every panel
    spin_pos_color = EXTROPIC_DIVERGING(1.0)
    spin_neg_color = EXTROPIC_DIVERGING(0.0)
    free_color = "#e7e3dc"
    values = np.asarray(values)
    for i in range(grid * grid):
        r, c = divmod(i, grid)
        if free_mask is not None and free_mask[i]:
            fill = free_color
        else:
            fill = spin_pos_color if values[i] > 0 else spin_neg_color
        ax.add_patch(
            mpatches.Rectangle(
                (c, grid - 1 - r),
                1,
                1,
                facecolor=fill,
                edgecolor="white",
                linewidth=2.0,
            )
        )
    ax.set_xlim(-0.1, grid + 0.1)
    ax.set_ylim(-0.1, grid + 0.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_pattern_completion(field, one_sample, average, *, grid=GRID):
    """Three 4x4 grids: field bias, one field-conditioned sample, and mean spin.

    The first panel discloses a uniform nonzero field magnitude. The third
    panel shows the actual mean spin in ``[-1, 1]`` on a diverging scale.
    """
    field = np.asarray(field)
    free_mask = field == 0.0
    active_magnitudes = np.unique(np.abs(field[~free_mask]))
    field_title = "bias sign"
    if active_magnitudes.size == 1:
        field_title = rf"bias sign ($|b_i|={active_magnitudes[0]:g}$)"

    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.8), constrained_layout=True)
    _completion_grid(
        axes[0],
        np.sign(field),
        free_mask=free_mask,
        title=field_title,
        grid=grid,
    )
    _completion_grid(axes[1], one_sample, title="field-conditioned sample", grid=grid)

    mean_grid = np.asarray(average, dtype=float).reshape(grid, grid)
    im = axes[2].imshow(
        mean_grid, cmap=EXTROPIC_DIVERGING, vmin=-1.0, vmax=1.0, origin="upper"
    )
    axes[2].set_title("mean spin", fontsize=10)
    axes[2].set_aspect("equal")
    axes[2].axis("off")
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, ticks=[-1, 0, 1])
    cbar.ax.tick_params(labelsize=7)

    handles = [
        mpatches.Patch(
            facecolor=EXTROPIC_DIVERGING(1.0),
            edgecolor="white",
            label="positive bias",
        ),
        mpatches.Patch(
            facecolor=EXTROPIC_DIVERGING(0.0),
            edgecolor="white",
            label="negative bias",
        ),
        mpatches.Patch(facecolor="#e7e3dc", edgecolor="white", label="zero field"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )
    return fig
