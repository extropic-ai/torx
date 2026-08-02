"""Field figures: densities, gaussian clouds, trajectories and marginals."""

import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from _affine_gaussian import (
    gaussian_density,
    mixture_density,
)
from _notebook_style import (
    COARSE_COLOR,
    EXACT_COLOR,
    EXTROPIC_BROWN,
    FIGSIZE_SINGLE,
    FIGURE_BG,
    format_mu_label,
    HEAT_CMAP,
    NEUTRAL_GRAY,
    REGIME_COLORS,
    TORX_COLOR,
)
from _plot_utils import (
    _cov_ellipse,
    _square_limits,
    _two_col_grid,
)
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

MIXTURE_BRANCH_COLORS = [EXACT_COLOR, TORX_COLOR, COARSE_COLOR]

_DENSITY_REFERENCE = EXTROPIC_BROWN

CLUSTER_COLORS = [EXACT_COLOR, TORX_COLOR, COARSE_COLOR]


def affine_gaussian_clouds(in_cloud, out_cloud, *, figsize=(6.0, 3.4)):
    """Two-panel scatter: input N(0,I) (gray) vs affine-Gaussian output (orange).

    Both panels share equal aspect and the same symmetric limits, with one
    shared x/y label and a single suptitle. Panel titles are the input/output
    expressions, sized to sit cleanly above each cloud.
    """
    lo, hi = _square_limits(in_cloud, out_cloud, cap=3.6)
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True, constrained_layout=True
    )
    panels = [
        (in_cloud, r"input  $x \sim \mathcal{N}(0, I)$", NEUTRAL_GRAY),
        (out_cloud, r"output  $x' = A x + b + \varepsilon$", TORX_COLOR),
    ]
    for ax, (pts, title, color) in zip(axes, panels, strict=True):
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.32, color=color, rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.tick_params(axis="both", length=0)
    fig.suptitle("Affine Gaussian gate: one sample cloud in, one out")
    fig.supxlabel(r"$x_0$")
    fig.supylabel(r"$x_1$")
    return fig


def mixture_clouds(samples, labels, *, figsize=(4.6, 3.6)):
    """Scatter of control-conditioned Gaussian branches with a legend.

    Each branch keeps its Extropic-cycle color (from ``MIXTURE_BRANCH_COLORS``)
    and is named in a legend, so no label sits on top of a dark cluster. The
    number of branches is read from the integer ``labels``.
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    handles = []
    for k in np.unique(labels):
        pts = samples[labels == k]
        color = MIXTURE_BRANCH_COLORS[k % len(MIXTURE_BRANCH_COLORS)]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=5,
            alpha=0.40,
            color=color,
            rasterized=True,
        )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                color=color,
                label=f"control {k}",
            )
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    ax.tick_params(axis="both", length=0)
    ax.set_title("Mixture Gaussian branches by control state")
    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        fontsize=8,
    )
    return fig


def affine_general_clouds(in_cloud, out_cloud, exact_mean, exact_cov, *, figsize):
    """Two-panel scatter: input N(0, I) and the affine-Gaussian output.

    Both panels share equal aspect and identical limits. Samples are orange; the
    slate ellipse on each panel is the exact analytic 1-sigma contour.
    """
    lo, hi = _square_limits(in_cloud, out_cloud)
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True, constrained_layout=True
    )
    in_mean = in_cloud.mean(axis=0)
    in_cov = np.cov(in_cloud, rowvar=False)
    panels = [
        (in_cloud, in_mean, in_cov, r"input  $x \sim \mathcal{N}(0,\,I)$"),
        (
            out_cloud,
            np.asarray(exact_mean),
            np.asarray(exact_cov),
            r"output  $x' = Ax + b + \varepsilon$",
        ),
    ]
    for ax, (pts, mean, cov, title) in zip(axes, panels, strict=True):
        ax.scatter(
            pts[:, 0], pts[:, 1], s=4, alpha=0.30, color=TORX_COLOR, rasterized=True
        )
        _cov_ellipse(ax, mean, cov, n_std=1.0, edgecolor=EXACT_COLOR, linewidth=1.8)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.tick_params(axis="both", length=0)
    fig.supxlabel(r"$x_0$")
    fig.supylabel(r"$x_1$")
    return fig


def gate_sequence_clouds(panels, *, figsize):
    """Compact 1+4 layout of gate-output clouds against the same input."""
    arrays = [pts for _, pts, _ in panels]
    lo, hi = _square_limits(*arrays, cap=5.0)
    n = len(panels)
    if n == 5:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(2, 3, width_ratios=(1.05, 1.0, 1.0))
        axes = [
            fig.add_subplot(grid[:, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]),
            fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[1, 2]),
        ]
        show_y_label = {0, 1, 3}
    else:
        ncols = 2
        fig, axes = _two_col_grid(n, figsize=figsize)
        show_y_label = set(range(0, n, ncols))

    for i, (title, pts, ref) in enumerate(panels):
        ax = axes[i]
        if ref is not None:
            ax.scatter(
                ref[:, 0], ref[:, 1], s=3, alpha=0.10, color="0.5", rasterized=True
            )
            _cov_ellipse(
                ax,
                ref.mean(axis=0),
                np.cov(ref, rowvar=False),
                n_std=1.0,
                edgecolor="0.4",
                linewidth=1.5,
                linestyle=(0, (4, 2)),
                zorder=4,
            )
        ax.scatter(
            pts[:, 0], pts[:, 1], s=4, alpha=0.32, color=TORX_COLOR, rasterized=True
        )
        _cov_ellipse(
            ax,
            pts.mean(axis=0),
            np.cov(pts, rowvar=False),
            n_std=1.0,
            edgecolor=EXACT_COLOR,
            linewidth=1.8,
            zorder=5,
        )
        ax.set_title(title, fontsize=9.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.tick_params(axis="both", length=0)
        if i in show_y_label:
            ax.set_ylabel(r"$x_1$")
        else:
            ax.tick_params(labelleft=False)
    fig.supxlabel(r"$x_0$")
    return fig


def prior_joint_density(prior, *, figsize):
    """Joint prior density with the prior mean marked by a white plus.

    Builds its own evaluation grid from hand-tuned display bounds, then evaluates
    the joint density straight from the prior moments.
    """
    # hand-tuned bounds framing the propagated prior (~mean +/- 3 sigma) for the docs figure
    x0 = np.linspace(-2.0, 2.6, 200)
    x1 = np.linspace(-2.0, 2.8, 200)
    X0, X1 = np.meshgrid(x0, x1)
    pts = np.stack([X0.ravel(), X1.ravel()], axis=-1)
    joint = np.asarray(gaussian_density(prior, grid=jnp.asarray(pts))).reshape(X0.shape)
    mean = prior.mean

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    cf = ax.contourf(X0, X1, joint, levels=14, cmap=HEAT_CMAP)
    ax.scatter(
        [float(mean[0])],
        [float(mean[1])],
        marker="+",
        s=90,
        color="white",
        linewidths=1.6,
        zorder=5,
    )
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    ax.set_title(r"joint prior $\rho(x_0, x_1)$")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("density")
    cbar.outline.set_visible(False)
    return fig


def prior_marginals(prior, *, figsize):
    """Marginal densities implied by the propagated prior moments."""
    # same hand-tuned ~mean +/- 3 sigma display bounds as the joint figure
    x0 = np.linspace(-2.0, 2.6, 300)
    x1 = np.linspace(-2.0, 2.8, 300)
    m0 = np.asarray(gaussian_density(prior, site=0, grid=jnp.asarray(x0)))
    m1 = np.asarray(gaussian_density(prior, site=1, grid=jnp.asarray(x1)))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.plot(x0, m0, color=EXACT_COLOR, lw=1.9, label=r"$\rho(x_0)$")
    ax.plot(x1, m1, color=TORX_COLOR, lw=1.9, label=r"$\rho(x_1)$")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.set_title("marginal densities")
    ax.legend(loc="upper right")
    return fig


def conditioning_cloud(samples, y_obs, prior, *, figsize):
    """Joint sample cloud with the observed line and the prior mean."""
    prior_mean = np.asarray(prior.mean)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.scatter(
        samples[:, 0], samples[:, 1], s=3, alpha=0.20, color=TORX_COLOR, rasterized=True
    )
    ax.axhline(
        y_obs,
        color=EXACT_COLOR,
        linewidth=1.8,
        linestyle="--",
        label=rf"$y_1 = {y_obs:g}$",
    )
    ax.scatter(
        [prior_mean[0]],
        [prior_mean[1]],
        marker="+",
        s=120,
        color="0.15",
        linewidths=2.0,
        zorder=5,
        label="prior mean",
    )
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel(r"$x_1$")
    ax.set_title("prior sample cloud")
    ax.legend(loc="upper left")
    return fig


def conditioning_posterior(
    prior,
    posterior,
    samples,
    observation,
    band,
    *,
    figsize,
):
    """Exact point-conditioned density with a finite-band sample approximation."""
    y_obs = float(observation)
    samples = np.asarray(samples)
    cond_samples = samples[np.abs(samples[:, 1] - y_obs) < band, 0]
    x_lo = float(samples[:, 0].min()) - 0.5
    x_hi = float(samples[:, 0].max()) + 0.5
    x_grid = np.linspace(float(x_lo), float(x_hi), 300)
    prior_pdf = np.asarray(gaussian_density(prior, site=0, grid=jnp.asarray(x_grid)))
    post_pdf = np.asarray(gaussian_density(posterior, site=0, grid=jnp.asarray(x_grid)))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if len(cond_samples) > 10:
        ax.hist(
            cond_samples,
            bins=24,
            density=True,
            color=NEUTRAL_GRAY,
            alpha=0.35,
            label=rf"finite-band MC approximation, $|x_1 - {y_obs:g}| < {band:g}$",
        )
    ax.plot(
        x_grid, prior_pdf, color=EXACT_COLOR, linewidth=2.0, label=r"prior $\rho(x_0)$"
    )
    ax.plot(
        x_grid,
        post_pdf,
        color=TORX_COLOR,
        linewidth=2.0,
        label=r"exact point-conditioned $\rho(x_0 \mid x_1 = y_1)$",
    )
    ax.set_xlabel(r"$x_0$")
    ax.set_ylabel("density")
    ax.set_title("Exact point conditioning and finite-band sampling")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.85")
    return fig


def plot_mixture_density(samples, means, log_vars, weights, *, figsize):
    """Histogram of MoG samples (orange) against the exact mixture PDF (slate).

    Builds the display grid and evaluates the mixture density on it internally,
    so the notebook only passes the gate parameters and its drawn samples.
    """
    # bimodal mixture sits within +/-4, the widest display window the gallery uses
    grid = np.linspace(-4.0, 4.0, 400)
    density = np.asarray(mixture_density(means, log_vars, weights, jnp.asarray(grid)))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.hist(
        samples,
        bins=70,
        density=True,
        color=TORX_COLOR,
        alpha=0.35,
        label="gate samples",
    )
    ax.plot(
        grid,
        density,
        color=EXACT_COLOR,
        lw=2.0,
        label=r"exact $\rho(x) = \sum_k \pi_k\,\mathcal{N}(\mu_k, \sigma_k^2)$",
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("density")
    ax.set_title("MixtureGaussianGate: a bimodal density")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="0.85")
    return fig


def plot_observations(observations, true_event, *, channel_cmap=HEAT_CMAP):
    """Stacked synthetic Gaussian observation channels with evaluation bands."""
    n_time, n_channels = observations.shape
    colors = plt.get_cmap(channel_cmap)(np.linspace(0.12, 0.72, n_channels))
    time = np.arange(n_time)
    event_times = np.flatnonzero(true_event)

    fig, axes = plt.subplots(n_channels, 1, figsize=(6.2, 7.0), sharex=True)
    axes = np.atleast_1d(axes)
    for ch, (ax, col) in enumerate(zip(axes, colors, strict=True)):
        for t in event_times:
            ax.axvspan(t - 0.5, t + 0.5, color=NEUTRAL_GRAY, alpha=0.20, lw=0)
        ax.plot(time, observations[:, ch], color=col, linewidth=1.3, zorder=3)
        ax.set_ylabel(f"ch {ch}", fontsize=9, labelpad=2)
        ax.tick_params(axis="y", labelsize=8)

    axes[-1].set_xlabel("time bin")
    axes[-1].tick_params(axis="x", labelsize=8)
    axes[0].set_title(f"{n_channels} synthetic Gaussian observation channels")
    event_patch = mpatches.Patch(
        color=NEUTRAL_GRAY, alpha=0.20, label="true event, evaluation only"
    )
    axes[0].legend(handles=[event_patch], loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout(h_pad=0.4)
    return fig


def plot_posterior_drive(
    true_drive,
    posterior_drive,
    posterior_drive_std,
    detection_threshold,
):
    """Headline plot: true latent drive vs Torx posterior mean and band."""
    time = np.arange(len(posterior_drive))

    fig, ax = plt.subplots(figsize=(6.2, 4.0), layout="constrained")
    ax.fill_between(
        time,
        posterior_drive - 2.0 * posterior_drive_std,
        posterior_drive + 2.0 * posterior_drive_std,
        color=TORX_COLOR,
        alpha=0.18,
        linewidth=0,
        label=r"posterior $\pm 2\sigma$",
        zorder=2,
    )
    ax.plot(
        time,
        true_drive,
        color=EXACT_COLOR,
        linewidth=2.0,
        label="true latent drive",
        zorder=3,
    )
    ax.plot(
        time,
        posterior_drive,
        color=TORX_COLOR,
        linewidth=2.4,
        label="posterior mean (Torx)",
        zorder=4,
    )
    ax.axhline(
        detection_threshold,
        color=NEUTRAL_GRAY,
        linestyle="--",
        linewidth=1.2,
        label="full-trial median threshold",
        zorder=1,
    )

    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("time bin")
    ax.set_ylabel("latent drive (intent-axis projection)")
    ax.set_title("Offline smoothed latent drive: Torx posterior vs truth")
    ax.legend(loc="lower left", ncols=2, fontsize=8)
    return fig


def plot_posterior_covariance(block_norms):
    """Frobenius norm of each timestep-by-timestep posterior covariance block."""
    fig, ax = plt.subplots(figsize=(4.0, 3.6), layout="constrained")
    im = ax.imshow(
        np.asarray(block_norms), cmap=HEAT_CMAP, aspect="equal", origin="upper"
    )
    ax.set_xlabel(r"time $t'$")
    ax.set_ylabel(r"time $t$")
    ax.set_title(r"Posterior covariance block norms $\|\Sigma_{t,t'}\|_F$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("3×3 block Frobenius norm")
    return fig


def plot_detector(time, uncalibrated_score, true_event, predicted_event, accuracy):
    """Two-panel offline readout with an uncalibrated score and event bars."""
    time = np.asarray(time)
    true_event = np.asarray(true_event, dtype=bool)
    predicted_event = np.asarray(predicted_event, dtype=bool)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(6.2, 4.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [2.5, 0.9], "hspace": 0.12},
    )
    ax_top.plot(
        time,
        uncalibrated_score,
        color=TORX_COLOR,
        linewidth=2.0,
        label="uncalibrated score",
        zorder=3,
    )
    ax_top.axhline(
        0.5,
        color=NEUTRAL_GRAY,
        linestyle="--",
        linewidth=1.0,
        label="decision threshold (0.5)",
    )
    ax_top.set_ylim(-0.04, 1.04)
    ax_top.set_ylabel("uncalibrated score")
    ax_top.set_title(f"One-seed oracle-axis sanity check (accuracy = {accuracy:.0%})")
    ax_top.legend(loc="upper left", fontsize=8)

    true_bars = [(float(t) - 0.5, 1.0) for t in time[true_event]]
    predicted_bars = [(float(t) - 0.5, 1.0) for t in time[predicted_event]]
    ax_bot.broken_barh(
        true_bars, (0.55, 0.30), facecolors=EXACT_COLOR, edgecolors="none"
    )
    ax_bot.broken_barh(
        predicted_bars, (0.05, 0.30), facecolors=TORX_COLOR, edgecolors="none"
    )
    ax_bot.set_ylim(0.0, 0.95)
    ax_bot.set_yticks([0.20, 0.70])
    ax_bot.set_yticklabels(["predicted", "true"], fontsize=8)
    # widen to cover the half-bin-wide endpoint event bars (shared x-axis)
    ax_bot.set_xlim(float(time[0]) - 0.5, float(time[-1]) + 0.5)
    ax_bot.set_xlabel("time bin")
    ax_bot.set_ylabel("event")
    event_handles = [
        mpatches.Patch(color=EXACT_COLOR, label="true event"),
        mpatches.Patch(color=TORX_COLOR, label="predicted"),
    ]
    ax_bot.legend(handles=event_handles, loc="center right", ncols=1, fontsize=8)
    return fig


def _regime_legend_labels(mu, sigma) -> list[str]:
    return [
        rf"regime {k}  ($\mu$={format_mu_label(mu[k])}, $\sigma$={float(sigma[k]):.1f})"
        for k in range(len(mu))
    ]


def plot_transition_density(
    conditional_increments,
    marginal_increments,
    x_grid,
    conditional_pdfs,
    marginal_pdf,
):
    """One-step transition density: the K regime-conditional panels plus the marginal."""
    K = len(conditional_increments)
    x_grid = np.asarray(x_grid)

    total = K + 1
    fig, axes = _two_col_grid(total)

    for k in range(K):
        ax = axes[k]
        ax.hist(
            conditional_increments[k],
            bins=80,
            density=True,
            color=REGIME_COLORS[k % len(REGIME_COLORS)],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.plot(
            x_grid,
            np.asarray(conditional_pdfs[k]),
            color=_DENSITY_REFERENCE,
            lw=1.8,
            label=rf"$\mathcal{{N}}(\mu_{k}\Delta t,\,\sigma_{k}^2\Delta t)$",
        )
        ax.set_title(f"new regime {k}", fontsize=9.5)
        ax.set_xlabel(r"$\Delta x$")
        ax.legend(
            loc="upper left",
            fontsize=7.0,
            frameon=False,
            handlelength=1.2,
            borderpad=0.2,
        )

    ax_mix = axes[K]
    ax_mix.hist(
        marginal_increments,
        bins=80,
        density=True,
        color=NEUTRAL_GRAY,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
    )
    ax_mix.plot(
        x_grid,
        np.asarray(marginal_pdf),
        color=_DENSITY_REFERENCE,
        lw=1.8,
        label=r"$\sum_k \pi_k\,\mathcal{N}(\mu_k\Delta t, \sigma_k^2\Delta t)$",
    )
    ax_mix.set_title(r"stationary marginal, $\pi = 1/K$", fontsize=9.5)
    ax_mix.set_xlabel(r"$\Delta x$")
    ax_mix.legend(
        loc="upper left", fontsize=7.0, frameon=False, handlelength=1.2, borderpad=0.2
    )

    # left-column panels carry the shared y-label
    for ax in axes[::2]:
        ax.set_ylabel("density")
    fig.suptitle(
        r"One-step transition density:  PditCycle $\rightarrow$ MixtureGaussianGate",
        fontsize=11,
    )
    return fig


def plot_sample_paths(
    time,
    positions,
    regimes,
    *,
    mu,
    sigma,
    n_show=6,
):
    """Show individually indexed paths, with each segment colored by regime."""
    K = len(mu)
    regime_palette = [REGIME_COLORS[k % len(REGIME_COLORS)] for k in range(K)]
    regime_labels = _regime_legend_labels(mu, sigma)
    n_show = min(int(n_show), len(positions))
    ncols = 2
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.8, 1.75 * nrows + 0.8),
        sharex=True,
        squeeze=False,
    )

    for traj_idx, ax in enumerate(axes.flat):
        if traj_idx >= n_show:
            ax.set_visible(False)
            continue
        pos = positions[traj_idx]
        reg = regimes[traj_idx]
        pts = np.stack([time, pos], axis=1)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_colors = [regime_palette[r] for r in reg[1:]]
        ax.add_collection(LineCollection(segments, colors=seg_colors, linewidths=1.25))
        ax.autoscale()
        ax.set_title(f"path {traj_idx}", fontsize=8.5, loc="left")
        ax.grid(True, alpha=0.16)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("time $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("position $x$")

    patches = [
        mpatches.Patch(color=regime_palette[k], label=regime_labels[k])
        for k in range(K)
    ]
    fig.legend(
        handles=patches,
        fontsize=7.5,
        loc="lower center",
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.4,
    )
    fig.suptitle("Representative regime-switching paths", fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96], pad=0.8)
    return fig


def plot_regime_occupancy(time, occupancy, *, num_trajectories):
    """Plot each regime's occupancy with Monte Carlo uncertainty."""
    time = np.asarray(time)
    occupancy = np.asarray(occupancy)
    K = occupancy.shape[0]
    regime_palette = [REGIME_COLORS[k % len(REGIME_COLORS)] for k in range(K)]

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for k in range(K):
        values = occupancy[k]
        se = np.sqrt(values * (1.0 - values) / num_trajectories)
        ax.plot(
            time,
            values,
            color=regime_palette[k],
            lw=1.6,
            label=f"regime {k}",
        )
        ax.fill_between(
            time,
            np.clip(values - 2 * se, 0, 1),
            np.clip(values + 2 * se, 0, 1),
            color=regime_palette[k],
            alpha=0.14,
            linewidth=0,
        )
    ax.axhline(
        1.0 / K,
        color=EXTROPIC_BROWN,
        lw=1.1,
        ls="--",
        path_effects=[pe.withStroke(linewidth=2.8, foreground=FIGURE_BG)],
        label=f"stationary reference 1/K = {1.0 / K:.3f}",
    )
    ax.set_xlabel("time $t$")
    ax.set_ylabel("fraction of paths")
    ax.set_ylim(0, 1)
    ax.set_title(r"Regime occupancy, lines with $\pm 2$ MC SE bands")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7.5,
        frameon=False,
        borderaxespad=0.0,
    )
    ax.grid(True, alpha=0.16)
    fig.tight_layout(pad=0.8)
    return fig


def _equal_limits(points, means, pad=0.7):
    pts = np.vstack([np.asarray(points), np.asarray(means)])
    center = pts.mean(axis=0)
    radius = float(np.max(np.abs(pts - center))) + pad
    return (center[0] - radius, center[0] + radius), (
        center[1] - radius,
        center[1] + radius,
    )


def cluster_scatter(
    points,
    labels,
    means,
    covs,
    *,
    figsize=(6.2, 5.2),
    title="Torx conditional Gaussian samples, with labels drawn by NumPy",
):
    """Scatter a labeled 2D mixture with analytic one-sigma component ellipses."""
    points = np.asarray(points)
    labels = np.asarray(labels)
    means = np.asarray(means)
    covs = np.asarray(covs)
    xlim, ylim = _equal_limits(points, means)
    n_clusters = means.shape[0]
    cluster_palette = [
        CLUSTER_COLORS[k % len(CLUSTER_COLORS)] for k in range(n_clusters)
    ]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    for k, color in enumerate(cluster_palette):
        mask = labels == k
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=5,
            alpha=0.34,
            color=color,
            rasterized=True,
        )
        _cov_ellipse(
            ax,
            means[k],
            covs[k],
            n_std=1.0,
            edgecolor="black",
            linewidth=1.6,
            path_effects=halo,
        )
        ax.scatter(
            [means[k, 0]],
            [means[k, 1]],
            marker="+",
            s=90,
            color="black",
            linewidths=1.8,
            zorder=5,
            path_effects=halo,
        )
    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=7,
            markerfacecolor=color,
            markeredgecolor="none",
            label=f"cluster {k}",
        )
        for k, color in enumerate(cluster_palette)
    ]
    legend_handles += [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor="black",
            label=r"1$\sigma$ contour",
        ),
        Line2D(
            [],
            [],
            marker="+",
            linestyle="none",
            markersize=9,
            color="black",
            markeredgewidth=1.8,
            label=r"mean $\mu_k$",
        ),
    ]
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$v_0$")
    ax.set_ylabel(r"$v_1$")
    ax.set_title(title)
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        labelspacing=0.7,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    return fig


def assignment_panels(
    points,
    true_labels,
    post_probs,
    means,
    *,
    posterior_fn,
    figsize=FIGSIZE_SINGLE,
):
    """Show the soft responsibility field with hard-assigned samples."""
    points = np.asarray(points)
    true_labels = np.asarray(true_labels)
    post_probs = np.asarray(post_probs)
    means = np.asarray(means)
    hard = post_probs.argmax(axis=1)
    accuracy = float((hard == true_labels).mean())
    xlim, ylim = _equal_limits(points, means)
    n_clusters = post_probs.shape[1]
    cluster_palette = [
        CLUSTER_COLORS[k % len(CLUSTER_COLORS)] for k in range(n_clusters)
    ]

    grid_size = 220
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])
    resp = np.asarray(posterior_fn(grid_pts))
    colors_rgb = np.asarray([to_rgb(color) for color in cluster_palette])
    rgb = (resp @ colors_rgb).reshape(grid_size, grid_size, 3)
    # pale tint so the field reads as a soft background and the samples stay crisp
    rgb = np.clip(0.46 * rgb + 0.54, 0.0, 1.0)
    region = resp.argmax(axis=1).reshape(grid_size, grid_size).astype(float)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(
        rgb,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
    )
    # clean decision-boundary lines between the responsibility regions
    boundary_levels = (np.arange(n_clusters - 1) + 0.5).tolist()
    ax.contour(
        xx,
        yy,
        region,
        levels=boundary_levels,
        colors="white",
        linewidths=1.2,
        alpha=0.85,
    )
    for k, color in enumerate(cluster_palette):
        mask = hard == k
        edge = tuple(0.55 * c for c in to_rgb(color))
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=8,
            alpha=0.85,
            color=color,
            edgecolors=[edge],
            linewidths=0.3,
            rasterized=True,
        )
    ax.scatter(
        means[:, 0],
        means[:, 1],
        marker="+",
        s=110,
        color="black",
        linewidths=2.0,
        zorder=5,
        path_effects=[pe.withStroke(linewidth=3.2, foreground="white")],
    )
    ax.text(
        0.025,
        0.975,
        f"hard recovery: {accuracy:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#D8D2C6",
            "linewidth": 0.8,
            "alpha": 0.95,
        },
    )
    responsibility_handles = [
        mpatches.Patch(
            facecolor=np.clip(0.46 * np.asarray(to_rgb(color)) + 0.54, 0.0, 1.0),
            edgecolor="none",
            label=f"component {k}",
        )
        for k, color in enumerate(cluster_palette)
    ]
    ax.legend(
        handles=responsibility_handles,
        title=r"field color blends $p(k\mid v)$",
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        fontsize=8,
        title_fontsize=8,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$v_0$")
    ax.set_ylabel(r"$v_1$")
    ax.set_title(r"soft responsibilities $p(k\mid v)$")
    return fig


def marginal_density(samples_1d, grid, density, *, figsize=FIGSIZE_SINGLE):
    """Histogram of one visible marginal against the exact mixture PDF."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.hist(
        samples_1d,
        bins=70,
        density=True,
        color=TORX_COLOR,
        alpha=0.36,
        edgecolor="white",
        linewidth=0.35,
        label="gate samples",
    )
    ax.plot(grid, density, color=EXACT_COLOR, lw=2.0, label=r"exact $\rho(v_0)$")
    ax.set_xlabel(r"$v_0$")
    ax.set_ylabel("density")
    ax.set_title("One visible marginal")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.16)
    ax.legend(loc="upper right", frameon=False)
    return fig
