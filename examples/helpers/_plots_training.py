"""Training figures: loss curves, reconstruction grids, FID, accuracy."""

import matplotlib.pyplot as plt
import numpy as np
from _notebook_style import (
    EXACT_COLOR,
    EXTROPIC_COPPER,
    FIGSIZE_BAR,
    FIGSIZE_CONVERGENCE,
    NEUTRAL_GRAY,
    TORX_COLOR,
)

# digit grids render at higher DPI than the notebook default so 28x28 reconstructions stay crisp
GRID_DPI = 200


def _style_axes(ax):
    ax.grid(axis="y", color=NEUTRAL_GRAY, alpha=0.18, linestyle=":", linewidth=0.6)
    ax.tick_params(axis="both", length=3, width=0.6)


def plot_diffusion_loss(loss_history, *, reference_loss=None):
    """Plot the saved UNet weighted-BCE training loss against gradient step.

    The raw per-step loss is drawn faintly; a rolling mean rides on top so the
    convergence trend stays legible through the sampling noise. ``reference_loss``
    draws a dashed horizontal reference (e.g. the checkpoint's final loss).
    """
    loss = np.asarray(loss_history, dtype=float)
    steps = np.arange(len(loss))

    fig, ax = plt.subplots(figsize=FIGSIZE_CONVERGENCE, constrained_layout=True)
    ax.plot(steps, loss, color=TORX_COLOR, linewidth=0.6, alpha=0.20)
    win = max(1, len(loss) // 120)
    if win > 1:
        smooth = np.convolve(loss, np.ones(win) / win, mode="valid")
        ax.plot(
            steps[win - 1 :],
            smooth,
            color=TORX_COLOR,
            linewidth=1.9,
            label="rolling mean",
        )
    if reference_loss is not None:
        ax.axhline(
            float(reference_loss),
            color=NEUTRAL_GRAY,
            linestyle="--",
            linewidth=0.9,
            label=f"final loss {float(reference_loss):.2f}",
        )
    ax.legend(loc="upper right")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("weighted BCE loss")
    ax.set_title("UNet denoiser training loss (offline, full MNIST)")
    ax.set_xlim(float(steps[0]), float(steps[-1]))
    # log scale so late-training decay stays visible after the initial drop
    ax.set_yscale("log")
    lo = float(loss.min())
    hi = float(np.percentile(loss, 99.5))
    ax.set_ylim(lo * 0.94, hi * 1.06)
    _style_axes(ax)
    return fig


def plot_reconstruction_grid(rows):
    """Small-multiples digit grid: one row per stage, stage label on the left.

    ``rows`` is a list of ``(label, batch)`` pairs. ``batch`` is a stack of
    binary images and ``label`` is the row name. Rendered larger and at a
    higher DPI than the notebook default so the 28x28 MNIST digits stay crisp.
    """
    n_rows = len(rows)
    n_cols = len(rows[0][1])

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 0.75 + 1.8, n_rows * 1.05 + 0.4),
        squeeze=False,
        dpi=GRID_DPI,
    )
    for row_idx, (label, batch) in enumerate(rows):
        for col_idx, image in enumerate(batch):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(NEUTRAL_GRAY)
                spine.set_linewidth(0.6)
        axes[row_idx, 0].set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
            labelpad=38,
            linespacing=1.35,
        )

    fig.subplots_adjust(
        left=0.30,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=0.05,
        hspace=0.05,
    )
    return fig


def plot_fid_drop(*, fid_corrupted, fid_denoised, title=None):
    """Two-bar FID figure: the corrupted ceiling dropping to the denoised value.

    The percentage reduction lives in the subtitle (outside the data field) so
    no annotation sits on the bars or a leader line.
    """
    labels = ["corrupted\n(input)", "denoised\n(UNet)"]
    vals = [float(fid_corrupted), float(fid_denoised)]
    x = np.arange(len(labels))
    drop = (vals[0] - vals[1]) / vals[0]

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR, constrained_layout=True, dpi=200)
    bars = ax.bar(x, vals, color=[NEUTRAL_GRAY, TORX_COLOR], width=0.56)
    for bar, value in zip(bars, vals, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02 * max(vals),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("FID (lower is better)")
    base = title or "Distribution quality: full-MNIST FID"
    ax.set_title(base, pad=18)
    ax.text(
        0.5,
        1.02,
        f"denoising cuts FID by {drop:.0%}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color=EXTROPIC_COPPER,
    )
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=3, width=0.6)
    return fig


def plot_flip_probability(forward_p, *, sigma_max=3.0):
    """Plot the per-bit flip probability p(sigma) = 0.5*(1 - exp(-2*sigma)).

    Draws the forward-process flip curve against noise time ``sigma``, the 0.5
    asymptote as a dashed reference, and marks the notebook's chosen operating
    point ``forward_p`` with a dot and a vertical guide down to the axis.
    """
    forward_p = float(forward_p)

    sigma = np.linspace(0.0, sigma_max, 400)
    p = 0.5 * (1.0 - np.exp(-2.0 * sigma))

    sigma_op = -0.5 * np.log(1.0 - 2.0 * forward_p)

    fig, ax = plt.subplots(figsize=FIGSIZE_CONVERGENCE, constrained_layout=True)

    ax.axhline(
        0.5,
        color=NEUTRAL_GRAY,
        linestyle="--",
        linewidth=0.9,
        label=r"$\frac{1}{2}$ (pure noise)",
    )

    ax.plot(sigma, p, color=TORX_COLOR, linewidth=1.8)

    ax.plot(
        [sigma_op, sigma_op],
        [0.0, forward_p],
        color=EXACT_COLOR,
        linestyle=":",
        linewidth=1.0,
    )
    ax.scatter(
        [sigma_op],
        [forward_p],
        s=46,
        color=EXACT_COLOR,
        zorder=5,
        edgecolors="white",
        linewidths=0.6,
        label=rf"operating point  $p={forward_p:.2f}$  ($\sigma={sigma_op:.2f}$)",
    )
    ax.set_xlabel(r"noise time $\sigma$")
    ax.set_ylabel("per-bit flip probability")
    ax.set_title(r"Forward flip probability $p(\sigma)=\frac{1}{2}\,(1-e^{-2\sigma})$")
    ax.set_xlim(0.0, sigma_max)
    ax.set_ylim(0.0, 0.56)
    _style_axes(ax)
    ax.legend(loc="lower right")
    return fig


def plot_training_curve(
    loss_history,
    acc_history,
    example_image,
    *,
    validation_n,
    majority_baseline=None,
    title=None,
):
    """Dual-axis validation curve for loss and accuracy."""
    loss_history = np.asarray(loss_history, dtype=float)
    acc_history = np.asarray(acc_history, dtype=float)
    epochs = np.arange(1, len(loss_history) + 1)

    fig, ax_loss = plt.subplots(figsize=FIGSIZE_BAR, constrained_layout=True)

    ax_loss.plot(epochs, loss_history, color=EXACT_COLOR, linewidth=1.5)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel(
        f"validation BCE loss, n={validation_n} (nats)", color=EXACT_COLOR
    )
    ax_loss.tick_params(axis="y", labelcolor=EXACT_COLOR)
    # frame the loss axis to the data span, rounded out to nearest 0.05
    loss_lo = max(0.0, np.floor(loss_history.min() * 20) / 20)
    loss_hi = np.ceil(loss_history.max() * 20) / 20
    ax_loss.set_ylim(loss_lo, loss_hi)
    ax_loss.set_xlim(epochs[0], epochs[-1])

    ax_acc = ax_loss.twinx()
    ax_acc.plot(epochs, acc_history, color=TORX_COLOR, linewidth=1.5)
    ax_acc.set_ylabel(f"validation accuracy, n={validation_n}", color=TORX_COLOR)
    ax_acc.tick_params(axis="y", labelcolor=TORX_COLOR)
    ax_acc.set_ylim(0.0, 1.05)
    ax_acc.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if majority_baseline is not None:
        ax_acc.axhline(
            majority_baseline,
            color=NEUTRAL_GRAY,
            linestyle=":",
            linewidth=0.9,
            label=f"validation majority baseline ({majority_baseline:.0%})",
        )
        ax_acc.legend(loc="lower right", frameon=False, fontsize=7)

    ax_acc.annotate(
        "validation accuracy",
        xy=(epochs[-1], acc_history[-1]),
        xytext=(epochs[-1] * 0.62, min(acc_history[-1] - 0.06, 0.94)),
        fontsize=8,
        color=TORX_COLOR,
        va="center",
        arrowprops={"arrowstyle": "-", "color": TORX_COLOR, "lw": 0.7},
    )
    ax_loss.annotate(
        f"validation loss {loss_history[-1]:.2f}",
        xy=(epochs[-1], loss_history[-1]),
        xytext=(epochs[-1] * 0.84, loss_history[-1] + 0.12),
        fontsize=8,
        color=EXACT_COLOR,
        ha="center",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": EXACT_COLOR, "lw": 0.7},
    )

    # bars-and-stripes example input as an inset
    inset_ax = ax_loss.inset_axes([0.60, 0.46, 0.15, 0.24])
    inset_ax.imshow(
        np.asarray(example_image),
        cmap="gray_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    inset_ax.set_aspect("equal")
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("0.55")
        spine.set_linewidth(0.7)
    inset_ax.set_title("bar: 1=black, 0=white", fontsize=7, color="0.40", pad=2)

    if title:
        ax_loss.set_title(title, fontsize=10)
    return fig
