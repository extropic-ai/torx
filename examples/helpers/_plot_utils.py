"""Shared plotting primitives reused across the per-notebook helpers."""

import matplotlib.pyplot as plt
import numpy as np
from _notebook_style import EXTROPIC_ORANGE, NEUTRAL_GRAY
from matplotlib.patches import Ellipse


def _cov_ellipse(ax, mean, cov, *, n_std=1.0, **kwargs):
    """Draw an ``n_std`` covariance ellipse for the given mean and covariance."""
    vals, vecs = np.linalg.eigh(np.asarray(cov))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    # only roundoff-scale negatives are acceptable; a material one means the
    # covariance is not PSD and the ellipse would be meaningless
    tol = 1e-9 * max(float(np.max(np.abs(vals))), 1.0)
    if np.min(vals) < -tol:
        raise ValueError("covariance has a material negative eigenvalue")
    width, height = 2 * n_std * np.sqrt(np.clip(vals, 0.0, None))
    ax.add_patch(
        Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs)
    )


def _clean_axes(ax):
    ax.tick_params(axis="both", direction="out", length=3, width=0.6)


def _square_limits(*clouds, pad=0.4, cap=None):
    """Return one symmetric (lo, hi) box covering the bulk of every cloud."""
    pts = np.vstack(clouds)
    hi = float(np.abs(pts).max()) + pad
    if cap is not None:
        hi = min(hi, cap)
    return -hi, hi


def _padded_range(*arrays, pad=0.08, include_zero=False):
    """(lo, hi) covering every array, expanded by ``pad`` of the span on each side."""
    parts = [np.asarray(a, dtype=float).ravel() for a in arrays]
    if include_zero:
        parts.append(np.array([0.0]))
    vals = np.concatenate(parts)
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo
    if span <= 0.0:
        # constant data: pad relative to its magnitude so limits stay readable
        return lo - pad * max(abs(lo), 1.0), hi + pad * max(abs(hi), 1.0)
    return lo - pad * span, hi + pad * span


def _parity_panel(
    ax,
    x,
    y,
    *,
    pad=0.08,
    color=EXTROPIC_ORANGE,
    marker_size=42,
    alpha=0.9,
    edgecolor="white",
    edge_width=0.5,
    scatter_zorder=2,
    scatter_label=None,
    line_style="--",
    line_width=1.0,
    line_zorder=1,
    line_label=None,
    clean=True,
):
    """Draw a square parity panel: a dashed ``y = x`` guide and an equal-aspect scatter.

    Computes one shared padded box from both series, draws the gray identity line
    and the orange (Torx) scatter with a white edge, and locks equal aspect.
    Returns ``(lo, hi)`` so callers can place extra artists (labels, rings, ...).
    """
    lo, hi = _padded_range(x, y, pad=pad)
    ax.plot(
        [lo, hi],
        [lo, hi],
        color=NEUTRAL_GRAY,
        linestyle=line_style,
        linewidth=line_width,
        zorder=line_zorder,
        label=line_label,
    )
    ax.scatter(
        x,
        y,
        color=color,
        s=marker_size,
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=edge_width,
        zorder=scatter_zorder,
        label=scatter_label,
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    if clean:
        _clean_axes(ax)
    return lo, hi


def _two_col_grid(n, *, figsize=None, width=6.0, row_height=2.7, **kwargs):
    """A 2-column subplot grid sized to ``n`` panels with trailing axes hidden.

    Notebooks lay panels out two-wide (not one n-wide strip) so each panel stays
    readable in the docs column. ``sharex``/``sharey`` default to ``True`` since
    callers compare panels on a common scale; pass them explicitly to override.
    Returns ``(fig, axes)`` with ``axes`` raveled; any axis past ``n`` is off.
    """
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (width, row_height * nrows)
    kwargs.setdefault("sharex", True)
    kwargs.setdefault("sharey", True)
    kwargs.setdefault("constrained_layout", True)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    axes = np.asarray(axes).ravel()
    for j in range(n, len(axes)):
        axes[j].axis("off")
    return fig, axes
