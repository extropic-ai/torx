"""Shared plotting style and rendering helpers for Torx notebooks."""

from collections.abc import Callable
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.figure import Figure

# --- Extropic brand palette -------------------------------------------------
# light-to-dark along the thermo gradient
EXTROPIC_SOLAR = "#FFF1C8"
EXTROPIC_GOLD = "#F5D64C"
EXTROPIC_YELLOW = "#FFB400"
EXTROPIC_ORANGE = "#FF8400"
EXTROPIC_COPPER = "#903001"
EXTROPIC_FUCHSIA = "#4E012A"
EXTROPIC_BROWN = "#1C0101"

# --- Semantic roles used across the notebooks -------------------------------
# slate = exact/ground-truth, orange = Torx, gold = coarse/intermediate
EXACT_COLOR = "#3D5A80"
TORX_COLOR = EXTROPIC_ORANGE
# deeper amber-gold than brand gold so coarse series stay legible on the light background
# brand `EXTROPIC_GOLD` is kept for the cycle and heat colormap
COARSE_COLOR = "#C8930A"
NEUTRAL_GRAY = "0.55"
BUNNY_BASE_COLOR = "#C8BFAE"
REGIME_COLORS = [EXACT_COLOR, TORX_COLOR, EXTROPIC_COPPER]

# soft off-white background so plots merge with the docs page, not harsh white cards on the dark theme
FIGURE_BG = "#E9E5DA"

# default color cycle for multi-series plots
EXTROPIC_CYCLE = [
    EXTROPIC_ORANGE,
    EXACT_COLOR,
    EXTROPIC_GOLD,
    EXTROPIC_COPPER,
    EXTROPIC_FUCHSIA,
    "#C77D3A",
    "#7A8FA6",
]

# dark-to-light thermo gradient for heat/density fields
EXTROPIC_THERMO = colors.LinearSegmentedColormap.from_list(
    "extropic_thermo",
    [
        EXTROPIC_BROWN,
        EXTROPIC_FUCHSIA,
        EXTROPIC_COPPER,
        EXTROPIC_ORANGE,
        EXTROPIC_YELLOW,
        EXTROPIC_GOLD,
        EXTROPIC_SOLAR,
    ],
)
# diverging map for exact-below / Torx-above a reference; centered on the background so near-zero cells merge in
EXTROPIC_DIVERGING = colors.LinearSegmentedColormap.from_list(
    "extropic_diverging", [EXACT_COLOR, FIGURE_BG, TORX_COLOR]
)
# register by name so `plt.get_cmap("extropic_thermo")` works too
for _cm in (EXTROPIC_THERMO, EXTROPIC_DIVERGING):
    if _cm.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cm)

HEAT_CMAP = EXTROPIC_THERMO

# docs column is ~752px = 6.27in at dpi 120, so authoring wider than ~6.2in downscales and crushes labels
FIGSIZE_SINGLE = (6.2, 4.2)
FIGSIZE_BAR = (6.2, 3.8)
FIGSIZE_CONVERGENCE = (6.2, 4.2)
FIGSIZE_MULTIPANEL = (6.2, 3.0)
FIGSIZE_PARITY = (4.4, 4.4)


def format_mu_label(mu_val: float) -> str:
    """Format a regime drift value for compact plot labels."""
    v = float(mu_val)
    if v == 0.0:
        return "0.0"
    if v > 0:
        return f"{v:.1f}"
    return f"−{abs(v):.1f}"


def apply_notebook_style() -> None:
    """Apply the shared Matplotlib style used by the notebooks."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.facecolor": FIGURE_BG,
            "axes.facecolor": FIGURE_BG,
            "savefig.facecolor": FIGURE_BG,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "axes.prop_cycle": plt.cycler(color=EXTROPIC_CYCLE),
        }
    )
    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        set_matplotlib_formats("retina")
    except (ImportError, ModuleNotFoundError):  # no inline backend installed
        pass


def savefig(fig: Figure, stem: str, figure_dir: Path) -> None:
    """Write ``fig`` into ``figure_dir`` as a PNG.

    The published docs are built from each notebook's embedded cell outputs (see
    ``display_and_close``), not from these files, so the on-disk copy is an
    optional convenience.
    """
    fig.savefig(
        figure_dir / f"{stem}.png",
        bbox_inches="tight",
        facecolor=FIGURE_BG,
    )


def make_savefig(figure_dir: Path) -> Callable[[Figure, str], None]:
    """Return a ``savefig(fig, stem)`` bound to ``figure_dir``.

    The returned callable writes the figure into ``figure_dir`` and then shows
    it inline (closing it), which is the one save-and-show idiom the gallery
    notebooks use.
    """

    def _savefig(fig: Figure, stem: str) -> None:
        savefig(fig, stem, figure_dir)
        display_and_close(fig)

    return _savefig


def display_and_close(fig: Figure) -> None:
    """Display a figure inline at 2x pixel density, then close it."""
    try:
        from io import BytesIO

        from IPython.display import display, Image
    except ImportError:
        plt.close(fig)
        return
    scale = 2
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=fig.get_dpi() * scale,
        bbox_inches="tight",
        facecolor=FIGURE_BG,
    )
    plt.close(fig)
    buf.seek(0)
    width: int | None = None
    try:
        from PIL import Image as _PILImage

        # never let an inline figure exceed the ~752px docs column
        width = min(_PILImage.open(buf).width // scale, 752)
    except (ImportError, ModuleNotFoundError):  # Pillow absent: full-size embed
        width = None
    buf.seek(0)
    display(Image(data=buf.read(), width=width))


def add_reference_slope(
    ax,
    *,
    slope: float,
    x_range: tuple[float, float],
    label: str,
    anchor_y: float | None = None,
    color: str = NEUTRAL_GRAY,
    linestyle: str = "--",
    linewidth: float = 0.9,
) -> np.ndarray:
    """Add a power-law reference line to a log-log plot, returning its y-values."""
    x0, x1 = x_range
    xs = np.array([x0, x1], dtype=float)
    if anchor_y is None:
        ymin, ymax = ax.get_ylim()
        anchor_y = np.sqrt(ymin * ymax)
    ys = anchor_y * (xs / xs[0]) ** slope
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
    return ys
