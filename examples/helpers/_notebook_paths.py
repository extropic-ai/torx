"""Path helpers for Torx example notebooks."""

from pathlib import Path


def figure_dir(root: Path) -> Path:
    """Return the generated figure directory under the given examples ``root``.

    ``root`` is the examples directory the notebook already resolved in its
    bootstrap cell (the sibling-``helpers/`` heuristic lives there, since the
    path must be known before any helper can be imported).
    """
    path = root / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def asset_dir(root: Path, name: str) -> Path:
    """Return a committed-asset directory (e.g. checkpoints) under ``examples/assets``."""
    return root / "assets" / name
