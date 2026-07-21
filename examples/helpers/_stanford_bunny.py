"""Private Stanford Bunny helpers scoped to the Torx example notebooks."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _notebook_style import BUNNY_BASE_COLOR, display_and_close, HEAT_CMAP
from matplotlib import cm, colors
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy import sparse
from scipy.sparse.csgraph import connected_components

Edge = tuple[int, int]

# rounds of 1-to-4 subdivision at render time so flat shading reads smooth
MESH_SUBDIVISION_ROUNDS = 1


def load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertex positions and triangle indices from an ASCII PLY file."""
    vertex_count = None
    face_count = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2])
            elif len(parts) == 3 and parts[:2] == ["element", "face"]:
                face_count = int(parts[2])
            elif line.strip() == "end_header":
                break

        if vertex_count is None or face_count is None:
            raise ValueError(f"{path} is missing vertex or face counts")

        vertices = np.array(
            [
                [float(value) for value in next(handle).split()[:3]]
                for _ in range(vertex_count)
            ],
            dtype=float,
        )

        faces: list[list[int]] = []
        for _ in range(face_count):
            values = [int(value) for value in next(handle).split()]
            corners = values[0]
            indices = values[1 : 1 + corners]
            if corners < 3:
                raise ValueError(f"face with {corners} corners")
            if corners == 3:
                faces.append(indices)
            else:
                faces.extend(
                    [indices[0], indices[i], indices[i + 1]]
                    for i in range(1, corners - 1)
                )

    return vertices, np.asarray(faces, dtype=np.int32)


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    """Return sorted undirected edges induced by triangular faces."""
    edge_set: set[Edge] = set()
    for a, b, c in np.asarray(faces, dtype=np.int32):
        edge_set.add(tuple(sorted((int(a), int(b)))))
        edge_set.add(tuple(sorted((int(b), int(c)))))
        edge_set.add(tuple(sorted((int(c), int(a)))))
    return np.asarray(sorted(edge_set), dtype=np.int32)


def largest_connected_mesh(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop unreferenced scan vertices and keep the largest vertex-edge connected component."""
    edges = mesh_edges(faces)
    adjacency = sparse.coo_matrix(
        (
            np.ones(2 * len(edges)),
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels)
    main_component = int(np.argmax(component_sizes))
    keep = np.flatnonzero(labels == main_component)

    old_to_new = np.full(len(vertices), -1, dtype=np.int32)
    old_to_new[keep] = np.arange(len(keep), dtype=np.int32)
    face_mask = np.all(old_to_new[faces] >= 0, axis=1)
    return vertices[keep], old_to_new[faces[face_mask]]


def display_vertices(vertices: np.ndarray) -> np.ndarray:
    """Map Stanford coordinates to a centered Matplotlib display frame."""
    shown = vertices[:, [0, 2, 1]]
    return shown - shown.mean(axis=0)


def subdivide_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Apply 1-to-4 triangle subdivision for smoother mesh rendering."""
    current_vertices = np.asarray(vertices, dtype=float)
    current_faces = np.asarray(faces, dtype=np.int32)
    edge_maps: list[np.ndarray] = []

    for _ in range(rounds):
        edges = np.concatenate(
            [
                current_faces[:, [0, 1]],
                current_faces[:, [1, 2]],
                current_faces[:, [2, 0]],
            ],
            axis=0,
        )
        edges_sorted = np.sort(edges, axis=1)
        unique_edges, inverse = np.unique(edges_sorted, axis=0, return_inverse=True)
        edge_maps.append(unique_edges)
        midpoints = current_vertices[unique_edges].mean(axis=1)
        n_old = len(current_vertices)
        current_vertices = np.concatenate([current_vertices, midpoints], axis=0)

        inverse = inverse.reshape(3, len(current_faces)).T
        e_ab = n_old + inverse[:, 0]
        e_bc = n_old + inverse[:, 1]
        e_ca = n_old + inverse[:, 2]
        a, b, c = current_faces[:, 0], current_faces[:, 1], current_faces[:, 2]
        current_faces = (
            np.stack(
                [
                    np.stack([a, e_ab, e_ca], axis=1),
                    np.stack([e_ab, b, e_bc], axis=1),
                    np.stack([e_ca, e_bc, c], axis=1),
                    np.stack([e_ab, e_bc, e_ca], axis=1),
                ],
                axis=1,
            )
            .reshape(-1, 3)
            .astype(np.int32)
        )
    return current_vertices, current_faces, edge_maps


def subdivide_scalar(values: np.ndarray, edge_maps: Sequence[np.ndarray]) -> np.ndarray:
    """Lift a per-vertex scalar field through ``subdivide_mesh`` edge maps."""
    current = np.asarray(values, dtype=float)
    for edges in edge_maps:
        current = np.concatenate([current, current[edges].mean(axis=1)], axis=0)
    return current


def add_isolines_3d(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    *,
    levels: int | Sequence[float] = 12,
    vmin: float = 0.0,
    vmax: float | None = None,
    color: str = "#333333",
    linewidth: float = 0.45,
) -> None:
    """Overlay approximate scalar isolines on a triangular 3D mesh."""
    values = np.asarray(values, dtype=float)
    if vmax is None:
        vmax = float(values.max())
    if isinstance(levels, int):
        level_values = np.linspace(vmin, vmax, levels + 2)[1:-1]
    else:
        level_values = np.asarray(levels, dtype=float)

    segments = []
    for face in np.asarray(faces, dtype=np.int32):
        tri = vertices[face]
        vals = values[face]
        for level in level_values:
            points = []
            for i, j in ((0, 1), (1, 2), (2, 0)):
                vi, vj = vals[i], vals[j]
                if (vi - level) * (vj - level) > 0 or vi == vj:
                    continue
                t = (level - vi) / (vj - vi)
                if 0.0 <= t <= 1.0:
                    points.append(tri[i] + t * (tri[j] - tri[i]))
            if len(points) == 2:
                segments.append(points)
    if not segments:
        return
    collection = Line3DCollection(
        segments,
        colors=colors.to_rgba(color, 0.55),
        linewidths=linewidth,
    )
    ax.add_collection3d(collection)


def save_torx_reference_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    panels: Sequence[tuple[str, np.ndarray]],
    output_stem: Path,
    *,
    ncols: int = 2,
    cmap: str | Colormap = HEAT_CMAP,
    colorbar_label: str = "normalized heat  (per-panel relative)",
    isolevels: int = 12,
) -> None:
    """Save a row of bunny heat panels with explicit per-panel titles.

    ``panels`` is a list of ``(title, field)`` pairs where ``field`` is a
    length-|V| nonnegative array; each panel is normalized to its own peak.
    """
    cmap = plt.get_cmap(cmap)
    titles = [title for title, _ in panels]
    fields = []
    for _title, field in panels:
        field = np.asarray(field, dtype=float).reshape(-1)
        peak = float(field.max())
        fields.append(field / peak if peak > 0.0 else field)

    sub_v, sub_f, sub_panels = _subdivide_for_render(vertices, faces, fields)
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    nrows = -(-len(panels) // ncols)
    fig_width = min(6.2, 3.0 * ncols)
    fig = plt.figure(figsize=(fig_width, 2.7 * nrows), dpi=300)
    rectangles, colorbar_rect = _draw_panel_grid(n_panels=len(panels), ncols=ncols)
    for title, values, rect in zip(titles, sub_panels, rectangles, strict=True):
        ax = fig.add_axes(rect, projection="3d")
        add_bunny_heat_mesh(
            ax,
            sub_v,
            sub_f,
            values,
            norm=norm,
            cmap=cmap,
            isolines=isolevels,
            isoline_vmin=0.0,
            isoline_vmax=1.0,
        )
        ax.set_title(title, fontsize=10, pad=-4)

    fig.patch.set_facecolor("white")
    cax = fig.add_axes(colorbar_rect)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(colorbar_label, labelpad=6, fontsize=9)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(labelsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{ext}"), facecolor="white")
    display_and_close(fig)


def visible_source_vertex(vertices: np.ndarray) -> int:
    """Pick a source vertex that is visible in the default camera."""
    shown = display_vertices(vertices)
    mins = shown.min(axis=0)
    spans = np.maximum(shown.max(axis=0) - mins, 1e-12)
    target = np.array(
        [
            mins[0] + 0.42 * spans[0],
            mins[1],
            mins[2] + 0.78 * spans[2],
        ]
    )
    screen_distance = (shown[:, 0] - target[0]) ** 2 + (shown[:, 2] - target[2]) ** 2
    return int(np.argmin(screen_distance))


def _smooth_face_shade(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Per-face shade from area-weighted per-vertex normals."""
    triangles = vertices[faces]
    raw_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    # don't normalize: `raw_normals` magnitude is 2 * face area, the right vertex-averaging weight
    vertex_normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(vertex_normals, faces[:, corner], raw_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(norms, 1e-12)

    light = np.array([-0.35, -0.55, 0.76])
    light = light / np.linalg.norm(light)
    vertex_shade = 0.55 + 0.45 * np.abs(vertex_normals @ light)
    return vertex_shade[faces].mean(axis=1)


def _frame_bunny_axes(
    ax: plt.Axes, shown: np.ndarray, elev: float = 14.0, azim: float = -72.0
) -> None:
    mins = shown.min(axis=0)
    maxes = shown.max(axis=0)
    center = (mins + maxes) / 2.0
    # half the largest extent plus a small margin so the mesh never clips
    radius = float(np.max(maxes - mins) * 0.53)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def add_bunny_heat_mesh(
    ax: plt.Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    *,
    norm: colors.Normalize,
    cmap: str | Colormap = HEAT_CMAP,
    elev: float = 14.0,
    azim: float = -72.0,
    base_color: str = BUNNY_BASE_COLOR,
    isolines: int | Sequence[float] | None = 12,
    isoline_vmin: float = 0.0,
    isoline_vmax: float | None = None,
) -> None:
    """Draw the bunny mesh as a light-gray base with a scalar field overlay.

    `vertices` / `faces` / `values` should already be subdivided; the
    save_*_figure helpers do this automatically.
    """
    cmap = plt.get_cmap(cmap)
    shown = display_vertices(vertices)
    triangles = shown[faces]
    face_values = values[faces].mean(axis=1)
    normalized = np.clip(np.asarray(norm(face_values), dtype=float), 0.0, 1.0)

    base_rgb = np.asarray(colors.to_rgb(base_color))
    shade = _smooth_face_shade(shown, faces)
    base = base_rgb[None, :] * shade[:, None]

    heat_rgb = cmap(normalized)[:, :3]
    alpha = (normalized**0.85)[:, None]
    facecolors = (1.0 - alpha) * base + alpha * heat_rgb
    facecolors = np.clip(facecolors, 0.0, 1.0)
    facecolors_rgba = np.concatenate(
        [facecolors, np.ones((len(facecolors), 1))], axis=1
    )

    _frame_bunny_axes(ax, shown, elev=elev, azim=azim)

    collection = Poly3DCollection(
        triangles,
        facecolors=facecolors_rgba,
        edgecolors=facecolors_rgba,
        linewidths=0.2,
        antialiased=False,
        rasterized=True,
    )
    collection.set_zsort("average")
    ax.add_collection3d(collection)

    if isolines is not None:
        vmax = isoline_vmax if isoline_vmax is not None else float(values.max())
        add_isolines_3d(
            ax,
            shown,
            faces,
            values,
            levels=isolines,
            vmin=isoline_vmin,
            vmax=vmax,
        )


def _subdivide_for_render(
    vertices: np.ndarray,
    faces: np.ndarray,
    scalar_fields: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Subdivide the mesh and lift each scalar field onto the finer mesh."""
    sub_vertices, sub_faces, edge_maps = subdivide_mesh(
        vertices, faces, rounds=MESH_SUBDIVISION_ROUNDS
    )
    sub_fields = [subdivide_scalar(f, edge_maps) for f in scalar_fields]
    return sub_vertices, sub_faces, sub_fields


def _draw_panel_grid(
    n_panels: int,
    *,
    ncols: int = 2,
) -> tuple[list[list[float]], list[float]]:
    """Return ``(panel_rects, colorbar_rect)`` for an N-panel grid.

    The panels are arranged in a ``ceil(n_panels / ncols)`` x ``ncols`` grid so
    the figure fits the rendered docs column; a horizontal colorbar sits along
    the bottom edge. Each rectangle is an ``add_axes()`` ``[left, bottom, w, h]``.
    """
    nrows = -(-n_panels // ncols)  # ceil
    left, right = 0.01, 0.99
    top, panel_bottom = 0.97, 0.10
    col_gap, row_gap = 0.012, 0.02
    cell_w = (right - left - (ncols - 1) * col_gap) / ncols
    cell_h = (top - panel_bottom - (nrows - 1) * row_gap) / nrows
    rects: list[list[float]] = []
    for i in range(n_panels):
        row, col = divmod(i, ncols)
        x = left + col * (cell_w + col_gap)
        # fill top-to-bottom so panels read in time order down the columns' rows
        y = panel_bottom + (nrows - 1 - row) * (cell_h + row_gap)
        rects.append([x, y, cell_w, cell_h])
    colorbar_rect = [0.25, 0.035, 0.5, 0.02]
    return rects, colorbar_rect
