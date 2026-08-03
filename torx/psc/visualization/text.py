"""ASCII text rendering for probabilistic circuits.

The visual style and architecture is inspired by Qiskit's circuit visualization.
See: https://github.com/Qiskit/qiskit
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .._circuit import AbstractPCircuit, DiscretePCircuit, HybridPCircuit
from ..gates._base import (
    AbstractControlledContinuousGate,
    AbstractDiscreteGate,
    AbstractHybridGate,
    AbstractPGate,
)


@dataclass
class GateDrawInfo:
    """Drawing information for a gate."""

    label: str
    all_sites: list[int]
    control_sites: list[int]
    target_sites: list[int]
    is_hybrid: bool
    discrete_sites: list[int]
    continuous_sites: list[int]


class TextDrawing:
    """ASCII circuit drawing that can be printed."""

    def __init__(self, lines: list[str]):
        self._lines = lines

    def __str__(self) -> str:
        return "\n".join(self._lines)

    def __repr__(self) -> str:
        return str(self)

    def _repr_html_(self) -> str:
        """Rich display for Jupyter notebooks."""
        return f"<pre>{self}</pre>"


def get_gate_label(gate: AbstractPGate) -> str:
    """Get display label for a gate."""
    label = gate._draw_label
    if label is not None:
        return label
    return gate.__class__.__name__


def _get_control_indices(gate: AbstractPGate, num_sites: int) -> list[int]:
    """Resolve _control_indices to actual indices."""
    ctrl = gate._control_indices
    if isinstance(ctrl, str):
        if ctrl == "all_but_last":
            return list(range(num_sites - 1))
        return []
    return list(ctrl)


def get_gate_draw_info(gate: AbstractPGate) -> GateDrawInfo:
    """Extract drawing info from a gate."""
    label = get_gate_label(gate)

    if isinstance(gate, AbstractDiscreteGate):
        sites = gate.sites if isinstance(gate.sites, list) else [gate.sites]

        control_indices = _get_control_indices(gate, len(sites))
        control_sites = [sites[i] for i in control_indices]
        target_sites = [s for i, s in enumerate(sites) if i not in control_indices]

        return GateDrawInfo(
            label=label,
            all_sites=sites,
            control_sites=control_sites,
            target_sites=target_sites,
            is_hybrid=False,
            discrete_sites=sites,
            continuous_sites=[],
        )

    elif isinstance(gate, AbstractHybridGate):
        gate_sites = gate.sites
        disc_sites = list(gate_sites.get("discrete", []))
        cont_sites = gate_sites.get("continuous", [])
        all_sites = disc_sites + cont_sites

        # For controlled continuous gates, discrete sites are controls
        if isinstance(gate, AbstractControlledContinuousGate):
            control_sites = disc_sites
            target_sites = cont_sites
        else:
            control_sites = []
            target_sites = cont_sites

        return GateDrawInfo(
            label=label,
            all_sites=all_sites,
            control_sites=control_sites,
            target_sites=target_sites,
            is_hybrid=True,
            discrete_sites=disc_sites,
            continuous_sites=cont_sites,
        )

    else:
        # Fallback for unknown gate types
        return GateDrawInfo(
            label=label,
            all_sites=[0],
            control_sites=[],
            target_sites=[0],
            is_hybrid=False,
            discrete_sites=[0],
            continuous_sites=[],
        )


def compute_layout(
    gate_infos: Sequence[GateDrawInfo],
) -> list[int]:
    """Assign each gate to a column, allowing parallel placement.

    Gates that don't share any sites can be placed in the same column.
    For controlled gates that span multiple wires, all intermediate wires
    are also blocked. Returns a list of column indices, one per gate.
    """
    site_frontier: dict[int, int] = {}
    gate_columns: list[int] = []

    for info in gate_infos:
        sites = info.all_sites

        if not sites:
            gate_columns.append(0)
            continue

        has_controls = len(info.control_sites) > 0 and len(info.target_sites) > 0
        spans_multiple = len(info.target_sites) > 1
        if has_controls or spans_multiple:
            min_wire = min(sites)
            max_wire = max(sites)
            blocked_sites = list(range(min_wire, max_wire + 1))
        else:
            blocked_sites = sites

        # Gate goes in the max of all blocked sites' frontiers
        col = max(site_frontier.get(s, 0) for s in blocked_sites)

        gate_columns.append(col)

        # Advance frontier for all blocked sites
        for s in blocked_sites:
            site_frontier[s] = col + 1

    return gate_columns


def draw_circuit(
    circuit: AbstractPCircuit, *, output: bool = True, unroll: bool = False
) -> TextDrawing | None:
    """Create ASCII representation of a circuit.

    **Arguments:**

    - `circuit`: The circuit to draw.
    - `output`: If True (default), print the circuit and return None.
        If False, return a TextDrawing object.
    - `unroll`: If True, repeat the circuit content `reps` times instead of
        showing a box with "xN". Default is False.

    **Returns:**

    A TextDrawing object if output=False, otherwise None.
    """
    if isinstance(circuit, DiscretePCircuit):
        drawing = _draw_discrete_circuit(circuit, unroll=unroll)
    elif isinstance(circuit, HybridPCircuit):
        drawing = _draw_hybrid_circuit(circuit, unroll=unroll)
    else:
        raise TypeError(f"Unsupported circuit type: {type(circuit)}")

    if output:
        print(drawing)
        return None
    return drawing


def _draw_discrete_circuit(
    circuit: DiscretePCircuit, *, unroll: bool = False
) -> TextDrawing:
    """Draw a discrete circuit."""
    num_wires = circuit.num_pdits
    if num_wires == 0:
        return TextDrawing(["(empty circuit)"])

    gates = circuit.gates
    reps = circuit.reps if unroll else 1

    if not gates:
        lines = []
        for i in range(num_wires):
            lines.append(f"p_{i}: " + "─" * 20)
        return TextDrawing(lines)

    unrolled_gates = list(gates) * reps
    gate_infos = [get_gate_draw_info(g) for g in unrolled_gates]

    gate_columns = compute_layout(gate_infos)
    num_columns = max(gate_columns) + 1 if gate_columns else 0

    # Compute width for each column
    col_widths = [0] * num_columns
    for col, info in zip(gate_columns, gate_infos):
        width = len(info.label) + 4
        col_widths[col] = max(col_widths[col], width)

    col_widths = [max(w, 5) for w in col_widths]

    grid = _build_grid(num_wires, col_widths, gate_columns, gate_infos, is_hybrid=False)
    lines = _grid_to_lines(grid, num_wires, "p")

    if not unroll and circuit.reps > 1:
        lines = _add_reps_box(lines, circuit.reps)

    return TextDrawing(lines)


def _draw_hybrid_circuit(
    circuit: HybridPCircuit, *, unroll: bool = False
) -> TextDrawing:
    """Draw a hybrid circuit with both discrete and continuous wires."""
    num_discrete = circuit.num_discrete_sites
    num_continuous = circuit.num_continuous_sites
    num_wires = num_discrete + num_continuous

    if num_wires == 0:
        return TextDrawing(["(empty circuit)"])

    gates = circuit.gates
    reps = circuit.reps if unroll else 1

    if not gates:
        lines = []
        for i in range(num_discrete):
            lines.append(f"d_{i}: " + "─" * 20)
        for i in range(num_continuous):
            lines.append(f"c_{i}: " + "═" * 20)
        return TextDrawing(lines)

    unrolled_gates = list(gates) * reps
    gate_infos = [get_gate_draw_info(g) for g in unrolled_gates]

    for info in gate_infos:
        if info.is_hybrid:
            info.continuous_sites = [s + num_discrete for s in info.continuous_sites]
            info.all_sites = info.discrete_sites + info.continuous_sites
            info.target_sites = info.continuous_sites

    gate_columns = compute_layout(gate_infos)
    num_columns = max(gate_columns) + 1 if gate_columns else 0

    col_widths = [0] * num_columns
    for col, info in zip(gate_columns, gate_infos):
        width = len(info.label) + 4
        col_widths[col] = max(col_widths[col], width)
    col_widths = [max(w, 5) for w in col_widths]

    grid = _build_grid(
        num_wires,
        col_widths,
        gate_columns,
        gate_infos,
        is_hybrid=True,
        num_discrete=num_discrete,
    )

    lines = _grid_to_lines_hybrid(grid, num_discrete, num_continuous)

    if not unroll and circuit.reps > 1:
        lines = _add_reps_box(lines, circuit.reps)

    return TextDrawing(lines)


def _build_grid(
    num_wires: int,
    col_widths: list[int],
    gate_columns: list[int],
    gate_infos: list[GateDrawInfo],
    is_hybrid: bool,
    num_discrete: int = 0,
) -> list[list[str]]:
    """Build character grid for the circuit.

    Each wire has 3 rows: top border, wire, bottom border.
    """
    total_width = sum(col_widths) + 2
    grid: list[list[str]] = []

    for wire in range(num_wires):
        is_continuous = is_hybrid and wire >= num_discrete
        wire_char = "═" if is_continuous else "─"

        grid.append([" "] * total_width)
        grid.append([wire_char] * total_width)
        grid.append([" "] * total_width)

    col_starts = [0]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)

    for col, info in zip(gate_columns, gate_infos):
        col_start = col_starts[col]
        col_width = col_widths[col]

        if not info.all_sites:
            continue

        has_controls = len(info.control_sites) > 0 and len(info.target_sites) > 0

        if has_controls:
            _draw_control_gate(
                grid, col_start, col_width, info, is_hybrid, num_discrete
            )
        elif len(info.target_sites) > 1:
            _draw_spanning_gate(
                grid, col_start, col_width, info, is_hybrid, num_discrete
            )
        else:
            _draw_single_gate(grid, col_start, col_width, info, is_hybrid, num_discrete)

    return grid


def _draw_single_gate(
    grid: list[list[str]],
    col_start: int,
    col_width: int,
    info: GateDrawInfo,
    is_hybrid: bool,
    num_discrete: int,
) -> None:
    """Draw a single-site gate as a box."""
    wire = info.target_sites[0] if info.target_sites else info.all_sites[0]
    is_continuous = is_hybrid and wire >= num_discrete

    top_row = wire * 3
    mid_row = wire * 3 + 1
    bot_row = wire * 3 + 2

    label = info.label
    box_width = len(label) + 4
    box_start = col_start + (col_width - box_width) // 2

    # Top: ┌───────┐
    grid[top_row][box_start] = "┌"
    for i in range(1, box_width - 1):
        grid[top_row][box_start + i] = "─"
    grid[top_row][box_start + box_width - 1] = "┐"

    # Middle: ┤ LABEL ├
    l_conn = "╡" if is_continuous else "┤"
    r_conn = "╞" if is_continuous else "├"
    grid[mid_row][box_start] = l_conn
    grid[mid_row][box_start + 1] = " "
    for i, char in enumerate(label):
        grid[mid_row][box_start + 2 + i] = char
    grid[mid_row][box_start + 2 + len(label)] = " "
    grid[mid_row][box_start + box_width - 1] = r_conn

    # Bottom: └───────┘
    grid[bot_row][box_start] = "└"
    for i in range(1, box_width - 1):
        grid[bot_row][box_start + i] = "─"
    grid[bot_row][box_start + box_width - 1] = "┘"


def _draw_control_gate(
    grid: list[list[str]],
    col_start: int,
    col_width: int,
    info: GateDrawInfo,
    is_hybrid: bool,
    num_discrete: int,
) -> None:
    """Draw a gate with control dot(s) and target box."""
    label = info.label
    box_width = len(label) + 4
    center = col_start + col_width // 2

    if len(info.target_sites) > 1:
        min_target = min(info.target_sites)
        max_target = max(info.target_sites)
        target_top = min_target * 3
        target_bot = max_target * 3 + 2

        _draw_spanning_gate(grid, col_start, col_width, info, is_hybrid, num_discrete)

        if min(info.control_sites) < min_target:
            grid[target_top][center] = "┴"
        if max(info.control_sites) > max_target:
            grid[target_bot][center] = "┬"

        for ctrl_wire in info.control_sites:
            ctrl_mid = ctrl_wire * 3 + 1
            grid[ctrl_mid][center] = "■"

            if ctrl_wire < min_target:
                for row in range(ctrl_mid + 1, target_top):
                    if grid[row][center] in " ─═":
                        grid[row][center] = "│"
            elif ctrl_wire > max_target:
                for row in range(target_bot + 1, ctrl_mid):
                    if grid[row][center] in " ─═":
                        grid[row][center] = "│"

        return

    target_wire = info.target_sites[0] if info.target_sites else info.all_sites[-1]
    is_target_continuous = is_hybrid and target_wire >= num_discrete

    target_top = target_wire * 3
    target_mid = target_wire * 3 + 1
    target_bot = target_wire * 3 + 2

    box_start = col_start + (col_width - box_width) // 2

    min_control = min(info.control_sites) if info.control_sites else target_wire
    max_control = max(info.control_sites) if info.control_sites else target_wire

    controls_above = min_control < target_wire
    controls_below = max_control > target_wire

    # Draw target box
    grid[target_top][box_start] = "┌"
    for i in range(1, box_width - 1):
        if controls_above and box_start + i == center:
            grid[target_top][box_start + i] = "┴"
        else:
            grid[target_top][box_start + i] = "─"
    grid[target_top][box_start + box_width - 1] = "┐"

    l_conn = "╡" if is_target_continuous else "┤"
    r_conn = "╞" if is_target_continuous else "├"
    grid[target_mid][box_start] = l_conn
    grid[target_mid][box_start + 1] = " "
    for i, char in enumerate(label):
        grid[target_mid][box_start + 2 + i] = char
    grid[target_mid][box_start + 2 + len(label)] = " "
    grid[target_mid][box_start + box_width - 1] = r_conn

    grid[target_bot][box_start] = "└"
    for i in range(1, box_width - 1):
        if controls_below and box_start + i == center:
            grid[target_bot][box_start + i] = "┬"
        else:
            grid[target_bot][box_start + i] = "─"
    grid[target_bot][box_start + box_width - 1] = "┘"

    # Draw control dots and vertical lines
    for ctrl_wire in info.control_sites:
        ctrl_mid = ctrl_wire * 3 + 1
        grid[ctrl_mid][center] = "■"

        if ctrl_wire < target_wire:
            for row in range(ctrl_mid + 1, target_top):
                if grid[row][center] in " ─═":
                    grid[row][center] = "│"
        else:
            for row in range(target_bot + 1, ctrl_mid):
                if grid[row][center] in " ─═":
                    grid[row][center] = "│"


def _draw_spanning_gate(
    grid: list[list[str]],
    col_start: int,
    col_width: int,
    info: GateDrawInfo,
    is_hybrid: bool,
    num_discrete: int,
) -> None:
    """Draw a gate that spans multiple wires."""
    label = info.label
    box_width = len(label) + 4
    box_start = col_start + (col_width - box_width) // 2

    min_wire = min(info.target_sites)
    max_wire = max(info.target_sites)

    top_row = min_wire * 3
    bot_row = max_wire * 3 + 2

    # Top border
    grid[top_row][box_start] = "┌"
    for i in range(1, box_width - 1):
        grid[top_row][box_start + i] = "─"
    grid[top_row][box_start + box_width - 1] = "┐"

    # Side borders and wire connections
    for wire in range(min_wire, max_wire + 1):
        wire_mid = wire * 3 + 1
        is_continuous = is_hybrid and wire >= num_discrete
        l_conn = "╡" if is_continuous else "┤"
        r_conn = "╞" if is_continuous else "├"

        grid[wire_mid][box_start] = l_conn
        grid[wire_mid][box_start + box_width - 1] = r_conn

        for i in range(1, box_width - 1):
            grid[wire_mid][box_start + i] = " "

        if wire < max_wire:
            wire_bot = wire * 3 + 2
            grid[wire_bot][box_start] = "│"
            grid[wire_bot][box_start + box_width - 1] = "│"
            for i in range(1, box_width - 1):
                grid[wire_bot][box_start + i] = " "

            next_top = (wire + 1) * 3
            grid[next_top][box_start] = "│"
            grid[next_top][box_start + box_width - 1] = "│"
            for i in range(1, box_width - 1):
                grid[next_top][box_start + i] = " "

    # Bottom border
    grid[bot_row][box_start] = "└"
    for i in range(1, box_width - 1):
        grid[bot_row][box_start + i] = "─"
    grid[bot_row][box_start + box_width - 1] = "┘"

    # Draw label in the middle
    mid_wire = (min_wire + max_wire) // 2
    label_row = mid_wire * 3 + 1
    label_start = box_start + (box_width - len(label)) // 2
    for i, char in enumerate(label):
        grid[label_row][label_start + i] = char


def _grid_to_lines(
    grid: list[list[str]],
    num_wires: int,
    prefix: str,
) -> list[str]:
    """Convert grid to lines with wire labels."""
    lines = []
    max_prefix_width = len(f"{prefix}_{num_wires - 1}: ")

    for wire in range(num_wires):
        top_row = wire * 3
        mid_row = wire * 3 + 1
        bot_row = wire * 3 + 2

        top_prefix = " " * max_prefix_width
        lines.append(top_prefix + "".join(grid[top_row]))

        mid_prefix = f"{prefix}_{wire}: ".rjust(max_prefix_width)
        lines.append(mid_prefix + "".join(grid[mid_row]))

        bot_prefix = " " * max_prefix_width
        lines.append(bot_prefix + "".join(grid[bot_row]))

    return lines


def _grid_to_lines_hybrid(
    grid: list[list[str]],
    num_discrete: int,
    num_continuous: int,
) -> list[str]:
    """Convert grid to lines for hybrid circuit with d_ and c_ prefixes."""
    lines = []
    total_wires = num_discrete + num_continuous
    max_prefix_width = max(
        len(f"d_{num_discrete - 1}: ") if num_discrete > 0 else 0,
        len(f"c_{num_continuous - 1}: ") if num_continuous > 0 else 0,
    )
    max_prefix_width = max(max_prefix_width, 5)

    for wire in range(total_wires):
        top_row = wire * 3
        mid_row = wire * 3 + 1
        bot_row = wire * 3 + 2

        if wire < num_discrete:
            prefix = f"d_{wire}"
        else:
            prefix = f"c_{wire - num_discrete}"

        top_prefix = " " * max_prefix_width
        lines.append(top_prefix + "".join(grid[top_row]))

        mid_prefix = f"{prefix}: ".rjust(max_prefix_width)
        lines.append(mid_prefix + "".join(grid[mid_row]))

        bot_prefix = " " * max_prefix_width
        lines.append(bot_prefix + "".join(grid[bot_row]))

    return lines


def _add_reps_box(lines: list[str], reps: int) -> list[str]:
    """Wrap the circuit in a box with repetition indicator."""
    if not lines:
        return lines

    content_width = max(len(line) for line in lines)

    new_lines = []
    for line in lines:
        padded = line.ljust(content_width)
        new_lines.append("║ " + padded + " ║")

    box_width = content_width + 4
    top_border = "╔" + "═" * (box_width - 2) + "╗"
    bot_border = "╚" + "═" * (box_width - 2) + "╝"

    reps_str = f" x{reps}"

    return [top_border] + new_lines + [bot_border + reps_str]
