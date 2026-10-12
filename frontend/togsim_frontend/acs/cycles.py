"""The cost of a kernel's compute nodes, as the cycle table and the unit table TOGSim reads. A node is one compute
per tile: its cycles go in the first column, the overlapping column is always 0, and the Systolic rows it holds go
to the unit table, where systolic_array_split spreads them over a core's arrays. Estimates, not gem5 measurements."""
from __future__ import annotations

import math
from typing import NamedTuple

from ..onnx_frontend.tiling import Hardware
from .node_mapping import ELEMENTWISE, REDUCTION, NodeDesc

#: vslidedown at 2,4,6,8...; each vfmax consumes the last -- one chain, so linear.
FOLD_WIDTH = 8

SYSTOLIC_UNIT = ("Systolic", "rows", 1)
VECTOR_UNIT = ("Vpu", "cycles", 1)


class Cost(NamedTuple):
    """One compute node: the cycles it holds its core for before the array split, the rows it feeds a systolic
    array (one per cycle) and its cycles on the vector unit."""
    cycles: int
    systolic_rows: int = 0
    vector_cycles: int = 0


def systolic_instructions(tile: dict, hw: Hardware) -> tuple[int, int]:
    """(preloads, matmuls) of a rows x depth x cols block: a preload per depth step and SA*VLANE columns, and
    under each a matmul per SA rows."""
    rows, depth, cols = (int(tile.get(k, hw.sa)) for k in ("rows", "depth", "cols"))
    preloads = depth * math.ceil(cols / (hw.sa * hw.vlane))
    return preloads, preloads * math.ceil(rows / hw.sa)


def systolic_cost(tile: dict, hw: Hardware) -> Cost:
    """A block's matmuls each feed min(SA, rows) rows, one per cycle; a preload hides behind the matmul before it,
    and the last matmul drains through the array's 2*SA - 2 stages. The rows are what the split divides."""
    _, matmuls = systolic_instructions(tile, hw)
    fed = matmuls * min(hw.sa, int(tile.get("rows", hw.sa)))
    return Cost(fed + 2 * hw.sa - 2, systolic_rows=fed)


def vector_cost(node: NodeDesc, tile_elems: int, hw: Hardware) -> Cost:
    total = 0
    for op in node.ops:
        if op.kind == ELEMENTWISE:
            total += math.ceil(tile_elems / hw.throughput) * op.n
        elif op.kind == REDUCTION:
            total += (FOLD_WIDTH - 1) * (1 + op.n)
        else:
            raise ValueError(f"unknown op kind {op.kind!r} in node {node.name!r}")
    return Cost(total, vector_cycles=total)


def default_cost(node: NodeDesc, tile: dict, hw: Hardware) -> Cost:
    if node.compute_type == "systolic":
        return systolic_cost(tile, hw)
    if node.compute_type == "vector":
        return vector_cost(node, int(tile.get("elems", 1)), hw)
    raise ValueError(f"unknown compute type {node.compute_type!r}")

_cost_function = default_cost


def set_cost_function(fn) -> None:
    global _cost_function
    _cost_function = fn


def costs(kernel_desc, tile: dict, hw: Hardware) -> list[Cost]:
    """One Cost per compute node, in tile_id order. A kernel without a compute node (concat, flatten) gets one
    zero tile: the unit table has to name the array unit for systolic_array_split to accept it."""
    return [Cost(*_cost_function(node, tile, hw)) for node in kernel_desc.nodes] or [Cost(0)]


def build_table(kernel_desc, tile: dict, hw: Hardware) -> list[tuple[int, int]]:
    """The cycle table rows, (cycles, overlapping): overlapping is always 0, a compute holds its core to its end."""
    return [(cost.cycles, 0) for cost in costs(kernel_desc, tile, hw)]


def build_units(kernel_desc, tile: dict, hw: Hardware) -> list[tuple]:
    """The unit table rows, (tile_id, unit, unit_of_work, capacity_per_cycle, admitted), two per tile."""
    rows = []
    for tile_id, cost in enumerate(costs(kernel_desc, tile, hw)):
        rows.append((tile_id, *SYSTOLIC_UNIT, cost.systolic_rows))
        rows.append((tile_id, *VECTOR_UNIT, cost.vector_cycles))
    return rows


def uncovered(tile_ids, table: list[tuple[int, int]], units: list[tuple]) -> list[int]:
    """The `tile_ids` the tables do not cover: no cycle table line (TOGSim refuses a compute that names one) or no
    unit table row (TOGSim warns, and the compute adds nothing to the units)."""
    with_units = {row[0] for row in units}
    return sorted(t for t in tile_ids if not 0 <= t < len(table) or t not in with_units)


def write_table(rows: list[tuple[int, int]], path: str, origin: str = "acs") -> None:
    with open(path, "w") as fh:
        for cycles, overlapping in rows:
            fh.write(f"{cycles}\t{overlapping}\n")
        fh.write(f"# origins: {origin}\n")


def write_units(rows: list[tuple], path: str) -> None:
    """Five tab-separated fields per line and nothing else: TOGSim rejects a header or a comment line."""
    with open(path, "w") as fh:
        for row in rows:
            fh.write("\t".join(str(field) for field in row) + "\n")
