"""The cost of a kernel's compute nodes, as the cycle table and the unit table TOGSim reads: a compute's cycles (the
overlapping column is always 0) and the Systolic rows systolic_array_split spreads over a core's arrays. A short
last tile is a tile id of its own, costed by the same formulas at what is left. Estimates, not gem5 measurements."""
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
    """The matmuls under a preload feed the block's rows, one per cycle (SA each, the last what is left); a preload
    hides behind the matmul before it, and the last matmul drains through the array's 2*SA - 2 stages. The rows
    are what the split divides."""
    preloads, _ = systolic_instructions(tile, hw)
    fed = preloads * int(tile.get("rows", hw.sa))
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


def costs(kernel_desc, tiles: dict, hw: Hardware) -> dict[int, Cost]:
    """The Cost of each tile id. `tiles` is the kernel's whole tile, for the ids of its nodes, or {tile id: the
    tile of its variant} (node + nodes * variant). A kernel without a compute node (concat, flatten) gets one zero
    tile: the unit table has to name the array unit for systolic_array_split to accept it."""
    nodes = kernel_desc.nodes
    if not (tiles and all(isinstance(tile_id, int) for tile_id in tiles)):
        tiles = {tile_id: tiles for tile_id in range(len(nodes))}
    return {tile_id: Cost(*_cost_function(nodes[tile_id % len(nodes)], tile, hw))
            for tile_id, tile in sorted(tiles.items())} or {0: Cost(0)}


def build_table(kernel_desc, tiles: dict, hw: Hardware) -> list[tuple[int, int]]:
    """The cycle table rows, (cycles, overlapping), line i for tile id i: overlapping is always 0, a compute holds
    its core to its end. A tile id below the last that no tile of the shape has is a line of zeros."""
    by_id = costs(kernel_desc, tiles, hw)
    return [(by_id[tile_id].cycles if tile_id in by_id else 0, 0) for tile_id in range(max(by_id) + 1)]


def build_units(kernel_desc, tiles: dict, hw: Hardware) -> list[tuple]:
    """The unit table rows, (tile_id, unit, unit_of_work, capacity_per_cycle, admitted), two per tile id costed."""
    rows = []
    for tile_id, cost in costs(kernel_desc, tiles, hw).items():
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
