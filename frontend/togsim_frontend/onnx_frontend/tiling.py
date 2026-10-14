"""The tiles the frontend cuts a node into, and the machines those tiles fit. The systolic tiles are the same on
every machine: a machine they do not fit is refused (unsupported), not tiled differently."""
from __future__ import annotations

from dataclasses import dataclass

#: The tile of a Gemm, MatMul or Conv, at most: rows x columns x depth, and the output rows of a conv tile.
TILE_ROWS, TILE_COLS, TILE_DEPTH = 32, 256, 64
CONV_TILE_ROWS = 8
#: The tile of an Attention: queries x keys.
ATTENTION_QUERIES, ATTENTION_KEYS = 8, 64
#: The rows of the tile of a second pass: layernorm's statistics, global_avgpool's scale. What the kernels default to.
SECOND_PASS_ROWS = 128
#: The kernels a work-item of which holds every spad buffer it states at once, at its matmul: the accumulator, the
#: weights and the input.
HOLDS_EVERY_BUFFER = frozenset({"gemm", "conv"})


@dataclass(frozen=True)


class Hardware:
    sa: int                 # systolic array dimension: vpu_num_lanes unless --sa
    vlane: int              # vpu_num_lanes
    vlen_bits: int          # zvl<N>b of the config's isa
    array_unit: str | None
    elem_bits: int = 32
    spad_bytes: int = 0     # a core's spad in bytes, as TOGSim reads the config (config.core_spad_kb); 0: no limit

    @property
    def lane_elems(self) -> int:
        return self.vlen_bits // self.elem_bits

    @property
    def throughput(self) -> int:
        return self.vlane * self.lane_elems


def _round_down(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def systolic_tile(m: int, n: int, k: int, hw: Hardware) -> tuple[int, int, int]:
    col_unit = hw.sa * hw.vlane
    tm = _round_down(min(m, TILE_ROWS), hw.sa)
    tn = _round_down(min(n, TILE_COLS), col_unit)
    tk = _round_down(min(k, TILE_DEPTH), hw.sa)
    return tm, tn, tk


def unsupported(hw: Hardware) -> list[str]:
    """Why the systolic tiles do not fit the machine `hw`, a line per fact; empty when they do. A tile is whole
    matmul instructions, SA rows by SA * lanes columns over SA-deep weights. Whether its buffers fit the spad depends
    on the node, which cuts a tile no larger than itself: spad_refusal."""
    col_unit = hw.sa * hw.vlane
    why = []
    for what, rows, cols, depth in (("Gemm, MatMul or Conv", TILE_ROWS, TILE_COLS, TILE_DEPTH),
                                    ("Attention", ATTENTION_QUERIES, ATTENTION_KEYS, None)):
        if rows % hw.sa:
            why.append(f"the {rows} rows of a {what} tile are not a multiple of the array's {hw.sa} rows (SA)")
        if depth is not None and depth % hw.sa:
            why.append(f"the {depth}-deep block of a {what} tile is not a multiple of the array's {hw.sa} (SA)")
        if cols % col_unit:
            why.append(f"the {cols} columns of a {what} tile are not a multiple of one matmul instruction's "
                       f"{col_unit} (SA * vpu_num_lanes = {hw.sa} * {hw.vlane})")
    return why


def spad_refusal(stem: str, stated: list[int], hw: Hardware) -> str | None:
    """Why TOGSim refuses kernel `stem` compiled to state the spad buffers `stated` on `hw`; None when it runs it. A
    core admits a work-item while the spad holds it and what is in flight beside it, and runs it alone when it takes
    more than half, so it refuses only a work-item that needs more of the spad at once than a core has. Of the kernels
    in HOLDS_EVERY_BUFFER that is every buffer stated; of the others the frontend does not say."""
    need = sum(stated)
    if stem not in HOLDS_EVERY_BUFFER or not hw.spad_bytes or need <= hw.spad_bytes:
        return None
    return (f"a work-item of kernel {stem} holds its {len(stated)} spad buffers at once, {need} bytes, and a core's "
            f"spad is {hw.spad_bytes}: TOGSim refuses the TOG")


def check_systolic(n: int, hw: Hardware) -> None:
    col_unit = hw.sa * hw.vlane
    if n < col_unit:
        raise ValueError(
            f"{n} columns is under one matmul instruction (SA*VLANE = {col_unit}); "
            f"the array cannot be driven at this shape")


def reducing_tile(rows: int, cols: int, hw: Hardware) -> tuple[int, int]:
    tn = min(cols, max(hw.throughput, hw.lane_elems * hw.vlane))
    tm = max(1, min(rows, max(1, (hw.throughput * 8) // max(1, tn))))
    return tm, tn


def pointwise_tile(rows: int, cols: int, hw: Hardware) -> tuple[int, int]:
    budget = hw.throughput * 32
    tn = min(cols, budget)
    tm = max(1, min(rows, max(1, budget // max(1, tn))))
    return tm, tn
