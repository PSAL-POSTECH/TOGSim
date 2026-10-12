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
ELEM_BYTES = 4
#: The spad buffers of the largest systolic tile, conv's: the accumulator, the weights and the input.
SYSTOLIC_TILE_SPAD_BYTES = (TILE_COLS * CONV_TILE_ROWS * TILE_ROWS + TILE_DEPTH * TILE_COLS
                            + CONV_TILE_ROWS * TILE_ROWS * TILE_DEPTH) * ELEM_BYTES


@dataclass(frozen=True)


class Hardware:
    sa: int                 # systolic array dimension (see note in config.py)
    vlane: int              # vpu_num_lanes
    vlen_bits: int          # vpu_vector_length_bits
    elem_bits: int = 32
    spad_bytes: int = 0     # core_spad_size_kb in bytes; 0: the config states none and TOGSim bounds nothing

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
    matmul instructions, SA rows by SA * lanes columns over SA-deep weights, and its buffers fit a stated spad."""
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
    if hw.spad_bytes and hw.spad_bytes < SYSTOLIC_TILE_SPAD_BYTES:
        why.append(f"core_spad_size_kb states a spad of {hw.spad_bytes} bytes and the buffers of a Conv tile take "
                   f"{SYSTOLIC_TILE_SPAD_BYTES}: TOGSim would stall the tile for ever")
    return why


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
