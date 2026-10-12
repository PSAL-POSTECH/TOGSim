"""The compute nodes of each kernel in kernels/, by tile_id, the operations a vector node runs per element, and the
axes a kernel walks a tile at a time: a last tile short on some of them is a variant, with tile ids of its own."""
from __future__ import annotations

from dataclasses import dataclass

ELEMENTWISE = "elementwise"
REDUCTION = "reduction"

#: max/min: compare, then select.
MAXMIN = 2


@dataclass(frozen=True)


class Op:
    kind: str
    n: int          # instructions per element (elementwise) or per fold step
    comment: str


@dataclass(frozen=True)


class NodeDesc:
    name: str
    compute_type: str       # "vector" | "systolic"
    ops: tuple = ()
    axes: tuple | None = None   # the tile macros of the axes its tile spans; None: every axis of the kernel


@dataclass(frozen=True)
class Axis:
    """An axis a kernel walks a tile at a time: the shape argument `dim`, `tile` (a -D macro) elements a step."""
    dim: str
    tile: str


@dataclass(frozen=True)


class KernelDesc:
    name: str
    nodes: tuple
    axes: tuple = ()        # in the bit order of the kernel's variant: bit i is set when the tile is short on axes[i]


def _vec(name, *ops, axes=None):
    return NodeDesc(name, "vector", tuple(ops), axes)


def _axes(*pairs):
    return tuple(Axis(dim, tile) for dim, tile in pairs)

#: Kernel name -> its compute nodes, by tile_id, and its axes. A "systolic" node is a whole block of the kernel's
#: K loop: every preload and matmul the array runs for it is one compute. A node outside an inner loop spans only
#: the axes of the loops around it (gemm's acc_init: M and N, not K).
KERNELS: dict[str, KernelDesc] = {

    "gemm": KernelDesc("gemm", (
        _vec("acc_init", Op(ELEMENTWISE, 1, "zero the accumulator"), axes=("TM", "TN")),
        NodeDesc("matmul", "systolic"),
    ), _axes(("M", "TM"), ("N", "TN"), ("K", "TK"))),

    "conv": KernelDesc("conv", (
        _vec("acc_init", Op(ELEMENTWISE, 1, "zero the accumulator"), axes=("TILE_N", "TILE_I_H", "TILE_M")),
        NodeDesc("matmul", "systolic"),
    ), _axes(("O_C", "TILE_N"), ("O_H", "TILE_I_H"), ("O_W", "TILE_M"), ("I_C", "TILE_K"))),

    "softmax": KernelDesc("softmax", (
        _vec("max_reduce", Op(ELEMENTWISE, MAXMIN, "running max")),
        _vec("max_write", Op(REDUCTION, MAXMIN, "fold the max"), axes=("TM",)),
        _vec("sum_reduce",
             Op(ELEMENTWISE, 1, "x - max"),
             Op(ELEMENTWISE, 1, "exp"),
             Op(ELEMENTWISE, 1, "accumulate")),
        _vec("sum_write", Op(REDUCTION, 1, "fold the sum"), axes=("TM",)),
        _vec("softmax",
             Op(ELEMENTWISE, 1, "x - max"),
             Op(ELEMENTWISE, 1, "exp"),
             Op(ELEMENTWISE, 1, "/ sum")),
    ), _axes(("M", "TM"), ("N", "TN"))),

    "layernorm": KernelDesc("layernorm", (
        _vec("stats_reduce",
             Op(ELEMENTWISE, 1, "accumulate x"),
             Op(ELEMENTWISE, 2, "accumulate x*x"), axes=("TM_STATS", "TN")),
        _vec("write_mean", Op(REDUCTION, 1, "fold the sum"), axes=("TM_STATS",)),
        _vec("write_var", Op(REDUCTION, 1, "fold the sum of squares"), axes=("TM_STATS",)),
        _vec("normalize",
             Op(ELEMENTWISE, 1, "x - mean"),
             Op(ELEMENTWISE, 1, "* rstd"),
             Op(ELEMENTWISE, 2, "scale and shift"), axes=("TM", "TN")),
    ), _axes(("M", "TM"), ("N", "TN"), ("M", "TM_STATS"))),

    "embed_layernorm": KernelDesc("embed_layernorm", (
        _vec("gather",
             Op(ELEMENTWISE, 1, "gather the embedding"),
             Op(ELEMENTWISE, 1, "x - mean"),
             Op(ELEMENTWISE, 1, "* rstd"),
             Op(ELEMENTWISE, 2, "scale and shift")),
    ), _axes(("SEQ", "TSEQ"), ("DIM", "TDIM"))),

    "global_avgpool": KernelDesc("global_avgpool", (
        _vec("sum_reduce", Op(ELEMENTWISE, 1, "accumulate"), axes=("TC", "THW")),
        _vec("sum_write", Op(REDUCTION, 1, "fold the sum"), axes=("TC",)),
        _vec("scale", Op(ELEMENTWISE, 1, "* 1/HW"), axes=("TC_SCALE",)),
    ), _axes(("C", "TC"), ("HW", "THW"), ("C", "TC_SCALE"))),

    "bias_act": KernelDesc("bias_act", (
        _vec("bias_act",
             Op(ELEMENTWISE, 1, "+ bias"),
             Op(ELEMENTWISE, 1, "max(x, 0)")),
    ), _axes(("M", "TM"), ("N", "TN"))),

    "bias_gelu": KernelDesc("bias_gelu", (
        _vec("bias_gelu",
             Op(ELEMENTWISE, 1, "+ bias"),
             Op(ELEMENTWISE, 1, "x / sqrt(2)"),
             Op(ELEMENTWISE, 2, "erf"),
             Op(ELEMENTWISE, 1, "1 + erf"),
             Op(ELEMENTWISE, 1, "* 0.5x")),
    ), _axes(("M", "TM"), ("N", "TN"))),

    "maxpool": KernelDesc("maxpool", (
        _vec("maxpool", Op(ELEMENTWISE, MAXMIN, "running max over the window")),
    ), _axes(("ROWS", "TROW"), ("COLS", "TCOL"))),

    "adaptive_avgpool": KernelDesc("adaptive_avgpool", (
        _vec("avgpool",
             Op(ELEMENTWISE, 1, "accumulate the window"),
             Op(ELEMENTWISE, 1, "* 1/n")),
    ), _axes(("ROWS", "TROW"), ("COLS", "TCOL"))),

    # pure DMA: no compute node
    "flatten": KernelDesc("flatten", ()),

    "attention": KernelDesc("attention", (
        NodeDesc("qk", "systolic"),
        _vec("rowmax", Op(REDUCTION, MAXMIN, "running row max")),
        _vec("sub", Op(ELEMENTWISE, 1, "s - max")),
        _vec("exp", Op(ELEMENTWISE, 1, "exp")),
        _vec("rowsum", Op(REDUCTION, 1, "fold the row sum")),
        _vec("mac", Op(ELEMENTWISE, 2, "running l")),
        _vec("rescale", Op(ELEMENTWISE, 2, "rescale the accumulator")),
        NodeDesc("sv", "systolic"),
        _vec("normalize",
             Op(ELEMENTWISE, 1, "/ l"),
             Op(ELEMENTWISE, 1, "scale"), axes=("Q_LEN",)),
    ), _axes(("SEQ", "Q_LEN"), ("SEQ", "KV_BLOCK"))),

    "concat": KernelDesc("concat", ()),

    "kvcache_concat": KernelDesc("kvcache_concat", (
        _vec("split", Op(ELEMENTWISE, 1, "split QKV into query, key, value")),
    ), _axes(("TOKENS", "TTOK"))),
}


def lookup(kernel_name: str) -> KernelDesc:
    try:
        return KERNELS[kernel_name]
    except KeyError:
        raise KeyError(
            f"no compute-node table for kernel {kernel_name!r}. Add one to "
            f"acs/node_mapping.py: without it the cost of every node in this "
            f"kernel would be invented.") from None


def variants_of(desc: KernelDesc, node: NodeDesc, values: dict[str, int]) -> list[int]:
    """The variants the tiles of `node` take at the dimensions and tiles in `values`: on each axis it spans, whole
    when the dimension holds a whole tile, short when the tile does not divide it, and every combination of those."""
    variants = [0]
    for bit, axis in enumerate(desc.axes):
        if node.axes is not None and axis.tile not in node.axes:
            continue
        dim, tile = values[axis.dim], values[axis.tile]
        states = ([0] if dim >= tile else []) + ([1 << bit] if dim % tile else [])
        variants = [v | state for v in variants for state in states]
    return variants


def tile_ids(desc: KernelDesc, values: dict[str, int]) -> list[int]:
    """The tile ids the kernel's computes name at `values`, as kernel_tile_id of kernels/kernel_shape.h numbers
    them: node + nodes * variant."""
    return sorted(index + len(desc.nodes) * variant
                  for index, node in enumerate(desc.nodes) for variant in variants_of(desc, node, values))


def at_variant(desc: KernelDesc, values: dict[str, int], variant: int) -> dict[str, int]:
    """`values` as a tile of `variant` has them: the tile of each axis the variant marks short is what is left of
    its dimension."""
    out = dict(values)
    for bit, axis in enumerate(desc.axes):
        if variant >> bit & 1:
            out[axis.tile] = values[axis.dim] % values[axis.tile]
    return out
