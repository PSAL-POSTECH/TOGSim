"""Which kernel of kernels/ an ONNX operator runs as, and the names that kernel is specialised by."""
from __future__ import annotations

from dataclasses import dataclass

SYSTOLIC = "systolic"
REDUCING = "reducing"
POINTWISE = "pointwise"


@dataclass(frozen=True)
class KernelRef:
    """A kernel, kernels/<stem>_tog.cpp: `size_names` are its shape arguments in the order it reads them,
    `tile_names` and `hw_names` the -D macros it is compiled with."""
    stem: str
    tile_class: str
    size_names: tuple = ()
    tile_names: tuple = ()
    hw_names: tuple = ()

#: op_type -> its kernel.
OPS: dict[str, list[KernelRef]] = {
    "Gemm":   [KernelRef("gemm", SYSTOLIC, ("M", "K", "N"), ("TM", "TN", "TK"))],
    "MatMul": [KernelRef("gemm", SYSTOLIC, ("M", "K", "N"), ("TM", "TN", "TK"))],

    "Conv": [KernelRef("conv", SYSTOLIC,
                       ("I_C", "I_H", "I_W", "O_C", "O_H", "O_W", "K_H", "K_W",
                        "S_H", "S_W", "D_H", "D_W", "P_T", "P_L", "P_B", "P_R"),
                       ("TILE_M", "TILE_N", "TILE_K", "TILE_I_H"), ("SA",))],

    "Attention": [KernelRef("attention", SYSTOLIC, ("HEADS", "KV_HEADS", "SEQ"),
                            ("Q_LEN", "KV_BLOCK", "DHEAD"))],
    "MultiHeadAttention": [KernelRef("attention", SYSTOLIC, ("HEADS", "KV_HEADS", "SEQ"),
                                     ("Q_LEN", "KV_BLOCK", "DHEAD"))],

    "Softmax": [KernelRef("softmax", REDUCING, ("M", "N"), ("TM", "TN"))],

    "LayerNormalization": [KernelRef("layernorm", REDUCING, ("M", "N"), ("TM", "TN", "TM_STATS"))],

    "EmbedLayerNormalization": [KernelRef("embed_layernorm", REDUCING, ("SEQ", "DIM"), ("TSEQ", "TDIM"))],

    "GlobalAveragePool": [KernelRef("global_avgpool", REDUCING, ("C", "HW"), ("TC", "THW", "TC_SCALE"))],

    "Relu":  [KernelRef("bias_act", POINTWISE, ("M", "N"), ("TM", "TN"))],
    "Add":   [KernelRef("bias_act", POINTWISE, ("M", "N"), ("TM", "TN"))],
    "Gelu":  [KernelRef("bias_gelu", POINTWISE, ("M", "N"), ("TM", "TN"))],

    "MaxPool": [KernelRef("maxpool", POINTWISE, ("ROWS", "COLS"), ("TROW", "TCOL"))],
    "AveragePool": [KernelRef("adaptive_avgpool", POINTWISE, ("ROWS", "COLS"), ("TROW", "TCOL"))],

    "Concat":  [KernelRef("concat", POINTWISE, ("A_ROWS", "B_ROWS", "D"), ("TROW", "TD"))],
    "Flatten": [KernelRef("flatten", POINTWISE, (), ())],
}

#: Fused nodes with no kernel, and the operators they decompose to.
DECOMPOSE: dict[str, list[str]] = {
    "FusedMatMul":              ["MatMul"],
    "FusedConv":                ["Conv", "Relu"],
    "BiasGelu":                 ["Add", "Gelu"],
    "FastGelu":                 ["Add", "Gelu"],
    "SkipLayerNormalization":   ["Add", "LayerNormalization"],
}

#: Metadata-only operators: no compute node to charge.
NO_COMPUTE = frozenset({
    "Reshape", "Squeeze", "Unsqueeze", "Identity", "Constant", "Shape",
    "Dropout", "Cast", "Transpose", "Gather", "Slice", "ConstantOfShape",
})


def resolve(op_type: str, _depth: int = 0) -> list[KernelRef] | None:
    if op_type in NO_COMPUTE:
        return []
    if op_type in OPS:
        return list(OPS[op_type])
    if op_type in DECOMPOSE and _depth < 4:
        out: list[KernelRef] = []
        for part in DECOMPOSE[op_type]:
            got = resolve(part, _depth + 1)
            if got is None:
                return None
            out.extend(got)
        return out
    return None
