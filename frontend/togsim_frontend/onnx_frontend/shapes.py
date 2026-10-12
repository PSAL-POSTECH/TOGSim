"""From an ONNX node and the inferred shapes to what its kernel is specialised by: the shape arguments, the tile
and hardware macros, the byte size of each tensor argument and the tile ACS costs."""
from __future__ import annotations

from .onnx_ops import KernelRef, POINTWISE, REDUCING, SYSTOLIC
from .tiling import (ATTENTION_KEYS, ATTENTION_QUERIES, CONV_TILE_ROWS, Hardware, check_systolic, pointwise_tile,
                     reducing_tile, systolic_tile)


def rows_cols(shape: list[int]) -> tuple[int, int]:
    if not shape:
        return 1, 1
    if len(shape) == 1:
        return 1, shape[0]
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return rows, shape[-1]


def attr(node, name, default):
    for a in node.attribute:
        if a.name == name:
            return list(a.ints) if a.ints else a.i
    return default


def _matmul_dims(node, shapes) -> tuple[int, int, int]:
    a = shapes[node.input[0]]
    b = shapes[node.input[1]]
    m, k = rows_cols(a)
    if len(b) == 1:
        n = b[0]
    else:
        n = b[-1]
        if attr(node, "transB", 0):
            n = b[-2]
    return m, k, n


def _text_attr(node, name, default: str) -> str:
    for a in node.attribute:
        if a.name == name:
            return a.s.decode()
    return default


def _conv_pads(node, in_hw, kernel_hw, strides, dilations) -> list[int]:
    """The four paddings of a Conv as ONNX orders them, top left bottom right: its `pads`, or what `auto_pad`
    stands for (VALID none; SAME_* an output of ceil(input / stride), the odd cell after or before)."""
    auto_pad = _text_attr(node, "auto_pad", "NOTSET")
    if auto_pad == "NOTSET":
        return list(attr(node, "pads", [0, 0, 0, 0]))
    if auto_pad == "VALID":
        return [0, 0, 0, 0]
    if auto_pad not in ("SAME_UPPER", "SAME_LOWER"):
        raise ValueError(f"auto_pad {auto_pad!r} is not one ONNX defines")
    begin, end = [], []
    for size, k, s, d in zip(in_hw, kernel_hw, strides, dilations):
        total = max(0, (-(-size // s) - 1) * s + (k - 1) * d + 1 - size)
        small, large = total // 2, total - total // 2
        begin.append(small if auto_pad == "SAME_UPPER" else large)
        end.append(large if auto_pad == "SAME_UPPER" else small)
    return begin + end


def _conv_dims(node, shapes) -> dict[str, int]:
    """The shape arguments of the conv kernel for a Conv node. The output extent is what the attributes give over
    the padded input, and has to be what ONNX shape inference recorded for the node's output, when it did."""
    x = shapes[node.input[0]]
    w = shapes[node.input[1]]
    if len(x) != 4 or len(w) != 4:
        raise ValueError(f"a convolution over {len(w) - 2} spatial axes: the conv kernel takes two")
    if attr(node, "group", 1) != 1:
        raise ValueError("grouped convolution: no library kernel takes a group count")
    o_c, _, k_h, k_w = w
    i_c, i_h, i_w = x[1], x[2], x[3]
    s_h, s_w = attr(node, "strides", [1, 1])
    d_h, d_w = attr(node, "dilations", [1, 1])
    p_t, p_l, p_b, p_r = _conv_pads(node, (i_h, i_w), (k_h, k_w), (s_h, s_w), (d_h, d_w))
    o_h = (i_h + p_t + p_b - (k_h - 1) * d_h - 1) // s_h + 1
    o_w = (i_w + p_l + p_r - (k_w - 1) * d_w - 1) // s_w + 1
    if o_h < 1 or o_w < 1:
        raise ValueError(f"a {k_h} x {k_w} kernel (dilations {d_h}, {d_w}) does not fit the padded input, "
                         f"{i_h + p_t + p_b} x {i_w + p_l + p_r}")
    inferred = shapes.get(node.output[0]) if node.output else None
    if inferred is not None and list(inferred[1:]) != [o_c, o_h, o_w]:
        raise ValueError(f"ONNX shape inference gives the output {' x '.join(map(str, inferred[1:]))}, and the "
                         f"strides, dilations and pads give {o_c} x {o_h} x {o_w}")
    return dict(I_C=i_c, I_H=i_h, I_W=i_w, O_C=o_c, O_H=o_h, O_W=o_w, K_H=k_h, K_W=k_w,
                S_H=s_h, S_W=s_w, D_H=d_h, D_W=d_w, P_T=p_t, P_L=p_l, P_B=p_b, P_R=p_r)


def _attention_heads(node, q_shape) -> int:
    n = attr(node, "num_heads", 0)
    return int(n) if n else 1


def _pool_dims(node, shapes) -> tuple[int, int]:
    x = shapes[node.input[0]]
    kernel = attr(node, "kernel_shape", [1, 1])
    strides = attr(node, "strides", [1, 1])
    s = strides[0] if isinstance(strides, list) else strides
    if len(x) == 4:
        return x[1] * max(1, x[2] // max(1, s)), max(1, x[3] // max(1, s))
    return rows_cols(x)


def constants(node, kernel: KernelRef, shapes: dict, hw: Hardware) -> dict[str, int]:
    """Every name of `kernel` (size, tile and hardware) with its value for `node`. conv walks the kernel
    positions itself, so its tile is O_W columns of O_C by I_C channels, TILE_I_H output rows at a time."""
    out: dict[str, int] = {}

    if kernel.stem == "conv":
        dims = _conv_dims(node, shapes)
        out.update(dims)
        check_systolic(dims["O_C"], hw)
        tm, tn, tk = systolic_tile(dims["O_W"], dims["O_C"], dims["I_C"], hw)
        out.update(TILE_M=tm, TILE_N=tn, TILE_K=tk, TILE_I_H=min(CONV_TILE_ROWS, dims["O_H"]))

    elif kernel.stem == "attention":
        q = shapes[node.input[0]]
        seq = q[1] if len(q) >= 3 else q[0]
        heads = _attention_heads(node, q)
        kv_heads = int(attr(node, "num_kv_heads", 0)) or heads
        dhead = max(1, q[-1] // max(1, heads))
        out.update(HEADS=heads, KV_HEADS=kv_heads, SEQ=seq, DHEAD=dhead, Q_LEN=ATTENTION_QUERIES,
                   KV_BLOCK=ATTENTION_KEYS)

    elif kernel.tile_class == SYSTOLIC:
        m, k, n = _matmul_dims(node, shapes)
        check_systolic(n, hw)
        tm, tn, tk = systolic_tile(m, n, k, hw)
        out.update(M=m, K=k, N=n, TM=tm, TN=tn, TK=tk)

    elif kernel.stem == "global_avgpool":
        x = shapes[node.input[0]]
        c = x[1] if len(x) >= 2 else x[0]
        hw_elems = (x[2] * x[3]) if len(x) == 4 else 1
        tc, thw = reducing_tile(c, hw_elems, hw)
        out.update(C=c, HW=hw_elems, TC=tc, THW=thw)

    elif kernel.stem == "concat":
        a_rows, d = rows_cols(shapes[node.input[0]])
        b_rows, _ = rows_cols(shapes[node.input[1]])
        trow, td = pointwise_tile(max(a_rows, b_rows), d, hw)
        out.update(A_ROWS=a_rows, B_ROWS=b_rows, D=d, TROW=trow, TD=td)

    elif not kernel.size_names:
        pass

    else:
        if kernel.stem in ("maxpool", "adaptive_avgpool"):
            rows, cols = _pool_dims(node, shapes)
        else:
            rows, cols = rows_cols(shapes[node.input[0]])
        if kernel.tile_class == REDUCING:
            tm, tn = reducing_tile(rows, cols, hw)
        else:
            tm, tn = pointwise_tile(rows, cols, hw)
        # each kernel names these its own way
        size = kernel.size_names
        tiles = kernel.tile_names
        out[size[0]] = rows
        if len(size) > 1:
            out[size[1]] = cols
        if tiles:
            out[tiles[0]] = tm
        if len(tiles) > 1:
            out[tiles[1]] = tn

    for name in kernel.hw_names:
        out[name] = hw.sa if name == "SA" else hw.vlane
    return out


def tile_for_cost(kernel: KernelRef, values: dict[str, int]) -> dict[str, int]:
    """The tile ACS costs: `elems` per vector tile and, for a systolic kernel, the rows x depth x cols block of
    one compute."""
    if kernel.stem == "conv":
        rows = values["TILE_I_H"] * values["TILE_M"]
        return {"rows": rows, "depth": values["TILE_K"], "cols": values["TILE_N"],
                "elems": rows * values["TILE_N"]}
    if kernel.stem == "attention":
        return {"rows": values["Q_LEN"], "depth": values["DHEAD"], "cols": values["KV_BLOCK"],
                "elems": values["Q_LEN"] * values["KV_BLOCK"]}
    if kernel.stem == "gemm":
        return {"rows": values["TM"], "depth": values["TK"], "cols": values["TN"],
                "elems": values["TM"] * values["TN"]}
    names = kernel.tile_names
    if not names:
        return {"rows": 1, "elems": 1}
    rows = values[names[0]]
    elems = rows * values[names[1]] if len(names) > 1 else rows
    return {"rows": rows, "elems": elems}


ELEM_BYTES = 4
INDEX_BYTES = 8


def tensor_bytes(kernel: KernelRef, v: dict[str, int]) -> list[int]:
    """The byte size of each tensor argument of `kernel`, in arg_id order: what tog_tensors.txt lists."""
    e = ELEM_BYTES
    if kernel.stem == "gemm":
        return [v["M"] * v["K"] * e, v["K"] * v["N"] * e, v["M"] * v["N"] * e]
    if kernel.stem == "conv":
        padded = (v["I_H"] + v["P_T"] + v["P_B"]) * (v["I_W"] + v["P_L"] + v["P_R"])
        return [padded * v["I_C"] * e, v["K_H"] * v["K_W"] * v["O_C"] * v["I_C"] * e,
                v["O_C"] * v["O_H"] * v["O_W"] * e]
    if kernel.stem == "attention":
        q = v["HEADS"] * v["SEQ"] * v["DHEAD"] * e
        kv = v["KV_HEADS"] * v["SEQ"] * v["DHEAD"] * e
        return [q, kv, kv, q]
    if kernel.stem == "softmax":
        return [v["M"] * v["N"] * e, v["M"] * e, v["M"] * e, v["M"] * v["N"] * e]
    if kernel.stem == "layernorm":
        x = v["M"] * v["N"] * e
        return [x, x, v["M"] * e, v["M"] * e, v["N"] * e, v["N"] * e, x]
    if kernel.stem == "embed_layernorm":
        return [v["SEQ"] * INDEX_BYTES, v["DIM"] * e, v["SEQ"] * v["DIM"] * e]
    if kernel.stem == "global_avgpool":
        return [v["C"] * v["HW"] * e, v["C"] * e, v["C"] * e]
    if kernel.stem in ("bias_act", "bias_gelu"):
        return [v["M"] * v["N"] * e, v["N"] * e, v["M"] * v["N"] * e]
    if kernel.stem in ("maxpool", "adaptive_avgpool"):
        return [v["ROWS"] * 4 * v["COLS"] * e, v["ROWS"] * v["COLS"] * e]
    if kernel.stem == "concat":
        return [v["A_ROWS"] * v["D"] * e, v["B_ROWS"] * v["D"] * e, (v["A_ROWS"] + v["B_ROWS"]) * v["D"] * e]
    if kernel.stem == "flatten":
        return []
    raise KeyError(f"no tensor sizes recorded for kernel {kernel.stem!r}; add them to shapes.tensor_bytes")
