"""The kernel library, kernels/*_tog.cpp, on tpuv6e: each kernel's instruction counts and total cycles at its
example shape, that a kernel reads its shape arguments, and the shape arguments it refuses. For conv, also the
elements its DMAs move at a stride, a dilation and a padding, against what a convolution of that shape reads."""
import itertools
import re
from collections import Counter
from pathlib import Path

from togtest import Case

KERNELS = Path(__file__).resolve().parents[2] / "kernels"
EXAMPLE = {
    "adaptive_avgpool": dict(tensors=(262144, 65536), cycles=[128], insts=(16, 4, 4, 0), total=2377),
    "attention": dict(tensors=(262144, 262144, 262144, 262144), cycles=[526, 21, 8, 8, 14, 16, 16, 526, 16],
                      rows={0: 512, 7: 512}, insts=(768, 128, 2176, 768), total=172025),
    "bias_act": dict(tensors=(131072, 1024, 131072), cycles=[256], insts=(8, 4, 4, 0), total=1498),
    "bias_gelu": dict(tensors=(131072, 1024, 131072), cycles=[768], insts=(8, 4, 4, 0), total=3481),
    "concat": dict(tensors=(131072, 131072, 262144), cycles=[], insts=(32, 32, 0, 0), total=1865),
    "conv": dict(tensors=(82944, 147456, 65536), cycles=[16, 8214], rows={1: 8192}, insts=(432, 2, 20, 36),
                 total=74771),
    "embed_layernorm": dict(tensors=(1024, 1024, 131072), cycles=[640], insts=(8, 4, 4, 0), total=2772),
    "flatten": dict(tensors=(), cycles=[], insts=(0, 0, 0, 0), total=1),
    "gemm": dict(tensors=(131072, 262144, 131072), cycles=[128, 8230], rows={1: 8192}, insts=(32, 4, 20, 32),
                 total=67320),
    "global_avgpool": dict(tensors=(65536, 1024, 1024), cycles=[256, 14, 256], insts=(3, 3, 4, 0), total=1101),
    "kvcache_concat": dict(tensors=(393216, 131072, 262144, 262144), cycles=[128], insts=(4, 12, 4, 0), total=4137),
    "layernorm": dict(tensors=(131072, 131072, 512, 512, 1024, 1024, 131072), cycles=[192, 14, 14, 256],
                      insts=(56, 10, 14, 0), total=3343),
    "maxpool": dict(tensors=(262144, 65536), cycles=[128], insts=(16, 4, 4, 0), total=2377),
    "softmax": dict(tensors=(131072, 512, 512, 131072), cycles=[256, 21, 384, 14, 384], insts=(24, 6, 14, 0),
                    total=4505),
}
GEMM = EXAMPLE["gemm"]
CONV_NAMES = "I_C I_H I_W O_C O_H O_W K_H K_W S_H S_W D_H D_W P_T P_L P_B P_R"


def replay(t, kernel, tensors, cycles, rows=None, shape=None, defines=(), log_level="info", **_):
    """`kernel` on tpuv6e under the default systolic_array_split, `tensors` the bytes of its tensor arguments: tile i
    runs cycles[i], holds rows[i] Systolic rows (0 without an entry) and, a tile without any, the vector unit."""
    rows = rows or {}
    units = [row for i, c in enumerate(cycles or [0])
             for row in ((i, "Systolic", "rows", 1, rows.get(i, 0)), (i, "Vpu", "cycles", 1, 0 if i in rows else c))]
    tog = t.tog(KERNELS / f"{kernel}_tog.cpp", defines, tensors=tensors, shape=shape)
    return t.simulate(tog, t.machine(), cycles or [0], units, log_level=log_level)


def insts(run):
    """(MOVIN, MOVOUT, COMP, MEMORY_BAR) of the `inst_count:` lines, summed over the cores."""
    return tuple(sum(int(n) for n in re.findall(rf"{name} +inst_count: (\d+)", run.log))
                 for name in ("MOVIN", "MOVOUT", "COMP", "MEMORY_BAR"))


def expect_insts(run, expected):
    """The run finished with those (MOVIN, MOVOUT, COMP, MEMORY_BAR) counts and without a warning: its tables
    and tog_tensors.txt cover every tile id and arg_id the kernel names."""
    run.expect_no_warning("")
    if insts(run) != expected:
        run.fail(f"expected (MOVIN, MOVOUT, COMP, MEMORY_BAR) inst_count {expected}, got {insts(run)}")
    return run


def example(kernel):
    """The kernel at its example shape (no tog_shape.txt) issues `insts`, its (MOVIN, MOVOUT, COMP, MEMORY_BAR)
    counts (the DMAs the original issued; one compute per tile for gemm, conv and attention), and replays to
    `total`, a golden. A typed TOG fails any case here."""
    how = EXAMPLE[kernel]
    return lambda t: expect_insts(replay(t, kernel, **how), how["insts"]).expect_total_cycles(how["total"])


def gemm_reads_its_shape_arguments(t):
    """M K N = 100 200 300 in 32 x 256 tiles of 64-deep K blocks is 4 x 2 work-items of 4 blocks: two loads and two
    barriers per block, a compute per block and one per work-item, a store per work-item."""
    work_items, blocks = 4 * 2, 4
    expected = (work_items * blocks * 2, work_items, work_items * (blocks + 1), work_items * blocks * 2)
    expect_insts(replay(t, "gemm", (80000, 240000, 120000), GEMM["cycles"], GEMM["rows"], shape=(100, 200, 300)),
                 expected)


def conv_reads_its_shape_and_its_tile(t):
    """A 3 -> 64 channel 3 x 3 convolution of a 32 x 32 image with -DTILE_K=8: 8 work-items (4 x 2 output tiles of
    8 x 16), each 9 kernel positions of 1 weight and 16 input loads, 2 barriers and a compute."""
    work_items, positions = 8, 9
    expected = (work_items * positions * 17, work_items, work_items * (positions + 1), work_items * positions * 2)
    tensors = (34 * 34 * 3 * 4, 9 * 64 * 3 * 4, 64 * 32 * 32 * 4)
    expect_insts(replay(t, "conv", tensors, [16, 8214], {1: 8192}, shape=(3, 32, 32, 64, 32, 32, 3, 3) + (1,) * 8,
                        defines=("TILE_K=8",)), expected)


CONV_TILE = dict(rows=8, cols=16, channels=64, sa=8)
TENSOR_ALIGN = 16384


def conv_tensors(shape):
    """The bytes of conv's tensor arguments at `shape`: the padded input, the weights, the output."""
    i_c, i_h, i_w, o_c, o_h, o_w, k_h, k_w, _, _, _, _, p_t, p_l, p_b, p_r = shape
    return ((i_h + p_t + p_b) * (i_w + p_l + p_r) * i_c * 4, k_h * k_w * o_c * i_c * 4, o_c * o_h * o_w * 4)


def moved(run, tensors):
    """Per tensor argument, {element offset: times a DMA of the run moved it}, from the address, size and stride of
    every MOVIN and MOVOUT issued (trace level). TOGSim lays the tensors back to back, each start 16 KiB aligned."""
    bases, at = [], 0
    for size in tensors:
        bases.append(at)
        at = -(-(at + size) // TENSOR_ALIGN) * TENSOR_ALIGN
    seen = [Counter() for _ in tensors]
    for e in run.events():
        if e.tag != "INST_ISSUED" or e.opcode not in ("MOVIN", "MOVOUT"):
            continue
        dram = int(e.fields["dram"], 16)
        arg = max(i for i, base in enumerate(bases) if base <= dram)
        extent, stride = ([int(n) for n in e.fields[key].strip("[]").split(",")] for key in ("size", "stride"))
        first = (dram - bases[arg]) // 4
        seen[arg].update(first + sum(i * step for i, step in zip(index, stride))
                         for index in itertools.product(*map(range, extent)))
    return seen


def convolved(shape, tiles, depth):
    """What a convolution of `shape` moves, as `moved` counts it: an input element once per output and kernel
    position that reads it, a weight once per output tile, an output element once. `depth` is the channels a
    load spans from one pixel or filter: I_C, or the tile's when I_C is under it."""
    i_c, i_h, i_w, o_c, o_h, o_w, k_h, k_w, s_h, s_w, d_h, d_w, p_t, p_l, p_b, p_r = shape
    x_w = i_w + p_l + p_r
    x = Counter(((oh * s_h + kh * d_h) * x_w + ow * s_w + kw * d_w) * i_c + c
                for oh in range(o_h) for ow in range(o_w) for kh in range(k_h) for kw in range(k_w)
                for c in range(depth))
    w = Counter(filt * i_c + c for filt in range(k_h * k_w * o_c) for c in range(depth) for _ in range(tiles))
    return [x, w, Counter(range(o_c * o_h * o_w))]


def conv_moves(shape, total, tile_k=CONV_TILE["channels"], past=(0, 0, 0)):
    """conv at `shape` (O_C a multiple of 64, O_H of 8 and O_W of 16: whole tiles) in TILE_K=`tile_k` blocks issues
    the DMAs of its tiles, moves exactly the elements a convolution of that shape reads and writes, ends `past`
    elements beyond each tensor, and replays to `total`, a golden."""
    def case(t):
        i_c, _, _, o_c, o_h, o_w, k_h, k_w = shape[:8]
        tiles = (o_c // 64) * (o_h // CONV_TILE["rows"]) * (o_w // CONV_TILE["cols"])
        blocks = k_h * k_w * -(-i_c // tile_k)
        loads = tile_k // CONV_TILE["sa"] + CONV_TILE["rows"] * CONV_TILE["cols"] // CONV_TILE["sa"]
        tensors = conv_tensors(shape)
        run = replay(t, "conv", tensors, [16, 8214], {1: 8192}, shape=shape, defines=(f"TILE_K={tile_k}",),
                     log_level="trace")
        expect_insts(run, (tiles * blocks * loads, tiles, tiles * (blocks + 1), tiles * blocks * 2))
        seen, expected = moved(run, tensors), convolved(shape, tiles, max(i_c, tile_k))
        for name, got, want, size in zip(("the input", "the weights", "the output"), seen, expected, tensors):
            if got != want:
                off = sorted((got - want) + (want - got))
                run.fail(f"{name}: the DMAs moved {sum(got.values())} elements, a convolution of this shape moves "
                         f"{sum(want.values())}; {len(off)} element offsets differ, the first {off[:4]}")
        beyond = tuple(max(0, max(got) + 1 - size // 4) for got, size in zip(seen, tensors))
        if beyond != past:
            run.fail(f"expected the DMAs to end {past} elements past (input, weights, output), got {beyond}")
        run.expect_total_cycles(total)
    return case


def attention_shares_a_kv_head(t):
    """HEADS KV_HEADS SEQ = 16 8 128: the 128 work-items stay, and each of their two key blocks runs the QK and the
    SV compute twice, once per query head of the KV head (2176 + 128 * 2 * 2 computes)."""
    how = EXAMPLE["attention"]
    expect_insts(replay(t, "attention", (524288, 262144, 262144, 524288), how["cycles"], how["rows"],
                        shape=(16, 8, 128)), (768, 128, 2688, 768))


def refused(shape, saying, kernel="gemm"):
    """`kernel` with `shape` in tog_shape.txt ends the run with exit 1 and `saying` on stderr, before any
    work-item."""
    def case(t):
        how = EXAMPLE[kernel]
        run = replay(t, kernel, how["tensors"], how["cycles"], how["rows"], shape=shape)
        if run.rc != 1 or saying not in run.log:
            run.fail(f'expected exit 1 saying "{saying}", got {run._exit()}: {run._seen()}')
    return case


CASES = [Case(f"example.{kernel}", example(kernel)) for kernel in EXAMPLE] + [
    Case("gemm_reads_its_shape_arguments", gemm_reads_its_shape_arguments),
    Case("conv_reads_its_shape_and_its_tile", conv_reads_its_shape_and_its_tile),
    Case("attention_shares_a_kv_head", attention_shares_a_kv_head),
    Case("refused.shape_argument_count", refused(
        (100, 200), "kernel gemm: it takes 3 shape arguments (M K N), tog_shape.txt gives 2")),
    Case("refused.shape_argument_below_one", refused(
        (128, 0, 256), "kernel gemm: shape argument 1 of (M K N) is 0, it must be at least 1")),
    Case("conv.stride_2", conv_moves((64, 32, 64, 64, 16, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1), 148838)),
    Case("conv.pad_0", conv_moves((64, 18, 18, 64, 16, 16, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0), 74771)),
    Case("conv.1x1_stride_2", conv_moves((64, 32, 32, 64, 16, 16, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0), 8817)),
    Case("conv.stride_2x1_pads_0_1_1_2", conv_moves((64, 32, 15, 64, 16, 16, 3, 3, 2, 1, 1, 1, 0, 1, 1, 2), 74731)),
    Case("conv.dilation_2x3_pads_2_3_2_3", conv_moves((64, 16, 16, 64, 16, 16, 3, 3, 1, 1, 2, 3, 2, 3, 2, 3), 74764)),
    Case("conv.7x7_stride_2_pad_3_of_3_channels", conv_moves(
        (3, 32, 32, 64, 16, 16, 7, 7, 2, 2, 1, 1, 3, 3, 3, 3), 404705, tile_k=8, past=(0, 5, 0))),
    Case("refused.conv_row_past_the_padded_input", refused(
        (64, 16, 16, 64, 16, 16, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1),
        "kernel conv: output row 15 reads input row 32, and the padded input has 18 rows", "conv")),
    Case("refused.conv_column_past_the_padded_input", refused(
        (64, 16, 16, 64, 16, 16, 3, 3, 1, 1, 1, 2, 1, 1, 1, 2),
        "kernel conv: output column 15 reads input column 19, and the padded input has 19 columns", "conv")),
    Case("refused.conv_stride_below_one", refused(
        (64, 16, 16, 64, 16, 16, 3, 3, 1, 0, 1, 1, 1, 1, 1, 1),
        f"kernel conv: shape argument 9 of ({CONV_NAMES}) is 0, it must be at least 1", "conv")),
    Case("refused.conv_pad_below_zero", refused(
        (64, 16, 16, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, -1, 1, 1),
        f"kernel conv: shape argument 13 of ({CONV_NAMES}) is -1, it must be at least 0", "conv")),
]
