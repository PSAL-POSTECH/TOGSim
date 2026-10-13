"""The kernel library, kernels/*_tog.cpp, on tpuv6e: each kernel's instruction counts and total cycles at its
example shape, that a kernel reads its shape arguments, and the shape arguments it refuses. At a shape the tile
does not divide, and for conv at a stride, a dilation and a padding: the elements the DMAs move, the tile ids named."""
import itertools
import re
from collections import Counter
from pathlib import Path

from togtest import Case

KERNELS = Path(__file__).resolve().parents[2] / "kernels"
EXAMPLE = {
    "adaptive_avgpool": dict(tensors=(262144, 65536), cycles=[128], insts=(16, 4, 4, 0), total=1632),
    "attention": dict(tensors=(262144, 262144, 262144, 262144), cycles=[526, 21, 8, 8, 14, 16, 16, 526, 16],
                      rows={0: 512, 7: 512}, insts=(768, 128, 2176, 768), total=166867),
    "bias_act": dict(tensors=(131072, 1024, 131072), cycles=[256], insts=(8, 4, 4, 0), total=1462),
    "bias_gelu": dict(tensors=(131072, 1024, 131072), cycles=[768], insts=(8, 4, 4, 0), total=3475),
    "concat": dict(tensors=(131072, 131072, 262144), cycles=[], insts=(32, 32, 0, 0), total=2031),
    "conv": dict(tensors=(82944, 147456, 65536), cycles=[16, 8214], rows={1: 8192}, insts=(432, 2, 20, 36),
                 total=74614),
    "embed_layernorm": dict(tensors=(1024, 1024, 131072), cycles=[640], insts=(8, 4, 4, 0), total=2780),
    "flatten": dict(tensors=(), cycles=[], insts=(0, 0, 0, 0), total=1),
    "gemm": dict(tensors=(131072, 262144, 131072), cycles=[128, 8230], rows={1: 8192}, insts=(32, 4, 20, 32),
                 total=66950),
    "global_avgpool": dict(tensors=(65536, 1024, 1024), cycles=[256, 14, 256], insts=(3, 3, 4, 0), total=1176),
    "kvcache_concat": dict(tensors=(393216, 131072, 262144, 262144), cycles=[128], insts=(4, 12, 4, 0), total=2293),
    "layernorm": dict(tensors=(131072, 131072, 512, 512, 1024, 1024, 131072), cycles=[192, 14, 14, 256],
                      insts=(56, 10, 14, 0), total=3211),
    "maxpool": dict(tensors=(262144, 65536), cycles=[128], insts=(16, 4, 4, 0), total=1632),
    "softmax": dict(tensors=(131072, 512, 512, 131072), cycles=[256, 21, 384, 14, 384], insts=(24, 6, 14, 0),
                    total=4506),
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


TILE_ID_CYCLES = 100


def replay_ids(t, kernel, tensors, ids, shape, defines=()):
    """`kernel` at `shape` against tables that hold the tile ids `ids` and no other: tile id i runs 100 + i cycles on
    the vector unit, so the cycles of a compute name its tile id. An id beyond the last is refused by TOGSim, one
    below it that is not listed has a line of zeros and no unit rows, which TOGSim warns about."""
    cycles = [TILE_ID_CYCLES + i if i in ids else 0 for i in range(max(ids, default=0) + 1)]
    units = [row for i in sorted(ids) or [0]
             for row in ((i, "Systolic", "rows", 1, 0), (i, "Vpu", "cycles", 1, cycles[i]))]
    tog = t.tog(KERNELS / f"{kernel}_tog.cpp", defines, tensors=tensors, shape=shape)
    return t.simulate(tog, t.machine(), cycles, units, log_level="trace")


def expect_ids(run, ids):
    """The tile ids the computes of the run named, read from the cycles of every COMP issued, are `ids`."""
    seen = {int(e.fields["compute_cycle"]) - TILE_ID_CYCLES
            for e in run.events() if e.tag == "INST_ISSUED" and e.opcode == "COMP"}
    if seen != set(ids):
        run.fail(f"expected computes of tile ids {sorted(ids)}, got {sorted(seen)}")
    return run


def gemm_reads_its_shape_arguments(t):
    """M K N = 100 200 300 in 32 x 256 tiles of 64-deep K blocks is 4 x 2 work-items of 4 blocks: two loads and two
    barriers per block, a compute per block and one per work-item, a store per work-item. Each axis has whole
    tiles and a short last one: acc_init in the 4 variants of M and N, matmul in the 8 of M, N and K."""
    work_items, blocks = 4 * 2, 4
    expected = (work_items * blocks * 2, work_items, work_items * (blocks + 1), work_items * blocks * 2)
    ids = {0, 2, 4, 6} | set(range(1, 16, 2))
    expect_ids(expect_insts(replay_ids(t, "gemm", (80000, 240000, 120000), ids, (100, 200, 300)), expected), ids)


def conv_reads_its_shape_and_its_tile(t):
    """A 3 -> 64 channel 3 x 3 convolution of a 32 x 32 image with -DTILE_K=8: 8 work-items (4 x 2 output tiles of
    8 x 16), each 9 kernel positions of 1 weight and 16 input loads, 2 barriers and a compute. The 3 channels are
    short of the 8-deep block, so the matmul is tile id 1 + 2 * 8."""
    work_items, positions = 8, 9
    expected = (work_items * positions * 17, work_items, work_items * (positions + 1), work_items * positions * 2)
    tensors = (34 * 34 * 3 * 4, 9 * 64 * 3 * 4, 64 * 32 * 32 * 4)
    run = replay_ids(t, "conv", tensors, {0, 17}, (3, 32, 32, 64, 32, 32, 3, 3) + (1,) * 8, ("TILE_K=8",))
    expect_ids(expect_insts(run, expected), {0, 17})


CONV_TILE = dict(rows=8, cols=16, channels=64, sa=8)
TENSOR_ALIGN = 16384


def conv_tensors(shape):
    """The bytes of conv's tensor arguments at `shape`: the padded input, the weights, the output."""
    i_c, i_h, i_w, o_c, o_h, o_w, k_h, k_w, _, _, _, _, p_t, p_l, p_b, p_r = shape
    return ((i_h + p_t + p_b) * (i_w + p_l + p_r) * i_c * 4, k_h * k_w * o_c * i_c * 4, o_c * o_h * o_w * 4)


def moved(run, tensors, elem_bytes=None):
    """Per tensor argument, {element offset: times a DMA of the run moved it}, from the address, size and stride of
    every MOVIN and MOVOUT issued (trace level). TOGSim lays the tensors back to back, each start 16 KiB aligned;
    an element is 4 bytes unless `elem_bytes` gives the tensor's."""
    elem_bytes = elem_bytes or (4,) * len(tensors)
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
        first = (dram - bases[arg]) // elem_bytes[arg]
        seen[arg].update(first + sum(i * step for i, step in zip(index, stride))
                         for index in itertools.product(*map(range, extent)))
    return seen


def cut(dim, tile):
    """The extents of the tiles an axis `dim` long is walked in, `tile` at a time: the last is what is left."""
    return [min(tile, dim - at) for at in range(0, dim, tile)]


def expect_moved(run, tensors, expected, names, elem_bytes=None):
    """No DMA of the run ends past its tensor, and per tensor the elements moved are `expected`, offsets with the
    times each is moved."""
    elem_bytes = elem_bytes or (4,) * len(tensors)
    seen = moved(run, tensors, elem_bytes)
    beyond = tuple(max(0, max(got, default=-1) + 1 - size // width)
                   for got, size, width in zip(seen, tensors, elem_bytes))
    if any(beyond):
        run.fail(f"expected every DMA inside its tensor, got DMAs ending {beyond} elements past {names}")
    for name, got, want in zip(names, seen, expected):
        if got != want:
            off = sorted((got - want) + (want - got))
            run.fail(f"{name}: the DMAs moved {sum(got.values())} elements, this shape moves {sum(want.values())}; "
                     f"{len(off)} element offsets differ, the first {off[:4]}")
    return run


def convolved(shape, planes):
    """What a convolution of `shape` moves, as `moved` counts it: an input element once per output, kernel position
    and tile of 64 filters that reads it, a weight once per tile of its output plane (`planes` of them), an output
    element once."""
    i_c, i_h, i_w, o_c, o_h, o_w, k_h, k_w, s_h, s_w, d_h, d_w, p_t, p_l, p_b, p_r = shape
    x_w = i_w + p_l + p_r
    x = Counter(((oh * s_h + kh * d_h) * x_w + ow * s_w + kw * d_w) * i_c + c
                for oh in range(o_h) for ow in range(o_w) for kh in range(k_h) for kw in range(k_w)
                for c in range(i_c) for _ in cut(o_c, 64))
    return [x, times(k_h * k_w * o_c * i_c, planes), times(o_c * o_h * o_w)]


def conv_moves(shape, total, tile_k=CONV_TILE["channels"], tile_m=CONV_TILE["cols"], ids=None, block=1):
    """conv at `shape` (-DTILE_K=`tile_k` -DTILE_M=`tile_m`) issues the DMAs of its tiles, SA-wide pieces of what
    each has, moves exactly the elements a convolution of that shape does, none past a tensor, and replays to `total`,
    a golden: at the example's costs with a block as tile id `block`, or naming exactly the tile ids `ids`."""
    def case(t):
        i_c, _, _, o_c, o_h, o_w, k_h, k_w = shape[:8]
        sa = CONV_TILE["sa"]
        planes = [(th, tw) for th in cut(o_h, CONV_TILE["rows"]) for tw in cut(o_w, tile_m)]
        tiles = [plane for _ in cut(o_c, 64) for plane in planes]
        blocks = cut(i_c, tile_k)
        loads = k_h * k_w * sum(-(-tk // sa) + th * -(-tw // sa) for th, tw in tiles for tk in blocks)
        computes = k_h * k_w * len(blocks)
        tensors = conv_tensors(shape)
        defines = (f"TILE_K={tile_k}", f"TILE_M={tile_m}")
        run = replay_ids(t, "conv", tensors, ids, shape, defines) if ids else replay(
            t, "conv", tensors, [16] + [0] * (block - 1) + [8214], {block: 8192}, shape=shape, defines=defines,
            log_level="trace")
        expect_insts(run, (loads, len(tiles), len(tiles) * (computes + 1), len(tiles) * computes * 2))
        expect_moved(run, tensors, convolved(shape, len(planes)), ("the input", "the weights", "the output"))
        if ids:
            expect_ids(run, ids)
        run.expect_total_cycles(total)
    return case


def times(elements, count=1):
    """{offset: count} for the first `elements` offsets of a tensor: each moved `count` times."""
    return Counter({offset: count for offset in range(elements)})


#: A kernel at a shape none of its tiles divides (the default tiles): the bytes of its tensors, the tile ids of its
#: variants (node + nodes * variant, a bit per short axis), its (MOVIN, MOVOUT, COMP, MEMORY_BAR), its total (a
#: golden, 100 + tile id cycles a compute) and how often each element of each tensor is moved.
SHORT = {
    "gemm": dict(shape=(40, 100, 300), tensors=(40 * 100 * 4, 100 * 300 * 4, 40 * 300 * 4),
                 ids={0, 2, 4, 6} | set(range(1, 16, 2)), insts=(16, 4, 12, 16), total=1540,
                 moves=[times(40 * 100, 2), times(100 * 300, 2), times(40 * 300)]),
    "attention": dict(shape=(2, 2, 100), tensors=(2 * 100 * 64 * 4,) * 4,
                      ids=set(range(8)) | set(range(9, 17)) | set(range(18, 26)) | set(range(27, 35)) | {8, 17},
                      insts=(156, 26, 442, 156), total=50378,
                      moves=[times(12800), times(12800, 13), times(12800, 13), times(12800)]),
    "bias_act": dict(shape=(130, 100), tensors=(52000, 400, 52000), ids={0, 1, 2, 3}, insts=(8, 4, 4, 0), total=951,
                     moves=[times(13000), times(100, 130), times(13000)]),
    "bias_gelu": dict(shape=(130, 100), tensors=(52000, 400, 52000), ids={0, 1, 2, 3}, insts=(8, 4, 4, 0), total=951,
                      moves=[times(13000), times(100, 130), times(13000)]),
    "softmax": dict(shape=(130, 100), tensors=(52000, 520, 520, 52000),
                    ids={0, 5, 10, 15, 1, 6, 2, 7, 12, 17, 3, 8, 4, 9, 14, 19}, insts=(24, 8, 16, 0), total=1946,
                    moves=[times(13000, 3), times(130, 201), times(130, 101), times(13000)]),
    "layernorm": dict(shape=(130, 100), tensors=(52000, 52000, 520, 520, 400, 400, 52000),
                      ids={0, 8, 16, 24, 1, 17, 2, 18, 3, 7, 11, 15}, insts=(44, 10, 14, 0), total=1865,
                      moves=[times(13000, 2), times(13000, 2), times(130, 101), times(130, 101), times(100, 130),
                             times(100, 130), times(13000)]),
    "embed_layernorm": dict(shape=(130, 100), tensors=(1040, 400, 52000), elem_bytes=(8, 4, 4), ids={0, 1, 2, 3},
                            insts=(8, 4, 4, 0), total=717, moves=[times(130, 100), times(100, 130), times(13000)]),
    "global_avgpool": dict(shape=(300, 100), tensors=(120000, 1200, 1200), ids={0, 3, 6, 9, 1, 4, 2, 14},
                           insts=(7, 5, 9, 0), total=1223, moves=[times(30000), times(300, 2), times(300)]),
    "maxpool": dict(shape=(300, 20), tensors=(96000, 24000), ids={0, 1, 2, 3}, insts=(16, 4, 4, 0), total=1148,
                    moves=[times(24000), times(6000)]),
    "adaptive_avgpool": dict(shape=(300, 20), tensors=(96000, 24000), ids={0, 1, 2, 3}, insts=(16, 4, 4, 0),
                             total=1148, moves=[times(24000), times(6000)]),
    "concat": dict(shape=(130, 70, 100), tensors=(52000, 28000, 80000), ids=set(), insts=(21, 21, 0, 0), total=1104,
                   moves=[times(13000), times(7000), times(20000)]),
    "kvcache_concat": dict(shape=(100, 28), tensors=(100 * 768 * 4, 100 * 256 * 4, 128 * 256 * 4, 128 * 256 * 4),
                           ids={0, 1}, insts=(4, 12, 4, 0), total=1810,
                           moves=[times(76800), times(25600), Counter(range(28 * 256, 128 * 256)),
                                  Counter(range(28 * 256, 128 * 256))]),
}


def short_last_tiles(kernel):
    """The kernel at its SHORT shape: no DMA ends past its tensor and each element is moved as often as the shape
    says, the computes name the tile ids of the variants and no other, and the counts and the total are those."""
    def case(t):
        how = SHORT[kernel]
        run = replay_ids(t, kernel, how["tensors"], how["ids"], how["shape"])
        expect_insts(run, how["insts"])
        names = [f"tensor argument {i}" for i in range(len(how["tensors"]))]
        expect_moved(run, how["tensors"], how["moves"], names, how.get("elem_bytes"))
        expect_ids(run, how["ids"]).expect_total_cycles(how["total"])
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
    Case(f"short_last_tiles.{kernel}", short_last_tiles(kernel)) for kernel in SHORT] + [
    Case("gemm_reads_its_shape_arguments", gemm_reads_its_shape_arguments),
    Case("conv_reads_its_shape_and_its_tile", conv_reads_its_shape_and_its_tile),
    Case("attention_shares_a_kv_head", attention_shares_a_kv_head),
    Case("refused.shape_argument_count", refused(
        (100, 200), "kernel gemm: it takes 3 shape arguments (M K N), tog_shape.txt gives 2")),
    Case("refused.shape_argument_below_one", refused(
        (128, 0, 256), "kernel gemm: shape argument 1 of (M K N) is 0, it must be at least 1")),
    Case("conv.stride_2", conv_moves((64, 32, 64, 64, 16, 32, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1), 148793)),
    Case("conv.pad_0", conv_moves((64, 18, 18, 64, 16, 16, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0), 74614)),
    Case("conv.1x1_stride_2", conv_moves((64, 32, 32, 64, 16, 16, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0), 8683)),
    Case("conv.stride_2x1_pads_0_1_1_2", conv_moves((64, 32, 15, 64, 16, 16, 3, 3, 2, 1, 1, 1, 0, 1, 1, 2), 74649)),
    Case("conv.dilation_2x3_pads_2_3_2_3", conv_moves((64, 16, 16, 64, 16, 16, 3, 3, 1, 1, 2, 3, 2, 3, 2, 3), 74603)),
    Case("conv.7x7_stride_2_pad_3_of_3_channels", conv_moves(
        (3, 32, 32, 64, 16, 16, 7, 7, 2, 2, 1, 1, 3, 3, 3, 3), 404627, tile_k=8, block=17)),
    Case("conv.short_on_every_axis", conv_moves(
        (20, 10, 20, 70, 10, 20, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1), 17941, tile_k=16,
        ids=set(range(0, 16, 2)) | set(range(1, 32, 2)))),
    Case("conv.112_columns_in_tiles_of_32", conv_moves(
        (64, 16, 224, 64, 8, 112, 3, 3, 2, 2, 1, 1, 1, 1, 0, 0), 7410, tile_m=32, ids={0, 1, 8, 9})),
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
