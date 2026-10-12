"""The kernel library, kernels/*_tog.cpp, on tpuv6e: each kernel's instruction counts and total cycles at its
example shape, that a kernel reads its shape arguments, and the shape arguments it refuses."""
import re
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


def replay(t, kernel, tensors, cycles, rows=None, shape=None, defines=(), **_):
    """`kernel` on tpuv6e under the default systolic_array_split, `tensors` the bytes of its tensor arguments: tile i
    runs cycles[i], holds rows[i] Systolic rows (0 without an entry) and, a tile without any, the vector unit."""
    rows = rows or {}
    units = [row for i, c in enumerate(cycles or [0])
             for row in ((i, "Systolic", "rows", 1, rows.get(i, 0)), (i, "Vpu", "cycles", 1, 0 if i in rows else c))]
    tog = t.tog(KERNELS / f"{kernel}_tog.cpp", defines, tensors=tensors, shape=shape)
    return t.simulate(tog, t.machine(), cycles or [0], units, log_level="info")


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


def attention_shares_a_kv_head(t):
    """HEADS KV_HEADS SEQ = 16 8 128: the 128 work-items stay, and each of their two key blocks runs the QK and the
    SV compute twice, once per query head of the KV head (2176 + 128 * 2 * 2 computes)."""
    how = EXAMPLE["attention"]
    expect_insts(replay(t, "attention", (524288, 262144, 262144, 524288), how["cycles"], how["rows"],
                        shape=(16, 8, 128)), (768, 128, 2688, 768))


def refused(shape, saying):
    """gemm with `shape` in tog_shape.txt ends the run with exit 1 and `saying` on stderr, before any work-item."""
    def case(t):
        run = replay(t, "gemm", GEMM["tensors"], GEMM["cycles"], GEMM["rows"], shape=shape)
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
]
