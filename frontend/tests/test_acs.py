"""ACS without the simulator: what a compute node costs and the two tables written from it."""
from togsim_frontend.acs import cycles as acs_cycles
from togsim_frontend.acs.__main__ import EXAMPLE_TILE
from togsim_frontend.acs.node_mapping import ELEMENTWISE, KERNELS, KernelDesc, NodeDesc, Op, lookup
from togsim_frontend.onnx_frontend.tiling import Hardware

HW = Hardware(sa=8, vlane=8, vlen_bits=256)
GEMM_TILE = {"rows": 32, "depth": 64, "cols": 256, "elems": 32 * 256}


def test_the_overlapping_column_is_always_zero():
    for stem, desc in KERNELS.items():
        assert all(overlapping == 0 for _, overlapping in acs_cycles.build_table(desc, EXAMPLE_TILE[stem], HW)), stem


def test_a_systolic_tile_is_its_matmul_rows_plus_the_array_drain():
    """64 K steps x 256 / (8 * 8) column groups = 256 preloads, x 32 / 8 row groups = 1024 matmuls of 8 rows."""
    assert acs_cycles.systolic_instructions(GEMM_TILE, HW) == (256, 1024)
    acc_init, matmul = acs_cycles.build_table(lookup("gemm"), GEMM_TILE, HW)
    assert matmul == (1024 * 8 + 2 * HW.sa - 2, 0)
    assert acc_init == (32 * 256 // HW.throughput, 0)


def test_the_unit_table_carries_the_rows_the_array_split_divides():
    units = acs_cycles.build_units(lookup("gemm"), GEMM_TILE, HW)
    assert units == [(0, "Systolic", "rows", 1, 0), (0, "Vpu", "cycles", 1, 128),
                     (1, "Systolic", "rows", 1, 8192), (1, "Vpu", "cycles", 1, 0)]


def test_a_kernel_without_a_compute_still_names_the_array_unit():
    """An empty unit table is refused under the default systolic_array_split."""
    assert acs_cycles.build_units(lookup("concat"), EXAMPLE_TILE["concat"], HW)[0] == (0, "Systolic", "rows", 1, 0)
    assert acs_cycles.build_table(lookup("flatten"), EXAMPLE_TILE["flatten"], HW) == [(0, 0)]


def test_the_tables_are_written_as_togsim_reads_them(tmp_path):
    desc = lookup("gemm")
    acs_cycles.write_table(acs_cycles.build_table(desc, GEMM_TILE, HW), str(tmp_path / "tog_cycles.tsv"))
    acs_cycles.write_units(acs_cycles.build_units(desc, GEMM_TILE, HW), str(tmp_path / "tog_units.tsv"))
    assert (tmp_path / "tog_cycles.tsv").read_text() == "128\t0\n8206\t0\n# origins: acs\n"
    lines = (tmp_path / "tog_units.tsv").read_text().splitlines()
    assert lines[2] == "1\tSystolic\trows\t1\t8192" and all(len(line.split("\t")) == 5 for line in lines)


def test_vector_work_is_charged_in_full():
    rows = acs_cycles.build_table(lookup("bias_act"), {"elems": 4096}, HW)
    cycles, overlapping = rows[0]
    assert overlapping == 0
    assert cycles == (4096 // HW.throughput) * 2      # + bias, then max(x, 0)


def test_gelu_costs_more_than_relu_on_the_same_tile():
    tile = {"elems": 4096}
    relu = acs_cycles.build_table(lookup("bias_act"), tile, HW)[0][0]
    gelu = acs_cycles.build_table(lookup("bias_gelu"), tile, HW)[0][0]
    assert gelu == 3 * relu              # six instructions against two


def test_the_cost_function_can_be_replaced():
    desc = KernelDesc("x", (NodeDesc("n", "vector", (Op(ELEMENTWISE, 1, "op"),)),))
    try:
        acs_cycles.set_cost_function(lambda node, tile, hw: acs_cycles.Cost(99, systolic_rows=7))
        assert acs_cycles.build_table(desc, {"elems": 1}, HW) == [(99, 0)]
        assert acs_cycles.build_units(desc, {"elems": 1}, HW)[0] == (0, "Systolic", "rows", 1, 7)
    finally:
        acs_cycles.set_cost_function(acs_cycles.default_cost)


def test_a_tile_id_without_a_cycle_line_or_a_unit_row_is_uncovered():
    """gemm names tile ids 0 and 1: both tables as built cover them, a table one line short or without the rows of
    a tile does not."""
    table = acs_cycles.build_table(lookup("gemm"), GEMM_TILE, HW)
    units = acs_cycles.build_units(lookup("gemm"), GEMM_TILE, HW)
    assert acs_cycles.uncovered(range(2), table, units) == []
    assert acs_cycles.uncovered(range(2), table[:1], units) == [1]
    assert acs_cycles.uncovered(range(2), table, [row for row in units if row[0] != 0]) == [0]


def test_a_short_block_is_costed_by_the_rows_depth_and_columns_it_has():
    """The formula of a whole block at what is left: 13 rows are a matmul of 8 and one of 5 under each preload, 13
    deep is 13 preloads per column group, 44 columns are one group of 64 where 256 were four."""
    def rows_fed(**short):
        return acs_cycles.systolic_cost({**GEMM_TILE, **short}, HW).systolic_rows
    assert rows_fed() == 64 * 4 * 32 == 8192
    assert rows_fed(rows=13) == 64 * 4 * 13
    assert acs_cycles.systolic_instructions({**GEMM_TILE, "rows": 13}, HW) == (256, 512)
    assert rows_fed(depth=13) == 13 * 4 * 32
    assert rows_fed(cols=44) == 64 * 1 * 32
    assert acs_cycles.systolic_cost({"rows": 13, "depth": 13, "cols": 44}, HW) == (13 * 13 + 14, 13 * 13, 0)


def test_a_block_shallower_than_the_array_is_a_preload_per_channel_it_has():
    """A conv tile of 8 x 16 outputs and 64 filters over the 3 channels of an image: 3 preloads of 128 rows, where
    the 8-deep block the tile macro names would be 8."""
    stem = {"rows": 128, "depth": 3, "cols": 64}
    assert acs_cycles.systolic_instructions(stem, HW) == (3, 48)
    assert acs_cycles.systolic_cost(stem, HW) == (3 * 128 + 14, 3 * 128, 0)
    assert acs_cycles.systolic_cost({**stem, "depth": 8}, HW).systolic_rows == 8 * 128
