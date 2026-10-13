"""ACS without the simulator: what a compute node costs and the two tables written from it."""
from togsim_frontend.acs import cycles as acs_cycles
from togsim_frontend.acs.__main__ import EXAMPLE_TILE
from togsim_frontend.acs.node_mapping import (ELEMENTWISE, KERNELS, KernelDesc, NodeDesc, Op, at_variant, lookup,
                                              tile_ids)
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
    assert units == [(0, "SYSTOLIC", "rows", 1, 0), (0, "Vpu", "cycles", 1, 128),
                     (1, "SYSTOLIC", "rows", 1, 8192), (1, "Vpu", "cycles", 1, 0)]


def test_a_kernel_without_a_compute_still_names_the_array_unit():
    """An empty unit table is refused under the default systolic_array_split."""
    assert acs_cycles.build_units(lookup("concat"), EXAMPLE_TILE["concat"], HW)[0] == (0, "SYSTOLIC", "rows", 1, 0)
    assert acs_cycles.build_table(lookup("flatten"), EXAMPLE_TILE["flatten"], HW) == [(0, 0)]


def test_the_tables_are_written_as_togsim_reads_them(tmp_path):
    desc = lookup("gemm")
    acs_cycles.write_table(acs_cycles.build_table(desc, GEMM_TILE, HW), str(tmp_path / "tog_cycles.tsv"))
    acs_cycles.write_units(acs_cycles.build_units(desc, GEMM_TILE, HW), str(tmp_path / "tog_units.tsv"))
    assert (tmp_path / "tog_cycles.tsv").read_text() == "128\t0\n8206\t0\n# origins: acs\n"
    lines = (tmp_path / "tog_units.tsv").read_text().splitlines()
    assert lines[2] == "1\tSYSTOLIC\trows\t1\t8192" and all(len(line.split("\t")) == 5 for line in lines)


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
        assert acs_cycles.build_units(desc, {"elems": 1}, HW)[0] == (0, "SYSTOLIC", "rows", 1, 7)
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


GEMM_ODD = {"M": 45, "K": 77, "N": 300, "TM": 32, "TN": 256, "TK": 64}


def test_a_kernel_names_a_tile_id_per_node_and_variant_its_shape_reaches():
    """node + nodes * variant, a bit per short axis in the kernel's order. gemm's acc_init spans M and N, its matmul
    K as well; a dimension under its tile has the short variant alone, one the tile divides the whole one alone."""
    gemm = lookup("gemm")
    assert tile_ids(gemm, {**GEMM_ODD, "M": 64, "K": 128, "N": 256}) == [0, 1]
    assert tile_ids(gemm, GEMM_ODD) == [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15]
    assert tile_ids(gemm, {**GEMM_ODD, "M": 5, "K": 128, "N": 256}) == [2, 3]
    conv = {"O_C": 64, "TILE_N": 64, "O_H": 16, "TILE_I_H": 8, "O_W": 112, "TILE_M": 32, "I_C": 3, "TILE_K": 8}
    assert tile_ids(lookup("conv"), conv) == [0, 8, 17, 25]
    attention = {"SEQ": 100, "Q_LEN": 8, "KV_BLOCK": 64}
    assert tile_ids(lookup("attention"), attention) == sorted(set(range(36)) - {26, 35})
    layernorm = {"M": 70, "N": 100, "TM": 8, "TN": 64, "TM_STATS": 128}
    assert tile_ids(lookup("layernorm"), layernorm) == [3, 7, 11, 15, 16, 17, 18, 24]
    assert tile_ids(lookup("concat"), {}) == []


def test_a_variant_is_the_tile_with_what_is_left_on_its_short_axes():
    assert at_variant(lookup("gemm"), GEMM_ODD, 0) == GEMM_ODD
    assert at_variant(lookup("gemm"), GEMM_ODD, 0b101) == {**GEMM_ODD, "TM": 13, "TK": 13}
    assert at_variant(lookup("gemm"), GEMM_ODD, 0b010) == {**GEMM_ODD, "TN": 44}


def test_the_tables_hold_a_row_per_tile_id_costed_and_zeros_between():
    """Tile ids 2 and 3 alone (a gemm of 5 rows): line i is tile id i, the lines of 0 and 1 are zeros and have no
    unit rows, so a compute that named them would make TOGSim warn."""
    short = {"rows": 5, "depth": 64, "cols": 256, "elems": 5 * 256}
    tiles = {2: short, 3: short}
    table = acs_cycles.build_table(lookup("gemm"), tiles, HW)
    units = acs_cycles.build_units(lookup("gemm"), tiles, HW)
    assert table == [(0, 0), (0, 0), (20, 0), (256 * 5 + 14, 0)]
    assert units == [(2, "SYSTOLIC", "rows", 1, 0), (2, "Vpu", "cycles", 1, 20),
                     (3, "SYSTOLIC", "rows", 1, 1280), (3, "Vpu", "cycles", 1, 0)]
    assert acs_cycles.uncovered([2, 3], table, units) == [] and acs_cycles.uncovered([0, 2, 5], table, units) == [0, 5]
