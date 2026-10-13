"""DEPRECATED typed grouping: every TOG here uses compute_type 1, 2 or 3. Delete this directory with the mode.
What they pin is not typed-only (a semaphore re-wait, the spad a work-item needs at once, one compute per core,
systolic_array_split); the TOGs are.
Every run here loads its TOG: the refusals that end before one is loaded are in tests/array_split/."""
from pathlib import Path

from togtest import Case, read_table, spad_alone

HERE = Path(__file__).resolve().parent
TYPED_GROUPING = True
MATMUL_TILE = 5
GEMM512_CYCLES = [183, 16, 6, 14, 51, 1589]
GEMM512_UNITS = read_table(HERE / "gemm512_units.tsv")


def rewait(t, *defines):
    """The re-wait TOG on tpuv6e without the array split: the in-place cross-lane tile 2000 cycles, its reader 100."""
    return t.simulate(t.tog("rewait_follows_inplace_write_tog.cpp", defines),
                      t.machine(systolic_array_split="none"), cycles=[2000, 100])


def rewait_follows_inplace_write(t):
    """The compute that reads the buffer after the re-wait issues no earlier than the in-place write finishes."""
    run = rewait(t).expect_ok()
    written, read = run.when("INST_FINISHED", 2), run.when("INST_ISSUED", 3)
    if read < written:
        run.fail(f"expected the reader after the re-wait to issue after the in-place write finished @{written}, "
                 f"it issued @{read}")


def rewait_with_wrong_count(t):
    rewait(t, "WRONG_COUNT").expect_refused("re-wait expects another element count", exit=1)


def held_through_overlap_refused(t):
    """A 24 KB spad against the 32 KB the matmul finds held: one of the two loads reaches it only through the
    preload it overlaps (an edge released at issue), and the refusal counts that load too."""
    run = t.simulate(t.tog("held_through_overlap_tog.cpp"),
                     t.machine(**spad_alone(24), systolic_array_split="none"), cycles=[200], timeout=60)
    run.expect_refused("work-item 0 (key 0) needs 32768 bytes in the spad at once, and a core's spad holds 24576",
                       exit=1)


def gemm512(t, grid=(1, 1), cycles=GEMM512_CYCLES, units=GEMM512_UNITS, **machine):
    """The gemm512 TOG as a `grid` of work-items on tpuv6e (2 arrays per core, 128 lanes) with `machine` set;
    `cycles` are those of tiles 0-5, `units` the unit table (tile 5: 1024 TPUV6_MXU rows, 201 Vpu cycles)."""
    return t.simulate(t.tog("gemm512_tog.cpp", shape=grid), t.machine(**machine), cycles, units)


def crosslane_and_vector(t, crosslane_cycles, **machine):
    """The cross-lane TOG without the array split: tile 0 (cross-lane) `crosslane_cycles`, tile 1 (vector) 100."""
    return t.simulate(t.tog("crosslane_and_vector_tog.cpp"), t.machine(systolic_array_split="none", **machine),
                      cycles=[crosslane_cycles, 100])


def expect_one_compute_per_core(run):
    """The run finished, computes ran, and no two compute windows of one core overlap."""
    run.expect_ok()
    ran = sorted(c for c in run.computes() if c.cycles > 0)
    clash = [(a, b) for a, b in zip(ran, ran[1:]) if a.core == b.core and b.start < a.finish]
    if not run.computes() or clash:
        run.fail(f"expected the computes of a core one at a time, got {len(run.computes())} computes, "
                 f"{len(clash)} overlapping{f', e.g. {clash[0][0]} and {clash[0][1]}' if clash else ''}")
    return ran


def gemm512_one_work_item(t):
    """With a unit table and the split off the gemm runs to its Total line, one compute at a time."""
    run = gemm512(t, systolic_array_split="none")
    expect_one_compute_per_core(run)
    run.total_cycles()


def gemm512_two_work_items(t):
    expect_one_compute_per_core(gemm512(t, grid=(2, 1), systolic_array_split="none"))


def crosslane_beside_vector(t):
    expect_one_compute_per_core(crosslane_and_vector(t, 150000))


def two_cores_run_side_by_side(t):
    """On two cores each core still runs one compute at a time, and the two cores' windows overlap."""
    run = crosslane_and_vector(t, 300, num_cores=2)
    ran = expect_one_compute_per_core(run)
    cores = sorted({c.core for c in ran})
    side_by_side = any(a.core != b.core and a.start < b.finish and b.start < a.finish for a in ran for b in ran)
    if cores != [0, 1] or not side_by_side:
        run.fail(f"expected computes on cores [0, 1] with overlapping windows, got cores {cores}, "
                 f"cross-core overlap {side_by_side}")


def admitted(tile, unit, count, units=GEMM512_UNITS):
    """`units` (GEMM512_UNITS) with the admitted count of (tile, unit) set to `count`."""
    hit = [r for r in units if r[:2] == [str(tile), unit]]
    assert len(hit) == 1, f"gemm512_units.tsv has {len(hit)} rows for tile {tile} unit {unit}"
    return [r[:4] + [str(count)] if r is hit[0] else r for r in units]


def matmul_cycles(cycles):
    return GEMM512_CYCLES[:MATMUL_TILE] + [cycles]


SPLIT = {
    "matmul_over_two_arrays": dict(
        matmul=1077, why="1589 - 1024 + 1024/2"),
    "one_subtile_is_not_split": dict(
        matmul=1589, why="128 rows are one subtile, N_eff 1", units=admitted(MATMUL_TILE, "TPUV6_MXU", 128)),
    "vector_unit_bounds_the_split": dict(
        matmul=1265, why="1589 - 1024 + 700, Vpu 700 > 1024/2", units=admitted(MATMUL_TILE, "Vpu", 700)),
    "shorter_matmul_over_two_arrays": dict(
        matmul=588, why="1100 - 1024 + 1024/2", cycles=matmul_cycles(1100)),
}
OVER_ONE_ARRAY = dict(
    saying="computes tile id 5, whose unit table row admits 128 rows to TPUV6_MXU, and the unit takes 100 in the "
           "tile's 100 cycles (1 per cycle)",
    exit=1, num_systolic_array_per_core=4, cycles=matmul_cycles(100),
    units=admitted(MATMUL_TILE, "TPUV6_MXU", 128, admitted(MATMUL_TILE, "Vpu", 100)))
SPLIT_REFUSED = {
    "refused.spread_exceeds_one_array": OVER_ONE_ARRAY,
    "refused.spread_exceeds_one_array.icnt_32_ports_at_1751_mhz": dict(
        OVER_ONE_ARRAY, icnt_injection_ports_per_core=32, core_freq_mhz=1751),
    "refused.spread_exceeds_one_array.icnt_32_ports_at_1751_mhz_statistics_every_100_cycles": dict(
        OVER_ONE_ARRAY, icnt_injection_ports_per_core=32, core_freq_mhz=1751, core_stats_print_period_cycles=100),
    "refused.spread_exceeds_one_array.statistics_every_10000_cycles": dict(
        OVER_ONE_ARRAY, core_stats_print_period_cycles=10000),
}


def split(matmul, why, **given):
    """Every matmul (compute_type 1) states `matmul` as its compute_cycle: cc - B_sys + max(B_sys / N_eff, the other
    units' B), B_sys the TPUV6_MXU rows, N_eff = min(arrays, ceil(rows / lanes)) (Core::split_compute_cycles)."""
    def case(t):
        run = gemm512(t, **given).expect_ok()
        seen = sorted({c.cycles for c in run.computes() if c.compute_type == 1})
        if seen != [matmul]:
            run.fail(f"expected every matmul to run {matmul} cycles ({why}), got {seen}")
    return case


def split_refused(saying, exit, **given):
    """The TOG is refused with `saying` and that exit status while it is loaded: no instruction is traced."""
    def case(t):
        run = gemm512(t, **given).expect_refused(saying, exit)
        if run.events():
            run.fail("expected the refusal before anything was simulated, found instruction trace lines")
    return case


CASES = [
    Case("semaphore.rewait_follows_inplace_write", rewait_follows_inplace_write),
    Case("semaphore.refused.rewait_with_wrong_count", rewait_with_wrong_count),
    Case("spad_capacity.refused.held_through_overlap", held_through_overlap_refused),
    Case("one_compute_per_core.gemm512_one_work_item", gemm512_one_work_item),
    Case("one_compute_per_core.gemm512_two_work_items", gemm512_two_work_items),
    Case("one_compute_per_core.crosslane_beside_vector", crosslane_beside_vector),
    Case("one_compute_per_core.two_cores_run_side_by_side", two_cores_run_side_by_side),
] + [Case(f"systolic_split.{name}", split(**how)) for name, how in SPLIT.items()] + [
    Case(f"systolic_split.{name}", split_refused(**how)) for name, how in SPLIT_REFUSED.items()]
