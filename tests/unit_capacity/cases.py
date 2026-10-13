"""A compute tile whose unit table rows ask more of a unit than the unit takes in the tile's cycles is refused while
its TOG is loaded, in both modes and whatever the timing of the run; a tile that fills its cycles exactly runs."""
from pathlib import Path

from togtest import MIB, Case, Failure, write_table

TOG = Path(__file__).resolve().parents[1] / "cycle_table" / "names_tile_id_tog.cpp"
TILE = 1                    # the tile id the TOG's one compute names


def rows(tile=TILE, **admitted):
    """Unit table rows of `tile`: `admitted` of each unit named, the array (1 row per cycle) first."""
    units = dict(Systolic=("rows", 1), Vpu=("cycles", 1), Issue=("micro-ops", 12))
    return [(tile, unit, *units[unit], admitted[unit]) for unit in units if unit in admitted]


def refusal(admitted, work, unit, takes, window, capacity=1, arrays=1, table=None, tile=TILE):
    """The refusal of work-item 0 (key 7) for `tile` admitting `admitted` `work` to `unit`, which takes `takes` in
    the `window` cycles the array split leaves of the cycle table's `table`, after the TOG's path."""
    each = f" on each of {arrays} arrays" if arrays > 1 else ""
    left = f"; the array split leaves {window} of the cycle table's {table} cycles" if table else ""
    return (f"work-item 0 (key 7) computes tile id {tile}, whose unit table row admits {admitted} {work} to {unit}, "
            f"and the unit takes {takes} in the tile's {window} cycles ({capacity} per cycle{each}{left})")


def replay(t, cycles, units, *defines, **machine):
    """--tog_so: the TOG on tpuv6e (2 arrays per core, 256 lanes) with `machine` set; tile 1 takes `cycles`, tile 0
    ten cycles, `units` is the unit table."""
    return t.simulate(t.tog(TOG, defines, tensors=(MIB, MIB)), t.machine(**machine), [10, cycles], units)


def session(t, *kernels):
    """--models_list: one LAUNCH_KERNEL per (cycles of tile 1, unit table) of `kernels`, the TOG from
    <work dir>/k<i>/, then one DEVICE_SYNC."""
    stream = t.work_dir / "commands.txt"
    with stream.open("w") as commands:
        for i, (cycles, units) in enumerate(kernels):
            home = t.tog(TOG, tensors=(MIB, MIB), into=f"k{i}").parent
            write_table(home / "tog_cycles.tsv", [(10, 0), (cycles, 0)])
            write_table(home / "tog_units.tsv", units)
            commands.write(f"LAUNCH_KERNEL,{i},0,0,{home}/kernel,{home}/attributes,0\n")
        commands.write("DEVICE_SYNC,0,0,0,-,-,0\n")
    return t.execute("Simulator", ["--config", t.machine(), "--models_list", stream, "--log_level", "trace"])


def expect_refused_before_the_run(run, saying, so):
    """The run was refused with exit 1 by a line that names the TOG `so` and says `saying`, and no instruction was
    traced."""
    run.expect_refused(f"TOG {so}: {saying}", exit=1)
    if run.events():
        run.fail("expected the refusal before anything was simulated, found instruction trace lines")


def expect_computes(run, *cycles):
    """The run finished and its computes ran `cycles` cycles, in issue order."""
    seen = [c.cycles for c in run.expect_ok().computes()]
    if seen != list(cycles):
        run.fail(f"expected computes of {list(cycles)} cycles, got {seen}")


LIMITS = {
    "array_unit_on_one_array": (
        230, dict(Systolic=230), 230, "Systolic", refusal(231, "rows", "Systolic", 230, 230),
        dict(num_systolic_array_per_core=4)),
    "array_unit_over_two_arrays": (
        512, dict(Systolic=512), 256, "Systolic", refusal(513, "rows", "Systolic", 512, 256, arrays=2, table=512),
        {}),
    "array_unit_without_the_split": (
        512, dict(Systolic=512), 512, "Systolic", refusal(513, "rows", "Systolic", 512, 512),
        dict(systolic_array_split="none")),
    "vector_unit": (
        600, dict(Systolic=0, Vpu=600), 600, "Vpu", refusal(601, "cycles", "Vpu", 600, 600), {}),
    "vector_unit_beside_a_split_array": (
        700, dict(Systolic=512, Vpu=700), 700, "Vpu", refusal(701, "cycles", "Vpu", 700, 700), {}),
    "unit_of_capacity_12": (
        600, dict(Systolic=0, Issue=7200), 600, "Issue", refusal(7201, "micro-ops", "Issue", 7200, 600, capacity=12),
        {}),
}


def filled_exactly_runs(cycles, admitted, runs, unit, saying, machine):
    """A row of LIMITS: tile 1 of `cycles` cycles with the rows `admitted`, which fill them exactly, on tpuv6e with
    `machine` set: its compute runs `runs` cycles."""
    return lambda t: expect_computes(replay(t, cycles, rows(**admitted), **machine), runs)


def one_more_refused(cycles, admitted, runs, unit, saying, machine):
    """The same row with one more admitted to `unit`: refused with `saying`."""
    over = rows(**{**admitted, unit: admitted[unit] + 1})
    return lambda t: expect_refused_before_the_run(replay(t, cycles, over, **machine), saying, t.work_dir / "tog.so")


TIMINGS = {
    "as_shipped": {},
    "icnt_32_ports": dict(icnt_injection_ports_per_core=32),
    "icnt_32_ports_at_1751_mhz": dict(icnt_injection_ports_per_core=32, core_freq_mhz=1751),
    "statistics_every_10000_cycles": dict(core_stats_print_period_cycles=10000),
    "statistics_every_100_cycles": dict(core_stats_print_period_cycles=100),
}


def refused_at_every_timing(machine):
    """256 rows on one array (of four) in 230 cycles, with a row of TIMINGS set: the refusal is the tile's, whenever
    its compute would have run and however long a statistics interval is."""
    saying = refusal(256, "rows", "Systolic", 230, 230)
    return lambda t: expect_refused_before_the_run(
        replay(t, 230, rows(Systolic=256), num_systolic_array_per_core=4, **machine), saying, t.work_dir / "tog.so")


def no_cycles_admit_nothing(t):
    """A tile of 0 cycles whose rows admit nothing runs; one Vpu cycle in it is refused."""
    replay(t, 0, rows(Systolic=0, Vpu=0)).expect_ok()
    expect_refused_before_the_run(replay(t, 0, rows(Systolic=0, Vpu=1)), refusal(1, "cycles", "Vpu", 0, 0),
                                  t.work_dir / "tog.so")


def tile_no_compute_names_is_not_measured(t):
    """Tile 0 admits 999 rows in its ten cycles and no compute names it: the run is that of the table without those
    rows. Named (TILE_ID=0), it is refused: the split leaves the tile one cycle, and two arrays take two rows."""
    fits, over = rows(Systolic=512), rows(Systolic=512) + rows(0, Systolic=999)
    totals = [replay(t, 600, units).expect_ok().total_cycles() for units in (fits, over)]
    if totals[0] != totals[1]:
        raise Failure(f"expected one total with and without the rows of tile 0, got {totals}")
    saying = refusal(999, "rows", "Systolic", 2, 1, arrays=2, table=10, tile=0)
    expect_refused_before_the_run(replay(t, 600, over, "TILE_ID=0"), saying, t.work_dir / "tog.so")


def expect_launch_refused(run, command, saying, so):
    """As expect_refused_before_the_run, the line being the error of LAUNCH_KERNEL command `command`."""
    expect_refused_before_the_run(run, saying, so)
    if f"[TOGSim] Error processing command {command} (type: LAUNCH_KERNEL)" not in run.said()[0]:
        run.fail(f"expected the refusal as the error of command {command}, the one that launched the kernel")


def session_filled_exactly_runs(t):
    expect_computes(session(t, (512, rows(Systolic=512)), (512, rows(Systolic=512))), 256, 256)


def session_one_more_refused(t):
    """The launch is refused: the line is the command's, and names the kernel's tog.so."""
    expect_launch_refused(session(t, (512, rows(Systolic=513))), 0,
                          refusal(513, "rows", "Systolic", 512, 256, arrays=2, table=512), t.work_dir / "k0" / "tog.so")


def session_second_kernel_refused(t):
    """Kernel 0 fills its tile and kernel 1 admits one row more: the session ends at kernel 1's launch, and the line
    names the tile id as kernel 1's TOG does, not its row in the joined table."""
    expect_launch_refused(session(t, (512, rows(Systolic=512)), (512, rows(Systolic=513))), 1,
                          refusal(513, "rows", "Systolic", 512, 256, arrays=2, table=512), t.work_dir / "k1" / "tog.so")


CASES = [case for name, how in LIMITS.items() for case in (
    Case(f"{name}.filled_exactly_runs", filled_exactly_runs(*how)),
    Case(f"{name}.refused.one_more", one_more_refused(*how)))] + [
    Case(f"refused.256_rows_in_230_cycles.{name}", refused_at_every_timing(machine))
    for name, machine in TIMINGS.items()] + [
    Case("no_cycles_admit_nothing", no_cycles_admit_nothing),
    Case("tile_no_compute_names_is_not_measured", tile_no_compute_names_is_not_measured),
    Case("session.filled_exactly_runs", session_filled_exactly_runs),
    Case("session.refused.one_more", session_one_more_refused),
    Case("session.refused.second_kernel", session_second_kernel_refused),
]
