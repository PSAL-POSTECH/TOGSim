"""--models_list: a command stream launches kernels, each from a directory holding tog.so and its tables."""
from togtest import MIB, TIMEOUT_S, Case, write_table

KERNELS = {0: dict(cycles=600, rows=256), 1: dict(cycles=900, rows=512)}


def units(kernel_id, array="TPUV6E_MXU"):
    """KERNELS[id]'s unit table: its rows of the machine's array unit `array`, and 100 cycles of the Vpu."""
    return [(0, array, "rows", 1, KERNELS[kernel_id]["rows"]), (0, "Vpu", "cycles", 1, 100)]


def kernel(t, kernel_id, table=True, tog=True, array="TPUV6E_MXU", items=2, defines=()):
    """<work dir>/k<id>/: tog.so (of `items` work-items), tog_tensors.txt, tog_cycles.tsv and tog_units.tsv of
    KERNELS[id] naming `array` (`table` may replace the unit table or, False, leave it out; `tog` False leaves tog.so
    out). Returns the directory."""
    home = t.tog("kernel_tog.cpp", (f"ITEMS={items}", *defines), tensors=(MIB, MIB), into=f"k{kernel_id}").parent
    write_table(home / "tog_cycles.tsv", [(KERNELS[kernel_id]["cycles"], 0)])
    if table is not False:
        write_table(home / "tog_units.tsv", units(kernel_id, array) if table is True else table)
    if not tog:
        (home / "tog.so").unlink()
    return home


def session(t, config, lines, *more, timeout=TIMEOUT_S):
    """Run the command stream `lines` on `config` at --log_level trace; `more` is appended to the command line."""
    stream = t.work_dir / "commands.txt"
    stream.write_text("".join(f"{line}\n" for line in lines))
    return t.execute("Simulator", ["--config", config, "--models_list", stream, "--log_level", "trace", *more],
                     timeout)


def launch(home, kernel_id, partition=0):
    return f"LAUNCH_KERNEL,{kernel_id},0,{partition},{home}/kernel,{home}/attributes,0"


SYNC = "DEVICE_SYNC,0,0,0,-,-,0"


def two_kernels_one_after_the_other(t):
    """On tpuv6e (2 arrays, 128 lanes) kernel 0 then kernel 1, a DEVICE_SYNC after each: both are enqueued from
    <dir>/tog.so, kernel 0's computes run 472 = 600 - 256 + 256/2, then kernel 1's run 644 = 900 - 512 + 512/2, which
    only its own tog_units.tsv gives (kernel 0's 256 rows would leave 772); and the session's closing lines are
    printed."""
    homes = [kernel(t, 0), kernel(t, 1)]
    run = session(t, t.machine(), ["# two kernels", launch(homes[0], 0), SYNC, launch(homes[1], 1), SYNC]).expect_ok()
    for kernel_id, home in enumerate(homes):
        enqueued = f"[Scheduler 0] Enqueued kernel_id: {kernel_id}, tog_path: {home}/tog.so, operation: kernel,"
        if enqueued not in run.log:
            run.fail(f'expected the line "{enqueued} ..."')
    ran = [(c.cycles, c.start, c.finish) for c in run.computes()]
    if [c[0] for c in ran] != [472, 472, 644, 644] or not max(c[2] for c in ran[:2]) <= min(c[1] for c in ran[2:]):
        run.fail(f"expected kernel 0's two computes of 472 cycles, then kernel 1's two of 644; got (cycles, start, "
                 f"finish) {ran}")
    for line in ("[Scheduler 0] Kernel 0 has completed", "Simulation finished", "Wall-clock time for simulation:"):
        if line not in run.log:
            run.fail(f'expected a line with "{line}"')
    run.total_cycles()


def two_partitions_side_by_side(t):
    """On tpuv3 with two cores, 32 DRAM channels and a partition per core (2 arrays, 128 lanes), kernel 0 goes to
    partition 0 and kernel 1 to partition 1 with no DEVICE_SYNC between: core 0 runs only kernel 0's computes
    (472 = 600 - 256 + 256/2), core 1 only kernel 1's (644), and the two cores' compute windows overlap."""
    homes = [kernel(t, 0, array="TPUV3_MXU"), kernel(t, 1, array="TPUV3_MXU")]
    machine = t.machine("tpuv3.yml", num_cores=2, dram_channels=32, num_partition=2, partition="{core_0: 0, core_1: 1}")
    run = session(t, machine, [launch(homes[0], 0, partition=0), launch(homes[1], 1, partition=1)]).expect_ok()
    ran = run.computes()
    by_core = {core: sorted(c.cycles for c in ran if c.core == core) for core in sorted({c.core for c in ran})}
    together = any(a.core != b.core and a.start < b.finish and b.start < a.finish for a in ran for b in ran)
    if by_core != {0: [472, 472], 1: [644, 644]} or not together:
        run.fail(f"expected core 0 to run two computes of 472 cycles and core 1 two of 644, overlapping; got "
                 f"{by_core}, cross-core overlap {together}")
    run.total_cycles()


def second_kernel_runs(items, cores):
    """Kernel 0 of `items` work-items, then kernel 1 of two, launched with no DEVICE_SYNC between on `cores` cores:
    every work-item of both is replayed, whatever `items` is, and each kernel's completion is logged."""
    def case(t):
        homes = [kernel(t, 0, items=items), kernel(t, 1)]
        run = session(t, t.machine(num_cores=cores), [launch(homes[0], 0), launch(homes[1], 1)], timeout=60)
        run.expect_ok()
        ran = sorted(c.cycles for c in run.computes())
        if ran != [472] * items + [644] * 2:
            run.fail(f"expected kernel 0's {items} computes of 472 cycles and kernel 1's two of 644, got {ran}")
        for kernel_id in (0, 1):
            if f"[Scheduler 0] Kernel {kernel_id} has completed" not in run.log:
                run.fail(f"expected kernel {kernel_id}'s completion line")
    return case


def second_kernel_runs_after_one_that_ends_at_once(t):
    """Kernel 0 is one work-item that makes no call, so it ends in the cycle it is handed to the core and leaves the
    core idle; kernel 1 was launched behind it with no DEVICE_SYNC between, and is replayed all the same."""
    homes = [kernel(t, 0, items=1, defines=("NO_CALLS",)), kernel(t, 1)]
    run = session(t, t.machine(), [launch(homes[0], 0), launch(homes[1], 1)], timeout=60).expect_ok()
    if sorted(c.cycles for c in run.computes()) != [644] * 2:
        run.fail(f"expected kernel 1's two computes of 644 cycles, got {sorted(c.cycles for c in run.computes())}")


def unit_table_flag_refused(t):
    home = kernel(t, 0)
    session(t, t.machine(), [launch(home, 0), SYNC], "--unit_table", home / "tog_units.tsv").expect_refused(
        "--unit_table is keyed by one TOG's tile_ids", exit=1)


def kernel_without_unit_table_refused(t):
    session(t, t.machine(), [launch(kernel(t, 0, table=False), 0), SYNC]).expect_refused(
        "no tog_units.tsv next to", exit=1)


def kernel_without_tog_refused(t):
    session(t, t.machine(), [launch(kernel(t, 0, tog=False), 0), SYNC]).expect_refused("no tog.so next to", exit=1)


def copy(t, kernel_id):
    """<work dir>/k<id>/: a kernel that copies and computes nothing, with an empty tog_units.tsv."""
    return kernel(t, kernel_id, table=[], defines=("NO_COMPUTE",))


def copy_kernel_first_runs(t):
    """A kernel with no computes, whose unit table is empty, launched before kernel 1: its table names no unit, so
    kernel 1's names the array unit, and kernel 1's two computes run 644 cycles, as when kernel 1 is launched first."""
    for order in ((copy(t, 0), kernel(t, 1)), (kernel(t, 1), copy(t, 0))):
        run = session(t, t.machine(), [launch(home, k) for k, home in enumerate(order)] + [SYNC]).expect_ok()
        if sorted(c.cycles for c in run.computes()) != [644, 644]:
            run.fail(f"expected kernel 1's two computes of 644 cycles, got {sorted(c.cycles for c in run.computes())}")


def second_kernel_adds_a_unit(t):
    """Kernel 1's table names XLU, which kernel 0's does not: the unit joins the table and is reported."""
    other = units(1) + [(0, "XLU", "elements", 1, 0)]
    run = session(t, t.machine(), [launch(kernel(t, 0), 0), SYNC, launch(kernel(t, 1, table=other), 1), SYNC])
    run.expect_ok()
    ran = sorted(c.cycles for c in run.computes())
    if ran != [472, 472, 644, 644] or "Core [0] : XLU utilization(%)" not in run.log:
        run.fail(f"expected computes of 472, 472, 644, 644 cycles and an XLU utilization line, got {ran}")


def array_unit_misspelled_after_an_empty_table_refused(t):
    """The first table with rows names the array unit by another name: refused as a first table would be."""
    misspelled = units(1, array="TPUV6_MXU")
    session(t, t.machine(), [launch(copy(t, 0), 0), launch(kernel(t, 1, table=misspelled), 1), SYNC]).expect_refused(
        "systolic_array_split: the unit table has no array unit TPUV6E_MXU", exit=1)


def restated_unit_disagrees_refused(t):
    """Kernel 1's table states the array unit with another capacity than kernel 0's."""
    other = [(0, "TPUV6E_MXU", "rows", 2, 512), (0, "Vpu", "cycles", 1, 100)]
    session(t, t.machine(), [launch(kernel(t, 0), 0), SYNC, launch(kernel(t, 1, table=other), 1), SYNC]).expect_refused(
        "unit TPUV6E_MXU disagrees with line", exit=1)


CASES = [
    Case("two_kernels_one_after_the_other", two_kernels_one_after_the_other),
    Case("two_partitions_side_by_side", two_partitions_side_by_side),
    *[Case(f"second_kernel_runs.after_{items}_items.{cores}_cores", second_kernel_runs(items, cores))
      for items in (1, 2, 3, 4, 5) for cores in (1, 2)],
    Case("second_kernel_runs.after_one_that_ends_at_once", second_kernel_runs_after_one_that_ends_at_once),
    Case("refused.unit_table_flag", unit_table_flag_refused),
    Case("refused.kernel_without_unit_table", kernel_without_unit_table_refused),
    Case("refused.kernel_without_tog", kernel_without_tog_refused),
    Case("copy_kernel_first_runs", copy_kernel_first_runs),
    Case("second_kernel_adds_a_unit", second_kernel_adds_a_unit),
    Case("refused.array_unit_misspelled_after_an_empty_table", array_unit_misspelled_after_an_empty_table_refused),
    Case("refused.restated_unit_disagrees", restated_unit_disagrees_refused),
]
