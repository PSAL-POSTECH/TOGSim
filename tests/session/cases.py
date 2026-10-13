"""--models_list: a command stream launches kernels, each from a directory holding tog.so and its tables."""
from togtest import MIB, REPO, Case, spad_alone, write_table

PARTITIONED = REPO / "configs" / "systolic_ws_128x128_c2_simple_noc_tpuv3_partition.yml"
KERNELS = {0: dict(cycles=600, rows=256), 1: dict(cycles=900, rows=512)}


def units(kernel_id, array="TPUV6_MXU"):
    """KERNELS[id]'s unit table: its rows of the machine's array unit `array`, and 100 cycles of the Vpu."""
    return [(0, array, "rows", 1, KERNELS[kernel_id]["rows"]), (0, "Vpu", "cycles", 1, 100)]


def kernel(t, kernel_id, table=True, tog=True, array="TPUV6_MXU"):
    """<work dir>/k<id>/: tog.so, tog_tensors.txt, tog_cycles.tsv and tog_units.tsv of KERNELS[id] naming `array`
    (`table` may replace the unit table or, False, leave it out; `tog` False leaves tog.so out). Returns the
    directory."""
    home = t.tog("kernel_tog.cpp", tensors=(MIB, MIB), into=f"k{kernel_id}").parent
    write_table(home / "tog_cycles.tsv", [(KERNELS[kernel_id]["cycles"], 0)])
    if table:
        write_table(home / "tog_units.tsv", units(kernel_id, array) if table is True else table)
    if not tog:
        (home / "tog.so").unlink()
    return home


def session(t, config, lines, *more):
    """Run the command stream `lines` on `config` at --log_level trace; `more` is appended to the command line."""
    stream = t.work_dir / "commands.txt"
    stream.write_text("".join(f"{line}\n" for line in lines))
    return t.execute("Simulator", ["--config", config, "--models_list", stream, "--log_level", "trace", *more])


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
    """On the two-partition config (core 0 / core 1, 2 arrays, 128 lanes) kernel 0 goes to partition 0 and kernel 1
    to partition 1 with no DEVICE_SYNC between: core 0 runs only kernel 0's computes (472 = 600 - 256 + 256/2),
    core 1 only kernel 1's (644), and the two cores' compute windows overlap."""
    homes = [kernel(t, 0, array="SYSTOLIC"), kernel(t, 1, array="SYSTOLIC")]
    run = session(t, PARTITIONED, [launch(homes[0], 0, partition=0), launch(homes[1], 1, partition=1)]).expect_ok()
    ran = run.computes()
    by_core = {core: sorted(c.cycles for c in ran if c.core == core) for core in sorted({c.core for c in ran})}
    together = any(a.core != b.core and a.start < b.finish and b.start < a.finish for a in ran for b in ran)
    if by_core != {0: [472, 472], 1: [644, 644]} or not together:
        run.fail(f"expected core 0 to run two computes of 472 cycles and core 1 two of 644, overlapping; got "
                 f"{by_core}, cross-core overlap {together}")
    run.total_cycles()


def two_loads(t, kernel_id, first, second=0):
    """<work dir>/k<id>/: the two-loads TOG with loads of `first` and `second` bytes, and KERNELS[id]'s tables."""
    home = t.tog("two_loads_tog.cpp", (f"FIRST_BYTES={first}", f"SECOND_BYTES={second}"), tensors=(MIB,),
                 into=f"k{kernel_id}").parent
    write_table(home / "tog_cycles.tsv", [(KERNELS[kernel_id]["cycles"], 0)])
    write_table(home / "tog_units.tsv", units(kernel_id))
    return home


def kernels_free_their_own_spad_bytes(t):
    """Two kernels at once on one core with a 32 KB spad, each numbering its buffer versions from 0. Kernel 0 loads
    8 KB and then 28 KB, kernel 1 16 KB, which is taken while the 28 KB load waits. Kernel 0's compute frees
    kernel 0's 8 KB only: the 28 KB load is issued once kernel 1's compute has freed the 16 KB, not before."""
    homes = [two_loads(t, 0, 8 * 1024, 28 * 1024), two_loads(t, 1, 16 * 1024)]
    machine = t.machine(**spad_alone(32), systolic_array_split="none")
    run = session(t, machine, [launch(homes[0], 0), launch(homes[1], 1), SYNC]).expect_ok()
    issued = [e for e in run.events() if e.tag == "INST_ISSUED"]
    big = [e.cycle for e in issued if e.opcode == "MOVIN" and e.fields.get("size") == f"[{28 * 1024}]"]
    freed = [e.cycle for e in issued if e.opcode == "COMP" and e.fields["compute_cycle"] == str(KERNELS[1]["cycles"])]
    if len(big) != 1 or len(freed) != 1 or big[0] < freed[0]:
        run.fail(f"expected the 28 KB load issued once kernel 1's compute was, got the load @{big} and the compute "
                 f"@{freed}")


def unit_table_flag_refused(t):
    home = kernel(t, 0)
    session(t, t.machine(), [launch(home, 0), SYNC], "--unit_table", home / "tog_units.tsv").expect_refused(
        "--unit_table is keyed by one TOG's tile_ids", exit=1)


def kernel_without_unit_table_refused(t):
    session(t, t.machine(), [launch(kernel(t, 0, table=False), 0), SYNC]).expect_refused(
        "no tog_units.tsv next to", exit=1)


def kernel_without_tog_refused(t):
    session(t, t.machine(), [launch(kernel(t, 0, tog=False), 0), SYNC]).expect_refused("no tog.so next to", exit=1)


def second_kernel_with_another_unit_refused(t):
    """A unit the first kernel's table does not name is an error: the kernels share one machine's units."""
    other = units(1) + [(0, "XLU", "elements", 1, 0)]
    session(t, t.machine(), [launch(kernel(t, 0), 0), SYNC, launch(kernel(t, 1, table=other), 1), SYNC]).expect_refused(
        "unit XLU is not one the first kernel's table has", exit=1)


CASES = [
    Case("two_kernels_one_after_the_other", two_kernels_one_after_the_other),
    Case("two_partitions_side_by_side", two_partitions_side_by_side),
    Case("kernels_free_their_own_spad_bytes", kernels_free_their_own_spad_bytes),
    Case("refused.unit_table_flag", unit_table_flag_refused),
    Case("refused.kernel_without_unit_table", kernel_without_unit_table_refused),
    Case("refused.kernel_without_tog", kernel_without_tog_refused),
    Case("refused.second_kernel_with_another_unit", second_kernel_with_another_unit_refused),
]
