"""A work-item that needs more of the spad at once than a core has is refused while its TOG is loaded, in both modes;
one that fits exactly runs, and so does one that names more bytes than the spad and never holds them together."""
from togtest import MIB, Case, write_table

TOG = "held_at_once_tog.cpp"
KB = 1024
HELD = 2 * 16 * KB          # what a work-item holds beside its big buffer: the 16 KB input and the 16 KB output
TIMEOUT_S = 60              # a TOG that is not refused does not finish: fail on that instead of waiting it out


def refusal(item, needed, spad_kb):
    """The refusal of work-item `item` (key 10 + item) needing `needed` bytes of a `spad_kb` KB spad, after the
    TOG's path."""
    return (f"work-item {item} (key {10 + item}) needs {needed} bytes in the spad at once, and a core's spad holds "
            f"{spad_kb * KB} (core_spad_size_kb: {spad_kb})")


def tog(t, big, defines, into=""):
    return t.tog(TOG, (f"BIG_BYTES={big}", *defines), tensors=(MIB, MIB, MIB), into=into)


def replay(t, big, spad_kb, *defines):
    """--tog_so: the TOG with a big buffer of `big` bytes on tpuv6e with a `spad_kb` KB spad (None: the config
    has no core_spad_size_kb line), every tile 200 cycles."""
    machine = t.machine(core_spad_size_kb=spad_kb, systolic_array_split="none")
    if spad_kb is None:
        machine.write_text("".join(line for line in machine.read_text().splitlines(keepends=True)
                                   if not line.startswith("core_spad_size_kb:")))
    return t.simulate(tog(t, big, defines), machine, cycles=[200], timeout=TIMEOUT_S)


def session(t, spad_kb, *bigs):
    """--models_list: one LAUNCH_KERNEL per entry of `bigs`, the same TOG with that big buffer from <work dir>/k<i>/,
    then one DEVICE_SYNC."""
    stream = t.work_dir / "commands.txt"
    with stream.open("w") as commands:
        for i, big in enumerate(bigs):
            home = tog(t, big, (), into=f"k{i}").parent
            write_table(home / "tog_cycles.tsv", [(200, 0)])
            commands.write(f"LAUNCH_KERNEL,{i},0,0,{home}/kernel,{home}/attributes,0\n")
        commands.write("DEVICE_SYNC,0,0,0,-,-,0\n")
    machine = t.machine(core_spad_size_kb=spad_kb, systolic_array_split="none")
    return t.execute("Simulator", ["--config", machine, "--models_list", stream, "--log_level", "trace"], TIMEOUT_S)


def expect_refused_before_the_run(run, saying, so):
    """The run was refused with exit 1 by a line that names the TOG `so` and says `saying`, and no instruction was
    traced."""
    run.expect_refused(f"TOG {so}: {saying}", exit=1)
    if run.events():
        run.fail("expected the refusal before anything was simulated, found instruction trace lines")


def expect_both_ran(run, total):
    """The run finished with `total` cycles, its two work-items scheduled one at a time."""
    run.expect_total_cycles(total)
    if [n for _, n in run.scheduled()] != [1, 1]:
        run.fail(f"expected two work-items scheduled one at a time, got {run.scheduled()}")


def fits_exactly_runs(t):
    """32 KB + 16384 bytes in a 48 KB spad: each work-item fills it to the byte, and the second waits for the first."""
    expect_both_ran(replay(t, 16 * KB, 48), 623)


def one_byte_over_refused(t):
    """One byte more in the big buffer: no schedule issues the compute, and the TOG is refused, not left running."""
    expect_refused_before_the_run(replay(t, 16 * KB + 1, 48), refusal(0, HELD + 16 * KB + 1, 48), t.work_dir / "tog.so")


def one_kb_short_refused(t):
    """The TOG that fills 48 KB against a 47 KB spad."""
    expect_refused_before_the_run(replay(t, 16 * KB, 47), refusal(0, HELD + 16 * KB, 47), t.work_dir / "tog.so")


def later_work_item_refused(t):
    """Work-item 0 loads no big buffer and fits; the refusal names work-item 1 and its key."""
    expect_refused_before_the_run(replay(t, 16 * KB + 1, 48, "SMALL_FIRST"), refusal(1, HELD + 16 * KB + 1, 48),
                                  t.work_dir / "tog.so")


def session_fits_exactly_runs(t):
    expect_both_ran(session(t, 48, 16 * KB), 623)


def expect_launch_refused(run, command, saying, so):
    """As expect_refused_before_the_run, the line being the error of LAUNCH_KERNEL command `command`."""
    expect_refused_before_the_run(run, saying, so)
    if f"[TOGSim] Error processing command {command} (type: LAUNCH_KERNEL)" not in run.said()[0]:
        run.fail(f"expected the refusal as the error of command {command}, the one that launched the kernel")


def session_one_byte_over_refused(t):
    """The launch is refused: the line is the command's, and names the kernel's tog.so."""
    expect_launch_refused(session(t, 48, 16 * KB + 1), 0, refusal(0, HELD + 16 * KB + 1, 48),
                          t.work_dir / "k0" / "tog.so")


def session_second_kernel_refused(t):
    """Kernel 0 fills the spad exactly and kernel 1 is one byte over: the session ends at kernel 1's launch."""
    expect_launch_refused(session(t, 48, 16 * KB, 16 * KB + 1), 1, refusal(0, HELD + 16 * KB + 1, 48),
                          t.work_dir / "k1" / "tog.so")


def names_more_than_it_holds_runs(t):
    """APART: 16 KB and 32 KB are named, 48 KB against a 32 KB spad, and each is consumed before the other is
    needed. The big load waits for the spad, and the run ends."""
    expect_both_ran(replay(t, 32 * KB, 32, "APART"), 977)


def apart_one_byte_over_refused(t):
    """APART with a big buffer one byte larger than the spad: its load alone does not fit."""
    expect_refused_before_the_run(replay(t, 32 * KB + 1, 32, "APART"), refusal(0, 32 * KB + 1, 32),
                                  t.work_dir / "tog.so")


def fill_not_waited_for_is_not_counted(t):
    """APART, REFILL: the 16 KB buffer is loaded twice and its compute waits for the second load alone, so the
    compute is counted 16 KB, not 32: the work-item needs its 20 KB load, and that is what a 16 KB spad refuses."""
    expect_refused_before_the_run(replay(t, 20 * KB, 16, "APART", "REFILL"), refusal(0, 20 * KB, 16),
                                  t.work_dir / "tog.so")


def version_left_open_runs(t):
    """APART, LEFT_OPEN: the TOG ends with a load nothing reads, so the measuring leaves a buffer version open. The
    run numbers its versions from the start all the same: every load is freed, and the run ends at its total."""
    expect_both_ran(replay(t, 32 * KB, 32, "APART", "LEFT_OPEN"), 977)


def no_spad_size_refuses_nothing(t):
    """core_spad_size_kb: 0 is a spad without a limit, and so is a config without the key: the TOG that 48 KB
    refuses runs, to one total."""
    runs = [replay(t, 16 * KB + 1, spad_kb).expect_ok() for spad_kb in (0, None)]
    if runs[0].total_cycles() != runs[1].total_cycles():
        runs[1].fail(f"expected one total with core_spad_size_kb: 0 and without the key, got "
                     f"{runs[0].total_cycles()} and {runs[1].total_cycles()}")


def measured_run_traces_as_unmeasured(t):
    """The TOG that fills 48 KB, with a spad it never fills (measured while loaded) and with no limit (not measured):
    the same instruction trace, line for line: ids, subgraphs, work-item numbers, cycles."""
    runs = [replay(t, 16 * KB, spad_kb).expect_ok() for spad_kb in (1024, 0)]
    traces = [[(e.cycle, e.core, e.tag, e.inst, e.opcode, e.fields) for e in run.events()] for run in runs]
    if not traces[0] or traces[0] != traces[1]:
        differ = sum(a != b for a, b in zip(*traces)) + abs(len(traces[0]) - len(traces[1]))
        runs[0].fail(f"expected one instruction trace of the measured and the unmeasured run, got {len(traces[0])} "
                     f"and {len(traces[1])} lines, {differ} differing")


CASES = [
    Case("fits_exactly_runs", fits_exactly_runs),
    Case("refused.one_byte_over", one_byte_over_refused),
    Case("refused.one_kb_short", one_kb_short_refused),
    Case("refused.later_work_item", later_work_item_refused),
    Case("session.fits_exactly_runs", session_fits_exactly_runs),
    Case("session.refused.one_byte_over", session_one_byte_over_refused),
    Case("session.refused.second_kernel", session_second_kernel_refused),
    Case("names_more_than_it_holds_runs", names_more_than_it_holds_runs),
    Case("refused.apart_one_byte_over", apart_one_byte_over_refused),
    Case("refused.fill_not_waited_for_is_not_counted", fill_not_waited_for_is_not_counted),
    Case("version_left_open_runs", version_left_open_runs),
    Case("no_spad_size_refuses_nothing", no_spad_size_refuses_nothing),
    Case("measured_run_traces_as_unmeasured", measured_run_traces_as_unmeasured),
]
