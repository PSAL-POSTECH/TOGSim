"""A work-item that needs more of the spad at once than a core has is refused while its TOG is loaded, in both modes;
one that fits exactly runs, and so does one that names more bytes than the spad and never holds them together. The
spad is vpu_spad_size_kb_per_lane x vpu_num_lanes, and a core_spad_size_kb beside them that differs is refused."""
from togtest import MIB, REPO, Case, spad_alone, write_table

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


def replay(t, big, spad_kb, *defines, **keys):
    """--tog_so: the TOG with a big buffer of `big` bytes on tpuv6e with a `spad_kb` KB spad stated alone (None: the
    config states no spad), or with the machine `keys` instead, every tile 200 cycles."""
    machine = t.machine(**(keys or spad_alone(spad_kb)), systolic_array_split="none")
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
    machine = t.machine(**spad_alone(spad_kb), systolic_array_split="none")
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
    """APART, LEFT_OPEN: the TOG ends with a 16 KB load nothing reads, so the measuring leaves a buffer version open.
    The run numbers its versions from the start all the same and ends at its total. That load keeps its bytes until
    its work-item ends, so the 32 KB load beside it needs a 48 KB spad."""
    expect_both_ran(replay(t, 32 * KB, 48, "APART", "LEFT_OPEN"), 977)


def version_left_open_refused(t):
    """The same TOG in a 32 KB spad: the 16 KB load nothing reads may be issued before the 32 KB one, which then never
    is. The TOG is refused, not left to stop mid-run."""
    expect_refused_before_the_run(replay(t, 32 * KB, 32, "APART", "LEFT_OPEN"), refusal(1, 48 * KB, 32),
                                  t.work_dir / "tog.so")


def work_items(t, body, sizes, spad_kb, items=1, cores=1):
    """--tog_so: `items` work-items of work_item_tog.cpp, each the calls `body`, with buffers of `sizes` bytes (buffer
    0 first), on `cores` tpuv6e cores with a `spad_kb` KB spad each."""
    tog = t.tog("work_item_tog.cpp", (f"BODY={body}", f"SIZES={','.join(str(b) for b in sizes)}", f"ITEMS={items}"),
                tensors=(256 * MIB, 256 * MIB))
    machine = t.machine(**spad_alone(spad_kb), systolic_array_split="none", num_cores=cores)
    return t.simulate(tog, machine, cycles=[200, 0], timeout=TIMEOUT_S)


def issued(run, opcode, core=0):
    """The cycles at which the `opcode` instructions of `core` were issued, in order of issue."""
    return [e.cycle for e in run.events() if (e.tag, e.opcode, e.core) == ("INST_ISSUED", opcode, core)]


UNREAD = "L(1); L(2); C({2}, {}); L(3); C({3}, {})"     # buffer 1 is loaded and nothing reads it


def unread_load_kept_to_the_end_of_its_work_item(t):
    """Three 16 KB loads in a 32 KB spad, the first read by nothing: it keeps its bytes while its work-item runs, so
    the third load is issued only once the compute that frees the second is; and it leaves with its work-item, so the
    second work-item finds the spad empty and runs the same way."""
    run = work_items(t, UNREAD, (0, 16 * KB, 16 * KB, 16 * KB), 32, items=2)
    expect_both_ran(run, 1148)
    loads, computes = issued(run, "MOVIN"), issued(run, "COMP")
    if len(loads) != 6 or len(computes) != 4 or loads[2] < computes[0] or loads[5] < computes[2]:
        run.fail(f"expected each work-item's third load issued once its first compute was, got loads @{loads} and "
                 f"computes @{computes}")


def unread_load_larger_than_spad_refused(t):
    """A work-item that only loads 64 KB, as a kernel that copies data onto the chip does, in a 48 KB spad."""
    expect_refused_before_the_run(work_items(t, "L(1)", (0, 64 * KB), 48), refusal(0, 64 * KB, 48),
                                  t.work_dir / "tog.so")


def unread_loads_fit_together_or_are_refused(t):
    """Three loads of 16 KB that nothing reads are all in the spad when the last of them is issued: 48 KB holds them,
    47 KB does not, though no load waits for another."""
    work_items(t, "L(1); L(2); L(3)", (0, 16 * KB, 16 * KB, 16 * KB), 48).expect_total_cycles(176)
    expect_refused_before_the_run(work_items(t, "L(1); L(2); L(3)", (0, 16 * KB, 16 * KB, 16 * KB), 47),
                                  refusal(0, 48 * KB, 47), t.work_dir / "tog.so")


def unread_load_counts_for_what_does_not_wait_for_it(body):
    """A 16 KB load nothing reads beside a 16 KB load and the compute that reads it, stated in the order `body`: no
    order between the two loads is stated, so the unread one may be in the spad when the other is issued. 32 KB runs
    it, and 31 KB is refused, not left to stop mid-run."""
    def case(t):
        sizes = (0, 16 * KB, 16 * KB)
        work_items(t, body, sizes, 32, items=3).expect_ok()
        expect_refused_before_the_run(work_items(t, body, sizes, 31, items=3), refusal(0, 32 * KB, 31),
                                      t.work_dir / "tog.so")
    return case


def unread_output_does_not_count_for_what_it_waits_for(t):
    """Work-item 0 reads two 16 KB loads in a compute, and a compute that waits for that one writes 16 KB nothing
    reads (buffer 3, which work-item 1 loads; buffer 4 has no size and only orders the two). The output is filled
    after the loads are freed, so the work-item needs 32 KB, not 48."""
    body = "if (i == 0) { L(1); L(2); C({1, 2}, {4}); C({4}, {3}); } else { L(3); C({3}, {}); }"
    sizes = (0, 16 * KB, 16 * KB, 16 * KB, 0)
    work_items(t, body, sizes, 32, items=2).expect_ok()
    expect_refused_before_the_run(work_items(t, body, sizes, 31, items=2), refusal(0, 32 * KB, 31),
                                  t.work_dir / "tog.so")


def updated_in_place_and_never_read(t):
    """Buffer 1 (16 KB) is loaded and then only updated in place: nothing frees it before its work-item ends. The
    compute that updates it waits for its load and for a second 16 KB load: 32 KB at once, not 16."""
    body, sizes = "L(1); L(2); C({1, 2}, {1})", (0, 16 * KB, 16 * KB)
    work_items(t, body, sizes, 32).expect_ok()
    expect_refused_before_the_run(work_items(t, body, sizes, 31), refusal(0, 32 * KB, 31), t.work_dir / "tog.so")


def two_in_flight_only_when_two_of_what_one_needs_fit(t):
    """Each work-item writes buffer 3 (8 KB) with a compute and loads it again, and nothing reads it: it keeps 16 KB
    until it ends, twice its footprint. In a 16 KB spad one is in flight at a time and the run ends; two in flight
    would each take one fill and wait for the other's. 32 KB holds two of them."""
    for spad_kb, in_flight in ((16, 1), (31, 1), (32, 2)):
        run = work_items(t, "C({}, {3}); L(3)", (0, 0, 0, 8 * KB), spad_kb, items=3)
        run.expect_ok()
        if run.scheduled() != [(8 * KB, in_flight)] * 3:
            run.fail(f"expected three work-items of spad_footprint 8192, {in_flight} in flight, in {spad_kb} KB; got "
                     f"{run.scheduled()}")


def buffer_ids_are_per_work_item(body):
    """Six work-items on two cores with a 16 KB spad each, the even ones `body` and the odd ones one compute that
    reads buffer 1 (8 KB), an id the even ones load: a work-item's buffer is no version of another's, so every
    load is freed on the core that took it and the run ends as with odd work-items that name no buffer."""
    def case(t):
        totals = []
        for odd in ("C({1}, {})", "C({}, {})"):
            run = work_items(t, f"if (i % 2 == 0) {{ {body}; }} else {odd}", (0, 8 * KB), 16, items=6, cores=2)
            totals.append(run.expect_ok().total_cycles())
        if totals[0] != totals[1]:
            run.fail(f"expected one total with and without the odd work-items' read of buffer 1, got {totals}")
    return case


def freed_when_every_reader_is_issued(t):
    """Buffer 1 (16 KB) has two readers. The one the TOG states first also waits for a 64 KB load; the one it states
    last is issued first and frees nothing, since the other still reads the buffer: the 16 KB load of buffer 3, which
    an 88 KB spad takes only without buffer 1, is issued once the first reader is."""
    run = work_items(t, "L(1); L(2); C({1, 2}, {}); C({1}, {}); L(3); C({3}, {})",
                     (0, 16 * KB, 64 * KB, 16 * KB), 88)
    run.expect_total_cycles(745)
    loads, computes = issued(run, "MOVIN"), issued(run, "COMP")
    if len(loads) != 3 or len(computes) != 3 or not computes[0] < computes[1] <= loads[2]:
        run.fail(f"expected the reader of buffer 1 alone issued first, then the reader of 1 and 2, then the load of "
                 f"buffer 3; got loads @{loads} and computes @{computes}")


FOUR_READERS = "L(1); L(2); C({1, 2}, {}); C({1, 2}, {}); C({1, 2}, {}); C({1}, {})"


def every_reader_holds_its_buffer(t):
    """Three computes read 64 KB and 512 KB together and a fourth, stated last, the 64 KB alone: each of the three
    is issued with both in the spad, whichever reader the TOG states last. 576 KB holds that and 575 KB does not."""
    sizes = (0, 64 * KB, 512 * KB)
    work_items(t, FOUR_READERS, sizes, 576).expect_total_cycles(1351)
    expect_refused_before_the_run(work_items(t, FOUR_READERS, sizes, 575), refusal(0, 576 * KB, 575),
                                  t.work_dir / "tog.so")


def no_spad_size_refuses_nothing(t):
    """core_spad_size_kb: 0 is a spad without a limit, and so is a config without that key and without
    vpu_spad_size_kb_per_lane: the TOG that 48 KB refuses runs, to one total."""
    runs = [replay(t, 16 * KB + 1, spad_kb).expect_ok() for spad_kb in (0, None)]
    if runs[0].total_cycles() != runs[1].total_cycles():
        runs[1].fail(f"expected one total with core_spad_size_kb: 0 and without the key, got "
                     f"{runs[0].total_cycles()} and {runs[1].total_cycles()}")


LANES = dict(vpu_spad_size_kb_per_lane=16, vpu_num_lanes=3)   # a 48 KB spad, stated by the lanes alone


def lanes_fit_exactly_runs(t):
    """16 KB on each of 3 lanes is a 48 KB spad: the TOG that fills 48 KB runs to the total of fits_exactly_runs."""
    expect_both_ran(replay(t, 16 * KB, None, **LANES), 623)


def lanes_one_byte_over_refused(t):
    """The same spad of 16 KB x 3 lanes, without core_spad_size_kb, refuses the TOG one byte over 48 KB."""
    expect_refused_before_the_run(replay(t, 16 * KB + 1, None, **LANES), refusal(0, HELD + 16 * KB + 1, 48),
                                  t.work_dir / "tog.so")


def lanes_restated_runs(t):
    """core_spad_size_kb: 48 beside 16 KB x 3 lanes restates the spad, and the run is fits_exactly_runs."""
    expect_both_ran(replay(t, 16 * KB, None, core_spad_size_kb=48, **LANES), 623)


def stated_twice_disagreeing_refused(t):
    """core_spad_size_kb: 48 on tpuv6e as shipped, whose 1024 KB x 128 lanes are 131072 KB: the config is refused
    before anything runs, naming both."""
    run = replay(t, 16 * KB, None, core_spad_size_kb=48)
    run.expect_refused("core_spad_size_kb: 48 is not vpu_spad_size_kb_per_lane x vpu_num_lanes = 1024 x 128 = 131072",
                       exit=134)


TPUV6E = REPO / "configs" / "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"


def tpuv6e_spad_is_its_lanes(t):
    """configs/..._tpuv6e.yml as shipped, with its TPUV6_MXU unit table, states 1024 KB on each of 128 lanes and no
    core_spad_size_kb: the TOG one byte over 131072 KB is refused, naming that spad."""
    big = 131072 * KB - HELD + 1
    so = t.tog(TOG, (f"BIG_BYTES={big}",), tensors=(MIB, 256 * MIB, MIB))
    run = t.simulate(so, TPUV6E, cycles=[200], units=[(0, "TPUV6_MXU", "rows", 1, 0), (0, "Vpu", "cycles", 1, 200)],
                     timeout=TIMEOUT_S)
    expect_refused_before_the_run(run, refusal(0, HELD + big, 131072), so)


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
    Case("refused.version_left_open", version_left_open_refused),
    Case("unread_load.kept_to_the_end_of_its_work_item", unread_load_kept_to_the_end_of_its_work_item),
    Case("refused.unread_load_larger_than_spad", unread_load_larger_than_spad_refused),
    Case("unread_load.fit_together_or_are_refused", unread_loads_fit_together_or_are_refused),
    Case("unread_load.counts_for_what_does_not_wait_for_it.stated_first",
         unread_load_counts_for_what_does_not_wait_for_it("L(1); L(2); C({2}, {})")),
    Case("unread_load.counts_for_what_does_not_wait_for_it.stated_last",
         unread_load_counts_for_what_does_not_wait_for_it("L(2); C({2}, {}); L(1)")),
    Case("unread_output_does_not_count_for_what_it_waits_for", unread_output_does_not_count_for_what_it_waits_for),
    Case("refused.updated_in_place_and_never_read", updated_in_place_and_never_read),
    Case("two_in_flight_only_when_two_of_what_one_needs_fit", two_in_flight_only_when_two_of_what_one_needs_fit),
    Case("buffer_ids_are_per_work_item.loaded_and_read", buffer_ids_are_per_work_item("L(1); C({1}, {})")),
    Case("buffer_ids_are_per_work_item.loaded_only", buffer_ids_are_per_work_item("L(1)")),
    Case("freed_when_every_reader_is_issued", freed_when_every_reader_is_issued),
    Case("refused.every_reader_holds_its_buffer", every_reader_holds_its_buffer),
    Case("no_spad_size_refuses_nothing", no_spad_size_refuses_nothing),
    Case("measured_run_traces_as_unmeasured", measured_run_traces_as_unmeasured),
    Case("lanes.fits_exactly_runs", lanes_fit_exactly_runs),
    Case("lanes.refused.one_byte_over", lanes_one_byte_over_refused),
    Case("lanes.restated_runs", lanes_restated_runs),
    Case("refused.stated_twice_disagreeing", stated_twice_disagreeing_refused),
    Case("lanes.refused.tpuv6e_one_byte_over", tpuv6e_spad_is_its_lanes),
]
