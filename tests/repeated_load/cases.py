"""A load of what its core's buffer already holds is not read again, in both modes; a load of anything else is. What
is held stays in the spad, and a work-item that needs more of it at once than a core has is refused as before."""
from collections import Counter

from togtest import MIB, Case, write_table

TOG = "weights_tog.cpp"
TWO_CORES = "systolic_ws_128x128_c2_simple_noc_tpuv3.yml"
KB = 1024
ITEMS = 4
WEIGHTS, SMALL_WEIGHTS, INPUT = MIB, 64 * KB, 4 * KB
TENSORS = (64 * MIB,) * 3
TIMEOUT_S = 60              # a TOG that is not refused and does not fit never ends: fail on that instead


def tog(t, defines, into=""):
    return t.tog(TOG, defines, tensors=TENSORS, into=into)


def replay(t, *defines, spad_kb=None, base=None):
    """--tog_so: the TOG with `defines` on tpuv6e (or the config `base`) without the array split, every tile 200
    cycles; `spad_kb` sets core_spad_size_kb."""
    keys = dict(systolic_array_split="none")
    if spad_kb is not None:
        keys["core_spad_size_kb"] = spad_kb
    machine = t.machine(base, **keys) if base else t.machine(**keys)
    return t.simulate(tog(t, defines), machine, cycles=[200], timeout=TIMEOUT_S)


def session(t, launches, *defines, spad_kb=None, base=None):
    """--models_list: the TOG launched `launches` times, each from a directory of its own, then one DEVICE_SYNC."""
    stream = t.work_dir / "commands.txt"
    with stream.open("w") as commands:
        for i in range(launches):
            home = tog(t, defines, into=f"k{i}").parent
            write_table(home / "tog_cycles.tsv", [(200, 0)])
            commands.write(f"LAUNCH_KERNEL,{i},0,0,{home}/kernel,{home}/attributes,0\n")
        commands.write("DEVICE_SYNC,0,0,0,-,-,0\n")
    keys = dict(systolic_array_split="none")
    if spad_kb is not None:
        keys["core_spad_size_kb"] = spad_kb
    machine = t.machine(base, **keys) if base else t.machine(**keys)
    return t.execute("Simulator", ["--config", machine, "--models_list", stream, "--log_level", "trace"], TIMEOUT_S)


def loads(run, tag):
    """{(core, bytes): count} of the loads with a `tag` trace line: INST_ISSUED is read from DRAM, INST_SKIP is not."""
    return Counter((e.core, int(e.fields["size"].strip("[]"))) for e in run.events()
                   if e.tag == tag and e.opcode == "MOVIN")


def expect_loads(run, read, found, inst_count):
    """The run finished having read `read` from DRAM and found `found` held, each {(core, bytes): count}, and every
    core's MOVIN line reads `inst_count` (its loads, those found held among them)."""
    run.expect_ok()
    if loads(run, "INST_ISSUED") != Counter(read) or loads(run, "INST_SKIP") != Counter(found):
        run.fail(f"expected the loads {dict(read)} read and {dict(found)} found held, as (core, bytes): count; got "
                 f"{dict(loads(run, 'INST_ISSUED'))} read and {dict(loads(run, 'INST_SKIP'))} found held")
    lines = [line.split(" : ")[1] for line in run.log.splitlines() if "MOVIN" in line and "inst_count" in line]
    if lines != inst_count:
        run.fail(f"expected the MOVIN lines {inst_count}, got {lines}")
    return run


def weights_read_once(t):
    """Four work-items load the same 1 MiB of weights: the first reads it, the other three find it, and each reads
    its own 4 KB of input. 1 MiB + 16 KB are read where v1.3.0 read 4 MiB + 16 KB, and the total is 1069, not 1073."""
    run = replay(t, f"W_BYTES={WEIGHTS}")
    expect_loads(run, {(0, WEIGHTS): 1, (0, INPUT): ITEMS}, {(0, WEIGHTS): ITEMS - 1},
                 ["MOVIN    inst_count: 8, skipped inst_count: 3"]).expect_total_cycles(1069)
    if sum(size * n for (_, size), n in loads(run, "INST_ISSUED").items()) != WEIGHTS + ITEMS * INPUT:
        run.fail("expected 1 MiB + 16 KB read from DRAM")


def nothing_found(define, total, weights_loads=ITEMS):
    """With `define` no load finds its data held: all `weights_loads` loads of 1 MiB are read, to `total`."""
    def case(t):
        expect_loads(replay(t, f"W_BYTES={WEIGHTS}", define), {(0, WEIGHTS): weights_loads, (0, INPUT): ITEMS}, {},
                     [f"MOVIN    inst_count: {weights_loads + ITEMS}"]).expect_total_cycles(total)
    return case


def twice_in_a_work_item(t):
    """TWICE: each work-item loads the weights twice on one semaphore. Seven of the eight loads find them, and the
    barrier still expects the elements of both."""
    expect_loads(replay(t, f"W_BYTES={WEIGHTS}", "TWICE"), {(0, WEIGHTS): 1, (0, INPUT): ITEMS},
                 {(0, WEIGHTS): 2 * ITEMS - 1}, ["MOVIN    inst_count: 12, skipped inst_count: 7"])


def session_each_launch_reads_once(t):
    """The kernel launched twice: what the first launch left is not the second's, which reads the weights once more."""
    expect_loads(session(t, 2, f"W_BYTES={WEIGHTS}"), {(0, WEIGHTS): 2, (0, INPUT): 2 * ITEMS},
                 {(0, WEIGHTS): 2 * (ITEMS - 1)}, ["MOVIN    inst_count: 16, skipped inst_count: 6"])


def each_core_reads_once(t):
    """Two cores in one partition take the work-items in turn: each core reads the weights once into its own spad
    and finds them once."""
    expect_loads(replay(t, f"W_BYTES={WEIGHTS}", base=TWO_CORES),
                 {(0, WEIGHTS): 1, (1, WEIGHTS): 1, (0, INPUT): 2, (1, INPUT): 2}, {(0, WEIGHTS): 1, (1, WEIGHTS): 1},
                 ["MOVIN    inst_count: 4, skipped inst_count: 1"] * 2)


def two_cores_free_what_they_hold(t):
    """Two cores of 72 KB, the kernel (64 KB weights, 72 KB at once) launched twice: the second launch finds each
    spad free of the first's weights and ends, at the total of v1.3.0."""
    expect_loads(session(t, 2, spad_kb=72, base=TWO_CORES),
                 {(0, SMALL_WEIGHTS): 2, (1, SMALL_WEIGHTS): 2, (0, INPUT): 4, (1, INPUT): 4},
                 {(0, SMALL_WEIGHTS): 2, (1, SMALL_WEIGHTS): 2},
                 ["MOVIN    inst_count: 8, skipped inst_count: 2"] * 2).expect_total_cycles(1160)


def refusal(item, needed, spad_kb):
    return (f"work-item {item} (key {10 + item}) needs {needed} bytes in the spad at once, and a core's spad holds "
            f"{spad_kb * KB} (core_spad_size_kb: {spad_kb})")


def runs_in(spad_kb, total, *defines):
    """The TOG with 64 KB weights and `defines` ends at `total` on a spad of `spad_kb` KB, one work-item at a time."""
    def case(t):
        run = replay(t, *defines, spad_kb=spad_kb).expect_total_cycles(total)
        if [n for _, n in run.scheduled()] != [1] * ITEMS:
            run.fail(f"expected {ITEMS} work-items scheduled one at a time, got {run.scheduled()}")
    return case


def refused_in(spad_kb, item, needed, *defines):
    """The TOG with 64 KB weights and `defines` is refused on `spad_kb` KB for work-item `item` needing `needed`
    bytes, before anything is simulated."""
    def case(t):
        run = replay(t, *defines, spad_kb=spad_kb)
        run.expect_refused(f"TOG {t.work_dir / 'tog.so'}: {refusal(item, needed, spad_kb)}", exit=1)
        if run.events():
            run.fail("expected the refusal before anything was simulated, found instruction trace lines")
    return case


def session_held_refused(t):
    """--models_list: the launch of the kernel that holds 72 KB at once is refused on 71 KB, as the command's error."""
    run = session(t, 1, spad_kb=71)
    run.expect_refused(f"TOG {t.work_dir / 'k0' / 'tog.so'}: {refusal(0, 72 * KB, 71)}", exit=1)
    if "[TOGSim] Error processing command 0 (type: LAUNCH_KERNEL)" not in run.said()[0] or run.events():
        run.fail("expected the refusal as the error of command 0, before anything was simulated")


def measured_run_traces_as_unmeasured(t):
    """The TOG whose weights are found held, with a spad it never fills (measured while loaded) and with no limit
    (not measured): the same instruction trace, line for line, the three loads found held among it."""
    runs = [replay(t, spad_kb=spad_kb).expect_ok() for spad_kb in (1024, 0)]
    traces = [[(e.cycle, e.core, e.tag, e.inst, e.opcode, e.fields) for e in run.events()] for run in runs]
    if not traces[0] or traces[0] != traces[1]:
        differ = sum(a != b for a, b in zip(*traces)) + abs(len(traces[0]) - len(traces[1]))
        runs[0].fail(f"expected one instruction trace of the measured and the unmeasured run, got {len(traces[0])} "
                     f"and {len(traces[1])} lines, {differ} differing")
    if sum(loads(runs[0], "INST_SKIP").values()) != ITEMS - 1:
        runs[0].fail(f"expected {ITEMS - 1} loads found held, got {dict(loads(runs[0], 'INST_SKIP'))}")


TAIL, LAST = "TAIL_BYTES=8192", "LAST_A_BYTES=8192"
CASES = [
    Case("weights_read_once", weights_read_once),
    Case("read.other_elements", nothing_found("MOVING", 1069)),
    Case("read.synchronous_load", nothing_found("SYNC", 1073)),
    Case("read.after_a_store_to_the_tensor", nothing_found("STORED", 1073)),
    Case("read.after_a_compute_wrote_the_buffer", nothing_found("WRITTEN", 1073)),
    Case("read.after_another_load_into_the_buffer", nothing_found("OTHER", 1873, 2 * ITEMS)),
    Case("twice_in_a_work_item", twice_in_a_work_item),
    Case("session.each_launch_reads_once", session_each_launch_reads_once),
    Case("two_cores.each_core_reads_once", each_core_reads_once),
    Case("two_cores.session_frees_what_it_holds", two_cores_free_what_they_hold),
    Case("held.fits_exactly_runs", runs_in(72, 1679)),
    Case("held.refused.one_kb_short", refused_in(71, 0, 72 * KB)),
    Case("held.synchronous_loads_free_their_own", runs_in(72, 1783, "SYNC")),
    Case("held.delays_a_load_that_does_not_fit_beside_it", runs_in(72, 2826, "SIDE_BYTES=8192")),
    Case("held.own_weights_delay_no_load", runs_in(72, 2469, "MOVING", "SIDE_BYTES=8192")),
    Case("held.loaded_twice_is_counted_once", runs_in(72, 1683, "TWICE")),
    Case("held.refused.loaded_twice_one_kb_short", refused_in(71, 0, 72 * KB, "TWICE")),
    Case("held.later_work_item_fits_exactly", runs_in(80, 1679, LAST)),
    Case("held.refused.later_work_item", refused_in(79, 3, 80 * KB, LAST)),
    Case("held.stays_after_its_reader_runs", runs_in(76, 2483, TAIL)),
    Case("held.refused.stays_after_its_reader", refused_in(75, 0, 76 * KB, TAIL)),
    Case("held.is_there_when_a_work_item_starts_runs", runs_in(76, 2483, "MOVING", TAIL)),
    Case("held.refused.is_there_when_a_work_item_starts", refused_in(75, 0, 76 * KB, "MOVING", TAIL)),
    Case("session.held.refused", session_held_refused),
    Case("measured_run_traces_as_unmeasured", measured_run_traces_as_unmeasured),
]
