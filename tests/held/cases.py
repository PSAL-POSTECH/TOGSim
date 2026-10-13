"""A work-item may read a buffer that nothing in it wrote: data that is on the chip already. Its core holds the
buffer's stated bytes for the whole kernel, and the kernel is refused, admitted and scheduled as in a spad that much
smaller; a TOG that loads everything it reads is not touched by any of it."""
from togtest import MIB, Case, spad_alone, write_table

TOG = "../spad_capacity/work_item_tog.cpp"
KB = 1024
TIMEOUT_S = 30              # a kernel that waits for spad it never gets does not end: fail on that instead
HOLDS = "holds {held} bytes for the kernel: buffers {ids}, which its work-items read without having written them"


def sizes(**of):
    """The buffer table: `b<id>=<bytes>` for each sized buffer, 0 for the ids between."""
    table = [0] * (max(int(name[1:]) for name in of) + 1)
    for name, size in of.items():
        table[int(name[1:])] = size
    return table


def machine(t, spad_kb, cores=1):
    return t.machine(**spad_alone(spad_kb), systolic_array_split="none", num_cores=cores)


def kernel_dir(t, i, body, table, items):
    """<work dir>/k<i>/: tog.so of `items` work-items of `body` with the buffer sizes `table`, and its tables."""
    defines = (f"BODY={body}", f"SIZES={','.join(str(b) for b in table)}", f"ITEMS={items}")
    home = t.tog(TOG, defines, tensors=(256 * MIB, 256 * MIB), into=f"k{i}").parent
    write_table(home / "tog_cycles.tsv", [(200, 0), (50, 0)])
    return home


def replay(t, body, table, spad_kb, items=1, cores=1, session=False):
    """One kernel of `items` work-items of `body`, on `cores` tpuv6e cores of `spad_kb` KB of spad each; tile 0 takes
    200 cycles. With --tog_so, or `session`: launched by a command stream. Returns the run and the TOG's path."""
    home = kernel_dir(t, 0, body, table, items)
    if session:
        return launch(t, spad_kb, [home], cores=cores), home / "tog.so"
    return t.execute("Simulator", ["--config", machine(t, spad_kb, cores), "--tog_so", home / "tog.so",
                                   "--cycle_table", home / "tog_cycles.tsv", "--log_level", "trace"],
                     TIMEOUT_S), home / "tog.so"


def launch(t, spad_kb, homes, cores=1, sync_between=False):
    """--models_list: one LAUNCH_KERNEL per directory of `homes`, in order, all on partition 0, a DEVICE_SYNC after
    the last and, with `sync_between`, after each."""
    lines = []
    for i, home in enumerate(homes):
        lines.append(f"LAUNCH_KERNEL,{i},0,0,{home}/kernel,{home}/attributes,0")
        if sync_between or i == len(homes) - 1:
            lines.append("DEVICE_SYNC,0,0,0,-,-,0")
    stream = t.work_dir / "commands.txt"
    stream.write_text("".join(f"{line}\n" for line in lines))
    return t.execute("Simulator", ["--config", machine(t, spad_kb, cores), "--models_list", stream, "--log_level",
                                   "trace"], TIMEOUT_S)


def in_both_modes(name, case):
    """`case(t, session)` once with --tog_so and once in a session."""
    return [Case(name, lambda t: case(t, False)), Case(f"session.{name}", lambda t: case(t, True))]


def said_held(run, so, core, held, ids):
    """The info line that names what `core` holds for the kernel `so`, once."""
    line = f"[info] [TOGSim] TOG {so}: core {core} " + HOLDS.format(held=held, ids=ids)
    if run.log.count(line) != 1:
        run.fail(f'expected once the line "{line}"')


def expect_holds(run, want, insts=None):
    """The cores' SPAD_HOLD lines are `want`, [(core, bytes)], and each has its SPAD_HOLD_END of the same bytes
    after the core's last instruction finished (of the instructions `insts`, the holding kernel's, when given)."""
    taken, ended = run.core_lines("SPAD_HOLD"), run.core_lines("SPAD_HOLD_END")
    if sorted((core, f["bytes"]) for _, core, f in taken) != sorted(want) or \
            sorted((core, f["bytes"]) for _, core, f in ended) != sorted(want):
        run.fail(f"expected the holds (core, bytes) {sorted(want)}, each taken and ended once; got SPAD_HOLD {taken} "
                 f"and SPAD_HOLD_END {ended}")
    for cycle, core, _ in ended:
        last = max(e.cycle for e in run.events() if e.core == core and e.tag == "INST_FINISHED" and
                   (insts is None or e.inst in insts))
        if cycle < last:
            run.fail(f"expected core {core}'s hold to end once its last instruction finished (@{last}), got @{cycle}")


def refusal(item, needed, held, ids, spad_kb, core=0):
    """The refusal of work-item `item` needing `needed` bytes beside the `held` bytes of the buffers `ids`."""
    return (f"work-item {item} (key {10 + item}) needs {needed} bytes in the spad at once beside the {held} bytes "
            f"core {core} holds for the kernel (buffers {ids}, read without having been written), and a core's "
            f"spad holds {spad_kb * KB} (core_spad_size_kb: {spad_kb})")


def expect_refused(run, so, saying, command=None):
    """Refused with exit 1 before anything was simulated, by a line naming the TOG `so` and `saying`; in a session
    the line is the error of LAUNCH_KERNEL command `command`."""
    run.expect_refused(f"TOG {so}: {saying}", exit=1)
    if run.events():
        run.fail("expected the refusal before anything was simulated, found instruction trace lines")
    if command is not None and f"Error processing command {command} (type: LAUNCH_KERNEL)" not in run.said()[0]:
        run.fail(f"expected the refusal as the error of command {command}, the one that launched the kernel")


def issued(run, opcode, core=0):
    return [e.cycle for e in run.events() if (e.tag, e.opcode, e.core) == ("INST_ISSUED", opcode, core)]


# ---- what a held buffer is ---------------------------------------------------------------------------------------

def read_waits_for_nothing(t, session):
    """A compute that reads buffer 9 (64 KB), which nothing in its work-item wrote, and writes buffer 2 (16 KB),
    which is stored. It is issued as soon as its work-item is scheduled: the read waits for no load, and no load is
    issued. The core holds 64 KB from then to the end of the work-item, and the work-item's own footprint is buffer
    2. The same work-item with a load of buffer 9 before the compute takes the cycles of that load longer."""
    table = sizes(b2=16 * KB, b9=64 * KB)
    run, so = replay(t, "C({9}, {2}); S(2)", table, 1024, session=session)
    run.expect_total_cycles(300)
    said_held(run, so, 0, 64 * KB, "9")
    expect_holds(run, [(0, 64 * KB)])
    scheduled, hold = run.core_lines("TILE_SCHEDULED"), run.core_lines("SPAD_HOLD")
    if issued(run, "MOVIN") or issued(run, "COMP") != [scheduled[0][0] + 1] or hold[0][0] != scheduled[0][0]:
        run.fail(f"expected no load, the hold taken as the work-item is scheduled (@{scheduled[0][0]}) and the "
                 f"compute issued the cycle after; got loads @{issued(run, 'MOVIN')}, computes "
                 f"@{issued(run, 'COMP')}, the hold @{hold[0][0]}")
    if run.scheduled() != [(16 * KB, 2)]:
        run.fail(f"expected a spad_footprint of buffer 2 alone, 16384, got {run.scheduled()}")
    loaded, _ = replay(t, "L(9); C({9}, {2}); S(2)", table, 1024, session=session)
    loaded.expect_total_cycles(491)
    if "bytes for the kernel" in loaded.log or loaded.core_lines("SPAD_HOLD"):
        loaded.fail("expected no held buffer in the work-item that loads buffer 9 itself")


def occupies_its_stated_bytes(t, session):
    """A 16 KB load and a compute that reads it and held buffer 9 (64 KB) into a 16 KB output: 32 KB at once beside
    the 64 KB held. A 96 KB spad holds that to the byte; one of 95 KB refuses the kernel, naming both parts."""
    body, table = "L(1); C({1, 9}, {2}); S(2)", sizes(b1=16 * KB, b2=16 * KB, b9=64 * KB)
    run, _ = replay(t, body, table, 96, session=session)
    run.expect_total_cycles(440)
    expect_holds(run, [(0, 64 * KB)])
    run, so = replay(t, body, table, 95, session=session)
    expect_refused(run, so, refusal(0, 32 * KB, 64 * KB, "9", 95), 0 if session else None)


def held_alone_larger_than_the_spad_refused(t, session):
    """A work-item that fills nothing and reads 64 KB of held data, in 64 KB and in 63 KB."""
    replay(t, "C({9}, {})", sizes(b9=64 * KB), 64, session=session)[0].expect_ok()
    run, so = replay(t, "C({9}, {})", sizes(b9=64 * KB), 63, session=session)
    expect_refused(run, so, refusal(0, 0, 64 * KB, "9", 63), 0 if session else None)


def a_later_work_item_refused(t):
    """Work-item 0 fits beside the 16 KB its core holds for work-item 1's read; work-item 1 loads 16 KB more."""
    body, table = "L(1); if (i == 1) { L(2); C({1, 2, 9}, {}); } else C({1}, {})", sizes(b1=8 * KB, b2=16 * KB,
                                                                                       b9=16 * KB)
    replay(t, body, table, 40, items=2)[0].expect_ok()
    run, so = replay(t, body, table, 39, items=2)
    expect_refused(run, so, refusal(1, 24 * KB, 16 * KB, "9", 39))


def session_second_kernel_refused(t):
    """Kernel 0 loads all it reads and fits; kernel 1 reads held data and is one KB over: the session ends at
    kernel 1's launch, before kernel 0 is simulated."""
    homes = [kernel_dir(t, 0, "L(1); C({1}, {})", sizes(b1=16 * KB), 1),
             kernel_dir(t, 1, "L(1); C({1, 9}, {})", sizes(b1=16 * KB, b9=32 * KB), 1)]
    launch(t, 48, homes).expect_ok()
    expect_refused(launch(t, 47, homes), homes[1] / "tog.so", refusal(0, 16 * KB, 32 * KB, "9", 47), command=1)


def written_later_takes_no_more(t):
    """Buffer 9 (16 KB) is read first, then loaded, then read again: it is held, the load into it takes no bytes of
    its own (a 16 KB spad holds the kernel) and the second read waits for the load."""
    run, so = replay(t, "C({9}, {}); L(9); C({9}, {})", sizes(b9=16 * KB), 16)
    run.expect_ok()
    said_held(run, so, 0, 16 * KB, "9")
    expect_holds(run, [(0, 16 * KB)])
    load_done = [e.cycle for e in run.events() if (e.tag, e.opcode) == ("INST_FINISHED", "MOVIN")]
    if len(load_done) != 1 or len(issued(run, "COMP")) != 2 or issued(run, "COMP")[1] <= load_done[0]:
        run.fail(f"expected the second compute issued after the load finished (@{load_done}), got computes "
                 f"@{issued(run, 'COMP')}")
    if run.scheduled() != [(0, 2)]:
        run.fail(f"expected a spad_footprint of 0, the work-item naming only a held buffer; got {run.scheduled()}")


def in_place_first_is_the_work_items_own(t):
    """A compute that reads and writes buffer 3 (16 KB), which nothing wrote before, writes it: the buffer is the
    work-item's own, not held, and nothing is held for the kernel. As in TOGSim 1.3.0 a compute that reads and
    writes a buffer starts no version of it, so the work-item also runs in a spad smaller than the buffer."""
    for spad_kb, in_flight in ((32, 2), (15, 1)):
        run, _ = replay(t, "C({3}, {3}); S(3)", sizes(b3=16 * KB), spad_kb, items=2)
        run.expect_ok()
        if "bytes for the kernel" in run.log or run.core_lines("SPAD_HOLD") or \
                run.scheduled() != [(16 * KB, in_flight)] * 2:
            run.fail(f"expected no held buffer and two work-items of spad_footprint 16384, {in_flight} in flight, "
                     f"got {run.scheduled()} and holds {run.core_lines('SPAD_HOLD')}")


def held_by_what_reads_it(name, body, table, held, ids, filled):
    """`body` reads held buffers through something other than one read of a compute: its core holds `held` bytes,
    the kernel runs in a spad of exactly that and the `filled` bytes it fills, and is refused in one KB less."""
    def case(t):
        spad_kb = (held + filled) // KB
        run, so = replay(t, body, table, spad_kb)
        run.expect_ok()
        said_held(run, so, 0, held, ids)
        expect_holds(run, [(0, held)])
        run, so = replay(t, body, table, spad_kb - 1)
        expect_refused(run, so, refusal(0, filled, held, ids, spad_kb - 1))
    return Case(f"held_by.{name}", case)


def store_of_unsized_held_buffer_refused(t):
    """A store drains its buffer, held or not: a held buffer the table gives no size is refused as any DMA'd one."""
    run, _ = replay(t, "dma(TOGSIM_DMA_STORE, 2, {9}, {})", sizes(b2=16 * KB, b9=0), 1024)
    run.expect_refused("[TOGSim] spad buffer 9 is DMA'd but the TOG states no size for it (togsim_spad_buffer_bytes "
                       "has 10 entries)", exit=1)


def unsized_held_buffer_occupies_nothing(t):
    """Buffer 9 has size 0 and buffer 40 is past the table: a compute may read them, they are held and occupy
    nothing, so the kernel runs in a 1 KB spad and no hold is traced."""
    run, so = replay(t, "C({9, 40}, {})", sizes(b9=0), 1)
    run.expect_ok()
    said_held(run, so, 0, 0, "9, 40")
    if run.core_lines("SPAD_HOLD") or run.core_lines("SPAD_HOLD_END"):
        run.fail("expected no SPAD_HOLD line for a hold of 0 bytes")


# ---- what the core holds, and for how long ----------------------------------------------------------------------

def held_once_per_core(cores):
    """Four work-items, each a 4 KB load and a compute; only work-item 1 reads held buffer 9 (16 KB). The core that
    runs work-item 1 holds 16 KB once, from the first work-item it is handed to the end of its last: before the
    reader arrives and after it ends. With two cores that is core 1 alone."""
    def case(t, session):
        body = "L(1); if (i == 1) C({1, 9}, {}); else C({1}, {})"
        run, so = replay(t, body, sizes(b1=4 * KB, b9=16 * KB), 64, items=4, cores=cores, session=session)
        run.expect_ok()
        core = cores - 1
        said_held(run, so, core, 16 * KB, "9")
        expect_holds(run, [(core, 16 * KB)])
        first = min(cycle for cycle, c, _ in run.core_lines("TILE_SCHEDULED") if c == core)
        if run.core_lines("SPAD_HOLD")[0][0] != first:
            run.fail(f"expected the hold taken with the core's first work-item (@{first}), got "
                     f"{run.core_lines('SPAD_HOLD')}")
        if len(run.computes()) != 4:
            run.fail(f"expected four computes, got {len(run.computes())}")
    return case


def each_core_holds_its_own(t, session):
    """Four work-items on two cores, all reading held buffer 9 (16 KB): each core holds 16 KB once."""
    run, so = replay(t, "L(1); C({1, 9}, {})", sizes(b1=4 * KB, b9=16 * KB), 64, items=4, cores=2, session=session)
    run.expect_ok()
    for core in (0, 1):
        said_held(run, so, core, 16 * KB, "9")
    expect_holds(run, [(0, 16 * KB), (1, 16 * KB)])


def one_copy_for_work_items_in_flight_together(t):
    """Four work-items, two in flight at once and in no order, each a 12 KB load and a compute that reads it and
    held buffer 9 (16 KB). The held bytes are there once: 16 + 2 x 12 = 40 KB holds two in flight, and in 39 KB the
    core keeps one in flight, since two would not fit beside what it holds."""
    body, table = "L(1); C({1, 9}, {})", sizes(b1=12 * KB, b9=16 * KB)
    for spad_kb, at_once in ((40, 2), (39, 1)):
        run, _ = replay(t, body, table, spad_kb, items=4)
        run.expect_ok()
        expect_holds(run, [(0, 16 * KB)])
        if run.scheduled() != [(12 * KB, at_once)] * 4:
            run.fail(f"expected four work-items of spad_footprint 12288, {at_once} in flight in {spad_kb} KB; got "
                     f"{run.scheduled()}")


def own_and_held_under_one_id(t):
    """Work-item 0 loads buffer 1 (16 KB) and reads it; work-item 1 reads buffer 1 and wrote nothing: that is held
    data, another buffer than work-item 0's. The core holds 16 KB for it and work-item 0's load takes 16 KB more:
    32 KB runs, 31 KB is refused at work-item 0."""
    body, table = "if (i == 0) { L(1); C({1}, {}); } else C({1}, {})", sizes(b1=16 * KB)
    run, so = replay(t, body, table, 32, items=2)
    run.expect_ok()
    said_held(run, so, 0, 16 * KB, "1")
    expect_holds(run, [(0, 16 * KB)])
    run, so = replay(t, body, table, 31, items=2)
    expect_refused(run, so, refusal(0, 16 * KB, 16 * KB, "1", 31))


# ---- a kernel that holds H bytes runs as one that holds none in a spad H bytes smaller ------------------------------

def as_in_a_smaller_spad(room_kb, at_once, then=""):
    """Four work-items of two 12 KB loads and a compute into a 24 KB output that is stored, 48 KB each, and what
    `then` adds. With a read of held buffer 9 (64 KB) in the compute and `room_kb` + 64 KB of spad, the run is the
    one without that read in `room_kb` KB: the same number in flight (`at_once`), the same trace cycle for cycle,
    the same total."""
    def case(t):
        table = sizes(b1=12 * KB, b2=12 * KB, b3=24 * KB, b4=32 * KB, b9=64 * KB)
        held, _ = replay(t, f"L(1); L(2); C({{1, 2, 9}}, {{3}}); S(3){then}", table, room_kb + 64, items=4)
        plain, _ = replay(t, f"L(1); L(2); C({{1, 2}}, {{3}}); S(3){then}", table, room_kb, items=4)
        held.expect_ok(), plain.expect_ok()
        expect_holds(held, [(0, 64 * KB)])
        footprint = (80 if then else 48) * KB
        if held.scheduled() != [(footprint, at_once)] * 4 or held.scheduled() != plain.scheduled():
            held.fail(f"expected four work-items of spad_footprint {footprint}, {at_once} in flight, in both runs; got "
                      f"{held.scheduled()} with the held read and {plain.scheduled()} without")
        traces = [[(e.cycle, e.tag, e.inst, e.opcode) for e in run.events()] for run in (held, plain)]
        if traces[0] != traces[1] or held.total_cycles() != plain.total_cycles():
            held.fail(f"expected one trace and one total, got totals {held.total_cycles()} and "
                      f"{plain.total_cycles()}, {sum(a != b for a, b in zip(*traces))} trace lines differing")
        return held
    return case


def a_compute_waits_for_room_beside_the_held_bytes(t):
    """The same with a 32 KB load and its compute after the store, in 64 + 64 KB: the three loads are in the spad
    when the first compute (instruction 2) wants 24 KB for its output, 80 KB beside the 64 KB held, so it waits
    until the last compute (instruction 5) has freed the 32 KB. The held bytes are in the spad the core counts."""
    run = as_in_a_smaller_spad(64, 1, "; L(4); C({4}, {})")(t)
    if run.when("INST_ISSUED", 5) >= run.when("INST_ISSUED", 2):
        run.fail(f"expected the compute stated last issued first (@{run.when('INST_ISSUED', 5)}), the first compute "
                 f"waiting for room (@{run.when('INST_ISSUED', 2)})")


def needs_at_once_measured_beside_the_held_bytes(t):
    """Two work-items that load buffer 1 (8 KB) three times, each load read before the next is needed, beside 16 KB
    of held data in a 48 KB spad. Their fills sum to 24 KB, more than half the 32 KB beside the hold, but one needs
    8 KB at once: two are in flight, as without the held read in 32 KB."""
    body = "L(1); C({{1{held}}}, {{}}); L(1); C({{1}}, {{}}); L(1); C({{1}}, {{}})"
    table = sizes(b1=8 * KB, b9=16 * KB)
    for held, spad_kb in ((", 9", 48), ("", 32)):
        run, _ = replay(t, body.format(held=held), table, spad_kb, items=2)
        run.expect_ok()
        if run.scheduled() != [(8 * KB, 2)] * 2:
            run.fail(f"expected two work-items of spad_footprint 8192 in flight at once in {spad_kb} KB, got "
                     f"{run.scheduled()}")


def loads_everything_it_reads_is_unchanged(t, session):
    """The same four work-items with every buffer loaded, the form every TOG had before: nothing is held, no hold
    is traced, and the footprint, the number in flight and the total are those of the tree before held buffers were
    defined (origin/main a1c4209, measured there)."""
    run, _ = replay(t, "L(1); L(2); C({1, 2}, {3}); S(3)", sizes(b1=12 * KB, b2=12 * KB, b3=24 * KB), 96, items=4,
                    session=session)
    run.expect_total_cycles(1171)
    if "bytes for the kernel" in run.log or run.core_lines("SPAD_HOLD") or run.core_lines("SPAD_HOLD_END") or \
            run.scheduled() != [(48 * KB, 2)] * 4:
        run.fail(f"expected nothing held and four work-items of spad_footprint 49152, two in flight; got "
                 f"{run.scheduled()}")


# ---- sessions: a kernel that holds bytes has its core to itself -----------------------------------------------------

LOADS = ("L(1); C({1}, {})", sizes(b1=8 * KB))                  # a kernel that loads what it reads
READS = ("C({9}, {2}); S(2)", sizes(b2=8 * KB, b9=24 * KB))     # a kernel that reads held data


def last_finish(run, insts):
    """The cycle the last of the instructions `insts` finished."""
    return max(e.cycle for e in run.events() if e.tag == "INST_FINISHED" and e.inst in insts)


def holder_waits_for_the_core(t):
    """Kernel 0, two work-items that load what they read, then kernel 1, which reads 24 KB of held data, with no
    DEVICE_SYNC between. Kernel 1 does not enter beside kernel 0's last work-item: its hold is taken once kernel
    0's instructions (0 to 3) have all finished."""
    run = launch(t, 64, [kernel_dir(t, 0, *LOADS, 2), kernel_dir(t, 1, *READS, 1)]).expect_ok()
    expect_holds(run, [(0, 24 * KB)])
    if run.core_lines("SPAD_HOLD")[0][0] <= last_finish(run, range(4)) or len(run.computes()) != 3:
        run.fail(f"expected three computes and kernel 1's hold taken after kernel 0's last instruction finished "
                 f"(@{last_finish(run, range(4))}), got {run.core_lines('SPAD_HOLD')}")


def nobody_joins_a_holder(t):
    """Kernel 0, two work-items that read 24 KB of held data, then kernel 1, which loads what it reads. Kernel 1's
    work-item is handed to the core once kernel 0's hold has ended, not beside kernel 0's last work-item."""
    run = launch(t, 64, [kernel_dir(t, 0, *READS, 2), kernel_dir(t, 1, *LOADS, 1)]).expect_ok()
    expect_holds(run, [(0, 24 * KB)], insts=range(4))
    ended, scheduled = run.core_lines("SPAD_HOLD_END")[0][0], [c for c, _, _ in run.core_lines("TILE_SCHEDULED")]
    if len(scheduled) != 3 or scheduled[2] < ended or len(run.computes()) != 3:
        run.fail(f"expected kernel 1's work-item scheduled once kernel 0's hold ended (@{ended}), got work-items "
                 f"scheduled @{scheduled}")


def kernels_that_hold_nothing_overlap(t):
    """Two kernels that load what they read, with no DEVICE_SYNC between: kernel 1's work-item is handed to the core
    while kernel 0's last one is still running, as before."""
    run = launch(t, 64, [kernel_dir(t, 0, *LOADS, 2), kernel_dir(t, 1, *LOADS, 1)]).expect_ok()
    scheduled = [c for c, _, _ in run.core_lines("TILE_SCHEDULED")]
    if len(scheduled) != 3 or scheduled[2] >= last_finish(run, range(4)) or run.core_lines("SPAD_HOLD"):
        run.fail(f"expected kernel 1's work-item scheduled before kernel 0's last instruction finished "
                 f"(@{last_finish(run, range(4))}) and no hold, got work-items scheduled @{scheduled}")


def a_kernel_ending_beside_another_frees_only_its_own(t):
    """Kernel 0 fills a 32 KB spad with two 16 KB loads that one compute reads, and loads 16 KB more after it.
    Kernel 1, one compute of 50 cycles that names no buffer, runs beside it and ends first. Its end frees nothing
    of kernel 0: the third load is issued once the compute that reads the first two is, not when kernel 1 ends."""
    homes = [kernel_dir(t, 0, "L(1); L(2); C({1, 2}, {}); L(3); C({3}, {})",
                        sizes(b1=16 * KB, b2=16 * KB, b3=16 * KB), 1),
             kernel_dir(t, 1, "T(1, {}, {})", sizes(b1=16 * KB), 1)]
    run = launch(t, 32, homes).expect_ok()
    loads, computes = issued(run, "MOVIN"), sorted(run.computes(), key=lambda c: c.start)
    if len(loads) != 3 or [c.cycles for c in computes] != [50, 200, 200] or loads[2] < computes[1].start:
        run.fail(f"expected kernel 1's compute of 50 cycles first, and kernel 0's third load issued once its first "
                 f"compute was (@{computes[1].start if len(computes) > 1 else '?'}); got loads @{loads}")


def versions_of_two_kernels_at_once_are_apart(t):
    """Kernel 0 loads 8 KB that nothing reads and ends with that load. Kernel 1 runs beside it in a 40 KB spad: two
    16 KB loads of buffer 1, the compute that reads them, a third load and its compute. The buffers of the two
    kernels are numbered apart, so kernel 0's end frees its 8 KB and no more: the third load still waits for the
    compute that frees the first two."""
    homes = [kernel_dir(t, 0, "L(1)", sizes(b1=8 * KB), 1),
             kernel_dir(t, 1, "L(1); L(1); C({1}, {}); L(1); C({1}, {})", sizes(b1=16 * KB), 1)]
    run = launch(t, 40, homes).expect_ok()
    loads, computes = sorted(issued(run, "MOVIN")), sorted(issued(run, "COMP"))
    ended = last_finish(run, [0])
    if len(loads) != 4 or len(computes) != 2 or not ended < computes[0] <= loads[3]:
        run.fail(f"expected kernel 0 to end (@{ended}) before kernel 1's first compute is issued, and kernel 1's "
                 f"third load issued once that compute was; got loads @{loads} and computes @{computes}")


COPY = ("L(9)", sizes(b9=24 * KB))      # a kernel that only loads: what puts the data on the chip


def copy_then_use(sync, cores):
    """A copy kernel (one work-item that loads 24 KB and reads nothing), then the kernel that reads those 24 KB as
    held data, in a 32 KB spad: TOGSim relates the two by nothing. Both run whole; the copy's bytes leave with its
    work-item and the reader's core holds 24 KB from its own first work-item to its last, never 48 KB at once."""
    def case(t):
        run = launch(t, 32, [kernel_dir(t, 0, *COPY, 1), kernel_dir(t, 1, *READS, 1)], cores=cores,
                     sync_between=sync).expect_ok()
        expect_holds(run, [(0, 24 * KB)])
        copied = [e.cycle for e in run.events() if (e.tag, e.opcode) == ("INST_FINISHED", "MOVIN")]
        if len(copied) != 1 or len(run.computes()) != 1 or len(issued(run, "MOVOUT")) != 1 or \
                run.core_lines("SPAD_HOLD")[0][0] <= copied[0]:
            run.fail(f"expected one load, one compute and one store, and the hold taken after the copy's load "
                     f"finished (@{copied}); got the hold {run.core_lines('SPAD_HOLD')}")
    return Case(f"session.copy_then_use.{'sync_between' if sync else 'no_sync'}.{cores}_cores", case)


def use_then_copy(t):
    """The two kernels launched the other way round: nothing ties them, so the reader runs first and the copy after
    its hold has ended."""
    run = launch(t, 32, [kernel_dir(t, 0, *READS, 1), kernel_dir(t, 1, *COPY, 1)]).expect_ok()
    expect_holds(run, [(0, 24 * KB)], insts=range(2))
    if len(issued(run, "MOVIN")) != 1 or issued(run, "MOVIN")[0] <= run.core_lines("SPAD_HOLD_END")[0][0]:
        run.fail(f"expected the copy's load issued after the reader's hold ended, got the load "
                 f"@{issued(run, 'MOVIN')} and the end {run.core_lines('SPAD_HOLD_END')}")


def copy_larger_than_the_spad_refused(t):
    """The copy kernel is measured like any other: its 24 KB load does not fit a 23 KB spad."""
    homes = [kernel_dir(t, 0, *COPY, 1), kernel_dir(t, 1, *READS, 1)]
    run = launch(t, 23, homes)
    run.expect_refused(f"TOG {homes[0] / 'tog.so'}: work-item 0 (key 10) needs 24576 bytes in the spad at once, and "
                       f"a core's spad holds 23552 (core_spad_size_kb: 23)", exit=1)


def no_spad_size_holds_nothing(t):
    """Without a spad size nothing is accounted: the held buffers are named while the TOG is loaded, no hold is
    traced, and two kernels overlap on a core whatever they read."""
    run = launch(t, 0, [kernel_dir(t, 0, *READS, 2), kernel_dir(t, 1, *LOADS, 1)]).expect_ok()
    named = "read buffers 9 without having written them: 24576 bytes on the chip already, which nothing accounts"
    if run.core_lines("SPAD_HOLD") or named not in run.log or "bytes for the kernel" in run.log:
        run.fail("expected the held buffers named and no SPAD_HOLD line")
    scheduled = [c for c, _, _ in run.core_lines("TILE_SCHEDULED")]
    if len(scheduled) != 3 or scheduled[2] >= last_finish(run, range(4)):
        run.fail(f"expected kernel 1's work-item scheduled beside kernel 0's last, got @{scheduled}")


CASES = [
    *in_both_modes("read_waits_for_nothing", read_waits_for_nothing),
    *in_both_modes("occupies_its_stated_bytes", occupies_its_stated_bytes),
    *in_both_modes("refused.held_alone_larger_than_the_spad", held_alone_larger_than_the_spad_refused),
    Case("refused.a_later_work_item", a_later_work_item_refused),
    Case("session.refused.second_kernel", session_second_kernel_refused),
    Case("written_later_takes_no_more", written_later_takes_no_more),
    Case("in_place_first_is_the_work_items_own", in_place_first_is_the_work_items_own),
    held_by_what_reads_it("a_store", "dma(TOGSIM_DMA_STORE, 9, {9}, {})", sizes(b9=16 * KB), 16 * KB, "9", 0),
    held_by_what_reads_it("an_index_buffer_of_a_store", "C({}, {2}); dma(TOGSIM_DMA_STORE, 2, {2, 9}, {})",
                          sizes(b2=16 * KB, b9=4 * KB), 4 * KB, "9", 16 * KB),
    held_by_what_reads_it("the_read_bufs_of_a_load", "dma(TOGSIM_DMA_LOAD, 1, {9}, {1}); C({1}, {})",
                          sizes(b1=16 * KB, b9=4 * KB), 4 * KB, "9", 16 * KB),
    held_by_what_reads_it("two_buffers", "C({8, 9}, {2}); S(2)", sizes(b2=16 * KB, b8=8 * KB, b9=4 * KB), 12 * KB,
                          "8 to 9", 16 * KB),
    Case("refused.store_of_unsized_held_buffer", store_of_unsized_held_buffer_refused),
    Case("unsized_held_buffer_occupies_nothing", unsized_held_buffer_occupies_nothing),
    *in_both_modes("held_once_per_core.one_core", held_once_per_core(1)),
    *in_both_modes("held_once_per_core.two_cores", held_once_per_core(2)),
    *in_both_modes("each_core_holds_its_own", each_core_holds_its_own),
    Case("one_copy_for_work_items_in_flight_together", one_copy_for_work_items_in_flight_together),
    Case("own_and_held_under_one_id", own_and_held_under_one_id),
    Case("as_in_a_smaller_spad.one_in_flight", as_in_a_smaller_spad(64, 1)),
    Case("as_in_a_smaller_spad.two_in_flight", as_in_a_smaller_spad(96, 2)),
    Case("as_in_a_smaller_spad.a_compute_waits_for_room", a_compute_waits_for_room_beside_the_held_bytes),
    Case("as_in_a_smaller_spad.needs_at_once_measured_beside_the_held_bytes",
         needs_at_once_measured_beside_the_held_bytes),
    *in_both_modes("loads_everything_it_reads_is_unchanged", loads_everything_it_reads_is_unchanged),
    Case("session.holder_waits_for_the_core", holder_waits_for_the_core),
    Case("session.nobody_joins_a_holder", nobody_joins_a_holder),
    Case("session.kernels_that_hold_nothing_overlap", kernels_that_hold_nothing_overlap),
    Case("session.a_kernel_ending_beside_another_frees_only_its_own",
         a_kernel_ending_beside_another_frees_only_its_own),
    Case("session.versions_of_two_kernels_at_once_are_apart", versions_of_two_kernels_at_once_are_apart),
    *[copy_then_use(sync, cores) for sync in (True, False) for cores in (1, 2)],
    Case("session.use_then_copy", use_then_copy),
    Case("session.refused.copy_larger_than_the_spad", copy_larger_than_the_spad_refused),
    Case("session.no_spad_size_holds_nothing", no_spad_size_holds_nothing),
]
