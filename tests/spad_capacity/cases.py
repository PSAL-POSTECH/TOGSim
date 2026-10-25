"""A dispatch states its work-item's spad in bytes per lane (.spad), and that is all TOGSim reads of the spad: more
than a lane's spad (vpu_spad_size_kb_per_lane) is refused while the TOG is loaded, in both modes; a work-item enters a
core while its spad fits beside those in flight, two at once only when twice its spad fits; no instruction waits for
spad. core_spad_size_kb is refused alone and must restate the lanes beside them."""
from togtest import MIB, REPO, Case, spad_per_lane, write_table

TOG = "work_item_tog.cpp"
KB = 1024
LANE = 1 * KB                                     # the lane's spad of every case but the shipped tpuv6e
BODY = "L(1); C({1}, {2}); S(2)"                  # a 16 KB load, a compute, a 16 KB store
SIZES = (0, 16 * KB, 16 * KB)
TIMEOUT_S = 60


def refusal(item, spad, lane):
    """The refusal of work-item `item` (key 10 + item) stating `spad` bytes per lane against a lane of `lane` bytes."""
    return (f"work-item {item} (key {10 + item}) states {spad} bytes per lane of spad, and a lane's spad holds {lane} "
            f"(vpu_spad_size_kb_per_lane: {lane // KB})")


def tog(t, spad, items=1, sizes=SIZES, into=""):
    return t.tog(TOG, (f"BODY={BODY}", f"SIZES={','.join(map(str, sizes))}", f"ITEMS={items}", f"SPAD={spad}"),
                 tensors=(256 * MIB, 256 * MIB), into=into)


def replay(t, spad, items=1, sizes=SIZES, **keys):
    """--tog_so: `items` work-items stating `spad` on tpuv6e with a 1 KB lane, or the machine `keys` instead."""
    machine = t.machine(**(keys or spad_per_lane(1)), systolic_array_split="none")
    return t.simulate(tog(t, spad, items, sizes), machine, cycles=[200], timeout=TIMEOUT_S)


def session(t, *spads):
    """--models_list: one LAUNCH_KERNEL per entry of `spads`, one work-item stating it, then one DEVICE_SYNC."""
    stream = t.work_dir / "commands.txt"
    with stream.open("w") as commands:
        for i, spad in enumerate(spads):
            home = tog(t, spad, into=f"k{i}").parent
            write_table(home / "tog_cycles.tsv", [(200, 0)])
            commands.write(f"LAUNCH_KERNEL,{i},0,0,{home}/kernel,{home}/attributes,0\n")
        commands.write("DEVICE_SYNC,0,0,0,-,-,0\n")
    machine = t.machine(**spad_per_lane(1), systolic_array_split="none")
    return t.execute("Simulator", ["--config", machine, "--models_list", stream, "--log_level", "trace"], TIMEOUT_S)


def expect_refused_before_the_run(run, saying, so):
    """Refused with exit 1 by a line that names the TOG `so` and says `saying`, before any instruction was traced."""
    run.expect_refused(f"TOG {so}: {saying}", exit=1)
    if run.events():
        run.fail("expected the refusal before anything was simulated, found instruction trace lines")


def expect_in_flight(run, spad, in_flight):
    """Two work-items scheduled stating `spad`, `in_flight` at once: one at a time, the second enters once the
    first's store has finished; two, it enters before (instructions 0 to 2 are the first's)."""
    run.expect_ok()
    if run.scheduled() != [(spad, in_flight)] * 2:
        run.fail(f"expected two work-items scheduled as (spad, max_dispatch) = ({spad}, {in_flight}), got "
                 f"{run.scheduled()}")
    second, first_done = run.core_lines("TILE_SCHEDULED")[1][0], run.when("INST_FINISHED", 2)
    if (second >= first_done) != (in_flight == 1):
        run.fail(f"expected the second work-item to enter {'after' if in_flight == 1 else 'before'} the first's store "
                 f"finished @{first_done}, it entered @{second}")


def fits_exactly_runs(t):
    """Each of two work-items states the whole lane: they run one after the other."""
    expect_in_flight(replay(t, LANE, items=2), LANE, 1)


def one_byte_over_refused(t):
    expect_refused_before_the_run(replay(t, LANE + 1), refusal(0, LANE + 1, LANE), t.work_dir / "tog.so")


def later_work_item_refused(t):
    """Work-item 0 fits; the refusal names work-item 1 and its key."""
    expect_refused_before_the_run(replay(t, f"(i == 0 ? {LANE} : {LANE + 1})", items=2), refusal(1, LANE + 1, LANE),
                                  t.work_dir / "tog.so")


def negative_refused(t):
    expect_refused_before_the_run(replay(t, -1), "work-item 0 (key 10) states a negative spad, -1 bytes per lane",
                                  t.work_dir / "tog.so")


def two_in_flight_when_twice_fits(t):
    expect_in_flight(replay(t, LANE // 2, items=2), LANE // 2, 2)


def one_in_flight_when_twice_does_not_fit(t):
    """One byte more than half the lane: twice it does not fit, and the work-items run one at a time."""
    expect_in_flight(replay(t, LANE // 2 + 1, items=2), LANE // 2 + 1, 1)


def no_instruction_waits_for_spad(t):
    """Work-items that state 1 byte per lane and move 256 KB each (2 KB a lane): in a 1 KB lane they run as without a
    spad limit,
    instruction for instruction."""
    big = (0, 256 * KB, 256 * KB)
    runs = [replay(t, 1, items=4, sizes=big, **keys) for keys in (spad_per_lane(1), spad_per_lane(None))]
    traces = [[(e.cycle, e.core, e.tag, e.inst) for e in run.expect_ok().events()] for run in runs]
    if not traces[0] or traces[0] != traces[1]:
        runs[0].fail(f"expected one instruction trace in a 1 KB lane and without a limit, totals "
                     f"{runs[0].total_cycles()} and {runs[1].total_cycles()}")


def no_spad_key_refuses_nothing(t):
    """A config without vpu_spad_size_kb_per_lane (or core_spad_size_kb) has no limit: any spad runs."""
    replay(t, 1 << 40, **spad_per_lane(None)).expect_ok()


def session_fits_exactly_runs(t):
    session(t, LANE).expect_ok()


def expect_launch_refused(run, command, saying, so):
    """As expect_refused_before_the_run, the line being the error of LAUNCH_KERNEL command `command`."""
    expect_refused_before_the_run(run, saying, so)
    if f"[TOGSim] Error processing command {command} (type: LAUNCH_KERNEL)" not in run.said()[0]:
        run.fail(f"expected the refusal as the error of command {command}, the one that launched the kernel")


def session_one_byte_over_refused(t):
    expect_launch_refused(session(t, LANE + 1), 0, refusal(0, LANE + 1, LANE), t.work_dir / "k0" / "tog.so")


def session_second_kernel_refused(t):
    """Kernel 0 fills the lane and kernel 1 is one byte over: the session ends at kernel 1's launch."""
    expect_launch_refused(session(t, LANE, LANE + 1), 1, refusal(0, LANE + 1, LANE), t.work_dir / "k1" / "tog.so")


def core_spad_alone_refused(t):
    replay(t, LANE, **spad_per_lane(None) | dict(core_spad_size_kb=48)).expect_refused(
        "core_spad_size_kb: a work-item states its spad per lane", exit=134)


def restated_runs(t):
    """core_spad_size_kb: 128 beside 1 KB on each of tpuv6e's 128 lanes restates the spad: fits_exactly_runs."""
    expect_in_flight(replay(t, LANE, items=2, **spad_per_lane(1) | dict(core_spad_size_kb=128)), LANE, 1)


def stated_twice_disagreeing_refused(t):
    """core_spad_size_kb: 48 on tpuv6e as shipped, whose 1024 KB x 128 lanes are 131072 KB."""
    replay(t, LANE, vpu_spad_size_kb_per_lane=1024, core_spad_size_kb=48).expect_refused(
        "core_spad_size_kb: 48 is not vpu_spad_size_kb_per_lane x vpu_num_lanes = 1024 x 128 = 131072", exit=134)


def tpuv6e_one_byte_over_refused(t):
    """configs/tpuv6e.yml as shipped states 1024 KB on each lane: a dispatch one byte over is refused, naming it."""
    so = tog(t, 1024 * KB + 1)
    run = t.simulate(so, REPO / "configs" / "tpuv6e.yml", cycles=[200],
                     units=[(0, "TPUV6E_MXU", "rows", 1, 0), (0, "Vpu", "cycles", 1, 200)], timeout=TIMEOUT_S)
    expect_refused_before_the_run(run, refusal(0, 1024 * KB + 1, 1024 * KB), so)


CASES = [
    Case("fits_exactly_runs", fits_exactly_runs),
    Case("refused.one_byte_over", one_byte_over_refused),
    Case("refused.later_work_item", later_work_item_refused),
    Case("refused.negative", negative_refused),
    Case("two_in_flight_when_twice_fits", two_in_flight_when_twice_fits),
    Case("one_in_flight_when_twice_does_not_fit", one_in_flight_when_twice_does_not_fit),
    Case("no_instruction_waits_for_spad", no_instruction_waits_for_spad),
    Case("no_spad_key_refuses_nothing", no_spad_key_refuses_nothing),
    Case("session.fits_exactly_runs", session_fits_exactly_runs),
    Case("session.refused.one_byte_over", session_one_byte_over_refused),
    Case("session.refused.second_kernel", session_second_kernel_refused),
    Case("config.refused.core_spad_alone", core_spad_alone_refused),
    Case("config.restated_runs", restated_runs),
    Case("config.refused.stated_twice_disagreeing", stated_twice_disagreeing_refused),
    Case("config.refused.tpuv6e_one_byte_over", tpuv6e_one_byte_over_refused),
]
