"""A TOG is replayed against a cycle table: a run without one, or with one that does not open, is refused, and so is
a compute that names a tile id the table does not hold, in both modes."""
from togtest import MIB, Case, Failure, write_table

TOG = "names_tile_id_tog.cpp"


def refusal(tile_id, held):
    """The refusal of work-item 0 (key 7) naming `tile_id` against a table of `held` tile ids, after the TOG's path."""
    has = f"it has {held} tile ids, 0 to {held - 1}" if held else "it has no line"
    return f"work-item 0 (key 7) computes tile id {tile_id}, which the cycle table does not hold ({has})"


def replay(t, cycles, *defines, units=None, **machine):
    """--tog_so: the TOG (naming tile id 1 unless a define says otherwise) with `cycles` as its cycle table."""
    machine = machine or dict(systolic_array_split="none")
    return t.simulate(t.tog(TOG, defines, tensors=(MIB, MIB)), t.machine(**machine), cycles, units)


def session(t, cycles, *defines, link=None):
    """--models_list: one LAUNCH_KERNEL of the TOG from <work dir>/k/, `cycles` as its tog_cycles.tsv (None: the
    directory has none), or tog_cycles.tsv a symlink to `link`."""
    home = t.tog(TOG, defines, tensors=(MIB, MIB), into="k").parent
    if link is not None:
        (home / "tog_cycles.tsv").symlink_to(link)
    elif cycles is not None:
        write_table(home / "tog_cycles.tsv", [(c, 0) for c in cycles])
    stream = t.work_dir / "commands.txt"
    stream.write_text(f"LAUNCH_KERNEL,0,0,0,{home}/kernel,{home}/attributes,0\nDEVICE_SYNC,0,0,0,-,-,0\n")
    return t.execute("Simulator", ["--config", t.machine(systolic_array_split="none"), "--models_list", stream,
                                   "--log_level", "trace"])


def expect_one_compute(run, cycles):
    """The run finished and its one compute ran `cycles` cycles."""
    seen = [c.cycles for c in run.expect_ok().computes()]
    if seen != [cycles]:
        run.fail(f"expected one compute of {cycles} cycles, got {seen}")


def tile_id_at_table_length_refused(t):
    """Tile id 1 against a one-line table: the message names the TOG, the tile id and how many the table has."""
    run = replay(t, [200]).expect_refused(refusal(1, 1), exit=1)
    if f"[TOGSim] TOG {t.work_dir / 'tog.so'}: work-item" not in run.log:
        run.fail("expected the refusal to name the TOG's path")


def table_one_line_longer_runs(t):
    """The same TOG against a two-line table runs, and its compute takes line 1's 300 cycles."""
    expect_one_compute(replay(t, [200, 300]), 300)


def session_tile_id_at_table_length_refused(t):
    run = session(t, [200]).expect_refused(refusal(1, 1), exit=1)
    if f"TOG {t.work_dir / 'k' / 'tog.so'}: work-item" not in run.log:
        run.fail("expected the refusal to name the kernel's tog.so")


def session_table_one_line_longer_runs(t):
    expect_one_compute(session(t, [200, 300]), 300)


def negative_tile_id_refused(t):
    """tile_id is a uint64_t: -1 arrives as 2^64 - 1, beyond every table, and is refused as that number."""
    replay(t, [200, 300], "TILE_ID=-1").expect_refused(refusal(2**64 - 1, 2), exit=1)


def tile_id_past_32_bits_refused(t):
    """2^32 + 1 is not tile id 1: the id is compared whole, not through an int32_t."""
    replay(t, [200, 300], "TILE_ID=4294967297ull").expect_refused(refusal(2**32 + 1, 2), exit=1)


def without(t, *table):
    """--tog_so with `table` in place of a cycle table that opens: nothing, or --cycle_table and a path."""
    tog = t.tog(TOG, tensors=(MIB, MIB))
    return t.execute("Simulator", ["--config", t.machine(systolic_array_split="none"), "--tog_so", tog, *table,
                                   "--log_level", "trace"])


def expect_refused_before_the_run(run, saying):
    """The run was refused with exit 1 saying `saying`, and no instruction was traced."""
    run.expect_refused(saying, exit=1)
    if run.events():
        run.fail("expected the refusal before anything was simulated, found instruction trace lines")


def unopenable(path):
    """The refusal of a cycle table at `path` that does not open."""
    return f"[TOGSim] cycle table {path} cannot be opened"


def no_cycle_table_refused(t):
    """--tog_so without --cycle_table: there is no default table."""
    expect_refused_before_the_run(without(t), "[TOGSim] --tog_so needs --cycle_table")


def cycle_table_that_cannot_be_opened_refused(t):
    """--cycle_table naming a file that is not there: the message names the path."""
    absent = t.work_dir / "absent.tsv"
    expect_refused_before_the_run(without(t, "--cycle_table", absent), unopenable(absent))


def session_no_tog_cycles_refused(t):
    """A launched kernel whose directory has no tog_cycles.tsv: the launch is refused, naming the path it wants."""
    expect_refused_before_the_run(session(t, None), unopenable(t.work_dir / "k" / "tog_cycles.tsv"))


def session_tog_cycles_that_cannot_be_opened_refused(t):
    """A tog_cycles.tsv that is there and does not open (a symlink to nothing) is refused the same way."""
    expect_refused_before_the_run(session(t, None, link=t.work_dir / "absent.tsv"),
                                  unopenable(t.work_dir / "k" / "tog_cycles.tsv"))


def empty_table_holds_no_tile_id(t):
    """A table without a line is a table: a TOG with no compute replays against it, and a compute is refused by the
    rule of every table, in both modes."""
    if replay(t, [], "NO_COMPUTE").expect_ok().computes():
        raise Failure("expected the NO_COMPUTE TOG to run no compute")
    replay(t, [], "TILE_ID=0").expect_refused(refusal(0, 0), exit=1)
    session(t, [], "TILE_ID=0").expect_refused(refusal(0, 0), exit=1)


def session_that_launches_nothing_needs_no_table(t):
    """A command stream without a LAUNCH_KERNEL loads no TOG and asks for no cycle table."""
    stream = t.work_dir / "commands.txt"
    stream.write_text("# nothing to launch\nDEVICE_SYNC,0,0,0,-,-,0\n")
    t.execute("Simulator", ["--config", t.machine(systolic_array_split="none"), "--models_list", stream]).expect_ok()


def unit_rows_beyond_the_cycle_table_are_ignored(t):
    """Unit table rows of tile id 3, which a one-line cycle table does not hold, are rows no compute reaches: the run
    is not refused and totals what it does without them."""
    rows = [(0, "TPUV6_MXU", "rows", 1, 512), (0, "Vpu", "cycles", 1, 50)]
    beyond = rows + [(3, "TPUV6_MXU", "rows", 1, 512), (3, "Vpu", "cycles", 1, 50)]
    totals = [replay(t, [600], "TILE_ID=0", units=u).expect_ok().total_cycles()
              for u in (rows, beyond)]
    if totals[0] != totals[1]:
        raise Failure(f"expected the same total with and without the rows of tile id 3, got {totals}")


CASES = [
    Case("refused.tile_id_at_table_length", tile_id_at_table_length_refused),
    Case("table_one_line_longer_runs", table_one_line_longer_runs),
    Case("session.refused.tile_id_at_table_length", session_tile_id_at_table_length_refused),
    Case("session.table_one_line_longer_runs", session_table_one_line_longer_runs),
    Case("refused.negative_tile_id", negative_tile_id_refused),
    Case("refused.tile_id_past_32_bits", tile_id_past_32_bits_refused),
    Case("refused.no_cycle_table", no_cycle_table_refused),
    Case("refused.cycle_table_that_cannot_be_opened", cycle_table_that_cannot_be_opened_refused),
    Case("session.refused.no_tog_cycles", session_no_tog_cycles_refused),
    Case("session.refused.tog_cycles_that_cannot_be_opened", session_tog_cycles_that_cannot_be_opened_refused),
    Case("empty_table_holds_no_tile_id", empty_table_holds_no_tile_id),
    Case("session.that_launches_nothing_needs_no_table", session_that_launches_nothing_needs_no_table),
    Case("unit_rows_beyond_the_cycle_table_are_ignored", unit_rows_beyond_the_cycle_table_are_ignored),
]
