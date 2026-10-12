"""A compute names a tile id the cycle table holds: one beyond it is refused, not charged 0 cycles, in both modes."""
from togtest import MIB, Case, Failure, write_table

TOG = "names_tile_id_tog.cpp"


def refusal(tile_id, held):
    """The refusal of work-item 0 (key 7) naming `tile_id` against a table of `held` tile ids, after the TOG's path."""
    return (f"work-item 0 (key 7) computes tile id {tile_id}, which the cycle table does not hold "
            f"(it has {held} tile ids, 0 to {held - 1})")


def replay(t, cycles, *defines, units=None, **machine):
    """--tog_so: the TOG (naming tile id 1 unless a define says otherwise) with `cycles` as its cycle table."""
    machine = machine or dict(systolic_array_split="none")
    return t.simulate(t.tog(TOG, defines, tensors=(MIB, MIB)), t.machine(**machine), cycles, units)


def session(t, cycles):
    """--models_list: one LAUNCH_KERNEL of the TOG from <work dir>/k/, `cycles` as its tog_cycles.tsv."""
    home = t.tog(TOG, tensors=(MIB, MIB), into="k").parent
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


def stub(t, tile_id):
    """--tog_so without --cycle_table: the flat stub, 128 cycles for tile ids 0 to 255."""
    tog = t.tog(TOG, [f"TILE_ID={tile_id}"], tensors=(MIB, MIB))
    return t.execute("Simulator", ["--config", t.machine(systolic_array_split="none"), "--tog_so", tog,
                                   "--log_level", "trace"])


def without_a_table_the_stub_holds_256_tile_ids(t):
    """No cycle table is still allowed: tile id 255 runs its 128 cycles, and 256 is beyond the stub as beyond a table."""
    expect_one_compute(stub(t, 255), 128)
    stub(t, 256).expect_refused(refusal(256, 256), exit=1)


def unit_rows_beyond_the_cycle_table_are_ignored(t):
    """Unit table rows of tile id 3, which a one-line cycle table does not hold, are rows no compute reaches: the run
    is not refused and totals what it does without them."""
    rows = [(0, "Systolic", "rows", 1, 512), (0, "Vpu", "cycles", 1, 50)]
    beyond = rows + [(3, "Systolic", "rows", 1, 512), (3, "Vpu", "cycles", 1, 50)]
    totals = [replay(t, [600], "TILE_ID=0", units=u, vpu_num_lanes=256).expect_ok().total_cycles()
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
    Case("without_a_table_the_stub_holds_256_tile_ids", without_a_table_the_stub_holds_256_tile_ids),
    Case("unit_rows_beyond_the_cycle_table_are_ignored", unit_rows_beyond_the_cycle_table_are_ignored),
]
