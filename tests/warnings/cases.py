"""What TOGSim keeps running and says once, at warn level: a compute whose tile id the unit table has no rows for."""
from togtest import MIB, Case, write_table

TOG = "names_ids_tog.cpp"
CYCLES = [300] * 5
NO_ROWS = "the unit table has no rows for tile ids"


def rows(*tile_ids):
    """Unit table rows of each of `tile_ids`: 50 Vpu cycles."""
    return [(tile_id, "Vpu", "cycles", 1, 50) for tile_id in tile_ids]


def replay(t, units, *defines):
    """--tog_so: the TOG against a five-line cycle table and `units` (None: no unit table), without the array split."""
    return t.simulate(t.tog(TOG, defines, tensors=(MIB, MIB)), t.machine(systolic_array_split="none"), CYCLES, units)


def session(t, units, launches=1):
    """--models_list: `launches` LAUNCH_KERNELs of the TOG from <work dir>/k/, `units` as its tog_units.tsv (None:
    the directory has none)."""
    home = t.tog(TOG, tensors=(MIB, MIB), into="k").parent
    write_table(home / "tog_cycles.tsv", [(c, 0) for c in CYCLES])
    if units is not None:
        write_table(home / "tog_units.tsv", units)
    stream = t.work_dir / "commands.txt"
    stream.write_text("".join(f"LAUNCH_KERNEL,{k},0,0,{home}/kernel,{home}/attributes,0\nDEVICE_SYNC,0,0,0,-,-,0\n"
                              for k in range(launches)))
    return t.execute("Simulator", ["--config", t.machine(systolic_array_split="none"), "--models_list", stream,
                                   "--log_level", "trace"])


def no_rows(tog, ids):
    """The warning that the unit table has no rows for the tile ids `ids` (as the line lists them) of `tog`."""
    return (f"[TOGSim] TOG {tog}: {NO_ROWS} {ids}, which its computes name: those computes add nothing to the "
            f"units")


def tile_without_unit_rows_warned_once(t):
    """Two computes name tile id 1, which has no rows: one line names the TOG and the id, and the run finishes."""
    replay(t, rows(0)).expect_warned_once(no_rows(t.work_dir / "tog.so", "1"))


def tiles_without_unit_rows_listed_in_one_line(t):
    """Tile ids 1, 2 and 4 have no rows (2 inside the table, 4 beyond its last row): one line lists the three."""
    run = replay(t, rows(0, 3), "TILE_IDS={0, 1, 2, 4}")
    run.expect_warned_once(no_rows(t.work_dir / "tog.so", "1 to 2, 4")).expect_warned_once(NO_ROWS)


def every_tile_with_unit_rows_is_quiet(t):
    replay(t, rows(0, 1)).expect_no_warning(NO_ROWS)


def without_a_unit_table_is_quiet(t):
    """A run with no unit table sums nothing into units, and says nothing about them."""
    replay(t, None).expect_no_warning(NO_ROWS)


def session_tile_without_unit_rows_warned_once(t):
    """A launched kernel is checked against its own tog_units.tsv, and launching it again repeats nothing."""
    session(t, rows(0), launches=2).expect_warned_once(no_rows(t.work_dir / "k" / "tog.so", "1"))


def session_without_a_unit_table_is_quiet(t):
    session(t, None).expect_no_warning(NO_ROWS)


def session_every_tile_with_unit_rows_is_quiet(t):
    session(t, rows(0, 1)).expect_no_warning(NO_ROWS)


CASES = [
    Case("unit_rows.one_tile_without_them_warned_once", tile_without_unit_rows_warned_once),
    Case("unit_rows.tiles_without_them_listed_in_one_line", tiles_without_unit_rows_listed_in_one_line),
    Case("unit_rows.every_tile_with_them_is_quiet", every_tile_with_unit_rows_is_quiet),
    Case("unit_rows.without_a_unit_table_is_quiet", without_a_unit_table_is_quiet),
    Case("unit_rows.session.one_tile_without_them_warned_once", session_tile_without_unit_rows_warned_once),
    Case("unit_rows.session.without_a_unit_table_is_quiet", session_without_a_unit_table_is_quiet),
    Case("unit_rows.session.every_tile_with_them_is_quiet", session_every_tile_with_unit_rows_is_quiet),
]
