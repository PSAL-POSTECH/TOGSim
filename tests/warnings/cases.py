"""What TOGSim keeps running and says once, at warn level: a compute whose tile id the unit table has no rows for,
a DMA whose arg_id tog_tensors.txt does not list, a config key TOGSim does not read."""
from togtest import MIB, Case, write_table

TOG = "names_ids_tog.cpp"
CYCLES = [300] * 5
NO_ROWS = "the unit table has no rows for tile ids"
STRAY_ARG = "its DMAs name arg_id"
UNREAD_KEY = "is not read by TOGSim"


def rows(*tile_ids):
    """Unit table rows of each of `tile_ids`: 50 Vpu cycles."""
    return [(tile_id, "Vpu", "cycles", 1, 50) for tile_id in tile_ids]


def replay(t, units, *defines, tensors=(MIB, MIB)):
    """--tog_so: the TOG against a five-line cycle table and `units` (None: no unit table), without the array split;
    tog_tensors.txt lists `tensors`."""
    return t.simulate(t.tog(TOG, defines, tensors=tensors), t.machine(systolic_array_split="none"), CYCLES, units)


def session(t, units, *defines, launches=1):
    """--models_list: `launches` LAUNCH_KERNELs of the TOG from <work dir>/k/, `units` as its tog_units.tsv (None:
    the directory has none)."""
    home = t.tog(TOG, defines, tensors=(MIB, MIB), into="k").parent
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


def stray_arg(tog, ids, held):
    """The warning that the DMAs of `tog` name the arg_ids `ids` (as the line lists them) against `held` tensors."""
    return (f"[TOGSim] TOG {tog}: {STRAY_ARG} {ids}, and tog_tensors.txt lists {held} tensor(s): those DMAs are "
            f"addressed from base 0")


def store_addresses(run):
    """The DRAM address of every store the run issued, in issue order."""
    return [int(e.fields["dram"], 16) for e in run.events() if e.opcode == "MOVOUT" and e.tag == "INST_ISSUED"]


def arg_id_at_tensor_count_warned_once(t):
    """Four stores name arg_id 2 against two tensors: one line names the TOG, the arg_id and the two tensors, and
    the run finishes with the stores where they were, 16 KB apart from address 0."""
    run = replay(t, None, "STORE_ARG=2").expect_warned_once(stray_arg(t.work_dir / "tog.so", "2", 2))
    if store_addresses(run) != [i * 16384 for i in range(4)]:
        run.fail(f"expected the four stores at 0, 16384, 32768 and 49152, got {store_addresses(run)}")


def negative_arg_id_warned_with_the_others(t):
    """A load from arg_id -1 and a store to arg_id 3 are one line that lists both."""
    run = replay(t, None, "LOAD_ARG=-1", "STORE_ARG=3")
    run.expect_warned_once(stray_arg(t.work_dir / "tog.so", "-1, 3", 2)).expect_warned_once(STRAY_ARG)


def no_tensor_listed_warns_for_every_arg_id(t):
    """An empty tog_tensors.txt lists no tensor: arg_ids 0 and 1 are both beyond it."""
    replay(t, None, tensors=()).expect_warned_once(stray_arg(t.work_dir / "tog.so", "0 to 1", 0))


def every_arg_id_listed_is_quiet(t):
    replay(t, None).expect_no_warning(STRAY_ARG)


def session_arg_id_at_tensor_count_warned_once(t):
    """A launched kernel is checked against its own tog_tensors.txt, and launching it again repeats nothing."""
    session(t, None, "STORE_ARG=2", launches=2).expect_warned_once(stray_arg(t.work_dir / "k" / "tog.so", "2", 2))


def session_every_arg_id_listed_is_quiet(t):
    session(t, None).expect_no_warning(STRAY_ARG)


def misspelled_key_warned_once(t):
    """tpuv6e with core_freq_mhzz beside core_freq_mhz: the run goes on, and the line names the file and the key."""
    config = t.machine(systolic_array_split="none", core_freq_mhzz=1751)
    run = t.simulate(t.tog(TOG, (), tensors=(MIB, MIB)), config, CYCLES, None)
    run.expect_warned_once(f"[TOGSim] config {config}: key core_freq_mhzz {UNREAD_KEY}")
    if len(run.warned(UNREAD_KEY)) != 1:
        run.fail(f"expected only core_freq_mhzz named, got {run.warned(UNREAD_KEY)}")


def keys_other_tools_read_are_quiet(t):
    """tpuv6e as shipped carries isa (the compiler's, and the frontend's vector length), vcix and address_map: none
    of them is named."""
    replay(t, None).expect_no_warning(UNREAD_KEY)


def vector_length_key_warned_once(t):
    """vpu_vector_length_bits is no key of anyone's: the vector length is the zvl<N>b entry of isa."""
    config = t.machine(systolic_array_split="none", vpu_vector_length_bits=256)
    run = t.simulate(t.tog(TOG, (), tensors=(MIB, MIB)), config, CYCLES, None)
    run.expect_warned_once(f"[TOGSim] config {config}: key vpu_vector_length_bits {UNREAD_KEY}")


CASES = [
    Case("unit_rows.one_tile_without_them_warned_once", tile_without_unit_rows_warned_once),
    Case("unit_rows.tiles_without_them_listed_in_one_line", tiles_without_unit_rows_listed_in_one_line),
    Case("unit_rows.every_tile_with_them_is_quiet", every_tile_with_unit_rows_is_quiet),
    Case("unit_rows.without_a_unit_table_is_quiet", without_a_unit_table_is_quiet),
    Case("unit_rows.session.one_tile_without_them_warned_once", session_tile_without_unit_rows_warned_once),
    Case("unit_rows.session.without_a_unit_table_is_quiet", session_without_a_unit_table_is_quiet),
    Case("unit_rows.session.every_tile_with_them_is_quiet", session_every_tile_with_unit_rows_is_quiet),
    Case("arg_id.at_tensor_count_warned_once", arg_id_at_tensor_count_warned_once),
    Case("arg_id.negative_warned_with_the_others", negative_arg_id_warned_with_the_others),
    Case("arg_id.no_tensor_listed_warns_for_every_arg_id", no_tensor_listed_warns_for_every_arg_id),
    Case("arg_id.every_one_listed_is_quiet", every_arg_id_listed_is_quiet),
    Case("arg_id.session.at_tensor_count_warned_once", session_arg_id_at_tensor_count_warned_once),
    Case("arg_id.session.every_one_listed_is_quiet", session_every_arg_id_listed_is_quiet),
    Case("config_key.misspelled_warned_once", misspelled_key_warned_once),
    Case("config_key.read_by_other_tools_is_quiet", keys_other_tools_read_are_quiet),
    Case("config_key.vector_length_bits_warned_once", vector_length_key_warned_once),
]
