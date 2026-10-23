"""Runtime attributes: a TOG reads what the functional run recorded for each work-item key, through --attribute_path
or a LAUNCH_KERNEL's attribute_path; a missing, stale or malformed record, and every misuse, is refused by name."""
import re
from collections import Counter

from togtest import MIB, Case, Failure, programs_bytes, record_bytes, write_table

KEY1 = 1 << 21
OFFSETS = {(0, 0): 4096, (KEY1, 0): 20480}
WALK = {(0, 0): 3, (0, 1): [5, 2, 7], (1, 0): 0, (1, 1): ([0], [])}
ROWS = [[3, 0, 5, 1, 7, 2, 6, 4], [9, 8, 11, 10, 13, 12, 15, 14]]
ELEMS = [[100 * i + 7 * j for j in range(8)] for i in range(4)]
GATHER = {(0, 0): ROWS, (0, 1): ELEMS}
REQUEST = 32   # tpuv6e's DRAM request bytes ([Config/DRAM] line)
TO_MEMORY = re.compile(r"\[REQ_TO_MEM *\] MOVIN dram=0x([0-9a-f]+) channel=\d+$", re.M)


def machine(t):
    """tpuv6e without the array split: these TOGs bring no unit table."""
    return t.machine(systolic_array_split="none")


def replay(t, source, records, defines=(), attributes="attributes", tensors=(MIB, MIB, MIB)):
    """`source` with -D`defines`, its records written to <work dir>/<attributes> (None: no --attribute_path)."""
    path = None if records is None else t.attributes(records, into=attributes)
    return t.simulate(t.tog(source, defines, tensors=tensors), machine(t), cycles=[100], attributes=path)


def issued(run, opcode):
    """The dram address of every `opcode` instruction issued, in log order."""
    return [int(e.fields["dram"], 16) for e in run.events() if e.opcode == opcode and e.tag == "INST_ISSUED"]


def requests(run):
    """Counter of the request addresses the loads sent to memory."""
    return Counter(int(m[1], 16) for m in TO_MEMORY.finditer(run.log))


def scalar_offset_by_key(t):
    """Each work-item loads from the element offset its own key's record holds: 4096 and 20480 f32 elements."""
    run = replay(t, "scalar_offset_tog.cpp", OFFSETS).expect_ok()
    if issued(run, "MOVIN") != [4096 * 4, 20480 * 4]:
        run.fail(f"expected the loads at 0x4000 and 0x14000 (key 0, key {KEY1}), got {list(map(hex, issued(run, 'MOVIN')))}")


def runtime_trip_count(defines):
    """Key 0 walks 3 blocks, 5, 2 and 7, from the trip count and the block table it recorded; key 1 recorded 0
    trips and an empty table, so its work-item issues nothing, and the run still ends."""
    def case(t):
        run = replay(t, "array_loop_tog.cpp", WALK, defines).expect_ok()
        if issued(run, "MOVIN") != [b * 256 * 4 for b in (5, 2, 7)]:
            run.fail(f"expected loads of blocks 5, 2, 7 (0x1400, 0x800, 0x1c00), got "
                     f"{list(map(hex, issued(run, 'MOVIN')))}")
        if len(run.computes()) != 3 or len(issued(run, "MOVOUT")) != 3:
            run.fail(f"expected 3 computes and 3 stores, all of key 0, got {len(run.computes())} and "
                     f"{len(issued(run, 'MOVOUT'))}")
    return case


def gathered(rows, elems):
    """Counter of the request addresses the two row gathers and the element gather of gather_tog.cpp send."""
    want = Counter()
    for t in range(2):
        want.update({(4 * (r * 16 + c + 16 * rows[t][r])) // REQUEST * REQUEST for r in range(8) for c in range(16)})
    want.update({(MIB + 4 * (i * 8 + j + elems[i][j])) // REQUEST * REQUEST for i in range(4) for j in range(8)})
    return want


def gather_offsets(t):
    """Every element of a gather moves by its entry: row r of trip t by 16 * ROWS[t][r] elements, element (i, j) of
    the element gather by ELEMS[i][j]; the requests are those addresses' 32-byte blocks and no others."""
    run = replay(t, "gather_tog.cpp", GATHER).expect_ok()
    seen, want = requests(run), gathered(ROWS, ELEMS)
    if seen != want:
        run.fail(f"expected {sum(want.values())} load requests at the gathered addresses, got {sum(seen.values())}; "
                 f"missing {sorted(want - seen)[:6]}, extra {sorted(seen - want)[:6]}")
    if seen == gathered([[0] * 8] * 2, [[0] * 8] * 4):
        run.fail("expected the gathers to move other blocks than dense tiles do")


def broken_dir(edit, records=OFFSETS):
    """The `records` (default: scalar_offset_tog.cpp's) written, then `edit(dir)`."""
    def make(t):
        out = t.attributes(records)
        edit(out)
        return out
    return make


def remove(name):
    return lambda out: (out / name).unlink()


def write(name, data):
    return lambda out: (out / name).write_bytes(data)


LOADER = {
    "e01.no_attribute_path": dict(
        make=lambda t: None, saying="it declares 1 runtime attribute(s) and was launched with no attribute_path"),
    "e02.attribute_path_does_not_open": dict(
        make=lambda t: t.work_dir / "absent", saying="attribute_path {work}/absent does not open"),
    "e03.no_programs_bin": dict(
        make=broken_dir(remove("programs.bin")),
        saying="{dir} has no programs.bin: the functional run recorded no program"),
    "e04.programs_bin_of_abi_17": dict(
        make=broken_dir(write("programs.bin", programs_bytes([0, KEY1], abi=17))),
        saying="{dir}/programs.bin was recorded at TOG ABI 17, TOGSim is version 1.0.0"),
    "e04.record_of_next_major": dict(
        make=broken_dir(write("attr_0_0.bin", record_bytes(0, 0, 4096, abi=0x20000))),
        saying="{dir}/attr_0_0.bin was recorded at version 2.0.0, TOGSim is version 1.0.0"),
    "e04.record_without_magic": dict(
        make=broken_dir(write("attr_0_0.bin", record_bytes(0, 0, 4096, magic=0))),
        saying="{dir}/attr_0_0.bin: not a runtime attribute file"),
    "e05.dispatched_key_not_listed": dict(
        make=broken_dir(write("programs.bin", programs_bytes([0]))),
        saying=f"work-item key {KEY1} was dispatched and {{dir}}/programs.bin does not list it"),
    "e05.listed_key_not_dispatched": dict(
        make=broken_dir(write("programs.bin", programs_bytes([0, 99, KEY1]))),
        saying="{dir}/programs.bin lists key 99, which no work-item was dispatched under"),
    "e05.listed_out_of_order": dict(
        make=broken_dir(write("programs.bin", programs_bytes([KEY1, 0]))),
        saying="{dir}/programs.bin lists key 0 out of ascending order or twice"),
    "e06.header_of_another_record": dict(
        make=broken_dir(write("attr_0_0.bin", record_bytes(0, 0, 4096, held=(5, 0)))),
        saying="{dir}/attr_0_0.bin holds key 5 id 0"),
    "e07.undeclared_id": dict(
        make=broken_dir(write("attr_0_1.bin", record_bytes(0, 1, 7))),
        saying="{dir}/attr_0_1.bin records attribute 1, which the TOG does not declare"),
    "e08.rank_mismatch": dict(
        make=broken_dir(write("attr_0_0.bin", record_bytes(0, 0, [4096]))),
        saying="{dir}/attr_0_0.bin records attribute 0 at rank 1, the TOG declares rank 0"),
    "e09.truncated": dict(
        make=broken_dir(write("attr_0_0.bin", record_bytes(0, 0, 4096)[:-4])),
        saying="{dir}/attr_0_0.bin is 28 bytes, its header says 32"),
    "e09.negative_extent": dict(
        source="array_loop_tog.cpp", make=broken_dir(write("attr_0_1.bin", record_bytes(0, 1, ([-1], []))), WALK),
        saying="{dir}/attr_0_1.bin records extent -1 on dim 0"),
    "e10.stale_key": dict(
        make=broken_dir(write("attr_99_0.bin", record_bytes(99, 0, 1))),
        saying="{dir}/attr_99_0.bin records key 99, which programs.bin does not list"),
    "e11.missing_record": dict(
        make=broken_dir(remove(f"attr_{KEY1}_0.bin")),
        saying=f"work-item 1 (key {KEY1}) reads attribute 0, which {{dir}} does not record for that key"),
    "e13.foreign_file": dict(
        make=broken_dir(write("notes.txt", b"x")),
        saying="{dir}/notes.txt is neither programs.bin nor attr_<key>_<id>.bin"),
    "e13.record_name_with_a_sign": dict(
        make=broken_dir(write("attr_-1_0.bin", record_bytes(0, 0, 1))),
        saying="{dir}/attr_-1_0.bin is neither programs.bin nor attr_<key>_<id>.bin"),
}

TOG_HPP = {
    "dispatch_without_key": dict(define="COUNTED", saying="a TOG with runtime attributes dispatches without .key"),
    "undeclared_id": dict(define="UNDECLARED", saying="runtime attribute 1 is not declared"),
    "declared_rank_differs": dict(define="RANK_DIFFERS", saying="runtime attribute 0 is not declared"),
    "scalar_of_an_array": dict(source="array_loop_tog.cpp", define="SCALAR_OF_ARRAY",
                               saying="scalar(w, a) of attribute 1 of rank 1"),
    "element_index_count": dict(source="array_loop_tog.cpp", define="TWO_INDICES",
                                saying="element(w, a, i...) of attribute 1: 2 indices for rank 1"),
    "element_past_its_extent": dict(source="array_loop_tog.cpp", define="PAST_EXTENT",
                                    saying="element(w, a, i...) of attribute 1: index 3 on dim 0 outside its extent 3"),
    "extent_dim_outside_rank": dict(source="array_loop_tog.cpp", define="EXTENT_DIM",
                                    saying="extent(w, a, d) of attribute 1: dim 1 outside rank 1"),
    "gather_at_past_its_extent": dict(source="gather_tog.cpp", define="AT_PAST_EXTENT",
                                      saying="a gather on attribute 0 at [2]: index 2 on dim 0 outside its extent 2"),
    "gather_axis_past_the_tile": dict(source="gather_tog.cpp", define="AXIS_PAST_TILE",
                                      saying="a gather on attribute 0 on axis 2 of a 2-dim tile"),
    "gather_without_its_at": dict(source="gather_tog.cpp", define="NO_AT",
                                  saying="a gather on attribute 0 of rank 2: 0 indices at and a row of 1 dims on axis 0"),
    "gather_row_shorter_than_the_tile": dict(
        source="gather_tog.cpp", records={(0, 0): [r[:7] for r in ROWS], (0, 1): ELEMS},
        saying="a gather on attribute 0 at [0]: its row holds 7 entries for 8 tile elements on axis 0"),
    "gather_elements_of_another_shape": dict(
        source="gather_tog.cpp", records={(0, 0): ROWS, (0, 1): [[0] * 4] * 8},
        saying="a gather on attribute 1 at []: its row holds [8, 4] entries for [4, 8] tile elements on axis -1"),
}

RAW = {
    "read_outside_a_work_item": dict(define="IN_KERNEL", saying="[TOGSim] TOG {tog}: an attribute read outside a work-item"),
    "read_of_an_undeclared_id": dict(
        define="UNDECLARED", saying="[TOGSim] TOG {tog}: work-item 0 (key 0) reads attribute 1, which the TOG does not declare"),
    "offsets_of_another_count": dict(
        define="SHORT_OFFSETS", saying="[TOGSim] TOG {tog}: work-item 0 (key 0) gives a DMA 15 element offsets for 16 elements"),
}

DECLARATIONS = {"BAD_ID": "an attribute id outside 0..63", "GAP": "attribute ids are not 0..n-1 once each",
                "TWICE": "attribute ids are not 0..n-1 once each", "BAD_RANK": "an attribute of rank outside 0..8"}


def session(t, tog, attributes):
    """--models_list: one LAUNCH_KERNEL of `tog` (its directory gets the cycle table) with `attributes` as its
    attribute_path ("-" when None), then a DEVICE_SYNC."""
    home = tog.parent
    write_table(home / "tog_cycles.tsv", [(100, 0)])
    stream = t.work_dir / "commands.txt"
    stream.write_text(f"LAUNCH_KERNEL,0,0,0,{home}/kernel,{attributes or '-'},0\nDEVICE_SYNC,0,0,0,-,-,0\n")
    return t.execute("Simulator", ["--config", machine(t), "--models_list", stream, "--log_level", "trace"])


def loader_refuses(make, saying, in_session, source="scalar_offset_tog.cpp"):
    """The TOG is refused before anything is simulated, exit 1: with --tog_so, or as the error of the LAUNCH_KERNEL."""
    def case(t):
        tog = t.tog(source, into="k")
        path = make(t)
        said = saying.format(dir=path, work=t.work_dir)
        if in_session:
            session(t, tog, path).expect_refused(
                f"Error processing command 0 (type: LAUNCH_KERNEL): [TOGSim] TOG {tog}: {said}", exit=1)
        else:
            args = [] if path is None else ["--attribute_path", path]
            t.execute("Simulator", ["--config", machine(t), "--tog_so", tog, "--cycle_table",
                                    t.table("cycles", [(100, 0)]), *args]).expect_refused(
                f"[TOGSim] TOG {tog}: {said}", exit=1)
    return case


def header_refuses(saying, define=None, source="scalar_offset_tog.cpp", records=None, in_session=False):
    """togsim.hpp refuses the work-item's call on stderr, exit 1, before anything is simulated."""
    def case(t):
        tog = t.tog(source, (define,) if define else (), tensors=(MIB, MIB, MIB), into="k")
        path = t.attributes(records or {"scalar_offset_tog.cpp": OFFSETS, "array_loop_tog.cpp": WALK,
                                        "gather_tog.cpp": GATHER}[source])
        if in_session:
            run = session(t, tog, path)
        else:
            run = t.execute("Simulator", ["--config", machine(t), "--tog_so", tog, "--cycle_table",
                                          t.table("cycles", [(100, 0)]), "--attribute_path", path])
        run.expect_refused(f"togsim.hpp: {saying}", exit=1)
    return case


def raw_refuses(define, saying):
    """The raw C ABI misuses togsim.hpp cannot make: refused by TOGSim, exit 1."""
    def case(t):
        tog = t.tog("raw_read_tog.cpp", (define,), tensors=(MIB,))
        path = t.attributes({(0, 0): 0})
        t.execute("Simulator", ["--config", machine(t), "--tog_so", tog, "--cycle_table",
                                t.table("cycles", [(100, 0)]), "--attribute_path", path]).expect_refused(
            saying.format(tog=tog), exit=1)
    return case


def declaration_refused(define, generated):
    """A struct TOG's attribute list that breaks a rule does not compile, the rule's message in g++'s output; the
    same list through register_tog is refused when the .so is loaded."""
    def case(t):
        saying = DECLARATIONS[define]
        try:
            tog = t.tog("declare_tog.cpp", (define, "GENERATED") if generated else (define,))
        except Failure as failure:
            if generated:
                raise
            if f'fail_("{saying}")' not in str(failure):
                raise Failure(f'expected declare_tog.cpp -D{define} not to compile, quoting fail_("{saying}"); g++ '
                              f"said: {str(failure)[:400]}")
            return
        if not generated:
            raise Failure(f"expected declare_tog.cpp -D{define} not to compile, and it did")
        t.execute("Simulator", ["--config", machine(t), "--tog_so", tog, "--cycle_table",
                                t.table("cycles", [(100, 0)])]).expect_refused(f"togsim.hpp: {saying}", exit=1)
    return case


def no_attributes_ignore_a_path_without_records(t):
    """A TOG that declares none runs with no attribute_path, "-", a directory that is not there, and one that
    holds only programs.bin."""
    tog = t.tog("scalar_offset_tog.cpp", ("NO_ATTRIBUTES",), into="k")
    empty = t.attributes({}, programs=[0])
    for path in (None, "-", t.work_dir / "absent", empty):
        args = [] if path is None else ["--attribute_path", path]
        t.execute("Simulator", ["--config", machine(t), "--tog_so", tog, "--cycle_table",
                                t.table("cycles", [(100, 0)]), *args]).expect_ok()
        session(t, tog, path).expect_ok()


def no_attributes_refuse_records(in_session):
    """E12: a TOG that declares none, given a directory that holds records, is refused."""
    def case(t):
        tog = t.tog("scalar_offset_tog.cpp", ("NO_ATTRIBUTES",), into="k")
        path = t.attributes(OFFSETS)
        said = f"[TOGSim] TOG {tog}: it declares no runtime attribute and {path} holds 2 record(s)"
        if in_session:
            session(t, tog, path).expect_refused(f"Error processing command 0 (type: LAUNCH_KERNEL): {said}", exit=1)
        else:
            t.execute("Simulator", ["--config", machine(t), "--tog_so", tog, "--cycle_table",
                                    t.table("cycles", [(100, 0)]), "--attribute_path", path]).expect_refused(said, exit=1)
    return case


def session_reads_its_launch_path(t):
    """--models_list: the records come from the LAUNCH_KERNEL's attribute_path, as with --tog_so."""
    tog = t.tog("scalar_offset_tog.cpp", into="k")
    run = session(t, tog, t.attributes(OFFSETS)).expect_ok()
    if issued(run, "MOVIN") != [4096 * 4, 20480 * 4]:
        run.fail(f"expected the loads at 0x4000 and 0x14000, got {list(map(hex, issued(run, 'MOVIN')))}")


def attribute_path_belongs_to_tog_so(t):
    """--attribute_path names one TOG's records: with --models_list it is refused, each launch names its own."""
    stream = t.work_dir / "commands.txt"
    stream.write_text("DEVICE_SYNC,0,0,0,-,-,0\n")
    t.execute("Simulator", ["--config", machine(t), "--models_list", stream, "--attribute_path",
                            t.work_dir]).expect_refused("[TOGSim] --attribute_path belongs to one TOG", exit=1)


CASES = [
    Case("scalar_offset_by_key", scalar_offset_by_key),
    Case("runtime_trip_count", runtime_trip_count(())),
    Case("runtime_trip_count_from_extent", runtime_trip_count(("BY_EXTENT",))),
    Case("gather_offsets", gather_offsets),
    Case("session_reads_its_launch_path", session_reads_its_launch_path),
    Case("no_attributes_ignore_a_path_without_records", no_attributes_ignore_a_path_without_records),
    Case("attribute_path_belongs_to_tog_so", attribute_path_belongs_to_tog_so),
] + [Case(f"refused.{name}{where}", loader_refuses(**how, in_session=bool(where)))
     for name, how in LOADER.items() for where in ("", ".session")] + [
    Case(f"refused.e12.no_attributes_with_records{where}", no_attributes_refuse_records(bool(where)))
    for where in ("", ".session")] + [
    Case(f"refused.togsim_hpp.{name}{where}", header_refuses(**how, in_session=bool(where)))
    for name, how in TOG_HPP.items() for where in ("", ".session")] + [
    Case(f"refused.raw.{name}", raw_refuses(**how)) for name, how in RAW.items()] + [
    Case(f"refused.declaration.{define.lower()}{'.generated' if generated else ''}", declaration_refused(define, generated))
    for define in DECLARATIONS for generated in (False, True)]
