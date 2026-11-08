"""Where a launch's tensors sit: at the addresses tog_tensor_addrs.txt gives when it is beside tog.so, else back to
back from 0 (each run says which); a file that disagrees with tog_tensors.txt is refused; --version prints 1.1.0."""
import re

from togtest import MIB, VERSION, Case, write_table

TOG = "../kernel_order/chain_tog.cpp"
ADDRS = "tog_tensor_addrs.txt"
GIVEN = f"tensors at the addresses {ADDRS} gives"
FROM_0 = f"no {ADDRS}, 3 tensors laid out from address 0"
TO_MEMORY = re.compile(r"\[REQ_TO_MEM *\] (MOVIN|MOVOUT) dram=0x([0-9a-f]+) channel=\d+$", re.M)
A, X, B, UNUSED = 0x40000000, 0x200000000, 0x100000, 0xC0000000
TILE_REACH = "'s tiles span"


def kernel(t, k, addrs=None, tensors=(MIB, MIB, MIB)):
    """<work dir>/k<k>/: the chain's kernel `k` (loads argument k, stores argument k + 1) over three tensors, with
    `addrs` (lines of tog_tensor_addrs.txt) beside it when given. Returns the directory."""
    home = t.tog(TOG, (f"IN={k}",), tensors=tensors, into=f"k{k}").parent
    write_table(home / "tog_cycles.tsv", [(600, 0)])
    write_table(home / "tog_units.tsv", [(0, "TPUV6E_MXU.rows", "rows", 1, 256), (0, "Vpu", "cycles", 1, 100)])
    if addrs is not None:
        (home / ADDRS).write_text("".join(f"{line}\n" for line in addrs))
    return home


def session(t, homes, machine=None):
    """--models_list: each of `homes` launched in order on partition 0, a DEVICE_SYNC after each."""
    stream = t.work_dir / "commands.txt"
    stream.write_text("".join(f"LAUNCH_KERNEL,{k},0,0,{h}/kernel,{h}/attributes,0\nDEVICE_SYNC,0,0,0,-,-,0\n"
                              for k, h in enumerate(homes)))
    return t.execute("Simulator", ["--config", machine or t.machine(), "--models_list", stream,
                                   "--log_level", "trace"])


def requests(run):
    """{opcode: set of addresses} of the requests that entered the memory partition."""
    out = {"MOVIN": set(), "MOVOUT": set()}
    for m in TO_MEMORY.finditer(run.log):
        out[m[1]].add(int(m[2], 16))
    return out


def inside(addrs, base):
    return all(base <= a < base + MIB for a in addrs)


def given_addresses_place_each_tensor(t):
    """Kernel 0 reads A and writes X, kernel 1 reads X and writes B, X at one address in both files and at another
    argument slot's address in neither: every request lands in its tensor's range, and kernel 1 reads exactly the
    addresses kernel 0 wrote. Breaks when the bases ignore the file or are re-derived from 0 per launch."""
    homes = [kernel(t, 0, [f"a {A} {MIB} 16384 in", f"x {X} {MIB} 16384 out", f"u {UNUSED} {MIB} 16384 out"]),
             kernel(t, 1, [f"u {UNUSED} {MIB} 16384 in", f"x {X} {MIB} 16384 in", f"b {B} {MIB} 16384 out"])]
    run = session(t, homes).expect_ok()
    got = requests(run)
    loads_x = {a for a in got["MOVIN"] if X <= a < X + MIB}
    stores_x = {a for a in got["MOVOUT"] if X <= a < X + MIB}
    if (not inside(got["MOVIN"] - loads_x, A) or not inside(got["MOVOUT"] - stores_x, B) or not loads_x
            or loads_x != stores_x or run.log.count(GIVEN) != 2):
        run.fail(f"expected loads in A ({A:#x}) and X ({X:#x}), stores in X and B ({B:#x}), X's loads the addresses "
                 f"stored, and two lines saying \"{GIVEN}\"; got {len(got['MOVIN'])} loads, {len(got['MOVOUT'])} "
                 f"stores, {len(loads_x)} loads of X, {len(stores_x)} stores to X, {run.log.count(GIVEN)} lines")


def absent_lays_out_from_0(t):
    """Without the file kernel 0's three tensors sit at 0, 1 MiB and 2 MiB, as before 1.1.0, and the run says so."""
    run = session(t, [kernel(t, 0)]).expect_ok()
    got = requests(run)
    if not got["MOVIN"] or not inside(got["MOVIN"], 0) or not inside(got["MOVOUT"], MIB) or FROM_0 not in run.log:
        run.fail(f"expected loads in [0, 1 MiB), stores in [1 MiB, 2 MiB) and a line \"{FROM_0}\"; got loads "
                 f"{sorted(got['MOVIN'])[:2]}..., stores {sorted(got['MOVOUT'])[:2]}...")


def refused(lines, saying):
    """A file of `lines` beside kernel 0 is refused with `saying` before anything runs, exit 1."""
    return lambda t: session(t, [kernel(t, 0, lines)]).expect_refused(saying, exit=1)


REFUSED = {
    "refused.fewer_lines": ([f"a {A} {MIB} 16384 in", f"x {X} {MIB} 16384 out"],
                            f"{ADDRS} lists 2 tensors, tog_tensors.txt 3"),
    "refused.other_bytes": ([f"a {A} {MIB} 16384 in", f"x {X} 4096 16384 out", f"u {UNUSED} {MIB} 16384 out"],
                            f"tensor 1 (x) is 4096 bytes there and {MIB} in tog_tensors.txt"),
    "refused.malformed_line": ([f"a {A} {MIB} 16384 in", f"x {X} {MIB}", f"u {UNUSED} {MIB} 16384 out"],
                               f"line 2: \"x {X} {MIB}\" is not `name addr bytes align role`"),
}


def tiled(t, gap):
    """Kernel 0 over one-row tensors tiled [8, 128] (each padded to 8 MiB), tensor 1 placed `gap` bytes after
    tensor 0."""
    home = kernel(t, 0, [f"a {A} {MIB} 16384 in", f"x {A + gap} {MIB} 16384 out", f"u {UNUSED} {MIB} 16384 out"])
    (home / "tog_tensor_shapes.txt").write_text("1 262144 4\n" * 3)
    return session(t, [home], t.machine(tog_tensor_tile="[8, 128]"))


def tiles_reaching_the_next_tensor_warned(t):
    """Tensor 0's padded 8 MiB reach tensor 1, 2 MiB after it: one warning, and the run finishes."""
    tiled(t, 2 * MIB).expect_warned_once(TILE_REACH)


def tiles_clear_of_the_next_tensor_quiet(t):
    tiled(t, 16 * MIB).expect_no_warning(TILE_REACH)


def version_printed(t):
    """--version prints the version alone and exits 0: what a producer reads to know the file is honoured."""
    run = t.execute("Simulator", ["--version"]).expect_ok()
    if run.log.strip() != VERSION:
        run.fail(f"expected --version to print {VERSION} alone, got {run.log.strip()!r}")


CASES = [Case("given_addresses_place_each_tensor", given_addresses_place_each_tensor),
         Case("absent_lays_out_from_0", absent_lays_out_from_0)] + [
    Case(name, refused(*how)) for name, how in REFUSED.items()] + [
    Case("tiles_reaching_the_next_tensor_warned", tiles_reaching_the_next_tensor_warned),
    Case("tiles_clear_of_the_next_tensor_quiet", tiles_clear_of_the_next_tensor_quiet),
    Case("version_printed", version_printed)]
