"""--result <path>: the run's results as JSON, each value the one its end-of-run line prints (the line rounds what
the file holds); a session's file also holds each kernel's completion and the wall clock. No --result, no file;
a path that cannot be written is refused."""
import json
import re

from togtest import MIB, Case, write_table

UNITS = [(0, "TPUV6E_MXU.rows", "rows", 1, 256), (0, "Vpu", "cycles", 1, 100), (0, "Issue", "micro-ops", 12, 24)]
UNIT = re.compile(r"\] (Core \[\d+\]|Total) : (.+?) [Uu]tilization\(%\): (\d+\.\d+), active[_ ]cycles?: (\d+), "
                  r"idle[_ ]cycles?: (\d+)$", re.M)
DMA = re.compile(r"\] Core \[(\d+)\] : DMA active_cycles: (\d+), DMA idle_cycles: (\d+), DRAM BW: (\d+\.\d+) GB/s "
                 r"\((\d+) responses\)$", re.M)
CHANNEL = re.compile(r"\[DRAM\] channel (\d+) \| (\d+\.\d+) GB/s avg\., (\d+\.\d+)% of utilization \| (\d+) reads, "
                     r"(\d+) writes$", re.M)
COMBINED = re.compile(r"\[DRAM\] channels \S+ combined \| (\d+\.\d+) GB/s aggregate, (\d+\.\d+)% of utilization "
                      r"\(avg\. per channel\) \| (\d+) reads, (\d+) writes$", re.M)
DONE = re.compile(r"\[Scheduler (\d+)\] Kernel (\d+) has completed .* finished at cycle (\d+)$", re.M)


def read(run, path):
    if not path.is_file():
        run.fail(f"expected --result to write {path}")
    return json.loads(path.read_text())


def close(printed, held):
    """`printed` is `held` as the line rounds it, to its decimals."""
    decimals = len(printed.split(".")[1])
    return held is not None and abs(float(printed) - held) <= 0.5 * 10 ** -decimals + 1e-12


def expect_lines_held(run, got):
    """Every end-of-run unit, DMA and DRAM line, and the total, is in `got` with the same values."""
    final = run.log[run.log.index("Simulation finished"):]
    seen = 0
    for who, name, pct, active, idle in UNIT.findall(final):
        held = got["units"].get(who, {}).get(name)
        if held is None or not close(pct, held["utilization_pct"]) or (held["active_cycles"], held["idle_cycles"]) \
                != (int(active), int(idle)):
            run.fail(f"{who} : {name} prints {pct}% {active} {idle}; the file holds {held}")
        seen += 1
    for core, active, idle, bw, responses in DMA.findall(final):
        held = got["dma"].get(core)
        if held is None or (held["active_cycles"], held["idle_cycles"], held["responses"]) \
                != (int(active), int(idle), int(responses)) or not close(bw, held["dram_bw_gbps"]):
            run.fail(f"Core [{core}] DMA prints {active} {idle} {bw} {responses}; the file holds {held}")
        seen += 1
    for ch, bw, pct, reads, writes in CHANNEL.findall(final):
        held = got["dram"]["channels"].get(ch)
        if held is None or (held["reads"], held["writes"]) != (int(reads), int(writes)) \
                or not close(bw, held["bandwidth_gbps"]) or not close(pct, held["utilization_pct"]):
            run.fail(f"DRAM channel {ch} prints {bw} {pct} {reads} {writes}; the file holds {held}")
    for bw, pct, reads, writes in COMBINED.findall(final):
        held = got["dram"]["combined"]
        if held is None or (held["reads"], held["writes"]) != (int(reads), int(writes)) \
                or not close(bw, held["bandwidth_gbps"]) or not close(pct, held["utilization_pct"]):
            run.fail(f"DRAM combined prints {bw} {pct} {reads} {writes}; the file holds {held}")
    if got["total_cycles"] != run.total_cycles():
        run.fail(f"the run's total is {run.total_cycles()} cycles; the file holds {got['total_cycles']}")
    if not seen:
        run.fail("expected unit or DMA lines after Simulation finished")


def tog_so_with_unit_table(t):
    """--tog_so on tpuv6e with a unit table: each unit of the table, the DMA, the DRAM and the total."""
    path = t.work_dir / "result.json"
    run = t.execute("Simulator", ["--config", t.machine(), "--tog_so", t.tog("../outputs/every_instruction_tog.cpp",
                                                                             tensors=[1 << 20] * 5),
                                  "--cycle_table", t.table("cycles", [(600, 0)]), "--unit_table",
                                  t.table("units", UNITS), "--result", path]).expect_ok()
    got = read(run, path)
    expect_lines_held(run, got)
    if set(got["units"].get("Core [0]", {})) != {"TPUV6E_MXU.rows", "Vpu", "Issue"} or got["wall_clock_s"] is not None:
        run.fail(f"expected Core [0]'s three units and no wall clock (--tog_so prints none); got {got}")


def tog_so_without_array_split(t):
    """--tog_so with systolic_array_split: none and no unit table: each array, the vector and cross-lane units."""
    path = t.work_dir / "result.json"
    run = t.execute("Simulator", ["--config", t.machine(systolic_array_split="none"), "--tog_so",
                                  t.tog("../outputs/every_instruction_tog.cpp", tensors=[1 << 20] * 5),
                                  "--cycle_table", t.table("cycles", [(600, 0)]), "--result", path]).expect_ok()
    got = read(run, path)
    expect_lines_held(run, got)
    if not {"Vector unit", "Cross-lane unit", "Systolic array [0]"} <= set(got["units"].get("Core [0]", {})):
        run.fail(f"expected Core [0]'s arrays, vector unit and cross-lane unit; got {list(got['units'])}")


def session_of_two_kernels(t):
    """--models_list with two kernels: their completions, as the scheduler logs them, and the wall clock."""
    homes = []
    for kernel_id, cycles in ((0, 600), (1, 900)):
        home = t.tog("../session/kernel_tog.cpp", ("ITEMS=2",), tensors=(MIB, MIB), into=f"k{kernel_id}").parent
        write_table(home / "tog_cycles.tsv", [(cycles, 0)])
        write_table(home / "tog_units.tsv", [(0, "TPUV6E_MXU.rows", "rows", 1, 256), (0, "Vpu", "cycles", 1, 100)])
        homes.append(home)
    stream = t.work_dir / "commands.txt"
    stream.write_text("".join(f"LAUNCH_KERNEL,{k},0,0,{h}/kernel,{h}/attributes,0\nDEVICE_SYNC,0,0,0,-,-,0\n"
                              for k, h in enumerate(homes)))
    path = t.work_dir / "result.json"
    run = t.execute("Simulator", ["--config", t.machine(), "--models_list", stream, "--result", path]).expect_ok()
    got = read(run, path)
    expect_lines_held(run, got)
    done = [(int(s), int(k), int(c)) for s, k, c in DONE.findall(run.log)]
    held = [(k["scheduler"], k["kernel_id"], k["finished"]) for k in got["kernels"]]
    if held != done or len(held) != 2 or not isinstance(got["wall_clock_s"], (int, float)):
        run.fail(f"expected the two completions {done} and a wall clock; the file holds {held}, "
                 f"wall clock {got['wall_clock_s']}")


def no_option_no_file(t):
    """Without --result the run writes no JSON."""
    run = t.simulate(t.tog("../outputs/every_instruction_tog.cpp", tensors=[1 << 20] * 5), t.machine(),
                     cycles=[600], units=UNITS, log_level="info").expect_ok()
    if list(t.work_dir.glob("*.json")):
        run.fail(f"expected no JSON without --result; found {list(t.work_dir.glob('*.json'))}")


def unwritable_path_is_refused(t):
    """A --result in a directory that does not exist is refused, exit 1."""
    run = t.execute("Simulator", ["--config", t.machine(), "--tog_so", t.tog("../outputs/every_instruction_tog.cpp",
                                                                             tensors=[1 << 20] * 5),
                                  "--cycle_table", t.table("cycles", [(600, 0)]), "--unit_table",
                                  t.table("units", UNITS), "--result", t.work_dir / "absent" / "result.json"])
    run.expect_refused("--result: cannot write", exit=1)


CASES = [Case("tog_so_with_unit_table", tog_so_with_unit_table),
         Case("tog_so_without_array_split", tog_so_without_array_split),
         Case("session_of_two_kernels", session_of_two_kernels),
         Case("no_option_no_file", no_option_no_file),
         Case("unwritable_path_is_refused", unwritable_path_is_refused)]
