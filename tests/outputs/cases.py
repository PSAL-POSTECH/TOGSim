"""The lines TOGSim prints, as a consumer parses them: the shapes of a run's lines, compared with a golden file.
A shape is a line without its timestamp, each absolute path, hex value and number replaced by a placeholder; a path
under the work directory, the build or this repository is recognized by that root, whatever characters it holds."""
import difflib
import re
from pathlib import Path

from togtest import REPO, Case, Failure

HERE = Path(__file__).resolve().parent
SPDLOG = re.compile(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+\] (\[\w+\] .*)$")
MASKS = [
    (re.compile(r"(?<![\w.])/[\w.+-]+(?:/[\w.+-]+)+"), "<path>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "<hex>"),
    (re.compile(r"\btag\d+"), "tagN"),
    (re.compile(r"(?<![\w.])\d+\.\d+"), "N.N"),
    (re.compile(r"(?<![A-Za-z_\d])\d+"), "N"),
]
UNITS = [(0, "TPUV6E_MXU.rows", "rows", 1, 256), (0, "Vpu", "cycles", 1, 100), (0, "Issue", "micro-ops", 12, 24)]


def root_masks(*roots):
    """One mask per distinct root, as given and resolved, longest first: the root and what follows it of a path."""
    spelled = {str(r) for root in roots for r in (root, Path(root).resolve())}
    return [(re.compile(re.escape(r) + r"(?:/[\w.+-]+)*"), "<path>") for r in sorted(spelled, key=len, reverse=True)]


def shapes(log, roots=()):
    """The golden text of a log: the shapes of its spdlog lines and of the config echo (other bare lines belong to
    Ramulator2 and the cache model), each once per section. Before and after the run they keep their order;
    during the run, where what comes first depends on the timing, they are sorted."""
    names = ["before the run, in order", "during the run, sorted", "after the run, in order"]
    sections, at, echo = {name: [] for name in names}, 0, False
    for line in log.splitlines():
        logged = SPDLOG.match(line)
        if not logged and not echo:
            continue
        text = logged[1] if logged else line
        echo = text.endswith("PyTorchSim config:") if logged else echo
        if at == 1 and re.match(r"\[info\] (\[TOGSim\] Total cycles:|Simulation finished)", text):
            at = 2
        for mask, placeholder in root_masks(*roots) + MASKS:
            text = mask.sub(placeholder, text)
        if text not in sections[names[at]]:
            sections[names[at]].append(text)
        if at == 0 and text.endswith("======Start Simulation====="):
            at = 1
    sections[names[1]].sort()
    return "".join(f"# {name}\n" + "".join(f"{shape}\n" for shape in found) for name, found in sections.items())


def expect_golden(t, run, name):
    """The shapes of `run` are tests/outputs/<name>.golden; on a difference the shapes seen are left beside the log."""
    golden, seen = HERE / f"{name}.golden", t.work_dir / f"{name}.seen"
    seen.write_text(shapes(run.log, (t.work_dir, t.build_dir, REPO)))
    want = golden.read_text() if golden.is_file() else ""
    if seen.read_text() != want:
        diff = list(difflib.unified_diff(want.splitlines(), seen.read_text().splitlines(), "golden", "printed", n=0,
                                         lineterm=""))[2:]
        raise Failure(f"the lines printed differ in shape from tests/outputs/{name}.golden in {len(diff)} diff lines "
                      f"(- golden, + printed); to accept them: cp {seen} {golden}\n" + "\n".join(diff[:40]))


def tog_so_with_unit_table(t):
    """--tog_so on tpuv6e as shipped, with a unit table, at --log_level trace: the per-unit utilization lines."""
    run = t.simulate(t.tog("every_instruction_tog.cpp", tensors=[1 << 20] * 5), t.machine(), cycles=[600],
                     units=UNITS).expect_ok()
    expect_golden(t, run, "tog_so_with_unit_table")


def tog_so_without_array_split(t):
    """--tog_so with systolic_array_split: none and no unit table, at info: the per-array and per-unit lines."""
    run = t.simulate(t.tog("every_instruction_tog.cpp", tensors=[1 << 20] * 5),
                     t.machine(systolic_array_split="none"), cycles=[600], log_level="info").expect_ok()
    expect_golden(t, run, "tog_so_without_array_split")


CASES = [Case("tog_so_with_unit_table", tog_so_with_unit_table),
         Case("tog_so_without_array_split", tog_so_without_array_split)]
