"""What every tests/*/cases.py shares: build a TOG, run the Simulator on it, read its log, say what is expected.
An expectation that does not hold raises Failure: what was expected, what was seen, where the log is."""
import hashlib
import os
import re
import resource
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TPUV6E = "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"
MIB = 1 << 20
TIMEOUT_S = 600
TYPED_WARNING = "uses the typed grouping"
EVENT = re.compile(r"\[(\d+)\]\[Core (\d+)\]\[(\w+) *\]\[INST_ID=(\d+)\] ([A-Z_]+)(?: \((.*)\))?$", re.M)
SCHEDULED = re.compile(r"TILE_SCHEDULED *\] spad_footprint=(\d+) max_dispatch=(\d+)")
SAID = re.compile(r"^(?:\[[^\]]*\] \[(?:error|critical)\] |\s*what\(\):\s*)(.*)$|^(togsim: .*)$", re.M)
WARNED = re.compile(r"^\[[^\]]*\] \[warning\] (.*)$", re.M)
_built, _locks, _guard = {}, {}, threading.Lock()


def no_core_dumps():
    """Keep a run that aborts (a rejected config exits 134) from dumping core, for this process and its children.
    A piped core_pattern (systemd-coredump) still runs its helper at RLIMIT_CORE 0; 1 is what the kernel skips."""
    try:
        piped = Path("/proc/sys/kernel/core_pattern").read_text().startswith("|")
    except OSError:
        piped = False
    hard = resource.getrlimit(resource.RLIMIT_CORE)[1]
    want = 1 if piped else 0
    resource.setrlimit(resource.RLIMIT_CORE, (want if hard == resource.RLIM_INFINITY else min(want, hard), hard))


class Failure(Exception):
    """An expectation that did not hold; its text is what the runner prints after FAIL."""


@dataclass(frozen=True)
class Case:
    """One line of the runner's output: `name` within its directory, and the function (of a Ctx) that states it."""
    name: str
    run: object


@dataclass(frozen=True)
class Event:
    """One trace line, `[cycle][Core c][TAG][INST_ID=n] OPCODE (key=value ...)`."""
    cycle: int
    core: int
    tag: str
    inst: int
    opcode: str
    fields: dict


@dataclass(frozen=True, order=True)
class Compute:
    """A compute that finished: `cycles` is what its INST_ISSUED line states, `start` is finish - cycles."""
    core: int
    start: int
    finish: int
    compute_type: int
    cycles: int


def write_table(path, rows):
    """Write `rows` as tab-separated lines (a cycle table or a unit table) and return the path."""
    path.write_text("".join("\t".join(str(f) for f in row) + "\n" for row in rows))
    return path


def read_table(path):
    """The tab-separated rows of `path`, each a list of strings."""
    return [line.split("\t") for line in Path(path).read_text().splitlines() if line]


def _compile(source, defines, cache_dir):
    """`source` compiled against include/togsim_runtime.h with -D<define> each, once per run of the suite:
    the .so, or the Failure that carries what g++ said."""
    key = (str(source), tuple(sorted(defines)))
    with _guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if key not in _built:
            tag = hashlib.sha1(repr(key).encode()).hexdigest()[:10]
            out = cache_dir / f"{source.stem}-{tag}.so"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cmd = [os.environ.get("CXX", "g++"), "-std=c++17", "-shared", "-fPIC", "-O1", f"-I{REPO / 'include'}",
                   *[f"-D{d}" for d in sorted(defines)], str(source), "-o", str(out)]
            done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            _built[key] = out if done.returncode == 0 else Failure(
                f"{source.name} does not compile ({' '.join(cmd)}):\n{done.stdout.rstrip()}")
        return _built[key]


class Run:
    """One finished run of a binary: its exit status, everything it printed, and the expectations on them."""

    def __init__(self, cmd, rc, log_path, seconds, timeout_s):
        self.cmd, self.rc, self.log_path, self.seconds = cmd, rc, log_path, seconds
        self.timed_out = rc is None
        self.timeout_s = timeout_s
        self.log = log_path.read_text(errors="replace")

    def fail(self, what):
        """Raise the Failure `what`, with the log it was read from."""
        raise Failure(f"{what}  [log: {self.log_path}]")

    def said(self):
        """The error lines of the log: spdlog errors and an uncaught exception without their prefix, the loader's
        stderr lines whole (`togsim: ...`)."""
        return [m[1] if m[2] is None else m[2] for m in SAID.finditer(self.log)]

    def _exit(self):
        return f"timed out after {self.timeout_s} s" if self.timed_out else f"exit {self.rc}"

    def _seen(self):
        """The error lines joined, or the last line of the log when it has none (Ramulator2 and BookSim2 print bare)."""
        last = self.log.strip().splitlines()[-1:] or ["nothing printed"]
        return "; ".join(self.said()) or f"no error line, the log ends: {last[0]}"

    def expect_ok(self):
        """The run finished with exit 0."""
        if self.rc != 0:
            self.fail(f"expected exit 0, got {self._exit()}: {self._seen()}")
        return self

    def expect_refused(self, saying, exit=None):
        """The run ended non-zero (with `exit` exactly, when given) and its first error line contains `saying`."""
        seen = self._seen()
        if self.timed_out or self.rc == 0:
            self.fail(f'expected a refusal saying "{saying}", got {self._exit()}: {seen}')
        if saying not in (self.said()[:1] or [""])[0]:
            self.fail(f'expected the refusal to say "{saying}", got {self._exit()}: {seen}')
        if exit is not None and self.rc != exit:
            self.fail(f'expected the refusal "{saying}" to exit {exit}, got {self._exit()}')
        return self

    def warned(self, saying):
        """The warn-level lines that contain `saying`, without their prefix."""
        return [m[1] for m in WARNED.finditer(self.log) if saying in m[1]]

    def expect_warned_once(self, saying):
        """The run finished and exactly one warn-level line contains `saying`."""
        self.expect_ok()
        if len(self.warned(saying)) != 1:
            self.fail(f'expected one warning saying "{saying}", got {len(self.warned(saying))}; the warnings: '
                      f'{self.warned("") or "none"}')
        return self

    def expect_no_warning(self, saying):
        """The run finished and no warn-level line contains `saying`."""
        self.expect_ok()
        if self.warned(saying):
            self.fail(f'expected no warning saying "{saying}", got: {self.warned(saying)}')
        return self

    def total_cycles(self):
        """N of the `Total execution cycles: N` line."""
        m = re.search(r"Total execution cycles: (\d+)", self.log)
        if not m:
            self.fail(f"expected a 'Total execution cycles:' line, found none ({self._exit()})")
        return int(m[1])

    def expect_total_cycles(self, cycles):
        """The run finished and `Total execution cycles:` reads `cycles`."""
        self.expect_ok()
        if self.total_cycles() != cycles:
            self.fail(f"expected Total execution cycles: {cycles}, got {self.total_cycles()}")
        return self

    def events(self):
        """Every instruction trace line, in log order (needs --log_level trace)."""
        return [Event(int(m[1]), int(m[2]), m[3], int(m[4]), m[5],
                      dict(kv.split("=", 1) for kv in (m[6] or "").split(" ") if "=" in kv))
                for m in EVENT.finditer(self.log)]

    def when(self, tag, inst, core=0):
        """The cycle of the last `tag` line of instruction `inst` on `core`."""
        cycles = [e.cycle for e in self.events() if (e.tag, e.inst, e.core) == (tag, inst, core)]
        if not cycles:
            self.fail(f"expected a {tag} line of instruction {inst} on core {core}, found none ({self._exit()})")
        return cycles[-1]

    def computes(self):
        """Every compute that was issued and finished, in issue order."""
        issued, finished = {}, {}
        for e in self.events():
            if e.opcode == "COMP" and e.tag in ("INST_ISSUED", "INST_FINISHED"):
                (issued if e.tag == "INST_ISSUED" else finished)[(e.core, e.inst)] = e
        return [Compute(core, finished[core, inst].cycle - int(e.fields["compute_cycle"]), finished[core, inst].cycle,
                        int(e.fields["compute_type"]), int(e.fields["compute_cycle"]))
                for (core, inst), e in issued.items() if (core, inst) in finished]

    def scheduled(self):
        """(spad_footprint, max_dispatch) of every TILE_SCHEDULED line, in log order."""
        return [(int(m[1]), int(m[2])) for m in SCHEDULED.finditer(self.log)]


class Ctx:
    """What a case function is handed: the directory of its cases.py, a work directory of its own, the build."""

    def __init__(self, source_dir, work_dir, build_dir, tog_cache, typed=False):
        self.source_dir, self.work_dir, self.build_dir, self.tog_cache = source_dir, work_dir, build_dir, tog_cache
        self.typed = typed
        self._runs = 0

    def tog(self, source, defines=(), tensors=(MIB, MIB, MIB), shape=None, into=""):
        """`source` (beside cases.py) compiled with -D<define> each to <work dir>/<into>/tog.so, with the sidecars
        TOGSim reads beside it: tog_tensors.txt (bytes per tensor argument) and, given `shape`, tog_shape.txt."""
        built = _compile(self.source_dir / source, defines, self.tog_cache)
        if isinstance(built, Failure):
            raise built
        tog_dir = self.work_dir / into
        tog_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(built, tog_dir / "tog.so")
        (tog_dir / "tog_tensors.txt").write_text("".join(f"{b}\n" for b in tensors))
        if shape is not None:
            (tog_dir / "tog_shape.txt").write_text("".join(f"{v}\n" for v in shape))
        return tog_dir / "tog.so"

    def machine(self, base=TPUV6E, **keys):
        """<work dir>/machine.yml: configs/<base> with each of `keys` set (in place where the base has it, appended
        otherwise) and the Ramulator2 / BookSim2 paths made absolute, as the file leaves configs/."""
        src = REPO / "configs" / base
        lines, left = [], dict(keys)
        for line in src.read_text().splitlines():
            m = re.match(r"([A-Za-z_]\w*):\s*(.*)$", line)
            if m and m[1] in left:
                line = f"{m[1]}: {left.pop(m[1])}"
            elif m and m[1] in ("ramulator_config_path", "booksim_config_path") and not os.path.isabs(m[2]):
                line = f"{m[1]}: {os.path.normpath(src.parent / m[2])}"
            lines.append(line)
        lines += [f"{k}: {v}" for k, v in left.items()]
        out = self.work_dir / "machine.yml"
        out.write_text("\n".join(lines) + "\n")
        return out

    def table(self, name, rows):
        """<work dir>/<name>.tsv holding `rows`, tab-separated."""
        return write_table(self.work_dir / f"{name}.tsv", rows)

    def execute(self, binary, args, timeout=TIMEOUT_S):
        """Run build/bin/<binary> with `args` in the work directory: stdout and stderr to sim.log (sim2.log, ... for a
        case's later runs), the exit status as a shell reports it (134 for an abort) to sim.exit. A typed TOG fails
        a case outside tests/typed/, and a Simulator run without the typed warning fails a case inside it."""
        exe = self.build_dir / "bin" / binary
        if not exe.is_file():
            raise Failure(f"{exe} is absent: build the default target (make), not Simulator alone")
        self._runs += 1
        log_path = self.work_dir / ("sim.log" if self._runs == 1 else f"sim{self._runs}.log")
        cmd = [str(exe), *map(str, args)]
        start = time.monotonic()
        with open(log_path, "wb") as log:
            try:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=self.work_dir,
                                    timeout=timeout).returncode
                rc = 128 - rc if rc < 0 else rc
            except subprocess.TimeoutExpired:
                rc = None
        log_path.with_suffix(".exit").write_text(f"{'timeout' if rc is None else rc}\n")
        run = Run(cmd, rc, log_path, time.monotonic() - start, timeout)
        warned = TYPED_WARNING in run.log
        if warned and not self.typed:
            run.fail("the TOG uses the deprecated typed grouping (compute_type 1, 2 or 3): its case belongs in "
                     "tests/typed/")
        if self.typed and binary == "Simulator" and not warned:
            run.fail(f"the run ended ({run._exit()}) without the typed grouping warning: it loaded no typed TOG, so "
                     "its case does not belong in tests/typed/")
        return run

    def simulate(self, tog, machine, cycles, units=None, log_level="trace", timeout=TIMEOUT_S):
        """Replay `tog` on `machine`: `cycles` lists the cycles of tile 0, 1, ... (the cycle table, overlap 0),
        `units` the rows of the unit table when there is one."""
        args = ["--config", machine, "--tog_so", tog,
                "--cycle_table", self.table("cycles", [(c, 0) for c in cycles])]
        if units is not None:
            args += ["--unit_table", self.table("units", units)]
        return self.execute("Simulator", [*args, "--log_level", log_level], timeout)
