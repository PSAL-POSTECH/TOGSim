"""The examples of the README: examples/run.sh <name> prints <name>/expected.txt, and the documents quote only that.
A quote is a fenced block of a README.md or of docs/*.md: log lines as run.sh prints them, or lines of an example."""
import difflib
import re
import shutil
import subprocess

from togtest import REPO, TIMEOUT_S, Case, Failure

EXAMPLES = REPO / "examples"
NAMES = sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir())
BUILT = shutil.ignore_patterns("*.so", "*.gen.*", "no_tensors")
DOCUMENTS = [p for p in [REPO / "README.md", *sorted((REPO / "docs").glob("*.md")), EXAMPLES / "README.md",
                         REPO / "tests" / "README.md"] if p.is_file()]
FENCED = re.compile(r"^```(\w*)\n(.*?)^```$", re.M | re.S)
TIMESTAMP = re.compile(r"^\[\d{4}-\d\d-\d\d [\d:.]+\] ")
RUN_SH_LINE = re.compile(r"\[(trace|info|warning|error)\] |togsim: |\+ |# examples/|exit \d+$| *what\(\):|Config file ")
REFUSAL_ROW = re.compile(r"^\| `((?:togsim: |\[TOGSim\] |\[Simulator\] |Config file |what\(\):)[^`]*)` "
                         r"\| (\d+) \|", re.M)
PLACEHOLDER = re.compile(r"<\w+>|\\\(\\\.\\\.\\\.\\\)")


def prints_expected(name):
    """examples/run.sh <name> runs in the work directory, on a copy of examples/ beside links to include/, configs/
    and the build: the commands it echoes and the result lines of every run are those of <name>/expected.txt."""
    def case(t):
        shutil.copytree(EXAMPLES, t.work_dir / "examples", ignore=BUILT)
        for link, target in (("include", REPO / "include"), ("configs", REPO / "configs"), ("build", t.build_dir)):
            (t.work_dir / link).symlink_to(target, target_is_directory=True)
        expected = EXAMPLES / name / "expected.txt"
        if not expected.is_file():
            raise Failure(f"expected {expected} to exist: every directory of examples/ states what its run prints")
        try:
            done = subprocess.run([t.work_dir / "examples" / "run.sh", name], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise Failure(f"examples/run.sh {name} timed out after {TIMEOUT_S} s")
        log = t.work_dir / "run.log"
        log.write_text(done.stdout)
        want, got = expected.read_text().splitlines(), done.stdout.splitlines()
        if done.returncode != 0 or want != got:
            diff = list(difflib.unified_diff(want, got, "expected.txt", "run.sh", lineterm="", n=0))[2:22]
            raise Failure("\n".join([f"expected examples/run.sh {name} to exit 0 and print examples/{name}/"
                                     f"expected.txt, got exit {done.returncode} and these differences  [log: {log}]",
                                     *diff]))
    return case


def documents_quote_the_examples(t):
    """In a fenced block of the documents, a line in the form examples/run.sh prints (a log line with or without its
    timestamp, an echoed command, an exit status) is a line of some expected.txt, whole or up to a closing ` ...`;
    a cpp or yaml block is consecutive lines of an example's source, of minimal.yml or of togsim_runtime.h."""
    pinned = {line for name in NAMES for line in (EXAMPLES / name / "expected.txt").read_text().splitlines()}
    sources = [p.read_text() for p in [*sorted(EXAMPLES.glob("*/*.cpp")), EXAMPLES / "minimal.yml",
                                       REPO / "include" / "togsim_runtime.h"]]
    wrong = []
    for document in DOCUMENTS:
        where = document.relative_to(REPO)
        for language, body in FENCED.findall(document.read_text()):
            if language in ("cpp", "yaml") and not any(body in source for source in sources):
                wrong.append(f"{where}: the {language} block starting '{body.splitlines()[0]}' is in no example")
            for line in body.splitlines() if language not in ("cpp", "yaml") else []:
                line = TIMESTAMP.sub("", line)
                cut = line.endswith(" ...")
                quoted = line[:-4] if cut else line
                if RUN_SH_LINE.match(line) and not any(p.startswith(quoted) if cut else p == quoted for p in pinned):
                    wrong.append(f"{where}: '{line}' is in no expected.txt")
    if wrong:
        raise Failure("\n".join([f"expected every quote of the examples to be current, {len(wrong)} are not:", *wrong]))


def refusal_table_is_current(t):
    """Every row of the README's refusal table (a message in a code span, then its exit status) is a refusal that
    examples/run.sh mistakes prints: the message, with <placeholders> and (...) standing for any text, ends a line
    of mistakes/expected.txt that `exit <status>` follows."""
    printed = (EXAMPLES / "mistakes" / "expected.txt").read_text().splitlines()
    rows = REFUSAL_ROW.findall((REPO / "README.md").read_text())
    wrong = [f"'{message}' with exit {status}" for message, status in rows
             if not any(re.search(PLACEHOLDER.sub(".*", re.escape(message)) + "$", line) and after == f"exit {status}"
                        for line, after in zip(printed, printed[1:]))]
    if not rows or wrong:
        raise Failure("\n".join([f"expected the README's refusal table to list refusals of examples/mistakes/"
                                 f"expected.txt, found {len(rows)} rows and {len(wrong)} that are not:", *wrong]))


CASES = [Case(name, prints_expected(name)) for name in NAMES] + [
    Case("documents_quote_the_examples", documents_quote_the_examples),
    Case("refusal_table_is_current", refusal_table_is_current)]
