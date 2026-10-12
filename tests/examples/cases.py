"""The examples of the README: examples/run.sh <name> prints <name>/expected.txt."""
import difflib
import shutil
import subprocess

from togtest import REPO, TIMEOUT_S, Case, Failure

EXAMPLES = REPO / "examples"
NAMES = sorted(p.name for p in EXAMPLES.iterdir() if p.is_dir())
BUILT = shutil.ignore_patterns("*.so", "*.gen.*", "no_tensors")


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


CASES = [Case(name, prints_expected(name)) for name in NAMES]
