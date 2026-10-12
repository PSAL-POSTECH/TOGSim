#!/usr/bin/env python3
"""Run the cases of every tests/*/cases.py against a built tree: one PASS or FAIL line per case.
Exit 0 if all passed, 1 if a case failed, 2 if the runner could not run (no build, a filter that selects nothing)."""
import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "lib"))
import togtest  # noqa: E402


def broken(said):
    """A case that fails with `said`: what stands for a cases.py that could not be imported."""
    def case(t):
        raise togtest.Failure(said)
    return case


def discover():
    """[(case id, directory, Case, whether its TOGs are typed)] of every tests/*/cases.py, directories by name and
    cases in the order listed; and each directory's one-line description. A cases.py that raises is one failing case."""
    found, about = [], {}
    for path in sorted(HERE.glob("*/cases.py")):
        spec = importlib.util.spec_from_file_location(f"cases_{path.parent.name}", path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            where = traceback.extract_tb(error.__traceback__)[-1]
            said = f"it does not import: {type(error).__name__}: {error} ({Path(where.filename).name}:{where.lineno})"
            about[path.parent.name] = "its cases.py does not import"
            found.append((f"{path.parent.name}/cases.py", path.parent, togtest.Case("cases.py", broken(said)), False))
            continue
        about[path.parent.name] = (module.__doc__ or "").strip().splitlines()[0]
        names = [c.name for c in module.CASES]
        if len(set(names)) != len(names):
            sys.exit(f"{path}: two cases share a name")
        typed = getattr(module, "TYPED_GROUPING", False)
        found += [(f"{path.parent.name}/{c.name}", path.parent, c, typed) for c in module.CASES]
    return found, about


def run_case(case_id, source_dir, case, typed, workdir, build):
    """Run one case in its own fresh directory: (failure text or None, seconds)."""
    work = workdir / case_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    start = time.monotonic()
    try:
        case.run(togtest.Ctx(source_dir, work, build, workdir / ".togs", typed))
        said = None
    except togtest.Failure as failure:
        said = str(failure)
    except Exception as error:
        where = traceback.extract_tb(error.__traceback__)[-1]
        said = f"the case itself raised {type(error).__name__}: {error} ({Path(where.filename).name}:{where.lineno})"
    return said, time.monotonic() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("build", nargs="?", default=HERE.parent / "build", type=Path,
                        help="the build directory, where bin/Simulator is (default: this tree's build/)")
    parser.add_argument("--workdir", type=Path,
                        help="keep every case's files and log here, in <dir>/<case id>/ (default: a temporary "
                             "directory, removed unless a case failed)")
    parser.add_argument("-k", action="append", default=[], metavar="SUBSTR",
                        help="run only the cases whose id contains SUBSTR (repeatable: any of them)")
    parser.add_argument("-j", type=int, default=min(8, os.cpu_count() or 1), metavar="N",
                        help="cases run at once (default: %(default)s)")
    parser.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = parser.parse_args()
    togtest.no_core_dumps()

    cases, about = discover()
    selected = [c for c in cases if not args.k or any(k in c[0] for k in args.k)]
    if args.list:
        for directory in sorted(about):
            print(f"# {directory}/: {about[directory]}")
            print("".join(f"{case_id}\n" for case_id, *_ in selected if case_id.startswith(directory + "/")), end="")
        return 0
    if not selected:
        print(f"no case id contains any of {args.k}", file=sys.stderr)
        return 2
    build = args.build.resolve()
    if not (build / "bin" / "Simulator").is_file():
        print(f"{build}/bin/Simulator is absent: build this tree first, or name its build directory", file=sys.stderr)
        return 2
    workdir = args.workdir.resolve() if args.workdir else Path(tempfile.mkdtemp(prefix="togsim-tests-"))
    workdir.mkdir(parents=True, exist_ok=True)

    start, failed = time.monotonic(), []
    with ThreadPoolExecutor(max(1, args.j)) as pool:
        pending = [(case[0], pool.submit(run_case, *case, workdir, build)) for case in selected]
        for case_id, future in pending:
            said, seconds = future.result()
            if said is None:
                print(f"PASS {case_id} ({seconds:.1f}s)", flush=True)
            else:
                failed.append(case_id)
                first, *rest = said.splitlines() or [""]
                print(f"FAIL {case_id} ({seconds:.1f}s): {first}", flush=True)
                print("".join(f"     {line}\n" for line in rest), end="", flush=True)
    took = time.monotonic() - start
    if failed:
        print(f"FAILED {len(failed)} of {len(selected)} cases in {took:.1f}s: {' '.join(failed)}")
        print(f"files and logs: {workdir}")
        return 1
    if args.workdir:
        print(f"files and logs: {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    print(f"ALL PASS ({len(selected)} cases in {took:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
