"""Each exit status of status.py, reached through the two entry points as a caller runs them: `python -m` in a
process of its own. Only the runs marked `simulator` need build/bin/Simulator."""
import json
import os
import subprocess
import sys
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from togsim_frontend import paths, status

simulator = pytest.mark.skipif(not paths.simulator().is_file() and not os.environ.get("TOGSIM_REQUIRE_SIMULATOR"),
                               reason=f"{paths.simulator()} is absent: build the simulator first")
CONFIG = paths.root() / "configs" / "systolic_ws_8x8_c1_simple_noc_tpuv3.yml"
UNSUPPORTED = paths.root() / "configs" / "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"
FRONTEND, ACS = "togsim_frontend.onnx_frontend", "togsim_frontend.acs"


def run(module, *args, env=None, code=None):
    """Run `python -m <module> <args>`, or the Python `code` with <args>, with the package on the path and `env`
    over the environment: the finished process."""
    package = str(Path(__file__).resolve().parents[1])
    environ = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [package, os.environ.get("PYTHONPATH")])),
               **(env or {})}
    head = ["-c", code] if code else ["-m", module]
    return subprocess.run([sys.executable, *head, *map(str, args)], capture_output=True, text=True, env=environ)


def model(tmp_path, op):
    """An ONNX model of one `op` node over a 64 x 256 input."""
    io = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [64, 256]) for name in ("x", "y")]
    graph = helper.make_graph([helper.make_node(op, ["x"], ["y"], name=op.lower())], "g", io[:1], io[1:])
    path = tmp_path / f"{op.lower()}.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))
    return path


def frontend(tmp_path, graph, config=CONFIG, env=None):
    return run(FRONTEND, graph, "--config", config, "--out", tmp_path / "out", "--no-optimize", env=env)


def test_the_statuses_are_four_numbers_and_none_is_pythons_own():
    numbers = [status.EXIT_OK, status.EXIT_USAGE, status.EXIT_NODES_NOT_RUN,
               status.EXIT_UNSUPPORTED_MACHINE]
    assert numbers == [0, 2, 3, 5]


@simulator
def test_a_model_whose_nodes_all_ran_exits_0(tmp_path):
    done = frontend(tmp_path, model(tmp_path, "Relu"))
    assert done.returncode == status.EXIT_OK, done.stderr
    assert json.loads((tmp_path / "out" / "report.json").read_text())["nodes_run"] == 1


def test_acs_exits_0_with_the_two_tables_written(tmp_path):
    done = run(ACS, "gemm", "--config", CONFIG, "-o", tmp_path)
    assert done.returncode == status.EXIT_OK, done.stderr
    assert (tmp_path / "tog_cycles.tsv").exists() and (tmp_path / "tog_units.tsv").exists()


@pytest.mark.parametrize("module,args", [
    (FRONTEND, ["model.onnx"]),
    (FRONTEND, ["model.onnx", "--config", CONFIG, "--sa", "eight"]),
    (ACS, ["gemm"]),
])
def test_a_command_line_that_cannot_be_parsed_stays_argparses_2(module, args):
    done = run(module, *args)
    assert done.returncode == status.EXIT_USAGE and "usage:" in done.stderr and "error:" in done.stderr


def test_a_node_that_did_not_run_exits_3_with_the_report_written(tmp_path):
    done = frontend(tmp_path, model(tmp_path, "Sin"))
    assert done.returncode == status.EXIT_NODES_NOT_RUN, done.stderr
    assert [r["name"] for r in json.loads((tmp_path / "out" / "report.json").read_text())["refused"]] == ["sin"]


def test_an_unsupported_machine_exits_5_and_not_the_2_of_a_usage_error(tmp_path):
    done = frontend(tmp_path, tmp_path / "model.onnx", config=UNSUPPORTED)
    assert done.returncode == status.EXIT_UNSUPPORTED_MACHINE == 5 != status.EXIT_USAGE
    assert "is not a machine the frontend maps a model on" in done.stderr and "usage:" not in done.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("module,args", [(FRONTEND, ["model.onnx", "--config", CONFIG]),
                                         (ACS, ["gemm", "--config", CONFIG])])
def test_an_exception_no_status_names_leaves_with_pythons_1_and_its_traceback(module, args):
    code = ("import sys, runpy, togsim_frontend.onnx_frontend.config as config\n"
            "def broken(*args): raise AssertionError('not a failure any status names')\n"
            "config.read_hardware = broken\n"
            f"sys.argv[0] = {module!r}\n"
            f"runpy.run_module({module!r}, run_name='__main__')\n")
    done = run(None, *args, code=code)
    assert done.returncode == 1 and "Traceback" in done.stderr
    assert "AssertionError: not a failure any status names" in done.stderr
