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


def tree(tmp_path, simulator_script=None):
    """A TOGSim root with this tree's kernels/, include/ and configs/ and, as build/bin/Simulator, the shell
    script given; without one the root has no Simulator."""
    root = tmp_path / "root"
    root.mkdir()
    for name in ("kernels", "include", "configs"):
        (root / name).symlink_to(paths.root() / name, target_is_directory=True)
    if simulator_script is not None:
        binary = root / "build" / "bin" / "Simulator"
        binary.parent.mkdir(parents=True)
        binary.write_text(f"#!/bin/sh\n{simulator_script}\n")
        binary.chmod(0o755)
    return {"TOGSIM_DIR": str(root)}


def frontend(tmp_path, graph, config=CONFIG, env=None):
    return run(FRONTEND, graph, "--config", config, "--out", tmp_path / "out", "--no-optimize", env=env)


def test_the_statuses_are_six_numbers_and_none_is_pythons_own():
    numbers = [status.EXIT_OK, status.EXIT_USAGE, status.EXIT_NODES_NOT_RUN, status.EXIT_BAD_INPUT,
               status.EXIT_UNSUPPORTED_MACHINE, status.EXIT_TOOL_FAILED]
    assert numbers == [0, 2, 3, 4, 5, 6]


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
    (ACS, ["no_such_kernel", "--config", CONFIG]),
])
def test_a_command_line_that_cannot_be_parsed_stays_argparses_2(module, args):
    done = run(module, *args)
    assert done.returncode == status.EXIT_USAGE and "usage:" in done.stderr and "error:" in done.stderr


def test_a_node_that_did_not_run_exits_3_with_the_report_written(tmp_path):
    done = frontend(tmp_path, model(tmp_path, "Sin"))
    assert done.returncode == status.EXIT_NODES_NOT_RUN, done.stderr
    assert [r["name"] for r in json.loads((tmp_path / "out" / "report.json").read_text())["refused"]] == ["sin"]


@pytest.mark.parametrize("text,said", [
    (None, "No such file or directory"),
    ("vpu_num_lanes: [", "cannot be read"),
    ("- vpu_num_lanes\n", "not a mapping"),
    ("vpu_num_lanes: 8\n", "missing required key 'vpu_vector_length_bits'"),
    ("vpu_num_lanes: 8\nvpu_vector_length_bits:\n", "not all integers"),
    ("vpu_num_lanes: 8\nvpu_vector_length_bits: 256\ncore_spad_size_kb: [1]\n", "not all integers"),
    ("vpu_num_lanes: 0\nvpu_vector_length_bits: 256\n", "vpu_num_lanes is 0, below 1"),
    ("vpu_num_lanes: 8\nvpu_vector_length_bits: 16\n", "vpu_vector_length_bits is 16, below 32"),
    ("vpu_num_lanes: 8\nvpu_vector_length_bits: 256\ncore_spad_size_kb: -1\n", "core_spad_size_kb is -1, below 0"),
])
@pytest.mark.parametrize("module", [FRONTEND, ACS])
def test_a_config_that_cannot_be_read_exits_4(module, text, said, tmp_path):
    config = tmp_path / "systolic_ws_8x8_c1.yml"
    if text is not None:
        config.write_text(text)
    args = [model(tmp_path, "Relu"), "--out", tmp_path / "out"] if module == FRONTEND else ["gemm", "-o", tmp_path]
    done = run(module, *args, "--config", config)
    assert done.returncode == status.EXIT_BAD_INPUT, done.stderr
    assert done.stderr.startswith(f"error: config {config} cannot be read") and said in done.stderr
    assert "Traceback" not in done.stderr and not (tmp_path / "out").exists()


@pytest.mark.parametrize("module", [FRONTEND, ACS])
def test_a_config_whose_name_does_not_carry_the_array_dimension_exits_4_without_sa(module, tmp_path):
    config = tmp_path / "machine.yml"
    config.write_text(CONFIG.read_text())
    args = [model(tmp_path, "Sin"), "--out", tmp_path / "out"] if module == FRONTEND else ["gemm", "-o", tmp_path]
    done = run(module, *args, "--config", config)
    assert done.returncode == status.EXIT_BAD_INPUT and "Pass --sa" in done.stderr
    assert run(module, *args, "--config", config, "--sa", "8").returncode in (status.EXIT_OK, status.EXIT_NODES_NOT_RUN)


@pytest.mark.parametrize("sa", [0, -8])
@pytest.mark.parametrize("module", [FRONTEND, ACS])
def test_an_array_dimension_below_1_exits_4(module, sa, tmp_path):
    args = [model(tmp_path, "Relu"), "--out", tmp_path / "out"] if module == FRONTEND else ["gemm", "-o", tmp_path]
    done = run(module, *args, "--config", CONFIG, "--sa", sa)
    assert done.returncode == status.EXIT_BAD_INPUT, done.stderr
    assert f"the systolic array dimension (--sa, else the config's name) is {sa}, below 1" in done.stderr
    assert "Traceback" not in done.stderr and not (tmp_path / "out").exists()


@pytest.mark.parametrize("content", [None, b"\x00\xff not an ONNX model"])
def test_a_model_that_cannot_be_read_exits_4(content, tmp_path):
    graph = tmp_path / "model.onnx"
    if content is not None:
        graph.write_bytes(content)
    done = frontend(tmp_path, graph)
    assert done.returncode == status.EXIT_BAD_INPUT, done.stderr
    assert done.stderr.startswith(f"error: model {graph} cannot be read") and "Traceback" not in done.stderr
    assert not (tmp_path / "out").exists()


def test_an_unsupported_machine_exits_5_and_not_the_2_of_a_usage_error(tmp_path):
    done = frontend(tmp_path, tmp_path / "model.onnx", config=UNSUPPORTED)
    assert done.returncode == status.EXIT_UNSUPPORTED_MACHINE == 5 != status.EXIT_USAGE
    assert "is not a machine the frontend maps a model on" in done.stderr and "usage:" not in done.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("script,said", [
    (None, "the Simulator could not be started"),
    ("echo refused >&2; exit 1", "TOGSim reported no cycle count (exit 1"),
    ("kill -SEGV $$", "TOGSim reported no cycle count (exit -11"),
    ("echo 'ran, and says nothing of cycles'", "TOGSim reported no cycle count (exit 0"),
    ("echo 'Total execution cycles: 7'; exit 1", "TOGSim reported no cycle count (exit 1"),
    (r"printf '\377\376 refused\n' >&2; exit 1", "TOGSim reported no cycle count (exit 1"),
])
def test_a_simulator_that_is_absent_fails_or_reports_no_total_exits_6_with_no_report(script, said, tmp_path):
    done = frontend(tmp_path, model(tmp_path, "Relu"), env=tree(tmp_path, script))
    assert done.returncode == status.EXIT_TOOL_FAILED, done.stderr
    assert done.stderr.startswith("error: node relu (Relu): ") and said in done.stderr
    assert "NOT RUN" not in done.stdout and not (tmp_path / "out" / "report.json").exists()


@simulator
def test_a_config_the_simulator_itself_refuses_exits_6(tmp_path):
    """The two keys the frontend reads make a machine it accepts; the Simulator needs the rest of the config."""
    config = tmp_path / "systolic_ws_8x8_c1.yml"
    config.write_text("vpu_num_lanes: 8\nvpu_vector_length_bits: 256\n")
    done = frontend(tmp_path, model(tmp_path, "Relu"), config=config)
    assert done.returncode == status.EXIT_TOOL_FAILED, done.stderr
    assert "TOGSim reported no cycle count (exit " in done.stderr
    assert not (tmp_path / "out" / "report.json").exists()


@pytest.mark.parametrize("compiler,said", [
    ("false", "kernel bias_act did not compile with"),
    (os.path.join(os.sep, "no", "such", "compiler"), "kernel bias_act could not be compiled"),
])
def test_a_kernel_the_compiler_does_not_build_exits_6(compiler, said, tmp_path):
    done = frontend(tmp_path, model(tmp_path, "Relu"), env={"CXX": compiler})
    assert done.returncode == status.EXIT_TOOL_FAILED, done.stderr
    assert said in done.stderr and not (tmp_path / "out" / "report.json").exists()


def test_a_compiler_that_fails_with_words_that_are_not_text_exits_6(tmp_path):
    compiler = tmp_path / "cxx"
    compiler.write_text("#!/bin/sh\nprintf '\\377\\376 no such header\\n' >&2; exit 1\n")
    compiler.chmod(0o755)
    done = frontend(tmp_path, model(tmp_path, "Relu"), env={"CXX": str(compiler)})
    assert done.returncode == status.EXIT_TOOL_FAILED, done.stderr
    assert "kernel bias_act did not compile with" in done.stderr and "no such header" in done.stderr
    assert not (tmp_path / "out" / "report.json").exists()


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
