"""The frontend against a built simulator: every kernel replays with the tables ACS gives it, and a graph's total is
the sum of its nodes replayed on their own. Skipped when build/bin/Simulator is absent, unless
TOGSIM_REQUIRE_SIMULATOR is set (CI): then they fail."""
import json
import math
import os
import re
import subprocess

import onnx
import pytest
import yaml
from onnx import TensorProto, helper

from togsim_frontend import paths
from togsim_frontend.acs.node_mapping import KERNELS
from togsim_frontend.onnx_frontend import run as frontend
from togsim_frontend.onnx_frontend.config import read_hardware
from togsim_frontend.onnx_frontend.onnx_ops import OPS

pytestmark = pytest.mark.skipif(not paths.simulator().is_file() and not os.environ.get("TOGSIM_REQUIRE_SIMULATOR"),
                                reason=f"{paths.simulator()} is absent: build the simulator first")
CONFIG = paths.root() / "configs" / "systolic_ws_8x8_c1_simple_noc_tpuv3.yml"
MIB = 1 << 20
TOTAL = re.compile(r"Total execution cycles: (\d+)")


def standalone(tog_dir, stem, macros=(), shape=None, tensors=(MIB,) * 8, cycles=(), units=None, config=CONFIG,
               log_level="info"):
    """Compile kernels/<stem>_tog.cpp with g++ and replay it from `tog_dir`, laid out by hand: the run's log."""
    tog_dir.mkdir(parents=True)
    subprocess.run(["g++", "-shared", "-fPIC", "-std=gnu++17", "-O1", "-I", str(paths.include_dir()),
                    *[f"-D{m}" for m in macros], str(paths.kernel_source(stem)), "-o", str(tog_dir / "tog.so")],
                   check=True)
    (tog_dir / "tog_tensors.txt").write_text("".join(f"{b}\n" for b in tensors))
    if shape is not None:
        (tog_dir / "tog_shape.txt").write_text("".join(f"{v}\n" for v in shape))
    (tog_dir / "cycles.tsv").write_text("".join(f"{c}\t0\n" for c in cycles))
    cmd = [str(paths.simulator()), "--config", str(config), "--tog_so", str(tog_dir / "tog.so"),
           "--cycle_table", str(tog_dir / "cycles.tsv"), "--log_level", log_level]
    if units is not None:
        (tog_dir / "units.tsv").write_text("".join("\t".join(map(str, row)) + "\n" for row in units))
        cmd += ["--unit_table", str(tog_dir / "units.tsv")]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout + done.stderr)[-600:]
    return done.stdout


def units_of(cycles, systolic_rows=None):
    """The unit table ACS writes for tiles of `cycles`: tile i holds systolic_rows[i] rows, or the vector unit."""
    rows = systolic_rows or {}
    return [row for i, c in enumerate(cycles)
            for row in ((i, "Systolic", "rows", 1, rows.get(i, 0)), (i, "Vpu", "cycles", 1, 0 if i in rows else c))]


@pytest.fixture(scope="module")
def unsplit_config(tmp_path_factory):
    """The 8x8 config with systolic_array_split off, so a compute runs the cycles its table row gives."""
    config = yaml.safe_load(CONFIG.read_text())
    config["systolic_array_split"] = "none"
    config["ramulator_config_path"] = os.path.normpath(CONFIG.parent / config["ramulator_config_path"])
    path = tmp_path_factory.mktemp("config") / CONFIG.name
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("stem", sorted(KERNELS))
def test_the_tiles_a_kernel_computes_are_the_nodes_of_its_table(stem, tmp_path, unsplit_config):
    nodes = KERNELS[stem].nodes
    cycles = [1001 + i for i in range(len(nodes))]
    log = standalone(tmp_path / stem, stem, cycles=cycles, config=unsplit_config, log_level="trace")
    seen = {int(c) for c in re.findall(r"INST_ISSUED *\]\[INST_ID=\d+\] COMP \(compute_type=0 compute_cycle=(\d+)", log)}
    assert seen == set(cycles), f"{stem}: node_mapping.py lists {len(nodes)} nodes, the kernel computed tiles {seen}"
    assert "skipped inst_count" not in log


def save(tmp_path, nodes, inputs, outputs, initializers):
    """Write an ONNX model of `nodes`; `initializers` are (name, shape) of float weights."""
    weights = [helper.make_tensor(name, TensorProto.FLOAT, shape, [0.0] * math.prod(shape))
               for name, shape in initializers]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in inputs],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in outputs], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path = tmp_path / "model.onnx"
    onnx.save(model, str(path))
    return path


def warnings(out):
    """The warn-level lines TOGSim logged in the runs under `out`, each with the run it came from."""
    logs = sorted(out.rglob("togsim.log"))
    assert logs, f"no run under {out} kept a togsim.log"
    return [f"{log.parent.name}: {line}" for log in logs for line in log.read_text().splitlines()
            if "[warning]" in line]


def run_frontend(model, out, status=0):
    """Run the frontend on `model`, expecting its exit status to be `status`: the report it wrote. No run of it
    may have made TOGSim warn."""
    assert frontend.main([str(model), "--config", str(CONFIG), "--out", str(out), "--no-optimize"]) == status
    report = json.loads((out / "report.json").read_text())
    assert not report["nodes_run"] or warnings(out) == []
    return report


IMAGE, MATRIX = [1, 64, 16, 16], [64, 256]
#: An operator's input shapes and attributes, for one node of each operator the frontend maps.
NODE = {
    "Gemm": ([[64, 128], [128, 256]], {}), "MatMul": ([[64, 128], [128, 256]], {}),
    "Conv": ([IMAGE, [64, 64, 3, 3]], dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1])),
    "Attention": ([[1, 128, 512]], dict(num_heads=8)), "MultiHeadAttention": ([[1, 128, 512]], dict(num_heads=8)),
    "Softmax": ([MATRIX], {}), "LayerNormalization": ([MATRIX], {}), "EmbedLayerNormalization": ([[128, 64]], {}),
    "GlobalAveragePool": ([IMAGE], {}), "Relu": ([MATRIX], {}), "Add": ([MATRIX, MATRIX], {}), "Gelu": ([MATRIX], {}),
    "MaxPool": ([IMAGE], dict(kernel_shape=[2, 2], strides=[2, 2])),
    "AveragePool": ([IMAGE], dict(kernel_shape=[2, 2], strides=[2, 2])),
    "Concat": ([MATRIX, MATRIX], {}), "Flatten": ([IMAGE], {}),
}


@pytest.mark.parametrize("op", sorted(OPS))
def test_a_node_of_each_mapped_operator_replays_without_a_togsim_warning(op, tmp_path):
    """As the frontend lays the node out: its tog_tensors.txt lists every arg_id the kernel's DMAs name, and the
    tables ACS wrote have unit rows for every tile id it computes. Either miss is a warning in TOGSim's log."""
    assert sorted(NODE) == sorted(OPS)
    shapes, attributes = NODE[op]
    inputs = {f"in{i}": shape for i, shape in enumerate(shapes)}
    node = helper.make_node(op, list(inputs), ["out"], name=op.lower(), **attributes)
    hw = read_hardware(str(CONFIG))
    for kernel in OPS[op]:
        tog_dir = frontend.prepare_node(node, kernel, inputs, hw, str(tmp_path))
        assert frontend.simulate(tog_dir, str(CONFIG)) > 0
    assert warnings(tmp_path) == []


def test_a_run_with_a_tensor_or_a_unit_row_missing_is_seen_as_a_warning(tmp_path):
    """The check above reads something: a Relu laid out by the frontend, then with its last tensor and its unit
    rows taken away, makes TOGSim log the two warnings."""
    node = helper.make_node("Relu", ["x"], ["y"], name="relu")
    tog_dir = frontend.prepare_node(node, OPS["Relu"][0], {"x": MATRIX}, read_hardware(str(CONFIG)), str(tmp_path))
    tensors = (tmp_path / tog_dir / "tog_tensors.txt")
    tensors.write_text("".join(tensors.read_text().splitlines(keepends=True)[:-1]))
    (tmp_path / tog_dir / "tog_units.tsv").write_text("1\tSystolic\trows\t1\t0\n")
    frontend.simulate(tog_dir, str(CONFIG))
    said = warnings(tmp_path)
    assert len(said) == 2 and any("its DMAs name arg_id 2" in line for line in said)
    assert any("the unit table has no rows for tile ids 0" in line for line in said)


def total(log):
    return int(TOTAL.search(log).group(1))


def test_a_gemm_relu_add_graph_totals_its_nodes_replayed_alone(tmp_path):
    """Gemm 64 x 128 x 256 runs in 32 x 256 tiles of 64-deep blocks, 1024 matmuls of 8 rows each; Relu and Add
    in 8 x 256 tiles."""
    model = save(tmp_path, [
        helper.make_node("Gemm", ["x", "w", "b"], ["g"], name="gemm"),
        helper.make_node("Relu", ["g"], ["r"], name="relu"),
        helper.make_node("Add", ["r", "c"], ["y"], name="add"),
    ], [("x", [64, 128])], [("y", [64, 256])], [("w", [128, 256]), ("b", [256]), ("c", [64, 256])])
    report = run_frontend(model, tmp_path / "out")

    gemm = total(standalone(tmp_path / "gemm", "gemm", ["TM=32", "TN=256", "TK=64"], (64, 128, 256),
                            (64 * 128 * 4, 128 * 256 * 4, 64 * 256 * 4), [128, 8206], units_of([128, 8206], {1: 8192})))
    bias_act = total(standalone(tmp_path / "bias_act", "bias_act", ["TM=8", "TN=256"], (64, 256),
                                (64 * 256 * 4, 256 * 4, 64 * 256 * 4), [64], units_of([64])))

    assert [(n["op"], n["cycles"]) for n in report["nodes"]] == [("Gemm", gemm), ("Relu", bias_act), ("Add", bias_act)]
    assert report["total_cycles"] == gemm + 2 * bias_act
    assert report["unmapped"] == {} and report["nodes_run"] == 3


def test_a_conv_relu_graph_totals_its_nodes_replayed_alone(tmp_path):
    """64 -> 64 channels over 16 x 16 run in 8 x 16 output tiles of 64 channels, 1024 matmuls of 8 rows per
    kernel position."""
    model = save(tmp_path, [
        helper.make_node("Conv", ["x", "w"], ["c"], name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"], name="relu"),
    ], [("x", [1, 64, 16, 16])], [("y", [1, 64, 16, 16])], [("w", [64, 64, 3, 3])])
    report = run_frontend(model, tmp_path / "out")

    conv = total(standalone(tmp_path / "conv", "conv", ["TILE_M=16", "TILE_N=64", "TILE_K=64", "TILE_I_H=8", "SA=8"],
                            (64, 16, 16, 64, 16, 16, 3, 3) + (1,) * 8, (18 * 18 * 64 * 4, 9 * 64 * 64 * 4, 64 * 16 * 16 * 4),
                            [128, 8206], units_of([128, 8206], {1: 8192})))
    bias_act = total(standalone(tmp_path / "bias_act", "bias_act", ["TM=128", "TN=16"], (1024, 16),
                                (1024 * 16 * 4, 16 * 4, 1024 * 16 * 4), [64], units_of([64])))

    assert [(n["op"], n["cycles"]) for n in report["nodes"]] == [("Conv", conv), ("Relu", bias_act)]
    assert report["total_cycles"] == conv + bias_act


def test_an_operator_without_a_kernel_is_listed_not_counted_and_fails_the_run(tmp_path, capsys):
    """The Relu still runs and is totalled, the report is written, and the exit status is 3: the total is not the
    model's."""
    model = save(tmp_path, [
        helper.make_node("Relu", ["x"], ["r"], name="relu"),
        helper.make_node("Sin", ["r"], ["y"], name="sin"),
    ], [("x", [64, 256])], [("y", [64, 256])], [])
    report = run_frontend(model, tmp_path / "out", status=frontend.EXIT_NODES_NOT_RUN)
    assert frontend.EXIT_NODES_NOT_RUN == 3
    assert report["unmapped"] == {"Sin": 1} and report["nodes_run"] == 1
    assert report["refused"] == [{"name": "sin", "op": "Sin", "reason": "no kernel in kernels/ runs this operator"}]
    assert report["total_cycles"] == report["nodes"][0]["cycles"] > 0
    said = capsys.readouterr().out
    assert "  sin (Sin): no kernel in kernels/ runs this operator" in said
    assert said.rstrip().splitlines()[-1].startswith("INCOMPLETE     : 1 node(s) did not run, so the total does not")


def test_a_node_its_kernel_cannot_run_at_its_shape_is_listed_with_the_reason_and_fails_the_run(tmp_path):
    """A Gemm of 32 output columns is under one matmul instruction of 64."""
    model = save(tmp_path, [helper.make_node("Gemm", ["x", "w", "b"], ["y"], name="narrow")],
                 [("x", [64, 128])], [("y", [64, 32])], [("w", [128, 32]), ("b", [32])])
    report = run_frontend(model, tmp_path / "out", status=frontend.EXIT_NODES_NOT_RUN)
    assert [(r["name"], r["op"]) for r in report["refused"]] == [("narrow", "Gemm")]
    assert "32 columns is under one matmul instruction" in report["refused"][0]["reason"]
    assert report["nodes_run"] == 0 and report["total_cycles"] == 0


def test_a_fully_mapped_graph_exits_zero_and_refuses_nothing(tmp_path, capsys):
    model = save(tmp_path, [helper.make_node("Relu", ["x"], ["y"], name="relu")], [("x", [64, 256])],
                 [("y", [64, 256])], [])
    report = run_frontend(model, tmp_path / "out", status=0)
    assert report["refused"] == [] and report["unmapped"] == {} and report["nodes_run"] == 1
    assert "INCOMPLETE" not in capsys.readouterr().out
