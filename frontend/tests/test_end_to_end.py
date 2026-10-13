"""The frontend against a built simulator: every kernel replays with the tables ACS gives it, and a graph's total is
the sum of its nodes replayed on their own. Skipped when build/bin/Simulator is absent, unless
TOGSIM_REQUIRE_SIMULATOR is set (CI): then they fail."""
import dataclasses
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
from togsim_frontend.acs.node_mapping import KERNELS, lookup, tile_ids
from togsim_frontend.onnx_frontend import emit
from togsim_frontend.onnx_frontend import run as frontend
from togsim_frontend.onnx_frontend import shapes as frontend_shapes
from togsim_frontend.onnx_frontend.config import read_hardware
from togsim_frontend.onnx_frontend.onnx_ops import OPS

pytestmark = pytest.mark.skipif(not paths.simulator().is_file() and not os.environ.get("TOGSIM_REQUIRE_SIMULATOR"),
                                reason=f"{paths.simulator()} is absent: build the simulator first")
CONFIG = paths.root() / "configs" / "systolic_ws_8x8_c1_simple_noc_tpuv3.yml"
MIB = 1 << 20
TOTAL = re.compile(r"Total execution cycles: (\d+)")


def standalone(tog_dir, stem, macros=(), shape=None, tensors=(MIB,) * 8, cycles=(), units=None, config=CONFIG,
               log_level="info"):
    """Compile kernels/<stem>_tog.cpp with g++ and replay it from `tog_dir`, laid out by hand: the run's log.
    `cycles` lists the cycles of tile 0, 1, ..., or is {tile id: cycles} with zeros for the ids between."""
    tog_dir.mkdir(parents=True)
    if isinstance(cycles, dict):
        cycles = [cycles.get(i, 0) for i in range(max(cycles) + 1)]
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
    """The unit table ACS writes for tiles of `cycles` (a list, or {tile id: cycles}): tile i holds
    systolic_rows[i] rows, or the vector unit."""
    rows = systolic_rows or {}
    return [row for i, c in (cycles.items() if isinstance(cycles, dict) else enumerate(cycles))
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


def save(tmp_path, nodes, inputs, outputs, initializers, check=True):
    """Write an ONNX model of `nodes`; `initializers` are (name, shape) of float weights. `check` off lets through an
    operator opset 13 does not define (Attention)."""
    weights = [helper.make_tensor(name, TensorProto.FLOAT, shape, [0.0] * math.prod(shape))
               for name, shape in initializers]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in inputs],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in outputs], weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    if check:
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


#: The tables ACS writes for a conv tile of 8 x 16 outputs: 64 or 128 filters over 64 channels, and 64 over the 3
#: of an image, short of the 8-deep block: its matmul is tile id 1 + 2 * 8, three preloads of 128 rows.
CONV_64, CONV_128, CONV_3_DEEP = ([128, 8206], {1: 8192}), ([256, 16398], {1: 16384}), ({0: 128, 17: 398}, {17: 384})
#: A Conv: its attributes, input and weight shapes, the output's rows and columns; then the kernel run it has to
#: be, written out by hand: the TILE_N and TILE_K macros, the shape arguments after I_C I_H I_W O_C and the tables.
CONVS = {
    "stride_2": (dict(strides=[2, 2], pads=[1, 1, 1, 1]), [1, 64, 32, 32], [64, 64, 3, 3], (16, 16),
                 (64, 64), (16, 16, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1), CONV_64),
    "pad_0": ({}, [1, 64, 18, 18], [64, 64, 3, 3], (16, 16),
              (64, 64), (16, 16, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0), CONV_64),
    "1x1_stride_2": (dict(strides=[2, 2]), [1, 64, 32, 32], [128, 64, 1, 1], (16, 16),
                     (128, 64), (16, 16, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0), CONV_128),
    "7x7_stride_2_pad_3_of_3_channels": (dict(strides=[2, 2], pads=[3, 3, 3, 3]), [1, 3, 32, 32], [64, 3, 7, 7],
                                         (16, 16), (64, 8), (16, 16, 7, 7, 2, 2, 1, 1, 3, 3, 3, 3), CONV_3_DEEP),
    "stride_2x1_pads_0_1_1_2": (dict(strides=[2, 1], pads=[0, 1, 1, 2]), [1, 64, 32, 15], [64, 64, 3, 3], (16, 16),
                                (64, 64), (16, 16, 3, 3, 2, 1, 1, 1, 0, 1, 1, 2), CONV_64),
    "dilation_2x3": (dict(dilations=[2, 3], pads=[2, 3, 2, 3]), [1, 64, 16, 16], [64, 64, 3, 3], (16, 16),
                     (64, 64), (16, 16, 3, 3, 1, 1, 2, 3, 2, 3, 2, 3), CONV_64),
    "same_upper_stride_2": (dict(strides=[2, 2], auto_pad="SAME_UPPER"), [1, 64, 32, 32], [64, 64, 3, 3], (16, 16),
                            (64, 64), (16, 16, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1), CONV_64),
}


@pytest.mark.parametrize("name", sorted(CONVS))
def test_a_conv_with_a_stride_a_dilation_or_pads_is_the_kernel_run_at_those_shape_arguments(name, tmp_path):
    """The graph is mapped whole (exit 0, no TOGSim warning) and totals its nodes; its Conv costs what the conv
    kernel does when replayed by hand at the shape arguments and the padded input size the attributes stand for."""
    attributes, x, w, (o_h, o_w), (tile_n, tile_k), rest, (cycles, rows) = CONVS[name]
    (i_c, i_h, i_w), (o_c, k_h, k_w), (p_t, p_l, p_b, p_r) = x[1:], (w[0], *w[2:]), rest[-4:]
    model = save(tmp_path, [
        helper.make_node("Conv", ["x", "w"], ["c"], name="conv", kernel_shape=[k_h, k_w], **attributes),
        helper.make_node("Relu", ["c"], ["y"], name="relu"),
    ], [("x", x)], [("y", [1, o_c, o_h, o_w])], [("w", w)])
    report = run_frontend(model, tmp_path / "out")

    tensors = ((i_h + p_t + p_b) * (i_w + p_l + p_r) * i_c * 4, k_h * k_w * o_c * i_c * 4, o_c * o_h * o_w * 4)
    conv = total(standalone(tmp_path / "conv", "conv",
                            ["TILE_M=16", f"TILE_N={tile_n}", f"TILE_K={tile_k}", "TILE_I_H=8", "SA=8"],
                            (i_c, i_h, i_w, o_c) + rest, tensors, cycles, units_of(cycles, rows)))
    assert report["refused"] == [] and report["unmapped"] == {} and report["nodes_run"] == 2
    assert [(n["op"], n["cycles"]) for n in report["nodes"]][0] == ("Conv", conv)
    assert report["total_cycles"] == sum(n["cycles"] for n in report["nodes"]) > conv


def test_a_grouped_conv_is_listed_with_the_reason_and_fails_the_run(tmp_path):
    model = save(tmp_path, [helper.make_node("Conv", ["x", "w"], ["y"], name="depthwise", kernel_shape=[3, 3],
                                             pads=[1, 1, 1, 1], group=64)],
                 [("x", [1, 64, 16, 16])], [("y", [1, 64, 16, 16])], [("w", [64, 1, 3, 3])])
    report = run_frontend(model, tmp_path / "out", status=frontend.EXIT_NODES_NOT_RUN)
    assert report["refused"] == [{"name": "depthwise", "op": "Conv",
                                  "reason": "grouped convolution: no library kernel takes a group count"}]
    assert report["nodes_run"] == 0 and report["total_cycles"] == 0


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


def spad_alone(directory, kb):
    """The 8x8 config with a spad of `kb` stated by core_spad_size_kb alone."""
    config = yaml.safe_load(CONFIG.read_text())
    del config["vpu_spad_size_kb_per_lane"]
    config["core_spad_size_kb"] = kb
    config["ramulator_config_path"] = os.path.normpath(CONFIG.parent / config["ramulator_config_path"])
    directory.mkdir(parents=True)
    (directory / CONFIG.name).write_text(yaml.safe_dump(config))
    return directory / CONFIG.name


def togsim_runs(tog_dir, config):
    """Whether TOGSim replays the TOG laid out in `tog_dir` on `config` to the end: its exit status, 0."""
    return subprocess.run([str(paths.simulator()), "--config", str(config), "--tog_so", f"{tog_dir}/tog.so",
                           "--cycle_table", f"{tog_dir}/tog_cycles.tsv", "--unit_table", f"{tog_dir}/tog_units.tsv"],
                          capture_output=True).returncode == 0


@pytest.mark.parametrize("op,shapes,attributes", [
    ("Conv", [[1, 64, 16, 32], [256, 64, 3, 3]], dict(pads=[1, 1, 1, 1])),
    ("Gemm", [[64, 64], [64, 256]], {}),
])
def test_the_frontend_refuses_a_conv_or_gemm_on_the_spad_togsim_refuses_its_tog_on_and_no_larger(op, shapes,
                                                                                                attributes, tmp_path):
    """Of the largest tile, 384 and 104 KB: on a spad that size TOGSim replays it and the frontend lays it out, on
    one KB less TOGSim refuses it and the frontend refuses the node."""
    inputs = {f"in{i}": shape for i, shape in enumerate(shapes)}
    node = helper.make_node(op, list(inputs), ["out"], name=op.lower(), **attributes)
    hw = read_hardware(str(CONFIG))
    tog_dir = frontend.prepare_node(node, OPS[op][0], inputs, dataclasses.replace(hw, spad_bytes=0),
                                    str(tmp_path / "unbounded"))
    need = sum(emit.spad_buffer_bytes(f"{tog_dir}/tog.so"))
    assert need == {"Conv": 384, "Gemm": 104}[op] * 1024
    for kb, fits in ((need // 1024, True), (need // 1024 - 1, False)):
        config = spad_alone(tmp_path / f"{kb}kb", kb)
        assert togsim_runs(tog_dir, config) == fits
        try:
            frontend.prepare_node(node, OPS[op][0], inputs, read_hardware(str(config)), str(tmp_path / f"{kb}kb"))
            refused = False
        except ValueError as exc:
            refused = f"{need} bytes, and a core's spad is {kb * 1024}: TOGSim refuses the TOG" in str(exc)
        assert refused == (not fits)


def test_a_fully_mapped_graph_exits_zero_and_refuses_nothing(tmp_path, capsys):
    model = save(tmp_path, [helper.make_node("Relu", ["x"], ["y"], name="relu")], [("x", [64, 256])],
                 [("y", [64, 256])], [])
    report = run_frontend(model, tmp_path / "out", status=0)
    assert report["refused"] == [] and report["unmapped"] == {} and report["nodes_run"] == 1
    assert "INCOMPLETE" not in capsys.readouterr().out


DMA = re.compile(r"INST_ISSUED *\]\[INST_ID=\d+\] MOV(?:IN|OUT) \((.*)\)$", re.M)
TENSOR_ALIGN = 16384


def past_its_tensor(log, tensors):
    """The DMAs of a trace-level `log` that end past the tensor they start in, as (tensor, bytes past its end).
    TOGSim lays the tensors of tog_tensors.txt back to back, each start 16 KiB aligned."""
    bases, at = [], 0
    for size in tensors:
        bases.append(at)
        at = -(-(at + size) // TENSOR_ALIGN) * TENSOR_ALIGN
    past, dmas = [], [dict(field.split("=", 1) for field in fields.split(" ")) for fields in DMA.findall(log)]
    assert dmas or not tensors
    for dma in dmas:
        first, width = int(dma["dram"], 16), int(dma["elem_bits"]) // 8
        arg = max(i for i, base in enumerate(bases) if base <= first)
        size, stride = (dma[key].strip("[]").split(",") for key in ("size", "stride"))
        last = first + sum((int(n) - 1) * int(step) for n, step in zip(size, stride)) * width
        if last + width > bases[arg] + tensors[arg]:
            past.append((arg, last + width - bases[arg] - tensors[arg]))
    return past


ROWS_48, ROWS_100, STEM = [128, 48], [70, 100], [1, 3, 32, 32]
#: A node whose shape its tile does not divide, per mapped kernel: the operator, its input shapes and attributes.
SHORT = {
    "relu_128_rows_in_tiles_of_42": ("Relu", [ROWS_48], {}),
    "add": ("Add", [ROWS_48, ROWS_48], {}), "gelu": ("Gelu", [ROWS_48], {}),
    "softmax": ("Softmax", [ROWS_100], {}), "layernorm": ("LayerNormalization", [ROWS_100], {}),
    "embed_layernorm": ("EmbedLayerNormalization", [ROWS_100], {}),
    "global_avgpool": ("GlobalAveragePool", [[1, 300, 10, 10]], {}),
    "maxpool": ("MaxPool", [[1, 64, 16, 100]], dict(kernel_shape=[2, 2], strides=[2, 2])),
    "avgpool": ("AveragePool", [[1, 64, 16, 100]], dict(kernel_shape=[2, 2], strides=[2, 2])),
    "concat": ("Concat", [[70, 48], [30, 48]], {}),
    "gemm_45x77x300": ("Gemm", [[45, 77], [77, 300]], {}), "matmul_5_rows": ("MatMul", [[5, 128], [128, 100]], {}),
    "conv_7x7_stride_2_pad_3_of_3_channels": ("Conv", [STEM, [64, 3, 7, 7]], dict(
        kernel_shape=[7, 7], strides=[2, 2], pads=[3, 3, 3, 3])),
    "conv_10x44_outputs_of_70_filters": ("Conv", [[1, 12, 10, 44], [70, 12, 3, 3]], dict(
        kernel_shape=[3, 3], pads=[1, 1, 1, 1])),
    "attention_100_tokens": ("Attention", [[1, 100, 512]], dict(num_heads=8)),
}


@pytest.mark.parametrize("name", sorted(SHORT))
def test_a_node_its_tile_does_not_divide_computes_the_tile_ids_predicted_and_stays_in_its_tensors(name, tmp_path,
                                                                                                unsplit_config):
    """As the frontend lays the node out it replays without a warning; replayed again against a table whose line i
    says 1001 + i cycles, the tile ids its computes name are the ones node_mapping.tile_ids predicts, some of them
    of a short variant; and no DMA ends past the tensor it starts in."""
    op, in_shapes, attributes = SHORT[name]
    inputs = {f"in{i}": shape for i, shape in enumerate(in_shapes)}
    node = helper.make_node(op, list(inputs), ["out"], name=name, **attributes)
    hw = read_hardware(str(CONFIG))
    kernel, = OPS[op]
    tog_dir = tmp_path / frontend.prepare_node(node, kernel, inputs, hw, str(tmp_path))
    assert frontend.simulate(str(tog_dir), str(CONFIG)) > 0 and warnings(tmp_path) == []

    desc = lookup(kernel.stem)
    predicted = tile_ids(desc, frontend_shapes.constants(node, kernel, inputs, hw))
    assert not desc.nodes or max(predicted) >= len(desc.nodes), f"{name}: no tile of this shape is short"
    table = tog_dir / "by_id.tsv"
    table.write_text("".join(f"{1001 + i}\t0\n" for i in range(max(predicted, default=0) + 1)))
    done = subprocess.run([str(paths.simulator()), "--config", str(unsplit_config), "--tog_so", str(tog_dir / "tog.so"),
                           "--cycle_table", str(table), "--log_level", "trace"], capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout + done.stderr)[-600:]
    issued = re.findall(r"INST_ISSUED *\]\[INST_ID=\d+\] COMP \(compute_type=0 compute_cycle=(\d+)", done.stdout)
    seen = {int(cycles) - 1001 for cycles in issued}
    assert sorted(seen) == predicted
    tensors = [int(line) for line in (tog_dir / "tog_tensors.txt").read_text().split()]
    assert past_its_tensor(done.stdout, tensors) == []


def test_the_dma_check_sees_a_load_and_a_store_that_leave_their_tensor():
    """A 2 x 3 DMA at stride 4, 1 from byte 8 of a 32-byte tensor ends 4 bytes past it; one row less stays in.
    The lines are as TOGSim writes them: a MOVOUT's fields come in another order than a MOVIN's."""
    movin = ("[1][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000008 rank=2 "
             "size=[{},3] stride=[4,1] elem_bits=32 async=false indirect=false tag_id=[0x0,0x0])")
    movout = ("[9][Core 0][INST_ISSUED    ][INST_ID=1] MOVOUT (addr_name=tag1 dram=0x0000000000004008 rank=2 "
              "elem_bits=32 async=false indirect=false tag=0x0 stride=[4,1] size=[{},3] tag_idx=[0] tile=0)")
    assert past_its_tensor(movin.format(2), [32, 64]) == [(0, 4)]
    assert past_its_tensor(movin.format(1) + "\n" + movout.format(2), [32, 32]) == [(1, 4)]
    assert past_its_tensor(movin.format(1) + "\n" + movout.format(1), [32, 32]) == []


def hand_costs(tiles, sa=8, lanes=8, throughput=64, instructions=0):
    """What ACS charges, written out again: {tile id: (cycles, Systolic rows or None)} for `tiles`, {tile id: (rows,
    depth, cols)} of a systolic block or {tile id: elements} of a vector tile running `instructions` per element."""
    costs = {}
    for tile_id, tile in tiles.items():
        if isinstance(tile, tuple):
            rows, depth, cols = tile
            fed = depth * math.ceil(cols / (sa * lanes)) * rows
            costs[tile_id] = (fed + 2 * sa - 2, fed)
        else:
            costs[tile_id] = (math.ceil(tile / throughput) * instructions, None)
    return costs


def by_hand(tog_dir, stem, macros, shape, tensors, costs):
    """The total of kernels/<stem>_tog.cpp replayed by hand against the tables `costs` stands for."""
    cycles = {tile_id: cost[0] for tile_id, cost in costs.items()}
    rows = {tile_id: cost[1] for tile_id, cost in costs.items() if cost[1] is not None}
    return total(standalone(tog_dir, stem, macros, shape, tensors, cycles, units_of(cycles, rows)))


def test_a_relu_of_128_rows_in_tiles_of_42_runs_three_whole_tiles_and_one_of_2_rows(tmp_path):
    """128 x 48 in tiles of 42 x 48: the whole tile is 2016 elements, 32 cycles an instruction; the last has 2 rows,
    96 elements and 2 cycles an instruction, as tile id 1. The graph costs what that replay does by hand."""
    model = save(tmp_path, [helper.make_node("Relu", ["x"], ["y"], name="relu")], [("x", [128, 48])],
                 [("y", [128, 48])], [])
    report = run_frontend(model, tmp_path / "out")
    costs = hand_costs({0: 42 * 48, 1: 2 * 48}, instructions=2)
    assert costs == {0: (64, None), 1: (4, None)}
    relu = by_hand(tmp_path / "bias_act", "bias_act", ["TM=42", "TN=48"], (128, 48),
                   (128 * 48 * 4, 48 * 4, 128 * 48 * 4), costs)
    assert report["refused"] == [] and report["unmapped"] == {}
    assert [(n["op"], n["cycles"]) for n in report["nodes"]] == [("Relu", relu)] and report["total_cycles"] == relu
    whole = by_hand(tmp_path / "whole", "bias_act", ["TM=42", "TN=48"], (126, 48),
                    (126 * 48 * 4, 48 * 4, 126 * 48 * 4), {0: (64, None)})
    assert whole < relu < whole * 4 // 3


def test_a_gemm_with_odd_extents_costs_each_block_by_what_is_left_of_it(tmp_path):
    """Gemm 45 x 77 x 300 in 32 x 256 tiles of 64-deep blocks: 13 rows, 44 columns and a 13-deep block are left.
    Its sixteen tile ids are costed by their extents, and the graph totals its nodes."""
    model = save(tmp_path, [
        helper.make_node("Gemm", ["x", "w", "b"], ["g"], name="gemm"),
        helper.make_node("Relu", ["g"], ["y"], name="relu"),
    ], [("x", [45, 77])], [("y", [45, 300])], [("w", [77, 300]), ("b", [300])])
    report = run_frontend(model, tmp_path / "out")

    extent = lambda variant, bit, whole, left: left if variant >> bit & 1 else whole
    blocks = {1 + 2 * v: (extent(v, 0, 32, 13), extent(v, 2, 64, 13), extent(v, 1, 256, 44)) for v in range(8)}
    inits = {2 * v: extent(v, 0, 32, 13) * extent(v, 1, 256, 44) for v in range(4)}
    costs = {**hand_costs(blocks), **hand_costs(inits, instructions=1)}
    assert costs[1] == (8206, 8192) and costs[15] == (13 * 13 + 14, 169) and costs[6] == (9, None)
    gemm = by_hand(tmp_path / "gemm", "gemm", ["TM=32", "TN=256", "TK=64"], (45, 77, 300),
                   (45 * 77 * 4, 77 * 300 * 4, 45 * 300 * 4), costs)
    assert report["refused"] == [] and report["unmapped"] == {} and report["nodes_run"] == 2
    assert [(n["op"], n["cycles"]) for n in report["nodes"]][0] == ("Gemm", gemm)
    assert report["total_cycles"] == sum(n["cycles"] for n in report["nodes"]) > gemm


def test_a_conv_whose_output_columns_its_tile_does_not_divide_costs_the_short_tile_less(tmp_path):
    """16 -> 64 channels over 8 x 40 outputs in tiles of 8 x 32: the last tile of a row has 8 columns, a quarter of
    the rows to feed, as tile ids 8 and 9. The graph costs what the conv kernel does replayed by hand."""
    model = save(tmp_path, [
        helper.make_node("Conv", ["x", "w"], ["c"], name="conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c"], ["y"], name="relu"),
    ], [("x", [1, 16, 8, 40])], [("y", [1, 64, 8, 40])], [("w", [64, 16, 3, 3])])
    report = run_frontend(model, tmp_path / "out")
    costs = {**hand_costs({1: (8 * 32, 16, 64), 9: (8 * 8, 16, 64)}),
             **hand_costs({0: 8 * 32 * 64, 8: 8 * 8 * 64}, instructions=1)}
    assert costs == {1: (4110, 4096), 9: (1038, 1024), 0: (256, None), 8: (64, None)}
    conv = by_hand(tmp_path / "conv", "conv", ["TILE_M=32", "TILE_N=64", "TILE_K=16", "TILE_I_H=8", "SA=8"],
                   (16, 8, 40, 64, 8, 40, 3, 3) + (1,) * 8, (10 * 42 * 16 * 4, 9 * 64 * 16 * 4, 64 * 8 * 40 * 4), costs)
    assert report["refused"] == [] and report["unmapped"] == {} and report["nodes_run"] == 2
    assert [(n["op"], n["cycles"]) for n in report["nodes"]][0] == ("Conv", conv)
    assert report["total_cycles"] == sum(n["cycles"] for n in report["nodes"]) > conv


def test_an_attention_over_a_token_count_its_tile_does_not_divide_is_mapped_whole(tmp_path):
    """100 tokens in tiles of 8 queries by 64 keys: 4 queries and 36 keys are left. Exit 0, no warning, and the total
    is the node's; fewer cycles than 104 tokens, which the whole tiles of before would have charged for."""
    def attention(seq):
        model = save(tmp_path, [helper.make_node("Attention", ["x"], ["y"], name="attention", num_heads=8)],
                     [("x", [1, seq, 512])], [("y", [1, seq, 512])], [], check=False)
        return run_frontend(model, tmp_path / f"out{seq}")
    report = attention(100)
    assert report["refused"] == [] and report["unmapped"] == {} and report["nodes_run"] == 1
    assert report["total_cycles"] == report["nodes"][0]["cycles"]
    assert attention(96)["total_cycles"] < report["total_cycles"] < attention(104)["total_cycles"]
