"""The frontend without the simulator: the operator table, the tiling and how a kernel is specialised."""
import os
import subprocess

import pytest
import yaml
from onnx import helper

from togsim_frontend import paths
from togsim_frontend.acs import cycles as acs_cycles
from togsim_frontend.acs.__main__ import EXAMPLE_TILE
from togsim_frontend.acs.node_mapping import KERNELS, lookup
from togsim_frontend.onnx_frontend import emit
from togsim_frontend.onnx_frontend import run as frontend
from togsim_frontend.onnx_frontend import shapes
from togsim_frontend.onnx_frontend import tiling
from togsim_frontend.onnx_frontend.config import core_spad_kb, read_hardware, vector_length
from togsim_frontend.onnx_frontend.onnx_ops import OPS, resolve
from togsim_frontend.onnx_frontend.tiling import Hardware, check_systolic, systolic_tile, unsupported

HW = Hardware(sa=8, vlane=8, vlen_bits=256, array_unit="TPUV3_MXU")
GEMM = OPS["Gemm"][0]


def test_tile_constants_reach_the_compiler_as_macros_not_as_source_edits():
    """The macros are gemm's tile; the shape arguments come in the order gemm reads them, M K N."""
    values = {"M": 512, "K": 64, "N": 256, "TM": 32, "TN": 256, "TK": 64}
    assert emit.defines(GEMM, values) == ["TM=32", "TN=256", "TK=64"]
    assert emit.shape_args(GEMM, values) == [512, 64, 256]


def test_a_name_the_frontend_has_no_value_for_is_refused():
    # silently skipping leaves the kernel at the example's size, and the run
    # still reports a cycle count -- for a shape nobody asked about
    with pytest.raises(KeyError):
        emit.defines(GEMM, {"M": 128, "K": 64, "N": 256, "TM": 32})
    with pytest.raises(KeyError):
        emit.shape_args(GEMM, {"M": 128, "TM": 32, "TN": 256, "TK": 64})


CONV = OPS["Conv"][0]
CONV_ARGS = ("S_H", "S_W", "D_H", "D_W", "P_T", "P_L", "P_B", "P_R")


def conv_values(x, w, out=None, **attributes):
    """What shapes.constants gives the conv kernel for a Conv of input `x` and weights `w`; `out` is the output
    shape ONNX shape inference recorded, when it did."""
    node = helper.make_node("Conv", ["x", "w"], ["y"], name="conv", **attributes)
    return shapes.constants(node, CONV, {"x": x, "w": w, **({"y": out} if out else {})}, HW)


def test_a_conv_passes_its_strides_dilations_and_four_pads_as_shape_arguments():
    """The pads stay in ONNX's order, top left bottom right, and the input's bytes are those of the padded input."""
    values = conv_values([1, 64, 32, 15], [64, 64, 3, 3], [1, 64, 16, 16], strides=[2, 1], pads=[0, 1, 1, 2])
    assert emit.shape_args(CONV, values) == [64, 32, 15, 64, 16, 16, 3, 3, 2, 1, 1, 1, 0, 1, 1, 2]
    assert CONV.size_names[8:] == CONV_ARGS
    assert shapes.tensor_bytes(CONV, values) == [33 * 18 * 64 * 4, 9 * 64 * 64 * 4, 64 * 16 * 16 * 4]
    dilated = conv_values([1, 64, 16, 16], [64, 64, 3, 3], dilations=[2, 3], pads=[2, 3, 2, 3])
    assert [dilated[n] for n in ("O_H", "O_W") + CONV_ARGS] == [16, 16, 1, 1, 2, 3, 2, 3, 2, 3]


def test_a_conv_without_attributes_has_stride_1_and_no_padding():
    values = conv_values([1, 64, 18, 18], [64, 64, 3, 3])
    assert [values[n] for n in ("O_H", "O_W") + CONV_ARGS] == [16, 16, 1, 1, 1, 1, 0, 0, 0, 0]


@pytest.mark.parametrize("auto_pad,size,pads,out", [
    ("VALID", 32, [0, 0, 0, 0], 15), ("SAME_UPPER", 32, [0, 0, 1, 1], 16), ("SAME_LOWER", 32, [1, 1, 0, 0], 16),
    ("SAME_UPPER", 31, [1, 1, 1, 1], 16)])
def test_auto_pad_stands_for_the_pads_onnx_defines(auto_pad, size, pads, out):
    """A 3 x 3 kernel at stride 2: SAME_* pads to an output of ceil(size / 2), the odd cell at the end or the start."""
    values = conv_values([1, 64, size, size], [64, 64, 3, 3], [1, 64, out, out], strides=[2, 2], auto_pad=auto_pad)
    assert [values[n] for n in ("P_T", "P_L", "P_B", "P_R")] == pads and values["O_H"] == values["O_W"] == out


def test_a_conv_the_kernel_does_not_take_is_refused_with_the_reason():
    """Groups, another count of spatial axes, a kernel larger than the padded input, and an output extent that is
    not the one ONNX shape inference recorded."""
    with pytest.raises(ValueError, match="grouped convolution"):
        conv_values([1, 64, 16, 16], [64, 1, 3, 3], group=64)
    with pytest.raises(ValueError, match="a convolution over 1 spatial axes: the conv kernel takes two"):
        conv_values([1, 64, 16], [64, 64, 3])
    with pytest.raises(ValueError, match="a 3 x 3 kernel .dilations 4, 4. does not fit the padded input, 8 x 8"):
        conv_values([1, 64, 8, 8], [64, 64, 3, 3], dilations=[4, 4])
    with pytest.raises(ValueError, match="shape inference gives the output 64 x 16 x 16, and the strides, "
                                         "dilations and pads give 64 x 8 x 8"):
        conv_values([1, 64, 16, 16], [64, 64, 3, 3], [1, 64, 16, 16], strides=[2, 2], pads=[1, 1, 1, 1])


@pytest.mark.parametrize("attributes,said", [
    (dict(strides=[0, 1]), "strides .0, 1. is not 2 integers of at least 1"),
    (dict(strides=[2]), "strides .2. is not 2 integers of at least 1"),
    (dict(dilations=[1, -1]), "dilations .1, -1. is not 2 integers of at least 1"),
    (dict(pads=[0, 0, 0, -1]), "pads .0, 0, 0, -1. is not 4 integers of at least 0"),
    (dict(pads=[1, 1]), "pads .1, 1. is not 4 integers of at least 0")])
def test_a_conv_with_a_stride_a_dilation_or_a_pad_no_convolution_has_is_refused(attributes, said):
    with pytest.raises(ValueError, match=said):
        conv_values([1, 64, 16, 16], [64, 64, 3, 3], **attributes)


@pytest.mark.parametrize("op,kernels", sorted(OPS.items()))
def test_a_node_over_a_tensor_without_elements_is_refused_whatever_its_kernel(op, kernels):
    node = helper.make_node(op, ["x", "w"], ["y"], name="empty")
    with pytest.raises(ValueError, match="input 'x' is 4 x 0 x 8: a tensor without elements"):
        shapes.constants(node, kernels[0], {"x": [4, 0, 8], "w": [8, 8], "y": [4, 8]}, HW)


def test_a_tog_directory_holds_the_sidecars_togsim_reads(tmp_path):
    so = tmp_path / "kernel.so"
    so.write_bytes(b"\x7fELF")
    tog = emit.write_tog_dir(str(tmp_path / "run"), str(so), [128, 256], [131072, 1024, 131072])
    assert tog == str(tmp_path / "run" / "tog.so")
    assert (tmp_path / "run" / "tog_shape.txt").read_text().split() == ["128", "256"]
    assert (tmp_path / "run" / "tog_tensors.txt").read_text().split() == ["131072", "1024", "131072"]


def test_a_node_whose_tile_ids_the_tables_miss_is_refused_by_name(tmp_path, monkeypatch):
    """A Relu's kernel names tile id 0. With the tables as ACS builds them the run directory is laid out; with a
    unit table that lost its rows the node is refused before anything is written for TOGSim to trip on."""
    relu = helper.make_node("Relu", ["x"], ["y"], name="relu_7")
    shapes = {"x": [64, 256], "y": [64, 256]}
    tog_dir = frontend.prepare_node(relu, OPS["Relu"][0], shapes, HW, str(tmp_path / "covered"))
    assert os.path.exists(os.path.join(tog_dir, "tog_units.tsv"))
    monkeypatch.setattr(acs_cycles, "build_units", lambda desc, tile, hw: [])
    with pytest.raises(ValueError, match=r"node 'relu_7': kernel bias_act names tile ids \[0\]"):
        frontend.prepare_node(relu, OPS["Relu"][0], shapes, HW, str(tmp_path / "missed"))
    assert not list((tmp_path / "missed").rglob("tog_units.tsv"))


@pytest.mark.parametrize("op,kernels", sorted(OPS.items()))
def test_every_mapped_kernel_exists_and_compiles_with_its_macros(op, kernels, tmp_path):
    for kernel in kernels:
        path = paths.kernel_source(kernel.stem)
        assert path.exists(), f"{op}: no kernel at {path}"
        macros = [f"{name}=8" for name in kernel.tile_names + kernel.hw_names]
        try:
            emit.compile_so(str(path), str(paths.include_dir()), str(tmp_path / f"{kernel.stem}.so"), macros)
        except subprocess.CalledProcessError as error:
            pytest.fail(f"{op}/{kernel.stem} does not compile with {macros}:\n{error.stderr}")


@pytest.mark.parametrize("op,kernels", sorted(OPS.items()))
def test_every_mapped_kernel_has_a_node_table(op, kernels):
    for kernel in kernels:
        lookup(kernel.stem)              # raises with a message if absent


@pytest.mark.parametrize("stem", sorted(KERNELS))
def test_every_kernel_with_a_node_table_exists_and_has_a_default_tile(stem):
    assert paths.kernel_source(stem).exists()
    assert stem in EXAMPLE_TILE


def test_a_fused_node_decomposes_to_kernels_that_exist():
    assert [k.stem for k in resolve("BiasGelu")] == ["bias_act", "bias_gelu"]
    assert [k.stem for k in resolve("SkipLayerNormalization")] == \
        ["bias_act", "layernorm"]


def test_a_fused_node_with_its_own_kernel_maps_straight_to_it():
    assert [k.stem for k in resolve("Attention")] == ["attention"]


def test_an_unsupported_op_is_reported_not_guessed():
    assert resolve("Einsum") is None     # None -> unmapped, never costed


def test_metadata_ops_carry_no_compute():
    assert resolve("Reshape") == []


def test_systolic_tile_is_a_multiple_of_the_array_geometry():
    """Under SA*VLANE columns a block has no whole matmul instruction."""
    tm, tn, tk = systolic_tile(512, 1024, 512, HW)
    assert tm % HW.sa == 0
    assert tn % (HW.sa * HW.vlane) == 0 and tn >= HW.sa * HW.vlane


def test_too_narrow_for_the_array_is_refused():
    check_systolic(64, HW)               # 64 == SA*VLANE, the minimum
    with pytest.raises(ValueError):
        check_systolic(63, HW)


def test_the_systolic_tile_is_the_same_on_every_machine():
    """At most 32 x 256 x 64, whatever the array: the machine decides whether that tile fits, not its size."""
    assert systolic_tile(512, 1024, 512, HW) == (32, 256, 64)
    hw = Hardware(sa=4, vlane=4, vlen_bits=128, array_unit="TPUV3_MXU")
    assert systolic_tile(512, 1024, 512, hw) == (32, 256, 64)
    assert systolic_tile(64, 128, 16, HW) == (32, 128, 16)


def test_a_machine_the_tiles_are_whole_matmul_instructions_on_is_supported():
    """Whatever its spad: a node's tile is held against that, and it is no larger than the node."""
    assert unsupported(HW) == []
    assert unsupported(Hardware(sa=4, vlane=16, vlen_bits=256, array_unit="TPUV3_MXU")) == []
    assert unsupported(Hardware(sa=8, vlane=8, vlen_bits=256, array_unit="TPUV3_MXU", spad_bytes=1024)) == []


def test_a_machine_is_refused_for_each_fact_the_tiles_do_not_fit():
    """SA has to divide a tile's rows and depth, and SA * lanes its columns."""
    rows, = unsupported(Hardware(sa=16, vlane=4, vlen_bits=256, array_unit="TPUV3_MXU"))
    assert "8 rows of a Attention tile are not a multiple of the array's 16 rows" in rows
    cols = unsupported(Hardware(sa=8, vlane=16, vlen_bits=256, array_unit="TPUV3_MXU"))
    assert len(cols) == 1 and "64 columns of a Attention tile" in cols[0] and "128 (SA * vpu_num_lanes = 8 * 16)" in cols[0]
    assert len(unsupported(Hardware(sa=256, vlane=256, vlen_bits=512, array_unit="TPUV3_MXU"))) == 5


def test_of_the_configs_here_only_the_8x8_one_is_a_supported_machine():
    configs = sorted((paths.root() / "configs").glob("*.yml"))
    assert len(configs) == 7
    assert [c.name for c in configs if not unsupported(read_hardware(str(c)))] == \
        ["tpuv3_8x8.yml"]


def test_an_unsupported_machine_is_an_error_before_anything_is_read_or_run(tmp_path, capsys):
    config = paths.root() / "configs" / "tpuv6e.yml"
    status = frontend.main([str(tmp_path / "model.onnx"), "--config", str(config), "--out", str(tmp_path / "out")])
    said = capsys.readouterr().err
    assert status == frontend.EXIT_UNSUPPORTED_MACHINE
    assert str(config) in said and "SA=128, vpu_num_lanes=128" in said
    assert "16384 (SA * vpu_num_lanes = 128 * 128)" in said
    assert not (tmp_path / "out").exists()


def stated_spad_bytes(tmp_path, stem, macros):
    """What kernels/<stem>_tog.cpp compiled with `macros` states in togsim_spad_buffer_bytes, buffer by buffer."""
    so = emit.compile_so(str(paths.kernel_source(stem)), str(paths.include_dir()), str(tmp_path / f"{stem}.so"), macros)
    return emit.spad_buffer_bytes(so)


#: The spad TOGSim gives a core of each config here, in KB. TOGSim states it nowhere a program reads it but its log,
#: so it is written out: vpu_spad_size_kb_per_lane x vpu_num_lanes, as each config gives both.
SPAD_KB = {**{f"{name}.yml": 128 * 128 for name in (
               "tpuv3_booksim", "tpuv3", "tpuv3_half", "tpuv4")},
           "tpuv6e.yml": 1024 * 128,
           "tpuv3_32x32.yml": 32 * 32, "tpuv3_8x8.yml": 32 * 8}


def test_the_frontend_reads_the_spad_of_each_config_here_as_togsim_does():
    configs = sorted((paths.root() / "configs").glob("*.yml"))
    assert {c.name: read_hardware(str(c)).spad_bytes // 1024 for c in configs} == SPAD_KB


def test_the_spad_is_its_lanes_and_a_core_spad_size_kb_beside_them_may_only_restate_it():
    """TOGSim's rule (core_spad_kb, src/Common.cc), on the machines its tests/spad_capacity run."""
    lanes = {"vpu_num_lanes": 3, "vpu_spad_size_kb_per_lane": 16}
    assert core_spad_kb(lanes) == core_spad_kb({**lanes, "core_spad_size_kb": 48}) == 48
    with pytest.raises(ValueError, match="core_spad_size_kb: 47 is not vpu_spad_size_kb_per_lane x vpu_num_lanes = "
                                         "16 x 3 = 48: state the spad once"):
        core_spad_kb({**lanes, "core_spad_size_kb": 47})
    assert core_spad_kb({"vpu_num_lanes": 256, "core_spad_size_kb": 48}) == 48
    assert core_spad_kb({"vpu_spad_size_kb_per_lane": 16, "core_spad_size_kb": 48}) == 48
    assert core_spad_kb({"vpu_num_lanes": 256}) == 0
    with pytest.raises(ValueError, match="65536 x 65536 = 4294967296 KB is more than a spad TOGSim holds"):
        core_spad_kb({"vpu_num_lanes": 65536, "vpu_spad_size_kb_per_lane": 65536})


def test_a_conv_or_gemm_holds_every_buffer_its_kernel_states_and_the_largest_conv_tile_takes_384_kb(tmp_path):
    """Its accumulator, weights and input, all read by its matmul; the others are not held against the spad."""
    conv = stated_spad_bytes(tmp_path, "conv", ["TILE_M=32", "TILE_N=256", "TILE_K=64", "TILE_I_H=8", "SA=8"])
    gemm = stated_spad_bytes(tmp_path, "gemm", ["TM=32", "TN=256", "TK=64"])
    assert sum(conv) == 393216 and sum(gemm) == 106496
    hw = Hardware(sa=8, vlane=8, vlen_bits=256, array_unit="TPUV3_MXU", spad_bytes=393216)
    assert tiling.spad_refusal("conv", conv, hw) is None and tiling.spad_refusal("gemm", gemm, hw) is None
    small = Hardware(sa=8, vlane=8, vlen_bits=256, array_unit="TPUV3_MXU", spad_bytes=393216 - 1024)
    assert tiling.spad_refusal("conv", conv, small) == ("a work-item of kernel conv holds its 3 spad buffers at once, "
                                                        "393216 bytes, and a core's spad is 392192: TOGSim refuses the "
                                                        "TOG")
    assert tiling.spad_refusal("conv", conv, HW) is None
    assert tiling.spad_refusal("attention", [1 << 30], small) is None


def test_a_node_whose_tile_the_spad_does_not_hold_is_refused_and_one_of_fewer_channels_is_not(tmp_path):
    """On the 8x8 config's 256 KB, a Conv of 256 output channels cuts the 384 KB tile; of 128, a tile of 224 KB."""
    hw = read_hardware(str(paths.root() / "configs" / "tpuv3_8x8.yml"))
    assert hw.spad_bytes == 256 * 1024
    for channels, refused in ((256, True), (128, False)):
        node = helper.make_node("Conv", ["x", "w"], ["y"], name="conv", pads=[1, 1, 1, 1])
        shapes_of = {"x": [1, 64, 16, 32], "w": [channels, 64, 3, 3]}
        if refused:
            with pytest.raises(ValueError, match="node 'conv': a work-item of kernel conv holds its 3 spad buffers at "
                                                 "once, 393216 bytes, and a core's spad is 262144"):
                frontend.prepare_node(node, CONV, shapes_of, hw, str(tmp_path))
        else:
            assert os.path.isdir(frontend.prepare_node(node, CONV, shapes_of, hw, str(tmp_path)))


def test_the_second_pass_of_layernorm_and_global_avgpool_keeps_the_128_row_tile_of_its_kernel(tmp_path):
    """The statistics pass and the scale pass run a tile of their own, TM_STATS and TC_SCALE. The frontend names it
    to predict its short last tile, and names the kernel's default: the kernel states the spad it does without it."""
    layernorm, avgpool = OPS["LayerNormalization"][0], OPS["GlobalAveragePool"][0]
    node = helper.make_node("LayerNormalization", ["x"], ["y"], name="ln")
    assert shapes.constants(node, layernorm, {"x": [64, 256]}, HW)["TM_STATS"] == tiling.SECOND_PASS_ROWS == 128
    node = helper.make_node("GlobalAveragePool", ["x"], ["y"], name="gap")
    assert shapes.constants(node, avgpool, {"x": [1, 64, 16, 16]}, HW)["TC_SCALE"] == 128
    for stem, tile, second in (("layernorm", ["TM=8", "TN=64"], "TM_STATS=128"),
                               ("global_avgpool", ["TC=8", "THW=64"], "TC_SCALE=128")):
        assert stated_spad_bytes(tmp_path / "named", stem, tile + [second]) == \
            stated_spad_bytes(tmp_path / "default", stem, tile)
        assert stated_spad_bytes(tmp_path / "other", stem, tile + [second[:-3] + "64"]) != \
            stated_spad_bytes(tmp_path / "default", stem, tile)


def test_the_array_dimension_is_vpu_num_lanes_unless_sa_is_given(tmp_path):
    config = tmp_path / "machine.yml"
    config.write_text("vpu_num_lanes: 8\nisa: [v, zvl256b]\nsystolic_array_unit: TPUV3_MXU\n")
    assert read_hardware(str(config)).sa == 8 and read_hardware(str(config), 16).sa == 16


def test_the_array_unit_is_the_configs_and_the_split_needs_one(tmp_path):
    """The unit table names systolic_array_unit; without the split there is none, and no array rows."""
    config = tmp_path / "machine.yml"
    config.write_text("vpu_num_lanes: 8\nisa: [v, zvl256b]\n")
    with pytest.raises(ValueError, match=r"round_robin \(the default\) needs systolic_array_unit"):
        read_hardware(str(config))
    config.write_text("vpu_num_lanes: 8\nisa: [v, zvl256b]\nsystolic_array_split: none\n")
    unsplit = read_hardware(str(config))
    assert unsplit.array_unit is None
    assert {row[1] for row in acs_cycles.build_units(lookup("gemm"), {"rows": 8}, unsplit)} == {"Vpu"}
    assert read_hardware(str(paths.root() / "configs" / "tpuv4.yml")).array_unit == "TPUV4_MXU"


@pytest.mark.parametrize("over,said", [
    ({"systolic_array_unit": ""}, "systolic_array_unit: '' is not a name"),
    ({"systolic_array_unit": None}, "systolic_array_unit: None is not a name"),
    ({"systolic_array_unit": 7}, "systolic_array_unit: 7 is not a name"),
    ({"systolic_array_split": "none"}, "systolic_array_unit: TPUV6E_MXU is ignored without systolic_array_split"),
    ({"systolic_array_split": "sideways"}, r"systolic_array_split: sideways \(none \| round_robin\)"),
])
def test_an_array_unit_or_split_togsim_refuses_is_refused(over, said, tmp_path):
    """tpuv6e with an array unit that is not a name, a unit beside the split off, or a split TOGSim does not know."""
    config, v6e = tmp_path / "machine.yml", paths.root() / "configs" / "tpuv6e.yml"
    config.write_text(yaml.safe_dump({**yaml.safe_load(v6e.read_text()), **over}))
    with pytest.raises(ValueError, match=said):
        read_hardware(str(config))


def test_the_vector_length_is_the_zvl_entry_of_isa():
    """VLEN is N of the one zvl<N>b entry; a config with isa and no such entry, or two, or no isa, is refused."""
    assert vector_length({"isa": ["m", "v", "zvl512b", "pytorchsim_tpuv6e"]}) == 512
    assert {read_hardware(str(c)).vlen_bits for c in (paths.root() / "configs").glob("*.yml")} == {256}
    for cfg, said in (({"isa": ["m", "v"]}, "has no zvl<N>b entries"), ({}, "missing required key 'isa'"),
                      ({"isa": ["zvl128b", "zvl256b"]}, "has 2 zvl<N>b entries")):
        with pytest.raises(ValueError, match=said):
            vector_length(cfg)


@pytest.mark.parametrize("op,kernels", sorted(OPS.items()))
def test_one_operator_maps_to_one_kernel(op, kernels):
    assert len(kernels) == 1, (
        f"{op} maps to {[k.stem for k in kernels]}; merge them into one kernel")


def test_the_package_finds_the_tree_it_sits_in(monkeypatch):
    monkeypatch.delenv("TOGSIM_DIR", raising=False)
    assert (paths.root() / "include" / "togsim_runtime.h").exists()
    monkeypatch.setenv("TOGSIM_DIR", os.sep + "somewhere")
    assert str(paths.simulator()) == os.path.join(os.sep, "somewhere", "build", "bin", "Simulator")
