"""The frontend without the simulator: the operator table, the tiling and how a kernel is specialised."""
import ctypes
import os
import subprocess

import pytest
from onnx import helper

from togsim_frontend import paths
from togsim_frontend.acs import cycles as acs_cycles
from togsim_frontend.acs.__main__ import EXAMPLE_TILE
from togsim_frontend.acs.node_mapping import KERNELS, lookup
from togsim_frontend.onnx_frontend import emit
from togsim_frontend.onnx_frontend import run as frontend
from togsim_frontend.onnx_frontend import tiling
from togsim_frontend.onnx_frontend.config import array_dim_from_name, read_hardware
from togsim_frontend.onnx_frontend.onnx_ops import OPS, resolve
from togsim_frontend.onnx_frontend.tiling import Hardware, check_systolic, systolic_tile, unsupported

HW = Hardware(sa=8, vlane=8, vlen_bits=256)
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
    assert systolic_tile(512, 1024, 512, Hardware(sa=4, vlane=4, vlen_bits=128)) == (32, 256, 64)
    assert systolic_tile(64, 128, 16, HW) == (32, 128, 16)


def test_a_machine_the_tiles_are_whole_matmul_instructions_on_is_supported():
    assert unsupported(HW) == []
    assert unsupported(Hardware(sa=4, vlane=16, vlen_bits=256)) == []
    assert unsupported(Hardware(sa=8, vlane=8, vlen_bits=256, spad_bytes=tiling.SYSTOLIC_TILE_SPAD_BYTES)) == []


def test_a_machine_is_refused_for_each_fact_the_tiles_do_not_fit():
    """SA has to divide a tile's rows and depth, SA * lanes its columns, and a stated spad has to hold a tile."""
    rows, = unsupported(Hardware(sa=16, vlane=4, vlen_bits=256))
    assert "8 rows of a Attention tile are not a multiple of the array's 16 rows" in rows
    cols = unsupported(Hardware(sa=8, vlane=16, vlen_bits=256))
    assert len(cols) == 1 and "64 columns of a Attention tile" in cols[0] and "128 (SA * vpu_num_lanes = 8 * 16)" in cols[0]
    spad, = unsupported(Hardware(sa=8, vlane=8, vlen_bits=256, spad_bytes=tiling.SYSTOLIC_TILE_SPAD_BYTES - 1024))
    assert "core_spad_size_kb" in spad
    assert len(unsupported(Hardware(sa=256, vlane=256, vlen_bits=512))) == 5


def test_of_the_configs_here_only_the_8x8_one_is_a_supported_machine():
    configs = sorted((paths.root() / "configs").glob("*.yml"))
    assert len(configs) == 19
    assert [c.name for c in configs if not unsupported(read_hardware(str(c)))] == \
        ["systolic_ws_8x8_c1_simple_noc_tpuv3.yml"]


def test_an_unsupported_machine_is_an_error_before_anything_is_read_or_run(tmp_path, capsys):
    config = paths.root() / "configs" / "systolic_ws_256x256_c1_simple_noc_tpuv6e.yml"
    status = frontend.main([str(tmp_path / "model.onnx"), "--config", str(config), "--out", str(tmp_path / "out")])
    said = capsys.readouterr().err
    assert status == frontend.EXIT_UNSUPPORTED_MACHINE == 2
    assert str(config) in said and "SA=256, vpu_num_lanes=256" in said
    assert "65536 (SA * vpu_num_lanes = 256 * 256)" in said
    assert not (tmp_path / "out").exists()


def stated_spad_bytes(tmp_path, stem, macros):
    """What kernels/<stem>_tog.cpp compiled with `macros` states in togsim_spad_buffer_bytes, summed."""
    so = emit.compile_so(str(paths.kernel_source(stem)), str(paths.include_dir()), str(tmp_path / f"{stem}.so"), macros)
    lib = ctypes.CDLL(so, mode=os.RTLD_LAZY)
    count = ctypes.c_int32.in_dll(lib, "togsim_spad_buffer_count").value
    return sum((ctypes.c_int64 * count).in_dll(lib, "togsim_spad_buffer_bytes"))


def test_the_spad_a_machine_needs_is_what_the_conv_kernel_states_for_the_largest_tile(tmp_path):
    conv = stated_spad_bytes(tmp_path, "conv", ["TILE_M=32", "TILE_N=256", "TILE_K=64", "TILE_I_H=8", "SA=8"])
    gemm = stated_spad_bytes(tmp_path, "gemm", ["TM=32", "TN=256", "TK=64"])
    assert conv == tiling.SYSTOLIC_TILE_SPAD_BYTES == 393216 and gemm < conv


def test_array_dim_comes_from_the_config_name():
    assert array_dim_from_name("systolic_ws_8x8_c1_simple_noc.yml") == 8
    assert array_dim_from_name("systolic_ws_128x128_c2_booksim_tpuv3.yml") == 128
    assert array_dim_from_name("no_array_here.yml") is None


@pytest.mark.parametrize("op,kernels", sorted(OPS.items()))
def test_one_operator_maps_to_one_kernel(op, kernels):
    assert len(kernels) == 1, (
        f"{op} maps to {[k.stem for k in kernels]}; merge them into one kernel")


def test_the_package_finds_the_tree_it_sits_in(monkeypatch):
    monkeypatch.delenv("TOGSIM_DIR", raising=False)
    assert (paths.root() / "include" / "togsim_runtime.h").exists()
    monkeypatch.setenv("TOGSIM_DIR", os.sep + "somewhere")
    assert str(paths.simulator()) == os.path.join(os.sep, "somewhere", "build", "bin", "Simulator")
