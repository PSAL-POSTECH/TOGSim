# The frontend layer (optional)

`frontend/` is a Python package, `togsim_frontend`, that sits on top of the simulator. The
simulator, its build and `tests/run.py` do not depend on it or on any Python package; it has its
own tests and its own CI job. Like the [kernel library](kernels.md), it comes from PyTorchSim PR
#315. It has two parts.

- **ACS** (`togsim_frontend.acs`), an analytic cycle model. For a kernel of `kernels/` and a
  machine config it writes `tog_cycles.tsv` and `tog_units.tsv` from formulas, where a compiler
  would have measured them with gem5. **Its numbers are estimates: the formulas have not been
  calibrated against gem5, and no result built on them should be read as a measurement.** A vector
  tile costs `ceil(elements / (lanes * elements per lane))` cycles per elementwise instruction; a
  systolic block costs the rows its matmuls feed (one per cycle) plus the `2*SA - 2` stages the
  last one drains through, and those rows are its rows of the config's `systolic_array_unit` in the unit table. The
  `overlapping` column is always `0`. A short last tile is costed by the same formulas at the
  extents it has: a vector tile by its elements, a block by `depth * ceil(columns / (SA * lanes))`
  preloads of its rows. So a block 3 channels deep is 3 preloads where an 8-deep one is 8 (the
  formula counts a preload per depth step and rounds nothing to `SA` there), 13 rows are fed as
  13 (a matmul of 8 and one of 5), and columns short of `SA * lanes` still take one preload per
  depth step. Run on its own (`python3 -m togsim_frontend.acs`), ACS writes the rows of the whole
  tile only, the tables of a shape the tile divides.
- **The ONNX frontend** (`togsim_frontend.onnx_frontend`). It loads a graph, infers its shapes,
  maps each node to a kernel (some operators to none, fused ones to several), compiles the kernel
  with the node's tile, writes the node's shape arguments and the ACS tables (a row per tile id
  the kernel names at that shape, the short variants at what is left), replays it with
  `--tog_so`, and sums the nodes' cycles. Each node runs on its own: the total is a sum of
  isolated kernels, not one session.

## Installing and running

It needs `python3` >= 3.9, `g++`, a build of the simulator, and `onnx` and `pyyaml`
(`onnxruntime` is optional and only fuses the graph first). It finds the tree through
`TOGSIM_DIR`, or, installed in place, through the checkout it sits in.

```bash
python3 -m pip install --upgrade pip    # Ubuntu 22.04's pip 22.0.2 cannot install a pyproject-only package
python3 -m pip install -e "./frontend[test]"
python3 -m togsim_frontend.onnx_frontend model.onnx \
  --config configs/tpuv3_8x8.yml --out result   # per node cycles, result/report.json
python3 -m togsim_frontend.acs gemm \
  --config configs/tpuv3_8x8.yml -o /tmp/gemm   # the two tables of one kernel
(cd frontend && python3 -m pytest tests)                                 # its tests
```

## Output and exit status

The frontend prints a line per node and the total, writes `<out>/report.json`, and keeps each
node's run in a directory of its own under `<out>` (the TOG, the files beside it, the two tables and
`togsim.log`, what TOGSim printed). Its exit status, and that of ACS (`0`, `1`, `2` and `4` there),
is one of the list in `frontend/togsim_frontend/status.py`; every status but `0` and `3` comes with
an `error:` line on stderr, or a traceback for `1`:

| Exit | Meaning |
|---|---|
| `0` | Every node with work ran; the total covers the model. ACS: the two tables are written. |
| `1` | An exception none of the statuses below names: Python's own status, with its traceback. A defect in the frontend, or a failure it does not classify (an output directory that cannot be written, a model ONNX shape inference rejects, a node without an input its operator reads). |
| `2` | The command line could not be parsed: argparse's own status, with its usage line. For ACS also a kernel name it has no node table for. Nothing else exits `2`. |
| `3` | At least one node did not run. The nodes that could were replayed and totalled and the report is written, but the total does not cover the whole model: each such node is listed by name with the reason, on stdout and under `refused` in the report, and the last line says so. |
| `4` | The config or the model cannot be read: missing, unreadable or malformed, a config without `vpu_num_lanes`, or without an `isa` of exactly one `zvl<N>b` entry (the vector length), or without `systolic_array_unit` under the array split, or a machine with less than one array row (`--sa` included), one lane or one 32-bit element per lane, or a spad TOGSim refuses ([Machines](#machines)). Nothing was run and no output directory is created. |
| `5` | The config is a machine the frontend does not map a model on ([Machines](#machines)). Nothing was read or run; the reasons are on stderr. |
| `6` | A tool the run depends on failed: the compiler (`$CXX`, else `g++`) is absent or did not build a kernel, or the Simulator is absent, exited non-zero (a config or a TOG it refuses, a crash) or printed no `Total execution cycles`. The run stops at that node, named on stderr with the tool's last output: no total is given and no report is written. The runs of the nodes before it, and the failing run's `togsim.log`, stay under `<out>`. |

## Machines

The systolic tiles are the same on every machine: a Gemm, MatMul or Conv is cut into
tiles of at most 32 rows x 256 columns x 64 deep (a Conv tile spans at most 8 output rows), an
Attention into 8 queries x 64 keys. The frontend does not tile differently for another machine;
it refuses a config whose machine those tiles do not fit. With `SA` the systolic array dimension,
a config is accepted when all of these hold:

- `SA` divides a tile's rows and its depth: `SA` divides 8 (and so 32 and 64).
- `SA * vpu_num_lanes`, the columns of one matmul instruction, divides a tile's columns: it
  divides 64 (and so 256).

The spad is not one of them: a node cuts a tile no larger than itself, and is held against the
spad on its own ([Nodes that do not run](#nodes-that-do-not-run)). The frontend reads a core's
spad as TOGSim does: `vpu_spad_size_kb_per_lane` x `vpu_num_lanes` when the config gives both
(a `core_spad_size_kb` beside them that is not their product is exit `4`, as TOGSim refuses it),
else `core_spad_size_kb` alone, else no limit.

Of `configs/`, `tpuv3_8x8.yml` is accepted. The 32-lane and 128-lane configs are refused: one matmul instruction is 1024 and 16384 columns there, wider than the
tile. ACS on its own (`python3 -m togsim_frontend.acs`) is not bound by
this: it writes the tables for the config and the tile it is given. `SA` is `vpu_num_lanes`, the
rows one array takes in TOGSim, unless `--sa` gives it.

## Nodes that do not run

Each of these is exit `3`: an operator without a kernel; a Gemm, MatMul or Conv with
fewer output columns than one matmul instruction (`SA * vpu_num_lanes`, 64 on 8x8); a Conv that
is grouped, is not over two spatial axes, has a kernel larger than its padded input, or whose
output extent by its attributes is not the one ONNX shape inference recorded; a Conv whose
`strides` or `dilations` are not two integers of at least 1, whose `pads` are not four of at least
0, or whose `auto_pad` is not one ONNX defines; a node with an input that has an extent of 0; a
Gemm, MatMul or Conv whose tile's spad buffers (accumulator, weights and input, all held at once
by its matmul) are more than a core's spad, as TOGSim refuses a TOG with a work-item that needs
more of the spad [at once](tog-abi.md#buffers-and-the-scratchpad) than a core has (the largest
Conv tile takes 384 KB, the 8x8 config's spad is 256 KB); a node whose kernel names a tile id
the tables ACS wrote do not cover. A node TOGSim did not replay to the end is not one of them:
that is exit `6`. Metadata operators (Reshape, Transpose, ...) have
no work and are counted as no-ops.

## Convolutions

A Conv passes its `strides`, `dilations` and `pads` to the kernel as shape arguments, per axis
and per side (`auto_pad` is turned into the pads it stands for), and sizes the input tensor as
the padded input, so its loads are at the addresses that convolution reads. The cost ACS gives a
conv tile does not depend on the stride, the dilation or the pads: a compute is the tile's outputs
by one kernel position's channels.

## Shapes the tiles do not divide

The tiles stay the fixed ones above, whatever the shape: a
Relu of 128 rows in tiles of 42 ends in a tile of 2 rows, a Conv of 112 output columns in tiles of
32 in one of 16, a 3-channel stem runs blocks 3 deep under the 8-deep tile, a Gemm of 5 rows has
its short tile alone. The frontend predicts the tile ids the kernel names at the node's shape
(`acs/node_mapping.py` lists each kernel's axes and the axes each node carries), has ACS cost each
at its extents, and refuses the node if a predicted id has no cycle line or no unit rows. The
tables have a line of zeros and no unit rows for an id no tile of the shape has, so a compute the
prediction missed is a TOGSim warning or refusal, not a silent cost. ACS takes one tile per
kernel: the nodes of layernorm's statistics pass and of global_avgpool's scale pass, which run
tiles of their own (`TM_STATS`, `TC_SCALE`, 128 rows), are costed by the kernel's main tile as
they were, whole or short.

## Tests

`frontend/tests/test_end_to_end.py` runs against `build/bin/Simulator` and is skipped without it,
unless `TOGSIM_REQUIRE_SIMULATOR` is set (CI sets it): then a missing simulator fails the tests.
`frontend/tests/test_exit_status.py` reaches each exit status through `python3 -m`; three of its
runs need the simulator in the same way, the others put a script in its place or stop before it.
