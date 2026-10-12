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
  last one drains through, and those rows are its `Systolic` rows in the unit table. The
  `overlapping` column is always `0`.
- **The ONNX frontend** (`togsim_frontend.onnx_frontend`). It loads a graph, infers its shapes,
  maps each node to a kernel (some operators to none, fused ones to several), compiles the kernel
  with the node's tile, writes the node's shape arguments and the ACS tables, replays it with
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
  --config configs/systolic_ws_8x8_c1_simple_noc_tpuv3.yml --out result   # per node cycles, result/report.json
python3 -m togsim_frontend.acs gemm \
  --config configs/systolic_ws_8x8_c1_simple_noc_tpuv3.yml -o /tmp/gemm   # the two tables of one kernel
(cd frontend && python3 -m pytest tests)                                 # its tests
```

## Output and exit status

The frontend prints a line per node and the total, writes `<out>/report.json`, and keeps each
node's run in a directory of its own under `<out>` (the TOG, the files beside it, the two tables and
`togsim.log`, what TOGSim printed). Its exit status:

| Exit | Meaning |
|---|---|
| `0` | Every node with work ran; the total covers the model. |
| `2` | The config is a machine the frontend does not map a model on ([Machines](#machines)). Nothing was read or run; the reasons are on stderr. Also a bad command line. |
| `3` | At least one node did not run. The nodes that could were replayed and totalled and the report is written, but the total does not cover the whole model: each such node is listed by name with the reason, on stdout and under `refused` in the report, and the last line says so. |

## Machines

The systolic tiles are the same on every machine: a Gemm, MatMul or Conv is cut into
tiles of at most 32 rows x 256 columns x 64 deep (a Conv tile spans at most 8 output rows), an
Attention into 8 queries x 64 keys. The frontend does not tile differently for another machine;
it refuses a config whose machine those tiles do not fit. With `SA` the systolic array dimension,
a config is accepted when all of these hold:

- `SA` divides a tile's rows and its depth: `SA` divides 8 (and so 32 and 64).
- `SA * vpu_num_lanes`, the columns of one matmul instruction, divides a tile's columns: it
  divides 64 (and so 256).
- `core_spad_size_kb`, if the config states it, is at least 384, the spad buffers of the largest
  tile (a Conv's). TOGSim never issues a tile whose buffers exceed the spad, so a smaller one
  would not finish.

Of `configs/`, `systolic_ws_8x8_c1_simple_noc_tpuv3.yml` is accepted. The 32x32, 128x128 and
256x256 (tpuv6e) configs are refused: one matmul instruction is 1024, 16384 and 65536 columns
there, wider than the tile. ACS on its own (`python3 -m togsim_frontend.acs`) is not bound by
this: it writes the tables for the config and the tile it is given. `SA` is not a config key; it
is read from the config's file name (`..._8x8_...`) or given with `--sa`.

## Nodes that do not run

Each of these is exit `3`: an operator without a kernel; a Gemm, MatMul or Conv with
fewer output columns than one matmul instruction (`SA * vpu_num_lanes`, 64 on 8x8); a grouped
convolution; a node whose kernel names a tile id the tables ACS wrote do not cover; a node TOGSim
did not replay to the end. Metadata operators (Reshape, Transpose, ...) have no work and are
counted as no-ops.

## Convolutions

A convolution is mapped by its output extent with stride-1 input addresses and a padding of one,
so what it moves is right for a 3x3, stride-1, pad-1 convolution and approximate otherwise.

## Tests

`frontend/tests/test_end_to_end.py` runs against `build/bin/Simulator` and is skipped without it,
unless `TOGSIM_REQUIRE_SIMULATOR` is set (CI sets it): then a missing simulator fails the tests.
