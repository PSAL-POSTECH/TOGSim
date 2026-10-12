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

## Machines

The systolic array dimension is not a config key; it is read from the config's file name
(`..._8x8_...`) or given with `--sa`. A Gemm, MatMul or Conv narrower than `SA * vpu_num_lanes`
output columns is reported as unmapped and left out of the total, as is an operator without a
kernel.

## Tests

`frontend/tests/test_end_to_end.py` runs against `build/bin/Simulator` and is skipped without it.
