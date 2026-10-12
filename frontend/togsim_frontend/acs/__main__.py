"""`python -m togsim_frontend.acs <kernel>`: write tog_cycles.tsv and tog_units.tsv for one kernel of kernels/."""
from __future__ import annotations

import argparse
import os

from ..onnx_frontend.config import read_hardware
from . import cycles as acs_cycles
from . import node_mapping

#: The tile each kernel is built with when no -D is given: rows x cols elements per vector tile ("elems"), and
#: for the systolic kernels the rows x depth x cols block of one compute.
EXAMPLE_TILE = {
    "gemm": dict(rows=32, depth=64, cols=256, elems=32 * 256),
    "conv": dict(rows=8 * 16, depth=64, cols=64, elems=8 * 16 * 64),
    "attention": dict(rows=8, depth=64, cols=64, elems=8 * 64),
    "softmax": dict(rows=128, elems=128 * 64),
    "layernorm": dict(rows=64, elems=64 * 64),
    "embed_layernorm": dict(rows=128, elems=128 * 64),
    "global_avgpool": dict(rows=256, elems=256 * 64),
    "bias_act": dict(rows=128, elems=128 * 64),
    "bias_gelu": dict(rows=128, elems=128 * 64),
    "maxpool": dict(rows=256, elems=256 * 16),
    "adaptive_avgpool": dict(rows=256, elems=256 * 16),
    "kvcache_concat": dict(rows=32, elems=32 * 256),
    "concat": dict(rows=128, elems=128 * 16),
    "flatten": dict(rows=1, elems=1),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Derive a TOGSim cycle table and unit table for one kernel, without gem5. "
                    "The numbers are estimates, not calibrated against gem5.")
    ap.add_argument("kernel", help="kernel name, as in kernels/<name>_tog.cpp")
    ap.add_argument("--config", required=True)
    ap.add_argument("-o", "--out", default=".",
                    help="directory for tog_cycles.tsv and tog_units.tsv (default: the current one)")
    ap.add_argument("--sa", type=int, default=None,
                    help="systolic array dimension; inferred from the config name")
    for name, what in (("rows", "rows"), ("elems", "element count"), ("depth", "reduction depth"),
                       ("cols", "column count")):
        ap.add_argument(f"--{name}", type=int, default=None,
                        help=f"the tile's {what}, when the kernel is built with another tile than its default")
    args = ap.parse_args(argv)

    hw = read_hardware(args.config, args.sa)
    desc = node_mapping.lookup(args.kernel)
    if args.kernel not in EXAMPLE_TILE:
        raise KeyError(f"no default tile recorded for kernel {args.kernel!r}; add it to EXAMPLE_TILE in "
                       f"acs/__main__.py")
    tile = dict(EXAMPLE_TILE[args.kernel])
    tile.update({k: v for k in ("rows", "elems", "depth", "cols") if (v := getattr(args, k)) is not None})

    costs = acs_cycles.costs(desc, tile, hw)
    os.makedirs(args.out, exist_ok=True)
    table = os.path.join(args.out, "tog_cycles.tsv")
    units = os.path.join(args.out, "tog_units.tsv")
    acs_cycles.write_table(acs_cycles.build_table(desc, tile, hw), table)
    acs_cycles.write_units(acs_cycles.build_units(desc, tile, hw), units)

    print(f"kernel   : {args.kernel} ({len(desc.nodes)} compute nodes)")
    print(f"hardware : SA={hw.sa} lanes={hw.vlane} vlen={hw.vlen_bits}b "
          f"({hw.throughput} elem/cycle)")
    print(f"tile     : {' '.join(f'{k}={v}' for k, v in tile.items())}")
    print(f"tables   : {table} {units}")
    print(f"  {'node':<14} {'type':<8} {'cycles':>7} {'rows':>7}")
    for cost, node in zip(costs, desc.nodes):
        print(f"  {node.name:<14} {node.compute_type:<8} {cost.cycles:>7} {cost.systolic_rows:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
