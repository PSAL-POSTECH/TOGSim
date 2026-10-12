"""Run an ONNX graph on TOGSim: each node's kernel is replayed on its own and the cycles are summed. A node that
does not run is listed by name with the reason and left out of the total, and the exit status then says so."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

from .. import paths
from ..acs import cycles as acs_cycles
from ..acs import node_mapping
from . import emit, graph, shapes, tiling
from .config import read_hardware
from .onnx_ops import resolve

CYCLE_RE = re.compile(r"Total execution cycles:\s*(\d+)")
#: Exit status when the config is a machine the tiles do not fit: nothing was run.
EXIT_UNSUPPORTED_MACHINE = 2
#: Exit status when a node did not run: the report is written, and its total does not cover the whole model.
EXIT_NODES_NOT_RUN = 3


def simulate(tog_dir: str, config: str) -> int:
    """Replay the kernel laid out in `tog_dir` and return its Total execution cycles."""
    tog_dir = os.path.abspath(tog_dir)
    cmd = [str(paths.simulator()), "--config", config,
           "--tog_so", os.path.join(tog_dir, "tog.so"),
           "--cycle_table", os.path.join(tog_dir, "tog_cycles.tsv"),
           "--unit_table", os.path.join(tog_dir, "tog_units.tsv")]
    res = subprocess.run(cmd, capture_output=True, text=True)
    m = CYCLE_RE.search(res.stdout + res.stderr)
    if res.returncode != 0 or not m:
        tail = (res.stderr or res.stdout)[-400:]
        raise RuntimeError(f"TOGSim reported no cycle count (exit {res.returncode}): {tail}")
    return int(m.group(1))


def prepare_node(node, kernel, shp, hw, out_dir) -> str:
    """Build what one kernel run of `node` needs, in a directory named by the kernel and its constants: tog.so
    compiled with the node's tile, the two sidecars, and the tables ACS writes. A tile id of the kernel that the
    tables do not cover refuses the node here, before TOGSim would."""
    values = shapes.constants(node, kernel, shp, hw)
    macros = emit.defines(kernel, values)
    shape = emit.shape_args(kernel, values)
    build = "_".join(m.split("=")[1] for m in macros)
    so = emit.compile_so(str(paths.kernel_source(kernel.stem)), str(paths.include_dir()),
                         os.path.join(out_dir, ".kernels", f"{kernel.stem}_{build}.so" if build else
                                      f"{kernel.stem}.so"), macros)

    tag = "_".join(str(v) for v in shape + [build] if str(v))
    tog_dir = os.path.join(out_dir, f"{kernel.stem}_{tag}" if tag else kernel.stem)
    emit.write_tog_dir(tog_dir, so, shape, shapes.tensor_bytes(kernel, values))

    desc = node_mapping.lookup(kernel.stem)
    tile = shapes.tile_for_cost(kernel, values)
    table = acs_cycles.build_table(desc, tile, hw)
    units = acs_cycles.build_units(desc, tile, hw)
    missing = acs_cycles.uncovered(range(len(desc.nodes)), table, units)
    if missing:
        raise ValueError(f"node {node.name!r}: kernel {kernel.stem} names tile ids {missing} and the tables ACS "
                         f"wrote have {len(table)} cycle lines and unit rows for tile ids "
                         f"{sorted({row[0] for row in units})}")
    acs_cycles.write_table(table, os.path.join(tog_dir, "tog_cycles.tsv"))
    acs_cycles.write_units(units, os.path.join(tog_dir, "tog_units.tsv"))
    return tog_dir


def run_node(node, kernel, shp, hw, config, out_dir, done: dict) -> int:
    """The cycles of one kernel run of `node`; a run whose directory was already replayed is not replayed again."""
    tog_dir = prepare_node(node, kernel, shp, hw, out_dir)
    if tog_dir not in done:
        done[tog_dir] = simulate(tog_dir, config)
    return done[tog_dir]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run an ONNX graph on TOGSim from the kernel library.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="togsim_onnx_result",
                    help="directory for the per-node runs and report.json (default: %(default)s)")
    ap.add_argument("--sa", type=int, default=None,
                    help="systolic array dimension; inferred from the config name")
    ap.add_argument("--no-optimize", action="store_true",
                    help="skip the ORT fusion pass")
    args = ap.parse_args(argv)

    hw = read_hardware(args.config, args.sa)
    why = tiling.unsupported(hw)
    if why:
        print(f"error: {args.config} (SA={hw.sa}, vpu_num_lanes={hw.vlane}) is not a machine the frontend maps a "
              f"model on. Its systolic tiles are the same on every machine, and here:", file=sys.stderr)
        for line in why:
            print(f"  - {line}", file=sys.stderr)
        return EXIT_UNSUPPORTED_MACHINE
    os.makedirs(args.out, exist_ok=True)
    model, shp = graph.load(args.model, optimize_graph=not args.no_optimize)

    print(f"config   : {os.path.basename(args.config)}")
    print(f"hardware : SA={hw.sa} lanes={hw.vlane} vlen={hw.vlen_bits}b "
          f"({hw.throughput} elem/cycle)")
    print()

    total = 0
    done: dict[str, int] = {}
    ran = no_compute = 0
    unmapped: dict[str, int] = {}
    refused = []
    rows = []

    for index, node in enumerate(model.graph.node):
        name = node.name or f"#{index}"
        kernels = resolve(node.op_type)
        if kernels is None:
            unmapped[node.op_type] = unmapped.get(node.op_type, 0) + 1
            refused.append({"name": name, "op": node.op_type, "reason": "no kernel in kernels/ runs this operator"})
            continue
        if not kernels:
            no_compute += 1
            continue
        node_cycles = 0
        try:
            for kernel in kernels:
                node_cycles += run_node(node, kernel, shp, hw, args.config,
                                        args.out, done)
        except (KeyError, ValueError, RuntimeError,
                subprocess.CalledProcessError) as exc:
            reason = f"{node.op_type} ({type(exc).__name__})"
            unmapped[reason] = unmapped.get(reason, 0) + 1
            refused.append({"name": name, "op": node.op_type, "reason": str(exc)})
            continue

        total += node_cycles
        ran += 1
        rows.append({"op": node.op_type, "name": name, "cycles": node_cycles})
        print(f"  {node.op_type:<20} {node_cycles:>12,} cycles")

    print()
    print(f"nodes run      : {ran}")
    print(f"nodes no-op    : {no_compute}")
    print(f"total cycles   : {total:,}")
    if refused:
        print(f"\nNOT RUN ({len(refused)} of {ran + len(refused)} nodes with work, not counted in the total):")
        for entry in refused:
            print(f"  {entry['name']} ({entry['op']}): {entry['reason']}")

    report = {"model": args.model, "config": args.config,
              "hardware": {"sa": hw.sa, "vlane": hw.vlane,
                           "vlen_bits": hw.vlen_bits},
              "total_cycles": total, "nodes_run": ran,
              "nodes_no_compute": no_compute, "unmapped": unmapped, "refused": refused,
              "nodes": rows}
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nreport         : {os.path.join(args.out, 'report.json')}")
    if refused:
        print(f"INCOMPLETE     : {len(refused)} node(s) did not run, so the total does not cover the whole model "
              f"(exit status {EXIT_NODES_NOT_RUN})")
        return EXIT_NODES_NOT_RUN
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
