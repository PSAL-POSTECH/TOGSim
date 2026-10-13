"""Run an ONNX graph on TOGSim: each node's kernel is replayed on its own and the cycles are summed. A node that
does not run is listed by name with the reason and left out of the total, and the exit status then says so. The
statuses are those of status.py."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

import yaml
from google.protobuf.message import DecodeError

from .. import paths
from ..acs import cycles as acs_cycles
from ..acs import node_mapping
from ..status import (EXIT_BAD_INPUT, EXIT_NODES_NOT_RUN, EXIT_OK, EXIT_TOOL_FAILED, EXIT_UNSUPPORTED_MACHINE,
                      fail)
from . import emit, graph, shapes, tiling
from .config import read_hardware
from .onnx_ops import resolve

CYCLE_RE = re.compile(r"Total execution cycles:\s*(\d+)")


class ToolError(RuntimeError):
    """The compiler or the Simulator failed: the run cannot say what the node costs, whatever the node is."""


def simulate(tog_dir: str, config: str) -> int:
    """Replay the kernel laid out in `tog_dir` and return its Total execution cycles; what TOGSim printed is kept
    beside the kernel as togsim.log, byte for byte. A Simulator that is absent, exits non-zero or prints no total
    is a ToolError, whatever it printed."""
    tog_dir = os.path.abspath(tog_dir)
    cmd = [str(paths.simulator()), "--config", config,
           "--tog_so", os.path.join(tog_dir, "tog.so"),
           "--cycle_table", os.path.join(tog_dir, "tog_cycles.tsv"),
           "--unit_table", os.path.join(tog_dir, "tog_units.tsv")]
    try:
        res = subprocess.run(cmd, capture_output=True)
    except OSError as exc:
        raise ToolError(f"the Simulator could not be started: {exc}") from None
    with open(os.path.join(tog_dir, "togsim.log"), "wb") as fh:
        fh.write(res.stdout + res.stderr)
    m = CYCLE_RE.search((res.stdout + res.stderr).decode(errors="replace"))
    if res.returncode != 0 or not m:
        tail = (res.stderr or res.stdout).decode(errors="replace")[-400:]
        raise ToolError(f"TOGSim reported no cycle count (exit {res.returncode}, log "
                        f"{os.path.join(tog_dir, 'togsim.log')}): {tail}")
    return int(m.group(1))


def prepare_node(node, kernel, shp, hw, out_dir) -> str:
    """Build what one kernel run of `node` needs, in a directory named by its kernel and constants: tog.so compiled
    with the node's tile, the two sidecars, and ACS's tables, a row per tile id the kernel names at this shape, a
    short last tile costed at what is left. A tile whose spad buffers TOGSim would refuse (tiling.spad_refusal) or an
    uncovered tile id refuses the node; a failed compile is a ToolError."""
    values = shapes.constants(node, kernel, shp, hw)
    macros = emit.defines(kernel, values)
    shape = emit.shape_args(kernel, values)
    build = "_".join(m.split("=")[1] for m in macros)
    try:
        so = emit.compile_so(str(paths.kernel_source(kernel.stem)), str(paths.include_dir()),
                             os.path.join(out_dir, ".kernels", f"{kernel.stem}_{build}.so" if build else
                                          f"{kernel.stem}.so"), macros)
    except subprocess.CalledProcessError as exc:
        raise ToolError(f"kernel {kernel.stem} did not compile with {macros} (exit {exc.returncode}): "
                        f"{(exc.stderr or '')[-400:]}") from None
    except OSError as exc:
        raise ToolError(f"kernel {kernel.stem} could not be compiled: {exc}") from None

    why = tiling.spad_refusal(kernel.stem, emit.spad_buffer_bytes(so), hw)
    if why:
        raise ValueError(f"node {node.name!r}: {why}")

    tag = "_".join(str(v) for v in shape + [build] if str(v))
    tog_dir = os.path.join(out_dir, f"{kernel.stem}_{tag}" if tag else kernel.stem)
    emit.write_tog_dir(tog_dir, so, shape, shapes.tensor_bytes(kernel, values))

    desc = node_mapping.lookup(kernel.stem)
    tile_ids = node_mapping.tile_ids(desc, values)
    tiles = {tile_id: shapes.tile_for_cost(kernel, node_mapping.at_variant(desc, values, tile_id // len(desc.nodes)))
             for tile_id in tile_ids}
    table = acs_cycles.build_table(desc, tiles, hw)
    units = acs_cycles.build_units(desc, tiles, hw)
    missing = acs_cycles.uncovered(tile_ids, table, units)
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

    try:
        hw = read_hardware(args.config, args.sa)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return fail(EXIT_BAD_INPUT, f"config {args.config} cannot be read: {exc}")
    why = tiling.unsupported(hw)
    if why:
        return fail(EXIT_UNSUPPORTED_MACHINE, "\n  - ".join(
            [f"{args.config} (SA={hw.sa}, vpu_num_lanes={hw.vlane}) is not a machine the frontend maps a model on. "
             f"Its systolic tiles are the same on every machine, and here:", *why]))
    try:
        model, shp = graph.load(args.model, optimize_graph=not args.no_optimize)
    except (OSError, DecodeError) as exc:
        return fail(EXIT_BAD_INPUT, f"model {args.model} cannot be read: {exc}")
    os.makedirs(args.out, exist_ok=True)

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
        except ToolError as exc:
            return fail(EXIT_TOOL_FAILED, f"node {name} ({node.op_type}): {exc}\nThe run stops here: no total is "
                                          f"given and no report is written.")
        except (KeyError, ValueError) as exc:
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
    return EXIT_OK

if __name__ == "__main__":
    raise SystemExit(main())
