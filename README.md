# TOGSim

TOGSim is a cycle-level simulator for a tile-operation-graph (TOG). A compiler lowers a kernel to
a TOG, a shared object whose calls are the kernel's DMAs, tile computes and barriers inside its
counted loops. TOGSim loads the TOG and replays it on a modeled accelerator (cores with systolic
arrays, a vector unit and a cross-lane unit; DRAM through Ramulator2 or a simple model; an
interconnect through BookSim2 or a fixed latency) and reports how many cycles the kernel took. The
TOG carries no timing model: what a tile compute costs comes from a cycle table measured elsewhere.
Inside the simulator a `TileGraph` is a TOG instantiated for one shape and one partition.

This repository was split out of
[PSAL-POSTECH/PyTorchSim](https://github.com/PSAL-POSTECH/PyTorchSim) at commit `7b61c1d2` of its
`pytorchsim-triton-backend` branch. The history of its `TOGSim/` directory is kept here; `configs/`
was imported as one snapshot of the same commit. The one change PyTorchSim made to `TOGSim/` after
that (`13540211`, in its tree up to `3e42fdc8`) is applied here as well.

## Layout

| Path | What it holds |
|---|---|
| `include/togsim_runtime.h` | The TOG C ABI. The only header a TOG includes. |
| `include/`, `src/` | The simulator. |
| `configs/` | Machine configs (`*.yml`) and the Ramulator2 / BookSim2 files they point at. |
| `extern/booksim`, `extern/ramulator2` | Submodules, built with the simulator. |
| `tests/` | Hand-written TOGs, the cases that replay them and their runner, `tests/run.py`. |

## Build

Requirements: a C++23 compiler (GCC 11 is what the tests ran on), `conan` 1.x and `cmake` >= 3.16.
The versions this tree was built and tested with are `conan==1.56.0` and `cmake==3.26.4`. The
configure step fetches yaml-cpp and fmt for Ramulator2, so it needs network access; the simulator
uses that same yaml-cpp, and conan supplies boost and spdlog.

`CMakeLists.txt` compiles with `_GLIBCXX_USE_CXX11_ABI=0`, so the conan profile must say
`compiler.libcxx=libstdc++` (conan 1.56's auto-detected default profile does; `conan profile
update settings.compiler.libcxx=libstdc++ default` sets it otherwise).

```bash
git submodule update --init --recursive
mkdir -p build && cd build
conan install .. --build=missing
cmake ..
make -j"$(nproc)"
```

The binary is `build/bin/Simulator`. It finds `build/lib/libramulator.so` by an absolute rpath, so
the build tree is used where it was built.

A consumer names this repository with one environment variable, `TOGSIM_DIR`, set to its root:
the header is `$TOGSIM_DIR/include/togsim_runtime.h` and the binary
`$TOGSIM_DIR/build/bin/Simulator`.

## Run

```bash
build/bin/Simulator \
  --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml \
  --tog_so path/to/tog.so \
  --cycle_table path/to/tog_cycles.tsv \
  --unit_table path/to/tog_units.tsv
```

`systolic_array_split` is `round_robin` unless the config says `none`, and it reads the unit table:
a `--tog_so` run without `--unit_table` is refused under the default. `--cycle_table` is required:
there is no default cycle table.

## Test

```bash
tests/run.py                      # every case, against build/
tests/run.py -k spad_footprint    # the cases whose id contains a substring
tests/run.py --list               # the case ids
```

The runner prints one line per case, `PASS <directory>/<case>` or `FAIL <directory>/<case>: `
followed by what was expected, what was seen and where the log is, then `ALL PASS` or the ids that
failed. It exits `0`, `1` if a case failed, or `2` if it could not run (no build). Each case runs
in a directory of its own, which holds the TOG, the machine config and the tables it was given,
`sim.log` and `sim.exit`; a run with a failure keeps them and says where, and `--workdir <dir>`
keeps them always.

| Directory | What its cases pin |
|---|---|
| `tests/unit/` | `TensorTiling` and the DRAM addresses of a load, without the simulator (`build/bin/tensor_tiling_test`). |
| `tests/abi/` | A TOG that states another ABI version, only the symbol of ABI 15, or none, is refused with the message that names both versions and exit `1`; the same TOG at this ABI replays. |
| `tests/array_split/` | `systolic_array_split` with a segment TOG: a compute's Systolic rows are spread over its core's arrays, and what the split refuses: a unit table with no array unit or none at all (exit `1`), `systolic_array_unit` named with the split off (the config is rejected, exit `134`), a spread over a unit's capacity. |
| `tests/configs/` | Every `configs/*.yml`, read where it is, replays a small TOG to the end; each is under the default `systolic_array_split`, so each is also refused without a unit table. |
| `tests/cycle_table/` | A TOG is replayed against a cycle table: a run without one, or with one that cannot be opened, is refused before anything is simulated (exit `1`), with `--tog_so` and with `--models_list`; a command stream that launches nothing needs none. A compute names a tile id the table holds: one at the table's length, a negative one or one past 32 bits is refused with the message that names the TOG, the tile id and how many the table has; a table one line longer runs it; a table without a line holds no tile id. |
| `tests/cycles/` | `Total execution cycles` of four of the hand-written TOGs on tpuv6e, as goldens: a change to the timing model, to that config or to a submodule moves them, and has to say so. |
| `tests/outputs/` | The [output lines](#4-output-lines): the shapes of the lines a `--tog_so` run logs and of its config echo (no timestamp; paths, hex values and numbers replaced by placeholders), each shape once, against a `*.golden` file. Not the values, nor how often a line is printed. |
| `tests/semaphore/` | A barrier waits every async load signaled on its semaphore; a TOG that breaks the contract is refused. |
| `tests/session/` | `--models_list`: a two-kernel command stream on one partition and on two, each kernel with its own `tog_cycles.tsv` and `tog_units.tsv`, and what the mode refuses. |
| `tests/spad_footprint/` | A scratchpad buffer is as large as the TOG states it; one the TOG does not size is refused. |
| `tests/warnings/` | What still runs and is said once at warn level, in one line per TOG that lists the ids, with `--tog_so` and with `--models_list`: a compute whose tile id the unit table has no rows for (nothing is said when every tile id has rows, or when the run has no unit table), and a DMA whose `arg_id` is negative or beyond the tensors of `tog_tensors.txt` (its address stays where it was, from base 0; nothing is said when every `arg_id` is listed). |
| `tests/typed/` | **Deprecated**, deleted with the [typed grouping](#deprecated-typed-grouping): every case whose TOG uses `compute_type` `1`, `2` or `3`, and no other: every run in it loads a typed TOG. What they pin is not typed-only: a semaphore re-wait, one compute at a time per core, `systolic_array_split` of a matmul. |

A case is a function in its directory's `cases.py`, beside the hand-written TOGs (`*_tog.cpp`) it
replays. `tests/lib/togtest.py` is what the cases share: it compiles a TOG against
`include/togsim_runtime.h` with `g++`, writes the machine config (one of `configs/` with the keys
the case sets) and the tables, runs the simulator and reads its log. The tests need `g++`,
`python3` (standard library only) and a build of the default target made in this tree.

## The TOG ABI

What a compiler, a driver or a script on the other side of this repository may depend on, in five
layers. Anything not named here (internal headers, the timing model, the cycle counts themselves,
other log lines, how the build is done) is not part of it.

### 1. TOG C ABI

`include/togsim_runtime.h`. The header is C++ (`<cstdint>`) with `extern "C"` linkage.

- `TOGSIM_ABI_VERSION` is `16`. A TOG that includes the header defines the weak symbol
  `togsim_abi_version` with that value. TOGSim opens the TOG with
  `dlopen(RTLD_NOW | RTLD_GLOBAL)` and refuses one that lacks the symbol or carries another value.
  16 is a rename, not a new call: up to 15 the symbol was `togsim_producer_abi_version`, the flag
  `--trace_so`, and the files and the config key below began with `trace` where they now begin
  with `tog`. A TOG built against 15 is refused with a message that names both versions.
- The header brackets the call formats a compiler copies into what it generates with two marker
  lines, `// --- BEGIN TOG call formats (copied verbatim into generated tog.cpp) ---` and
  `// --- END TOG call formats ---`.
- The TOG defines `togsim_kernel(ctx, shape_args, n_shape_args)`, its entry point, and
  `togsim_spad_buffer_count` / `togsim_spad_buffer_bytes[]`, the byte size of every scratchpad
  buffer id it names. A TOG without them is refused.
- TOGSim exports the four calls the TOG makes: `togsim_dma`, `togsim_compute`,
  `togsim_memory_barrier`, `togsim_dispatch`.
- Values: DMA direction `0` load / `1` store; `compute_type` `0` vector, and, deprecated, `1`
  matmul, `2` preload, `3` cross-lane (see [Deprecated: typed grouping](#deprecated-typed-grouping)).
- A TOG that breaks the semaphore contract (a wait whose element count differs from what was
  signaled, a wait on a semaphore never signaled, a load batch never waited) is refused with a
  `[TOGSim]` error line and a non-zero exit.
- A TOG whose `togsim_compute` names a `tile_id` the cycle table does not hold is refused while it
  is loaded, before anything is simulated, with exit `1` and the line `[TOGSim] TOG <so>: work-item
  <i> (key <k>) computes tile id <id>, which the cycle table does not hold (it has <n> tile ids, 0
  to <n-1>)`, or `(it has no line)` against a table without one. `tile_id` is a `uint64_t` and is compared whole: a negative value arrives as a number
  near 2^64 and is refused like any other. Up to `v1.2.0` such a compute was charged 0 cycles.
- A TOG is replayed against a cycle table. A run that gives none, or one that cannot be opened, is
  refused before anything is simulated, with exit `1` and the line `[TOGSim] --tog_so needs
  --cycle_table: ...` or `[TOGSim] cycle table <path> cannot be opened: ...`. Up to `v1.2.0` such
  a run replayed against a flat stub (128 cycles for tile ids 0 to 255) without a word.

The comments in the header are the reference for each argument.

### 2. Input sidecar files

Read from the directory of the TOG, by fixed name:

| File | Required | Contents |
|---|---|---|
| `tog_tensors.txt` | yes | One byte size per tensor argument, in `arg_id` order. Tensors are placed back to back in DRAM, each start aligned to 16 KiB. A `togsim_dma` whose `arg_id` is negative or beyond the tensors listed still runs, addressed from base 0, and the run says so once per TOG at warn level while the TOG is loaded: `[TOGSim] TOG <so>: its DMAs name arg_id <ids>, and tog_tensors.txt lists <n> tensor(s): those DMAs are addressed from base 0`. |
| `tog_shape.txt` | no | The kernel's shape arguments, whitespace-separated. Absent: the TOG carries its own constants. |
| `tog_tensor_shapes.txt` | when the config sets `tog_tensor_tile` | `rows cols elem_bytes` per tensor argument. |
| `runtime/indirect_access/indirect_index_<key>_<seq>.raw` | no | The indices of an indirect DMA, one `uint64` per element; `<key>` is the work-item's `togsim_dispatch` key, `<seq>` the DMA's rank among that work-item's indirect DMAs. Absent: the DMA is modeled as a dense tile, with a warning. |

Named on the command line:

- Cycle table (`--cycle_table`): one `cycle overlapping` pair per line; the line number, from 0, is
  the `tile_id` a `togsim_compute` names, so a table of `n` lines holds tile ids 0 to `n-1` and a
  TOG that names any other is refused (layer 1). Required with `--tog_so`: a run without it, or
  with a path that cannot be opened, is refused. A table without a line opens and holds no tile
  id: a TOG with no compute replays against it, and any compute is refused. Reading stops at the
  first token that is not a number, so what follows it is not part of the table.
- Unit table (`--unit_table`): five tab-separated fields per line,
  `tile_id unit unit_of_work capacity_per_cycle admitted`. Its rows are read for the tile ids the
  computes name; rows of a tile id the cycle table does not hold are reached by no compute and are
  not an error. A compute whose tile id has no rows still runs: it adds nothing to the units and
  `systolic_array_split` leaves it whole. The run says so once per TOG, at warn level, while the
  TOG is loaded: `[TOGSim] TOG <so>: the unit table has no rows for tile ids <ids>, which its
  computes name: those computes add nothing to the units`, `<ids>` with consecutive ids folded
  (`1, 3 to 5`). A run without a unit table says nothing.

Under `--models_list`, a kernel's TOG, cycle table and unit table are `tog.so`,
`tog_cycles.tsv` and `tog_units.tsv` in the directory of the path its command names. The unit
tables of the launched kernels join one table: a kernel's `tile_id`s are shifted behind the rows
already held, and a unit the first kernel's table does not name is an error. `tog.so` is
required; `tog_units.tsv` is required unless the config sets `systolic_array_split: none`;
`tog_cycles.tsv` is the kernel's cycle table under the same rule as `--cycle_table`, and is
required as well: a launch whose directory has none, or one that cannot be opened, is refused. A
command stream that launches no kernel loads no TOG and needs no table.

### 3. Command line

`build/bin/Simulator`, with:

| Flag | Meaning |
|---|---|
| `--config <yml>` | Machine config. Required. |
| `--tog_so <so>` | Replay one TOG as kernel 0 on partition 0. |
| `--cycle_table <tsv>` | Cycle table for `--tog_so`; required. A TOG that names a tile id beyond it is refused. |
| `--unit_table <tsv>` | Unit table for `--tog_so`; required unless the config sets `systolic_array_split: none`. Refused with `--models_list`, where each kernel brings its own `tog_units.tsv`. |
| `--models_list <file>` | A command stream (a file or a FIFO) instead of one TOG. |
| `--log_level <trace\|debug\|info>` | Default `info`. |
| `--help` | Print the options. |

A command stream line has seven comma-separated fields,
`command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp`, with
`command_type` one of `LAUNCH_KERNEL` or `DEVICE_SYNC`; lines starting with `#` are skipped. The
simulation ends when the stream reaches end of file.

Exit status:

| Status | When |
|---|---|
| `0` | The simulation finished (and `--help`). |
| `1` | A refused TOG (another ABI, no spad buffer sizes, a broken semaphore contract, a tile id the cycle table does not hold), no cycle table or one that cannot be opened, a failed check, a rejected unit table, a command-line error, no `--config`, a config file that cannot be opened or is not YAML, neither `--tog_so` nor `--models_list`. |
| `134` | An exception nothing catches ends in `abort`: a config without a required key or with a value TOGSim does not implement, a BookSim2 config that cannot be opened. |
| `255` | `ramulator_config_path` names a file that does not exist; Ramulator2 prints `Config file ... does not exist!` and exits `-1`. |

### 4. Output lines

Written to stdout through spdlog, each after a `[timestamp] [level]` prefix. The exception is the
TOG loader: its refusals (`togsim: refusing TOG ...`, `togsim: dlopen failed: ...`,
`togsim: dlsym togsim_kernel failed: ...`) go to stderr without a prefix. The lines consumers read
as of the split:

- In both modes: `[TOGSim] Command line: ...`, `Simulation finished`, `Total execution cycles: N`.
- Only with `--tog_so`: `[TOGSim] Total cycles: N`.
- Only with `--models_list`: `Wall-clock time for simulation: S seconds`.
- The three headings that open the final statistics, which PyTorchSim's `scripts/util_viewer.py`
  keys on: `=== DRAM statistics ===`, `[DRAM] Per-channel average bandwidth`,
  `===== Instructions count =====`.
- The per-core statistics, each starting `Core [i] :` &mdash; `Systolic array [k] utilization(%)`,
  `Vector unit utilization(%)`, `DMA active_cycles: ..., DMA idle_cycles: ..., DRAM BW: ...`,
  `Total_cycles: N`, `<name> inst_count: N`, and with `--unit_table` one
  `<unit> utilization(%): P, active_cycles: A, idle_cycles: I` line per unit (plus a `Total :` line).
- `[TOGSim] indirect dmas: X with their index dump, Y without`.
- `[DRAM] channel N | ...`, `[DRAM] all N channels combined | ...`,
  `[DRAM] channels 0..N combined | ...`, `[Config/DRAM] Total bandwidth ...`.
- `[Scheduler p] Kernel k has completed - ... finished at cycle N`.
- The config, echoed once at start as `key: value` lines.
- With `--log_level trace`: `[cycle][Core c][TAG][INST_ID=n] OPCODE (...)`, where `TAG` is padded
  to 15 characters and is one of the names in `include/TraceLogTags.h`.

### 5. Machine config keys

A config is a YAML map. The keys TOGSim reads:

| Group | Keys |
|---|---|
| Core | `num_cores`\*, `core_freq_mhz`\*, `core_stats_print_period_cycles`\*, `core_type`, `num_systolic_array_per_core`, `core_spad_size_kb`, `sa_weight_buffer_depth`, `systolic_array_split` (`round_robin`, the default \| `none`), `systolic_array_unit` (only with `round_robin`), `vpu_num_lanes` (required with `round_robin`) |
| DRAM | `dram_type`\* (`ramulator2` \| `simple`), `dram_channels`\*, `ramulator_config_path` (with `ramulator2`), `dram_freq_mhz`, `dram_stats_print_period_cycles`, `dram_num_partitions`; with `simple`: `dram_latency`\*, `dram_req_size_byte`, `dram_bandwidth_gbps_per_channel` or `dram_bandwidth_gbps_total` |
| L2 cache | `l2d_type` (`nocache` \| `datacache`), `l2d_config`, `l2d_hit_latency` |
| Interconnect | `icnt_type`\* (`simple` \| `booksim2`), `icnt_freq_mhz`\*, `icnt_latency_cycles`, `booksim_config_path` (with `booksim2`), `icnt_stats_print_period_cycles`, `icnt_injection_ports_per_core` |
| Scheduling | `num_partition`, `partition` (`core_<i>: <partition id>`) |
| TOG | `tog_tensor_tile` (`[rows, cols]`) |

\* required.

`ramulator_config_path` and `booksim_config_path` are resolved against the directory of the config
file unless absolute. The configs here write them as `../configs/...`, which holds as long as the
directory they sit in is named `configs`.

A key TOGSim does not read is ignored. The configs carry several for the other tools that read the
same file (`vpu_vector_length_bits`, `vpu_spad_size_kb_per_lane`, `tpu_*`, `pytorchsim_*`,
`codegen_*`); their meaning is not defined here.

## Deprecated: typed grouping

A compiler can group a kernel's compute into tiles in two ways, and the TOG says which by the
`compute_type` of its `togsim_compute` calls.

- **Segment grouping** emits `compute_type` `0` alone: a tile is whatever runs between two DMAs,
  waits or replayed loops, and it holds its core for the cycles the cycle table gives it. The
  compiler serializes a work-item's tiles through a buffer they all read and write. This is what
  replaces the typed grouping.
- **Typed grouping** classifies each tile by unit and opcode: `1` matmul and `2` preload go to a
  systolic array, `3` cross-lane to the cross-lane unit. For these TOGSim keeps one pipeline per
  systolic array and one for the cross-lane unit, a weight-slot throttle (`sa_weight_buffer_depth`
  weights per array; a preload stalls until one is free and pins its matmuls to its array),
  occupancy edges (a matmul is released when the matmul or preload it reads from is issued, not
  when it finishes) and commuting accumulator writes.

The typed grouping is **deprecated**. A typed TOG still replays, with the cycles it had, so the two
groupings can be compared; it will be removed. What is deprecated, exactly:

- `compute_type` `1`, `2` and `3` of `togsim_compute` (layer 1).
- The config key `sa_weight_buffer_depth` (layer 5), which only they read.
- `include/TypedGrouping.h` and `src/TypedGrouping.cc`, where everything only they reach lives.
- `tests/typed/`, the cases whose TOGs use them.

A run that loads a typed TOG logs one line at warn level, once, while the TOG is loaded and before
the simulation starts: `[TOGSim] <so> uses the typed grouping (compute_type 1, 2 or 3), which is
deprecated: ...`. No line listed under [Output lines](#4-output-lines) changes.

Not deprecated, because a segment TOG uses them as well: the unit table, `systolic_array_split`,
`systolic_array_unit`, `vpu_num_lanes` and `num_systolic_array_per_core` (the split spreads a
tile's array-unit rows over that many arrays whatever its `compute_type`), and one compute at a
time per core. The `Systolic array [k] utilization(%)` and `Cross-lane unit utilization(%)` lines
and the `GEMM`/`XLU` counts of `COMP inst_count` are printed for every run and read zero for a
segment TOG; they leave with the typed grouping.

## Versions

Tags are `vMAJOR.MINOR.PATCH`, continuing PyTorchSim's numbering (its last tag is `v1.1.0`; the first
here is `v1.2.0`, which carries TOG ABI 16).

- **MAJOR**: the TOG C ABI or a sidecar file changes incompatibly (layers 1 and 2).
  `TOGSIM_ABI_VERSION` moves with it, and a TOG built against the old header is refused.
- **MINOR**: the command line, an output line listed above, or a config key changes (layers 3, 4
  and 5).
- **PATCH**: no contract change.

Consumers pin this repository as a git submodule by commit and build it from source.

## License

MIT. See [LICENSE](LICENSE). The submodules carry their own terms: `extern/ramulator2` in its
`LICENSE` file, `extern/booksim` in the header of each source file.
