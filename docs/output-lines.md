# Output lines

What TOGSim prints, and which of its lines a script may rely on. [running.md](running.md#reading-the-output) explains how to read a result; this page is the
reference.

Everything goes to stdout through spdlog as `[YYYY-MM-DD HH:MM:SS.mmm] [level] text`, with these
exceptions:

- **On stderr, without a prefix:** the TOG loader's refusals (`togsim: refusing TOG ...`,
  `togsim: dlopen failed: ...`, `togsim: dlsym togsim_kernel failed: ...`); the two lines of an
  uncaught exception (`terminate called after throwing ...` and `  what():  <reason>`, exit 134);
  and Ramulator2's `Config file <path> does not exist!` (exit 255).
- **On stdout, as bare lines:** the statistics that Ramulator2, BookSim2 and the cache model print
  themselves.

Capture a log with `2>&1`, or the reason of a refusal is not in it.

## A whole log

The hello TOG on `examples/minimal.yml`, a machine small enough for its log to fit on a page
(`examples/run.sh hello` ends with this run):

```bash
build/bin/Simulator --config examples/minimal.yml --tog_so examples/hello/tog.so \
  --cycle_table examples/hello/tog_cycles.tsv
```

```text
[2026-10-12 13:22:43.516] [info] [TOGSim] Command line: build/bin/Simulator --config examples/minimal.yml --tog_so examples/hello/tog.so --cycle_table examples/hello/tog_cycles.tsv
[2026-10-12 13:22:43.516] [info] [LoadConfig] Loaded configuration file "examples/minimal.yml"
[2026-10-12 13:22:43.516] [info] PyTorchSim config:
num_cores: 1
core_freq_mhz: 1000
core_stats_print_period_cycles: 0
dram_type: simple
dram_channels: 1
dram_latency: 10
dram_stats_print_period_cycles: 0
icnt_type: simple
icnt_freq_mhz: 1000
icnt_latency_cycles: 1
systolic_array_split: none
[2026-10-12 13:22:43.516] [info] [Config/Core] core_id: 0, partition_id: 0 (no partition section)
[2026-10-12 13:22:43.516] [info] [Config/Core] Core 0: core_freq_mhz: 1000, systolic_arrays_per_core: 1
[2026-10-12 13:22:43.516] [info] [Config/DRAM] Total bandwidth 0.00 GB/s, 1000 MHz, 1 channels, 32 bytes per request
[2026-10-12 13:22:43.516] [info] [Config/L2Cache] No L2 cache
[2026-10-12 13:22:43.516] [info] [SimpleDRAM] DRAM latency: 10
[2026-10-12 13:22:43.516] [info] [SimpleDRAM] no bandwidth cap (latency-only); dram_latency 10 cycles, dram_freq_mhz 1000 for tick alignment
[2026-10-12 13:22:43.516] [info] [Config/Interconnect] interconnect_freq_mhz: 1000
[2026-10-12 13:22:43.516] [info] [Config/Interconnect] Simple interconnect selected
[2026-10-12 13:22:43.516] [info] ======Start Simulation=====
[2026-10-12 13:22:43.516] [info] [Scheduler 0] Kernel 0 has completed - TOG path: kernel operation: kernel finished at cycle 1645
[2026-10-12 13:22:43.516] [info] [Scheduler 0] Kernel 0 execution summary - Started at: 1 cycles, Total compute time: 1644 cycles
[2026-10-12 13:22:43.517] [info] [TOGSim] Total cycles: 1659
[2026-10-12 13:22:43.517] [info] Simulation finished
[2026-10-12 13:22:43.517] [info] ===== Instructions count =====
[2026-10-12 13:22:43.517] [info] Core [0] : MOVIN    inst_count: 1
[2026-10-12 13:22:43.517] [info] Core [0] : MOVOUT   inst_count: 1
[2026-10-12 13:22:43.517] [info] Core [0] : COMP     inst_count: 1 (GEMM: 0, Vector: 1, XLU: 0)
[2026-10-12 13:22:43.517] [info] Core [0] : MEMORY_BAR inst_count: 0
[2026-10-12 13:22:43.517] [info] ========= Core stat =========
[2026-10-12 13:22:43.517] [info] Core [0] : Systolic array [0] utilization(%): 0.00, active_cycles: 0, idle_cycles: 1659
[2026-10-12 13:22:43.517] [info] Core [0] : DMA active_cycles: 1024, DMA idle_cycles: 635, DRAM BW: 19.000 GB/s (1024 responses)
[2026-10-12 13:22:43.517] [info] Core [0] : Vector unit utilization(%): 36.23, active cycle: 601, idle_cycle: 0
[2026-10-12 13:22:43.517] [info] Core [0] : Cross-lane unit utilization(%): 0.00, active cycle: 0, idle_cycle: 1659
[2026-10-12 13:22:43.517] [info] Core [0] : NUMA local memory: 1024 requests, remote memory: 0 requests
[2026-10-12 13:22:43.517] [info] Core [0] : Total_cycles: 1659
[2026-10-12 13:22:43.517] [info] Core [0] : computes held at least once by the core's compute in flight: 0
[2026-10-12 13:22:43.517] [info] Total execution cycles: 1659
```

| Lines | What they are |
|---|---|
| `[TOGSim] Command line: ...` | The command, for rerunning it. |
| `[LoadConfig] ...`, `PyTorchSim config:` and the `key: value` lines | The machine config as TOGSim parsed it. |
| `[Config/Core]`, `[Config/DRAM]`, `[Config/L2Cache]`, `[Config/Interconnect]` | One summary per component: each core's partition and arrays, the DRAM clock, channels and request size, the cache, the interconnect. With Ramulator2 the whole Ramulator2 file is echoed after `[Config/DRAM] Ramulator2 configuration :`. |
| `======Start Simulation=====` | Printed by a `--tog_so` run before the first cycle. |
| `[Scheduler p] Kernel k has completed ... finished at cycle N` and the `execution summary` line | Printed by the scheduler of partition `p`. `N` is the cycle the kernel ended: its last work-item had finished and its last write had left the interconnect into a memory partition. Here 1645; the run ends at 1659, once that write is answered. The next kernel of the partition starts after it. |
| `[TOGSim] Total cycles: N` | The result, printed by a `--tog_so` run. |
| `Simulation finished` | The end of the simulation; the final statistics follow. |
| `===== Instructions count =====` and the `inst_count` lines | Per core: the loads (`MOVIN`), stores (`MOVOUT`), computes (`COMP`) and barriers (`MEMORY_BAR`) it ran. A compute of zero cycles is reported as `skipped inst_count`. `GEMM` and `XLU` count computes of the deprecated typed grouping. |
| `========= Core stat =========` and the `Core [i] :` lines | Per core: unit utilization, DMA activity and DRAM bandwidth, DRAM requests by partition, the core's cycles, and how many computes had to wait for the core. |
| `Total execution cycles: N` | The result. The last line of a `--tog_so` run; a `--models_list` run prints one more line after it, `Wall-clock time for simulation: S seconds`. |

This machine has no unit table, so its unit lines are the built-in `Systolic array [k]`,
`Vector unit` and `Cross-lane unit`. With a unit table each unit of the table has its own line
instead, `Core [i] : <unit> utilization(%): P, active_cycles: A, idle_cycles: I`.

## Lines a script may rely on

These are part of the contract (the fourth of the
[five things to depend on](compatibility.md#the-five-things-to-depend-on)); `tests/outputs/` pins
their shape.

- In both modes: `[TOGSim] Command line: ...`, `Simulation finished`, `Total execution cycles: N`.
- Only with `--tog_so`: `[TOGSim] Total cycles: N`.
- Only with `--models_list`: `Wall-clock time for simulation: S seconds`, after the total.
- The headings that open the final statistics: `===== Instructions count =====` and, with
  Ramulator2 only, `=== DRAM statistics ===` and `[DRAM] Per-channel average bandwidth`.
- The per-core statistics, each starting `Core [i] :` &mdash; `<name> inst_count: N`,
  `Systolic array [k] utilization(%)`, `Vector unit utilization(%)`,
  `DMA active_cycles: ..., DMA idle_cycles: ..., DRAM BW: ...`, `Total_cycles: N`, and with a unit
  table one `<unit> utilization(%): P, active_cycles: A, idle_cycles: I` line per unit, plus a
  `Total : <unit> ...` line per unit when there is more than one core.
- `[TOGSim] indirect dmas: X with their index dump, Y without`.
- With Ramulator2 only: `[DRAM] channel N | ...`, `[DRAM] all N channels combined | ...`,
  `[DRAM] channels 0..N combined | ...`.
- `[Config/DRAM] Total bandwidth ...`.
- `[Scheduler p] Kernel k has completed - ... finished at cycle N`.
- The config, echoed once at the start as `key: value` lines.
- With `--log_level trace`: `[cycle][Core c][TAG][INST_ID=n] OPCODE (...)`, where `TAG` is padded
  to 15 characters and is one of the names in `include/TraceLogTags.h`.

Other lines, and the numbers in any line, are not part of the contract.

## Warnings

A `[warning]` line names something that still runs and is probably a mistake. Each is printed once
per TOG, while the TOG is loaded, or per config key, and the run exits 0. They are not part of the contract.

| Line | Meaning |
|---|---|
| `[TOGSim] TOG <so>: the unit table has no rows for tile ids <ids>, which its computes name: those computes add nothing to the units` | Those computes count for no unit's utilization and are not split over the arrays. `<ids>` folds consecutive ids (`1, 3 to 5`). A run without a unit table says nothing. |
| `[TOGSim] TOG <so>: its DMAs name arg_id <ids>, and tog_tensors.txt lists <n> tensor(s): those DMAs are addressed from base 0` | Those DMAs go to address 0 plus their offset, on top of tensor 0. |
| `[TOGSim] config <path>: key <key> is not read by TOGSim` | A key of the config that TOGSim does not read and no [other tool](config-keys.md) owns: a misspelling, most often. |
| `[TOGSim] indirect dma without its index dump (...); modeled as a dense tile` | An indirect DMA whose index file is absent or of another size. |
| `[TOGSim] <so> uses the typed grouping (compute_type 1, 2 or 3), which is deprecated: ...` | See [compatibility.md](compatibility.md#deprecated-compute_type-1-2-and-3). |

`examples/run.sh mistakes` shows the first two.

An `[info]` line `[TOGSim] TOG <so>: core <c> holds <H> bytes for the kernel: buffers <ids>, ...`
names the [held buffers](tog-abi.md#data-that-is-already-on-the-chip) of each core.

## Periodic and final statistics

| Line | When | Meaning |
|---|---|---|
| `========= Core stat =========` block with `Total_cycles: N` | Every `core_stats_print_period_cycles` core cycles | The interval ending at cycle `N`: unit utilization, DMA cycles and DRAM bandwidth of that interval only. |
| `[DRAM] all N channels combined \| B GB/s aggregate, U% of utilization (avg. per channel) \| R reads, W writes (interval P cycles)` | With Ramulator2, every `dram_stats_print_period_cycles` | The DRAM traffic of that interval. |
| `=== DRAM statistics ===`, then one `--- channel N ---` dump per channel | With Ramulator2, at the end | Ramulator2's own counters, 58 lines per channel on tpuv6e. No key turns the dump off. |
| `[DRAM] channel N \| B GB/s avg., U% of utilization \| R reads, W writes` | With Ramulator2, at the end | The average bandwidth of each channel over the run. |
| `[DRAM] channels 0..N combined \| ...` | With Ramulator2, at the end | The sum over the channels. |

Reads and writes are counts of DRAM requests. With `dram_type: simple` none of the `[DRAM]`
statistics lines is printed. The end of the hello run on tpuv6e (32 channels, requests of 32
bytes); `examples/run.sh hello` prints these three lines above the final statistics:

```text
[info] [DRAM] Per-channel average bandwidth
[info] [DRAM] channel 0 | 1.35 GB/s avg., 2.63% of utilization | 16 reads, 16 writes
...
[info] [DRAM] channels 0..31 combined | 43.09 GB/s aggregate, 2.63% of utilization (avg. per channel) | 512 reads, 512 writes
```

**Why a log is long.** The log of that run is 2,102 lines. 1,857 of them are the
`=== DRAM statistics ===` dump (58 lines for each of the 32 channels and the heading), and 84 are
the periodic statistics of a run of 2437 cycles on a config that prints every 200. On a long run
the periodic blocks dominate, and `core_stats_print_period_cycles: 0` and
`dram_stats_print_period_cycles: 0` remove them; nothing removes the dump, so read a log through
`grep` or `tail`.

Some lines are spelled differently in the two places they appear: the final
`Vector unit utilization(%)` line says `active cycle` and `idle_cycle` where the periodic one says
`Vector unit Utilization(%)`, `active_cycles` and `idle_cycles`.

## Trace lines

`--log_level trace` adds one line per event, `[cycle][Core c][TAG][INST_ID=n] OPCODE (fields)`.
`TILE_SCHEDULED`, `SPAD_HOLD`, `SPAD_HOLD_END`, `REQ_TO_MEM`, `REQ_TO_DRAM` and `DRAM_WRITE_ACK` lines
have no `INST_ID`.

| Tag | Printed when | Fields worth reading |
|---|---|---|
| `TILE_SCHEDULED` | A work-item is handed to a core. | `spad_footprint` in bytes; `max_dispatch`, the work-items that core keeps in flight (1 or 2). |
| `SPAD_HOLD`, `SPAD_HOLD_END` | A core takes, and gives back, the bytes it holds for a kernel. | `bytes`. |
| `INST_ISSUED` | An instruction starts. | For a DMA: `dram` (its address), `size`, `stride`, `elem_bits`, `async`, `indirect`. For `COMP`: `compute_cycle` (after any array split), `tile` (the work-item index, not the `tile_id`). |
| `INST_FINISHED` | An instruction is done. | |
| `ASYNC_DMA_ISSUE` | An asynchronous load has sent all its requests. | |
| `DRAM_RESP_FIRST` | The first DRAM response of a DMA arrives. | |
| `DRAM_RESP_DONE` | An asynchronous load completes: all its data has arrived, and `dma_load_ready_cycles` have passed since. | |
| `DMA_LATENCY` | The DMA engine takes a DMA and holds it `cycles` before its first request (`dma_load_latency_cycles` or `dma_store_latency_cycles`, when it is not 0). | `cycles`. |
| `DRAM_NUMA` | A DMA is issued. | `numa_id`, `is_write`. |
| `L2CACHE_STAT` | A DMA is issued on a machine with the L2 data cache. | `cacheable`. |
| `REQ_TO_MEM` | One request of a DMA leaves the interconnect into the memory partition of its channel. | `MOVIN` or `MOVOUT`, `dram` (the request's address), `channel`. |
| `REQ_TO_DRAM`, `DRAM_WRITE_ACK` | One request of a DMA enters the DRAM model of its channel; the ack of a write leaves it. | `MOVIN` or `MOVOUT`, `dram` (the address the DRAM model sees: within the channel's partition on Ramulator2, the request's address on the simple DRAM), `channel`. |

`INST_ID` numbers instructions in the order TOGSim created them, over the whole run. The opcodes
are `MOVIN` (load), `MOVOUT` (store), `COMP` and `MEMORY_BAR`.

| Field | On | Meaning |
|---|---|---|
| `tile=N` | every instruction | The work-item the instruction belongs to, counted from 0 in dispatch order within its kernel. It is not the `tile_id`. |
| `addr_name=tagN` | DMAs, barriers | The DMA is the `N`th of its work-item, from 0. A barrier carries the name of the first load it waits for. |
| `dram`, `size`, `stride`, `elem_bits`, `rank` | DMAs | The byte address in DRAM, `dims` and `strides` as the TOG passed them, the element width, and the number of dimensions. |
| `async`, `indirect` | DMAs | The `is_async` and `indirect` arguments. |
| `compute_cycle` | `COMP` | The cycles the compute holds its core, after any array split. |
| `overlapping_cycle` | `COMP` | The second column of the cycle table. Deprecated: it is printed and changes nothing. |
| `compute_type`, `sa` | `COMP` | `compute_type` as the TOG passed it. `sa` is `-1` for `compute_type` 0. |
| `tag_id`, `tag_idx`, `tag_stride`, `tag`, `subgraph_id` | DMAs, barriers | Bookkeeping that pairs an asynchronous load with its barrier. Not part of the contract. |

## Exit status

| Status | When |
|---|---|
| 0 | The simulation finished (also `--help`). |
| 1 | A refused TOG (another ABI, no spad buffer sizes, a broken semaphore contract, a `tile_id` the cycle table does not hold, a compute tile whose unit table rows exceed what a unit takes in its cycles, a work-item that needs more of the spad at once than a core has), no cycle table or one that cannot be opened, a failed check, a rejected unit table, a command-line error, no `--config`, a config file that cannot be opened or is not YAML, neither `--tog_so` nor `--models_list`, a `LAUNCH_KERNEL` that fails, a simulation that is stuck for want of spad in a way the loader did not see (`[Simulator] simulation wedged at cycle N: ...`). |
| 134 | An exception nothing catches ends in `abort`: a config without a required key or with a value TOGSim does not implement, a BookSim2 config that cannot be opened. The reason is on stderr, as `what():  <reason>`. |
| 136 | `--models_list` names a file that cannot be opened: the log says `[TOGSim] Failed to open trace file: <path>` and the process dies while printing the statistics of a run of zero cycles. |
| 255 | `ramulator_config_path` names a file that does not exist; Ramulator2 prints `Config file ... does not exist!` on stderr. |
