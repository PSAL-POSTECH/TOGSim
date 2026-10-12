# Running TOGSim and reading its output

## Running TOGSim

```text
build/bin/Simulator --config <yml> --tog_so <so> --cycle_table <tsv> [--unit_table <tsv>] [--log_level <level>]
build/bin/Simulator --config <yml> --models_list <file> [--log_level <level>]
```

| Flag | Meaning |
|---|---|
| `--config <yml>` | The machine config. Required. |
| `--tog_so <so>` | Replay one TOG, as kernel 0 on partition 0. The path needs a `/` in it. |
| `--cycle_table <tsv>` | The cycle table of that TOG. Required. |
| `--unit_table <tsv>` | The unit table of that TOG. Required unless the config sets `systolic_array_split: none`. |
| `--models_list <file>` | Replay a command stream (a file or a FIFO) instead of one TOG. |
| `--log_level <level>` | `info` (the default), `debug` or `trace`. Any other value is treated as `info`. |
| `--help`, `-h` | Print the options. |

If both `--tog_so` and `--models_list` are given, the TOG is replayed and the stream ignored.

**The command stream.** One command per line, seven comma-separated fields:
`command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp`. Empty lines
and lines that start with `#` are skipped.

| Field | `LAUNCH_KERNEL` | `DEVICE_SYNC` |
|---|---|---|
| `kernel_id` | A number that labels the kernel in the log. | any number |
| `device_index` | Unused; appears in the log. | Unused; appears in the log. |
| `stream_index` | The partition the kernel runs on. | any number |
| `tog_path` | A path into the kernel's directory, such as `k0/tog.so`. Only its directory part is used, and it must have one: write `./tog.so`, not `tog.so`. | any text |
| `attribute_path` | Unused. | any text |
| `timestamp` | The core cycle at which the kernel arrives; 0 means now. | any number |

`LAUNCH_KERNEL` loads `tog.so`, `tog_cycles.tsv` and `tog_units.tsv` from the kernel's directory;
the first two are required, and the third unless the config sets `systolic_array_split: none`.
`DEVICE_SYNC` simulates until all launched work is done. At the end of the file the simulation runs
to completion as well. The kernels' unit tables are joined into one, so every kernel must use the
units the first kernel's table names; `--unit_table` is refused in this mode. A line with another
field count or an unknown command is reported and skipped.

**Exit status.**

| Status | When |
|---|---|
| 0 | The simulation finished (also `--help`). |
| 1 | A refusal of [the tables above](tog-abi.md#what-gets-a-run-refused), an unknown flag, no `--config`, a config that cannot be read, neither `--tog_so` nor `--models_list`, a `LAUNCH_KERNEL` that fails. |
| 134 | The machine config lacks a required key or has a value TOGSim does not implement. The process aborts with `what():  <reason>` on stderr. |
| 136 | `--models_list` names a file that cannot be opened. |
| 255 | `ramulator_config_path` names a file that does not exist. |

## Reading the output

Everything goes to stdout as `[date time] [level] text`, with two kinds of exception. Three
messages go to stderr without a prefix: the loader's `togsim:` lines, the `what():` line of an
abort, and Ramulator2's `Config file ... does not exist!`. And Ramulator2, BookSim2 and the cache
model print their statistics to stdout as bare lines.

A log has four parts: the command line and the config, echoed; one `[Config/...]` line per
component; periodic statistics while the simulation runs; and the final statistics, which end
with the total. [docs/output-lines.md](output-lines.md) walks through a whole log line by
line. The final statistics of the hello TOG on tpuv6e (the third run of `examples/run.sh hello`):

```text
[info] ===== Instructions count =====
[info] Core [0] : MOVIN    inst_count: 1
[info] Core [0] : MOVOUT   inst_count: 1
[info] Core [0] : COMP     inst_count: 1 (GEMM: 0, Vector: 1, XLU: 0)
[info] Core [0] : MEMORY_BAR inst_count: 0
[info] ========= Core stat =========
[info] Core [0] : Systolic utilization(%): 23.23, active_cycles: 256, idle_cycles: 846
[info] Core [0] : Vpu utilization(%): 9.07, active_cycles: 100, idle_cycles: 1002
[info] Core [0] : DMA active_cycles: 64, DMA idle_cycles: 1038, DRAM BW: 104.000 GB/s (1024 responses)
[info] Core [0] : NUMA local memory: 1024 requests, remote memory: 0 requests
[info] Core [0] : Total_cycles: 1102
[info] Core [0] : computes held at least once by the core's compute in flight: 0
[info] Total execution cycles: 1102
```

- **`Total execution cycles`** is the result: the cycles of the core clock (`core_freq_mhz`) until
  all work was done. Divide by `core_freq_mhz` for microseconds. It is the last line of a
  `--tog_so` log; a session prints one more line after it, `Wall-clock time for simulation`. A
  `--tog_so` run also prints the same number earlier, as `[TOGSim] Total cycles`.
- **`Instructions count`** is what each core ran: loads, stores, computes, barriers. A compute of
  zero cycles is not counted; it is reported as `skipped inst_count`. A load that found its data
  [already in its buffer](tog-abi.md#buffers-and-the-scratchpad) is counted, and again as
  `skipped inst_count` on the `MOVIN` line. All computes with `compute_type` 0 are under `Vector`.
- **`<unit> utilization(%)`**: one line per unit of the unit table. 256 rows at 1 per cycle are 256
  active cycles of 1102, or 23.23%. With more than one core each core prints its block, and a
  `Total : <unit> utilization(%)` line per unit follows.
- **`DMA active_cycles`** are the cycles the core's DMA engine spent sending requests: 2 DMAs of
  512 requests at 16 per cycle. **`DRAM BW`** is the bytes DRAM answered (32 x 1024 here) over the
  run, in GB/s, cut to a whole number.
- **`NUMA local / remote`** counts requests that went to a DRAM channel of the core's own
  partition or to another's (`dram_num_partitions`).
- **`computes held at least once ...`** is how many computes were ready while the core was still
  busy with another compute.

**Warnings.** A `[warning]` line before the simulation starts names something that still runs and
is probably a mistake: a compute whose `tile_id` has no unit-table rows, a DMA whose `arg_id`
`tog_tensors.txt` does not list, an indirect DMA without its index dump, a TOG that passes a
deprecated `compute_type`. Each is printed once per TOG. Grep a log for `\[warning\]` before you trust its
total.

**Without a unit table** (`systolic_array_split: none`) the unit lines are three built-in ones,
`Systolic array [k]`, `Vector unit` and `Cross-lane unit`, and only `Vector unit` is non-zero; the
last run of `examples/run.sh gemm` shows them.

**The trace.** `--log_level trace` adds one line per event of every instruction. To follow one
instruction, grep for its id:

```bash
build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml \
  --tog_so examples/hello/tog.so --cycle_table examples/hello/tog_cycles.tsv \
  --unit_table examples/hello/tog_units.tsv --log_level trace | grep -F 'INST_ID=0]'
```

```text
[2026-10-12 14:29:13.216] [trace] [2][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=false indirect=false tag_id=[0x0000000000000000,0x0000000000000000])
[2026-10-12 14:29:13.216] [trace] [2][Core 0][L2CACHE_STAT   ][INST_ID=0] dram=0x0000000000000000 cacheable=false
[2026-10-12 14:29:13.216] [trace] [2][Core 0][DRAM_NUMA      ][INST_ID=0] core_id=0 subgraph_id=0 numa_id=0 addr_name=tag0 is_write=false
[2026-10-12 14:29:13.219] [trace] [147][Core 0][DRAM_RESP_FIRST][INST_ID=0] MOVIN (addr_name=tag0 tile=0)
[2026-10-12 14:29:13.221] [trace] [310][Core 0][INST_FINISHED  ][INST_ID=0] MOVIN (addr_name=tag0 tile=0)
```

| Tag | Meaning |
|---|---|
| `TILE_SCHEDULED` | A work-item was handed to a core: `spad_footprint` in bytes, and `max_dispatch`, how many work-items that core keeps in flight (1 or 2). |
| `INST_ISSUED`, `INST_FINISHED` | An instruction started, finished. On a `COMP` line `compute_cycle` is its cycles after any array split, and `tile=N` the work-item index. |
| `ASYNC_DMA_ISSUE` | An asynchronous load has sent all its requests. |
| `DRAM_RESP_FIRST`, `DRAM_RESP_DONE` | The first response of a DMA arrived; all data of an asynchronous load arrived. |
| `DRAM_NUMA`, `L2CACHE_STAT` | Where a DMA goes: its DRAM partition, and (with the L2 data cache) whether its address is cacheable. |

`INST_ID` numbers instructions in the order TOGSim created them. The opcodes are `MOVIN`,
`MOVOUT`, `COMP` and `MEMORY_BAR`. [docs/output-lines.md](output-lines.md) explains every
field of a trace line, the periodic and DRAM statistics, why a log is long, and which lines a
script may rely on.

## Troubleshooting

A run that ends with a message: look the message up in the
[refusal tables](tog-abi.md#what-gets-a-run-refused). The symptoms below come without one.

| Symptom | Likely cause |
|---|---|
| The log shows only `[TOGSim] the TOG failed to load or replay`, or the run ends with `Aborted` and no reason | The reason went to stderr. Add `2>&1`. |
| The total is too small | A consumer called before its producer, or null `strides` ([tutorial step 6](tutorial.md#6-mistakes-what-togsim-refuses-and-what-it-does-not)). |
| A unit's utilization is lower than expected, or a compute is not split | The unit table has no rows for that `tile_id`. Look for the `[warning]` line that says so. |
| A compute takes fewer cycles than the cycle table says | The array split shortened it. `compute_cycle` in the trace is the value after the split. |
| The run never ends | Work-items wait for spad that is never freed, in an order the loader does not see ([known limitations](#known-limitations)). Raise `core_spad_size_kb`. |
| The second kernel of a session behaves like the first | Its body function is not `static`. |
| A changed config key has no effect | Misspelled keys are ignored. Compare with the echo at the top of the log. |
| A DMA goes to a wrong address | `offset` is in elements, not bytes; an `arg_id` past `tog_tensors.txt` is placed at address 0, with a `[warning]` line. Read `dram=` on its trace line. |
| The log is thousands of lines | Read it through `tail -n 1` or `grep`. Most of a short log is Ramulator2's dump per DRAM channel, which no key turns off ([why](output-lines.md#periodic-and-final-statistics)). |
| The binary does not start after the build tree was moved | It finds `build/lib/libramulator.so` by an absolute path. Build where you run. |

## Known limitations

Behaviour of the simulator that a user runs into. Each was reproduced with this tree.

- **A run can still stop for want of spad.** A work-item that needs more of the spad at once
  than a core has is [refused](tog-abi.md#buffers-and-the-scratchpad) while its TOG is loaded (the
  hello TOG and the tiles example in 16 KB). A stall that depends on the order of issue, or on
  what another work-item holds, is not seen then. If no other work-item is queued for that core,
  TOGSim stops with `simulation wedged at cycle N: ...` and exit 1: one work-item that loads an
  8 KB buffer twice before its one reader, in a spad of 8 KB. If more work-items are queued, the
  run never ends: three work-items whose 8 KB loads of later steps are taken ahead of a 56 KB
  load, in 72 KB, and a work-item of 64 KB with two of 24 KB dispatched beside it, in 64 KB, were
  each still running when stopped after 20 seconds.
- **An unreadable `--models_list` file ends in exit 136**, not in a clean refusal: the log says
  `[TOGSim] Failed to open trace file: <path>` and the process dies printing the statistics of a
  run of zero cycles.
- **Null `strides` are not "contiguous".** The header comment says so, but the DMA becomes a
  single DRAM request ([tutorial step 6](tutorial.md#6-mistakes-what-togsim-refuses-and-what-it-does-not)).
- **A non-static body function in a session** makes a later kernel run the first kernel's body. A
  session whose second kernel has two computes in its body ran one compute for it.
- **The `[Scheduler p] Kernel k has completed ... finished at cycle N` line** is printed near the
  start of the run and reads 1 for the hello run of 1659 cycles on `examples/minimal.yml`. It is
  not when the kernel finished; read the total.
- **Unknown config keys are ignored**, including misspelled ones:
  `configs/systolic_ws_128x128_c1_simple_noc_tpuv2.yml` spells a key
  `dram_stats_print_period_cycless`.
- **`--help` describes `--models_list` as `Path for the trace file (.trace)`**, and a failed open
  says `Failed to open trace file`. The file is the command stream.
- **The log calls a work-item a tile** (`TILE_SCHEDULED`, `tile=N`), and the type of a body is
  `togsim_tile_fn`.
