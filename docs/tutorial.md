# Write your first TOG

Each step below is a directory under `examples/` with a TOG source, the files it needs and
`expected.txt`, the output of `examples/run.sh <name>`. Build the simulator first
([Quick start](../README.md#quick-start)). Every run uses
`configs/tpuv6e.yml`, called **tpuv6e** in these pages.

The tests run every example against its `expected.txt` and check that each line quoted in a fenced
block here is still in those files; numbers in tables and prose were copied by hand.

## The model

Three things nest:

```
kernel       one TOG; its entry runs once
  work-item  one dispatch: a body function and its arguments, bound to one core
    instruction  one call in the body: load, store, compute or wait
```

**What a core can do at once.** The simulator enforces each of these limits:

| Resource of one core | Limit |
|---|---|
| work-items in flight | two, or one when a work-item's buffers need more than half the scratchpad |
| instructions issued | one per cycle |
| DMA engine | one DMA at a time, loads ahead of stores; it sends `icnt_injection_ports_per_core` DRAM requests per cycle (32 on tpuv6e). A load's wait after its data (`dma_load_ready_cycles`) and a store's after its requests (`dma_store_ready_cycles`) do not occupy it. |
| compute | one at a time, for the cycles the cycle table gives it |
| scratchpad (spad) | `core_spad_size_kb`; a load waits until its buffer fits |

All cores share one DRAM and one interconnect.

**What orders the instructions.** Nothing but the buffers. For each buffer an instruction reads,
it waits for the last instruction that wrote that buffer among those called before it in the same
work-item; then it waits for the resource it needs. The hello example of step 1 and the gemm
example of step 4, as graphs (an arrow is a buffer id):

```
hello:   load ──1──> compute ──2──> store

gemm:    load A ──1──┐
         load B ──2──┴──> compute (tile_id 0) ──3──┐
         load A ──1──┐                             │
         load B ──2──┴──> compute (tile_id 0) <────┘
                                │
                                3
                                v
                          compute (tile_id 1) ──4──> store
```

Six terms carry the rest of these pages; the [glossary](glossary.md#glossary) has the others.

| Term | Meaning |
|---|---|
| TOG | A shared object (`tog.so`) whose calls are one kernel's DMAs, computes and barriers. |
| work-item | One `dispatch`: the unit of parallelism. The log calls it a tile: `TILE_SCHEDULED`, `tile=N`. |
| DMA, data tile | A DMA is a transfer between DRAM and a core's scratchpad: a load (`MOVIN` in the log) or a store (`MOVOUT`). The block of a tensor it moves is a data tile. |
| buffer | A region of the scratchpad, named by a small integer id. Buffers are what DMAs fill and drain, and what orders instructions. |
| compute tile, `tile_id` | A kind of compute: a row of the cycle table. Many `compute` calls can name the same one. |
| cycle table, unit table | Per `tile_id`: how many cycles the compute takes, and what each hardware unit did during it. You supply both. |

## 1. Load, compute, store

`examples/hello/hello_tog.cpp` is a complete TOG:

```cpp
#include "togsim.hpp"
using namespace togsim;

struct Hello : Tog<Hello> {
  static constexpr Buf IN{.id = 1, .bytes = f32(64, 64)}, OUT{.id = 2, .bytes = f32(64, 64)};
  static constexpr std::array buffers{IN, OUT};
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr Desc SRC{.ptr = 0, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
  static constexpr Desc DST{.ptr = 1, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};

  void entry(Kernel& k) const { dispatch<&Hello::body>(k); }

  void body(Item& w) const {
    load(w, {.src = SRC, .dst = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = DST});
  }
};
```

Read it from the top down:

- **`buffers`** lists every spad buffer a call names: 1 and 2, each 64 x 64 elements of 4 bytes.
- **`STEP`** is compute node 0 of 1. A compute takes row `id + count * variant` of the cycle
  table; with the default variant 0, row 0.
- **`SRC`, `DST`** are DMA descriptors: tensor argument `ptr` (its line of `tog_tensors.txt`),
  64 x 64 `f32` elements from element 0, rows 64 apart: one contiguous block.
- **`entry`** runs once and dispatches the one work-item, **`body`**: three instructions, a load
  into buffer 1, a compute from 1 into 2, a store of 2.

[The TOGSim API](tog-api.md#togsim-api-v100) lists every call and field.

Three small files stand beside the source:

| File | Contents | Meaning |
|---|---|---|
| `tog_tensors.txt` | `16384`<br>`16384` | The TOG has two tensor arguments of 16384 bytes each. |
| `tog_cycles.tsv` | `600<TAB>0` | Compute tile 0 takes 600 cycles. |
| `tog_units.tsv` | `0<TAB>TPUV6E_MXU<TAB>rows<TAB>1<TAB>128`<br>`0<TAB>Vpu<TAB>cycles<TAB>1<TAB>100` | During one run of compute tile 0 the `TPUV6E_MXU` unit, the systolic array, took in 128 rows at 1 row per cycle, and the `Vpu` unit was busy for 100 cycles. |

`tog_tensors.txt` must be in the directory of `tog.so` under that name. The two tables are named on
the command line: `--cycle_table` is always required, `--unit_table` on every shipped config.

```bash
examples/run.sh hello
```

The script echoes each command after `+`. Its first run is the Quick start's run with
`--log_level trace`, filtered to the instructions (the dots are cuts made here):

```text
== hello
+ g++ -std=c++20 -shared -fPIC -O1 -Iinclude examples/hello/hello_tog.cpp -o examples/hello/tog.so
+ build/bin/Simulator --config configs/tpuv6e.yml --tog_so examples/hello/tog.so --cycle_table examples/hello/tog_cycles.tsv --unit_table examples/hello/tog_units.tsv --log_level trace
[trace] [1][Core 0][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [2][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=false ...
[trace] [1011][Core 0][INST_FINISHED  ][INST_ID=0] MOVIN (addr_name=tag0 tile=0)
[trace] [1012][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [1612][Core 0][INST_FINISHED  ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [1613][Core 0][INST_ISSUED    ][INST_ID=2] MOVOUT (addr_name=tag1 dram=0x0000000000004000 rank=2 elem_bits=32 async=false ...
[trace] [1653][Core 0][INST_FINISHED  ][INST_ID=2] MOVOUT (addr_name=tag1 tile=0)
[info] Total execution cycles: 2437
```

Each trace line is `[cycle][Core c][TAG][INST_ID=n] OPCODE (fields)`:

| Cycle | Line | What happened |
|---|---|---|
| 1 | `TILE_SCHEDULED spad_footprint=32768` | The work-item was handed to core 0. Its footprint is the two buffers, 2 x 16384 bytes. `max_dispatch=2`: this core keeps two work-items in flight. |
| 2 | `INST_ISSUED` `MOVIN` | The load starts. The DMA engine holds it 27 cycles (`dma_load_latency_cycles: 27` on tpuv6e), then sends 16384 bytes as 512 DRAM requests of 32 bytes. |
| 1011 | `INST_FINISHED` `MOVIN` | The load completes 842 cycles after its last response arrived (`dma_load_ready_cycles: 842` on tpuv6e). The DMA engine is free during that wait. |
| 1012 | `INST_ISSUED` `COMP` `compute_cycle=600` | The compute starts, with the 600 cycles of the cycle table. |
| 1612 | `INST_FINISHED` `COMP` | 1012 + 600. |
| 1613 | `INST_ISSUED` `MOVOUT` `dram=0x...4000` | The store starts. Tensor 1 sits at 0x4000, after the 16384 bytes of tensor 0. |
| 1653 | `INST_FINISHED` `MOVOUT` | The last write request was sent: the DMA engine held the store 24 cycles (`dma_store_latency_cycles: 24` on tpuv6e), then sent 512 requests at 32 per cycle. |
| 2437 | `Total execution cycles` | The simulation ends 784 cycles after the store's last request (`dma_store_ready_cycles: 784` on tpuv6e). |

In a trace line `tile=0` is the index of the work-item, not a `tile_id`
([all fields](output-lines.md#trace-lines)). The script's later runs print one instruction's
events, the final statistics ([Reading the output](running.md#reading-the-output)) and a whole log on a
machine without a unit table.

**A kernel of your own.** Copy the example and create the three files. The unit table must be
tab-separated, so `printf` is safer than an editor that turns tabs into spaces:

```bash
mkdir -p my_kernel
cp examples/hello/hello_tog.cpp my_kernel/my_tog.cpp
printf '16384\n16384\n' > my_kernel/tog_tensors.txt
printf '600\t0\n' > my_kernel/tog_cycles.tsv
printf '0\tTPUV6E_MXU\trows\t1\t128\n0\tVpu\tcycles\t1\t100\n' > my_kernel/tog_units.tsv
g++ -std=c++20 -shared -fPIC -O1 -Iinclude my_kernel/my_tog.cpp -o my_kernel/tog.so
build/bin/Simulator --config configs/tpuv6e.yml \
  --tog_so my_kernel/tog.so --cycle_table my_kernel/tog_cycles.tsv \
  --unit_table my_kernel/tog_units.tsv | tail -n 1
```

```text
[2026-10-12 14:29:13.020] [info] Total execution cycles: 2437
```

Change the 600 in `tog_cycles.tsv` to 1000 and the total becomes 2837.

## 2. Asynchronous loads and a barrier

A load can be synchronous or asynchronous.

- A **synchronous** load finishes when its data has arrived (on tpuv6e, 842 cycles after that);
  its readers wait for it.
- An **asynchronous** load (`.sem`) finishes once its last request is sent, without waiting for the
  data; the TOG `wait`s on its semaphore before using the data.

`examples/async_barrier/async_barrier_tog.cpp` loads two operands the second way:

```cpp
    Semaphore ab = semaphore(w);
    load(w, {.src = LHS, .dst = A, .sem = ab});
    load(w, {.src = RHS, .dst = B, .sem = ab});
    wait(w, {.sem = ab});
```

In the log the `wait` is a barrier, `MEMORY_BAR`; a compute that reads buffer 1 or 2 waits for it,
not for the loads.

```bash
examples/run.sh async_barrier
```

The first run of four (the dots are cuts made here):

```text
[trace] [2][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=true ...
[trace] [3][Core 0][INST_ISSUED    ][INST_ID=1] MOVIN (addr_name=tag1 dram=0x0000000000004000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=true ...
[trace] [45][Core 0][ASYNC_DMA_ISSUE][INST_ID=0] MOVIN (ASYNC subgraph_id=0 addr_name=tag0 ...
[trace] [88][Core 0][ASYNC_DMA_ISSUE][INST_ID=1] MOVIN (ASYNC subgraph_id=0 addr_name=tag1 ...
[trace] [89][Core 0][INST_ISSUED    ][INST_ID=2] MEMORY_BAR (addr_name=tag0 ...
[trace] [1011][Core 0][DRAM_RESP_DONE ][INST_ID=0] MOVIN (ASYNC subgraph_id=0 addr_name=tag0 ...
[trace] [1053][Core 0][DRAM_RESP_DONE ][INST_ID=1] MOVIN (ASYNC subgraph_id=0 addr_name=tag1 ...
[trace] [1054][Core 0][INST_FINISHED  ][INST_ID=2] MEMORY_BAR (addr_name=tag0 ...
[trace] [1054][Core 0][INST_ISSUED    ][INST_ID=3] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [1654][Core 0][INST_FINISHED  ][INST_ID=3] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
...
[info] Total execution cycles: 2479
```

The DMA engine holds each load 27 cycles before its requests, one load after the other, so the
loads have sent their requests at cycles 45 and 88 (`ASYNC_DMA_ISSUE`), and the barrier is
issued at 89. Each load then completes 842 cycles after its last data. That wait does not hold the
engine, so the two loads wait theirs at the same time: `DRAM_RESP_DONE` at 1011 and 1053, 42
cycles apart, not 842. The barrier finishes at 1054, one cycle after the second load completes,
and the compute starts in the same cycle.

**Which form to write.** The other three runs rebuild the TOG with `-DSYNC` (both loads
synchronous, no barrier) and with `-DTWO_STEPS` (a first compute that reads only operand A):

| TOG | Asynchronous loads and a barrier | Synchronous loads |
|---|---|---|
| one compute reads both operands | 2479 | 2479 |
| `-DTWO_STEPS`: a first compute reads only A | 3080, first compute at 1054 | 3038, first compute at 1012 |

A barrier holds every reader of its loads' buffers until all of them have completed; a synchronous
load releases its readers as soon as it completes (cycle 1011 for A). The asynchronous form is never
faster here: use it only to model a kernel that really issues several loads and then waits for all.

## 3. Many work-items, on one core, two and four

A real kernel is a loop over data tiles. `examples/tiles/tiles_tog.cpp` dispatches one work-item
per cell of an M x N grid, in ordinary C++ loops:

```cpp
  void entry(Kernel& k) const {
    const auto [M, N] = shape<2>(k, {.kernel = "tiles", .names = "M N", .example = {1, 1}});
    for (int64_t m = 0; m < M; m++)
      for (int64_t n = 0; n < N; n++) dispatch<&Tiles::body>(k, Cell{.m = m, .n = n, .N = N});
  }
```

Three things are new:

- **Shape arguments.** `shape<2>` reads the two integers of `tog_shape.txt`, a file beside
  `tog.so`. Here it contains `2 4`, so the kernel dispatches eight work-items. The same `tog.so`
  serves another grid when the file changes. Without the file the shape is `.example`.
- **Arguments.** `dispatch` copies the `Cell` (`int64_t` fields) for the body: the only way a body
  knows which work-item it is, here to find its data tile's offset.
- **Cores.** The TOG never names a core: TOGSim binds work-items to cores round-robin, in dispatch
  order.

All eight work-items run compute tile 0, so the cycle table still has one row. They also all use
buffer ids 1 and 2: buffer ids are local to a work-item, so these are sixteen separate buffers.

```bash
examples/run.sh tiles
```

The script runs the kernel on tpuv6e as shipped, which has one core, and on copies of it with
one key changed. The first three:

| Machine | `Total execution cycles` | What the trace shows |
|---|---|---|
| tpuv6e (`num_cores: 1`) | 8301 | Computes are issued at 1012, 1612, 2665, ...: one at a time, each after its work-item's load. |
| `num_cores: 2` | 4824 | Work-items 0, 2, 4, 6 are on core 0 and 1, 3, 5, 7 on core 1. The first computes start at 1009 and 1018. |
| `num_cores: 4` | 3086 | Two work-items per core. The first computes start at 1041 to 1061. |

The run with `num_cores: 2`, where `tile=N` is the work-item's index in dispatch order:

```text
[trace] [1][Core 0][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [1][Core 0][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [2][Core 1][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [2][Core 1][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [1009][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [1018][Core 1][INST_ISSUED    ][INST_ID=4] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=1)
[trace] [1609][Core 0][INST_ISSUED    ][INST_ID=7] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=2)
[trace] [1618][Core 1][INST_ISSUED    ][INST_ID=10] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=3)
...
[info] Total execution cycles: 4824
```

Two things to take from these runs:

- **More cores are not proportionally faster.** Two cores take 4824 cycles and four take 3086,
  against 8301 for one. All cores load through the same DRAM and interconnect: on four
  cores the first computes start at 1041 to 1061, against 1012 on one core.
- **Do not rely on the order of work-items.** On one core the computes run in the order
  `tile=` 0, 1, 2, 4, 5, 6, 7, 3. A TOG can order instructions inside a work-item, with buffers,
  and nothing across work-items.

The script's last two runs shrink the scratchpad and halve the DRAM channels;
[Machine configuration](machine.md#changing-the-machine) reads them.

## 4. Two kinds of compute, unit rows and the array split

`examples/gemm/gemm_tog.cpp` is one work-item of a matrix multiply C = A x B, in K = 2 steps:

```cpp
    for (int64_t k = 0; k < K; k++) {
      load(w, {.src = {.ptr = 0, .type = f32, .offset = k * T, .dims = {T, T}, .strides = {K * T, 1}}, .dst = A});
      load(w, {.src = {.ptr = 1, .type = f32, .offset = k * T * T, .dims = {T, T}, .strides = {T, 1}}, .dst = B});
      compute(w, {.node = MAC, .reads = {A, B, ACC}, .writes = {ACC}});
    }
    compute(w, {.node = EPILOGUE, .reads = {ACC}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = {.ptr = 2, .type = f32, .offset = 0, .dims = {T, T}, .strides = {T, 1}}});
```

**Two kinds of compute and an accumulator.** Each step loads a data tile of A and one of B and
runs compute tile 0, a multiply-accumulate (`MAC`, node 0 of 2). That compute lists `ACC`, buffer 3,
in both `.reads` and `.writes`: it updates the buffer in place, and because the next step reads
what this step wrote, the two steps run one after the other. A last compute, `EPILOGUE`, compute
tile 1, turns the accumulator into the output buffer 4, which is stored. The loads here take data tiles
out of larger tensors: A is 64 x 128 elements, so a data tile of A has `strides = {128, 1}` and
step `k` starts at element offset `k * 64`.

Two `tile_id`s mean two rows in the cycle table and rows for both in the unit table:

| File | Contents |
|---|---|
| `tog_cycles.tsv` | `600<TAB>0`<br>`150<TAB>0` |
| `tog_units.tsv` | `0<TAB>TPUV6E_MXU<TAB>rows<TAB>1<TAB>512`<br>`0<TAB>Vpu<TAB>cycles<TAB>1<TAB>50`<br>`1<TAB>Vpu<TAB>cycles<TAB>1<TAB>150` |

**Unit rows and the array split.** So far the unit table only fed the utilization lines. It also
changes how long a compute takes. A row says what one hardware unit did during one run of a
compute tile: here compute tile 0 pushes 512 rows through the unit named `TPUV6E_MXU`, the systolic
array, at one row per cycle, so 512 of its 600 cycles are array work; `Vpu`, the vector unit, is
busy for 50 cycles.

A core can have several systolic arrays (`num_systolic_array_per_core`, 2 on tpuv6e), and one
array takes `vpu_num_lanes` rows (128 on tpuv6e) before a compute tile is split. With 512 rows
TOGSim spreads the array work over both arrays, 256 rows each, and the compute takes
600 - 512 + 256 = 344 cycles ([the formula](tog-abi.md#the-unit-table-and-the-array-split)).

```bash
examples/run.sh gemm
```

The script runs that, then changes one thing at a time:

| Run | Compute tile 0 | Compute tile 1 | `Total execution cycles` |
|---|---|---|---|
| 512 rows, 2 arrays, 128 lanes (as shipped) | 600 - 512 + 256 = 344 | 150 | 2719 |
| `tog_units_128_rows.tsv`: 128 rows | 600 | 150 | 3231 |
| 512 rows, `num_systolic_array_per_core: 1` | 600 | 150 | 3231 |
| 128 rows, `vpu_num_lanes: 64` | 600 - 128 + 64 = 536 | 150 | 3103 |
| `systolic_array_split: none`, no unit table | 600 | 150 | 3231 |

The first run:

```text
[trace] [1054][Core 0][INST_ISSUED    ][INST_ID=2] COMP (compute_type=0 compute_cycle=344 overlapping_cycle=0 sa=-1 tile=0)
[trace] [1399][Core 0][INST_ISSUED    ][INST_ID=5] COMP (compute_type=0 compute_cycle=344 overlapping_cycle=0 sa=-1 tile=0)
[trace] [1744][Core 0][INST_ISSUED    ][INST_ID=6] COMP (compute_type=0 compute_cycle=150 overlapping_cycle=0 sa=-1 tile=0)
[info] Core [0] : TPUV6E_MXU utilization(%): 18.83, active_cycles: 512, idle_cycles: 2207
[info] Core [0] : Vpu utilization(%): 9.19, active_cycles: 250, idle_cycles: 2469
[info] Total execution cycles: 2719
```

All three lines say `tile=0`, the third too although it runs `tile_id` 1: `tile=N` in the log is
the work-item, and this TOG has one. Tell computes apart by `compute_cycle`. 128 rows fit one
array of 128 lanes, so nothing is spread and the compute keeps its 600 cycles; on a machine with
64 lanes the same 128 rows are split again.

## 5. Several kernels: session mode

`--models_list <file>` runs a command stream instead of one TOG. Each `LAUNCH_KERNEL` line
names the `tog.so` of a kernel, and TOGSim reads the kernel's other files from the same directory
under fixed names: `tog_tensors.txt`, `tog_cycles.tsv` and `tog_units.tsv`. `examples/session/`
has two kernel directories, `k0/` and `k1/`, both built from the hello TOG but with different
tables. `examples/session/commands.txt`:

```text
# command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp
LAUNCH_KERNEL,0,0,0,examples/session/k0/tog.so,-,0
DEVICE_SYNC,0,0,0,-,-,0
LAUNCH_KERNEL,1,0,0,examples/session/k1/tog.so,-,0
DEVICE_SYNC,0,0,0,-,-,0
```

A relative `tog_path` is relative to where you run the Simulator. `stream_index` is the partition
the kernel runs on. `DEVICE_SYNC` runs the simulation until everything launched so far is done.
`examples/session/commands_two_partitions.txt` sends kernel 1 to partition 1, with no
`DEVICE_SYNC` in between:

```text
# command_type,kernel_id,device_index,stream_index,tog_path,attribute_path,timestamp
LAUNCH_KERNEL,0,0,0,examples/session/k0/tog.so,-,0
LAUNCH_KERNEL,1,0,1,examples/session/k1/tog.so,-,0
DEVICE_SYNC,0,0,0,-,-,0
```

```bash
examples/run.sh session
```

```text
+ build/bin/Simulator --config configs/tpuv6e.yml --models_list examples/session/commands.txt --log_level trace
[info] [Scheduler 0] Enqueued kernel_id: 0, tog_path: examples/session/k0/tog.so, operation: kernel, request_time_cycles: 0
[trace] [1012][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[info] [Scheduler 0] Enqueued kernel_id: 1, tog_path: examples/session/k1/tog.so, operation: kernel, request_time_cycles: 0
[trace] [3447][Core 0][INST_ISSUED    ][INST_ID=4] COMP (compute_type=0 compute_cycle=644 overlapping_cycle=0 sa=-1 tile=0)
[info] Total execution cycles: 4916
# examples/session/two_partitions.gen.yml: configs/tpuv6e.yml with num_cores: 2, num_partition: 2, partition: {core_0: 0, core_1: 1}
+ build/bin/Simulator --config examples/session/two_partitions.gen.yml --models_list examples/session/commands_two_partitions.txt --log_level trace
[info] [Scheduler 0] Enqueued kernel_id: 0, tog_path: examples/session/k0/tog.so, operation: kernel, request_time_cycles: 0
[info] [Scheduler 1] Enqueued kernel_id: 1, tog_path: examples/session/k1/tog.so, operation: kernel, request_time_cycles: 0
[trace] [1014][Core 1][INST_ISSUED    ][INST_ID=4] COMP (compute_type=0 compute_cycle=644 overlapping_cycle=0 sa=-1 tile=0)
[trace] [1016][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[info] Total execution cycles: 2483
```

- **First run, one partition.** Kernel 1 starts after kernel 0 is done, and each kernel's compute
  follows its own tables: 600 cycles and 128 rows for `k0`, which fit one array of tpuv6e, so 600;
  900 cycles and 512 rows for `k1`, so 900 - 512 + 256 = 644.
- **Second run, two partitions.** A copy of tpuv6e with two cores, one in each partition
  (`num_partition: 2` and its `partition` map), and the kernels run side by side: 2483 cycles, each
  compute as in the first run.

In a session the total is followed by one more line, `Wall-clock time for simulation`.

## 6. Mistakes: what TOGSim refuses, and what it does not

`examples/mistakes/` holds two valid TOGs that a `-D<NAME>` flag breaks one way each; the script
also feeds a valid TOG wrong inputs and wrong machines.

```bash
examples/run.sh mistakes
```

**Refused.** A refused run exits non-zero with a line that names the cause, before anything is
simulated. For example:

```text
+ g++ -std=c++20 -shared -fPIC -O1 -Iinclude -DTILE_PAST_TABLE examples/mistakes/refused_tog.cpp -o examples/mistakes/TILE_PAST_TABLE.so
+ build/bin/Simulator --config configs/tpuv6e.yml --tog_so examples/mistakes/TILE_PAST_TABLE.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv
[error] [TOGSim] TOG examples/mistakes/TILE_PAST_TABLE.so: work-item 0 (key 0) computes tile id 1, which the cycle table does not hold (it has 1 tile ids, 0 to 0)
exit 1
```

The script prints 18 refusals; the [refusal table](tog-abi.md#what-gets-a-run-refused) lists each with its
cause and its fix.

**Said, but not refused.** Two mistakes still run to the end with exit 0, and TOGSim names them in
one `[warning]` line while it loads the TOG. The valid TOG of `examples/mistakes/silent_tog.cpp`
takes 2437 cycles.

| Mistake | Total | What happens | The warning |
|---|---|---|---|
| `TENSOR_PAST_LIST`: a `.ptr` `tog_tensors.txt` has no line for | 2437 | The tensor is placed at address 0, on top of tensor 0: `dram=0x0000000000000000` on the store's trace line. | `its DMAs name arg_id 2, and tog_tensors.txt lists 2 tensor(s): those DMAs are addressed from base 0` |
| a unit table whose rows are all for `tile_id` 1, while the TOG computes `tile_id` 0 | 2437 | The compute adds nothing to any unit, and the array split leaves it whole: `TPUV6E_MXU utilization(%): 0.00`. | `the unit table has no rows for tile ids 0, which its computes name: those computes add nothing to the units` |

```text
+ build/bin/Simulator --config configs/tpuv6e.yml --tog_so examples/mistakes/TENSOR_PAST_LIST.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv --log_level trace
[warning] [TOGSim] TOG examples/mistakes/TENSOR_PAST_LIST.so: its DMAs name arg_id 2, and tog_tensors.txt lists 2 tensor(s): those DMAs are addressed from base 0
```

**Valid, with another meaning.** `CONSUMER_FIRST` calls the store before the compute that writes
its buffer, so the store drains [data already on the chip](tog-abi.md#data-that-is-already-on-the-chip).
It is issued at cycle 3, an `[info]` line names the buffer, and a 16 KB spad refuses the TOG.

```text
+ build/bin/Simulator --config configs/tpuv6e.yml --tog_so examples/mistakes/CONSUMER_FIRST.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv --log_level trace
[info] [TOGSim] TOG examples/mistakes/CONSUMER_FIRST.so: core 0 holds 16384 bytes for the kernel: buffers 2, which its work-items read without having written them
[trace] [2][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=false ...
[trace] [3][Core 0][INST_ISSUED    ][INST_ID=1] MOVOUT (addr_name=tag1 dram=0x0000000000004000 rank=2 elem_bits=32 async=false ...
[trace] [1016][Core 0][INST_ISSUED    ][INST_ID=2] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[info] Core [0] : DMA active_cycles: 32, DMA idle_cycles: 1585, DRAM BW: 35.000 GB/s (1024 responses)
[info] Total execution cycles: 1617
+ build/bin/Simulator --config examples/mistakes/spad_16kb.gen.yml --tog_so examples/mistakes/CONSUMER_FIRST.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv
[error] [TOGSim] TOG examples/mistakes/CONSUMER_FIRST.so: work-item 0 (key 0) needs 16384 bytes in the spad at once beside the 16384 bytes core 0 holds for the kernel (buffers 2, read without having been written), and a core's spad holds 16384 (core_spad_size_kb: 16)
exit 1
```

So when you bring up a new TOG, read its log once at `--log_level trace`: check that it has no
`[warning]` line, that every compute is there with the cycles you expect, that each DMA has the
address and strides you meant, and that the instructions are issued in the order the data flows.
