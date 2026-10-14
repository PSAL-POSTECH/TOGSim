# Write your first TOG

Each step below is a directory under `examples/` with a TOG source, the files it needs and
`expected.txt`, the output of `examples/run.sh <name>`. Build the simulator first
([Quick start](../README.md#quick-start)). Every run uses
`configs/tpuv6e.yml`, called **tpuv6e** in these pages.

The tests replay every example against its `expected.txt`, and check that each log line and source
line quoted in a fenced block of these pages is still a line of those files. The numbers in the
other tables and in prose were copied from those files by hand.

## The model

Three things nest:

```
kernel       one TOG; its entry point togsim_kernel runs once
  work-item  one togsim_dispatch call: a body function and its loop indices, bound to one core
    instruction  one call in the body: togsim_dma, togsim_compute or togsim_memory_barrier
```

**What a core can do at once.** The simulator enforces each of these limits:

| Resource of one core | Limit |
|---|---|
| work-items in flight | two, or one when a work-item's buffers need more than half the scratchpad |
| instructions issued | one per cycle |
| DMA engine | one DMA at a time, loads ahead of stores; it sends `icnt_injection_ports_per_core` DRAM requests per cycle (32 on tpuv6e) |
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
| work-item | One `togsim_dispatch` call: the unit of parallelism. The log calls it a tile: `TILE_SCHEDULED`, `tile=N`. |
| DMA, data tile | A DMA is a transfer between DRAM and a core's scratchpad: a load (`MOVIN` in the log) or a store (`MOVOUT`). The block of a tensor it moves is a data tile. |
| buffer | A region of the scratchpad, named by a small integer id. Buffers are what DMAs fill and drain, and what orders instructions. |
| compute tile, `tile_id` | A kind of compute. Every `togsim_compute` call names one by its `tile_id`, and many calls can name the same one. |
| cycle table, unit table | Per `tile_id`: how many cycles the compute takes, and what each hardware unit did during it. You supply both. |

## 1. Load, compute, store

`examples/hello/hello_tog.cpp` is a complete TOG:

```cpp
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64

// The body of the work-item: each call is one modeled instruction, and the buffer ids order them.
static void body(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[2] = {ROWS, COLS}, strides[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, 0, 2, dims, strides, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, 1, 0, 2, dims, strides, 32, 0, 0, out, 1, nullptr, 0, 0);
}

// The entry point, called once when the TOG is loaded: it registers the one work-item.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  togsim_dispatch(ctx, body, nullptr, 0, 0);
}

// The byte size of each spad buffer id named above; id 0 is not a buffer.
extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
```

Read it from the bottom up:

- **The buffer table.** The TOG uses spad buffers 1 and 2. `togsim_spad_buffer_bytes[id]` is the
  size of buffer `id`, here 64 x 64 elements of 4 bytes each.
- **`togsim_kernel`** is the entry point. It calls `togsim_dispatch` once per work-item; this TOG
  has one.
- **`body`** is the body of that work-item. Its three calls are three instructions: a load into
  buffer 1, a compute that reads buffer 1 and writes buffer 2, and a store of buffer 2.

The calls take positional arguments. The load, argument by argument:

| # | Argument | Here | Meaning |
|---|---|---|---|
| 1 | `ctx` | `ctx` | The handle the body was given. |
| 2 | `dir` | `TOGSIM_DMA_LOAD` | A load: DRAM to spad. |
| 3 | `arg_id` | `0` | Tensor argument 0, the first line of `tog_tensors.txt`. |
| 4 | `offset` | `0` | The data tile starts at element 0 of that tensor. |
| 5, 6 | `ndim`, `dims` | `2`, `{64, 64}` | It is 64 x 64 elements. |
| 7 | `strides` | `{64, 1}` | Its rows are 64 elements apart, so it is one contiguous block. |
| 8 | `elem_bits` | `32` | An element is 4 bytes. |
| 9, 10 | `is_async`, `sem` | `0`, `0` | Synchronous; no semaphore (step 2). |
| 11, 12 | `read_bufs`, `n_read` | `nullptr`, `0` | A load reads no buffer. |
| 13, 14 | `write_bufs`, `n_write` | `in`, `1` | It fills one buffer, id 1. |
| 15 | `indirect` | `0` | Not a gather. |

The store is the same call with `TOGSIM_DMA_STORE`, tensor argument 1, and the buffer in
`read_bufs` instead of `write_bufs`. The compute:

| # | Argument | Here | Meaning |
|---|---|---|---|
| 2 | `tile_id` | `0` | Compute tile 0: row 0 of the cycle table gives its cycles. |
| 3 | `compute_type` | `0` | Always 0. |
| 4, 5 | `ndim`, `dims` | `0`, `nullptr` | Not used. |
| 6, 7 | `read_bufs`, `n_read` | `in`, `1` | It reads buffer 1, so it waits for the load that fills buffer 1. |
| 8, 9 | `write_bufs`, `n_write` | `out`, `1` | It writes buffer 2, so the store that reads buffer 2 waits for it. |

[The TOG ABI](tog-abi.md#the-tog-abi) has the full reference of each call.

Three small files stand beside the source:

| File | Contents | Meaning |
|---|---|---|
| `tog_tensors.txt` | `16384`<br>`16384` | The TOG has two tensor arguments of 16384 bytes each. |
| `tog_cycles.tsv` | `600<TAB>0` | Compute tile 0 takes 600 cycles. |
| `tog_units.tsv` | `0<TAB>TPUV6_MXU<TAB>rows<TAB>1<TAB>128`<br>`0<TAB>Vpu<TAB>cycles<TAB>1<TAB>100` | During one run of compute tile 0 the `TPUV6_MXU` unit, the systolic array, took in 128 rows at 1 row per cycle, and the `Vpu` unit was busy for 100 cycles. |

`tog_tensors.txt` must be in the directory of `tog.so` under that name. The two tables are named on
the command line: `--cycle_table` is always required, `--unit_table` on every shipped config. The
TOG links against nothing: TOGSim provides the four `togsim_*` functions when it loads the TOG.

```bash
examples/run.sh hello
```

The script echoes each command after `+`. Its first run is the replay of the Quick start with
`--log_level trace`, filtered to the instructions (the dots are cuts made here):

```text
== hello
+ g++ -std=c++17 -shared -fPIC -O1 -Iinclude examples/hello/hello_tog.cpp -o examples/hello/tog.so
+ build/bin/Simulator --config configs/tpuv6e.yml --tog_so examples/hello/tog.so --cycle_table examples/hello/tog_cycles.tsv --unit_table examples/hello/tog_units.tsv --log_level trace
[trace] [1][Core 0][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [2][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=false ...
[trace] [141][Core 0][INST_FINISHED  ][INST_ID=0] MOVIN (addr_name=tag0 tile=0)
[trace] [142][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [742][Core 0][INST_FINISHED  ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [743][Core 0][INST_ISSUED    ][INST_ID=2] MOVOUT (addr_name=tag1 dram=0x0000000000004000 rank=2 elem_bits=32 async=false ...
[trace] [759][Core 0][INST_FINISHED  ][INST_ID=2] MOVOUT (addr_name=tag1 tile=0)
[info] Total execution cycles: 840
```

Each trace line is `[cycle][Core c][TAG][INST_ID=n] OPCODE (fields)`:

| Cycle | Line | What happened |
|---|---|---|
| 1 | `TILE_SCHEDULED spad_footprint=32768` | The work-item was handed to core 0. Its footprint is the two buffers, 2 x 16384 bytes. `max_dispatch=2`: this core keeps two work-items in flight. |
| 2 | `INST_ISSUED` `MOVIN` | The load starts. 16384 bytes are 512 DRAM requests of 32 bytes. |
| 141 | `INST_FINISHED` `MOVIN` | The last response arrived. |
| 142 | `INST_ISSUED` `COMP` `compute_cycle=600` | The compute starts, with the 600 cycles of the cycle table. |
| 742 | `INST_FINISHED` `COMP` | 142 + 600. |
| 743 | `INST_ISSUED` `MOVOUT` `dram=0x...4000` | The store starts. Tensor 1 sits at 0x4000, after the 16384 bytes of tensor 0. |
| 759 | `INST_FINISHED` `MOVOUT` | The last write request was sent: 512 requests at 32 per cycle. |
| 840 | `Total execution cycles` | The simulation ends when DRAM has answered the last write. |

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
printf '0\tTPUV6_MXU\trows\t1\t128\n0\tVpu\tcycles\t1\t100\n' > my_kernel/tog_units.tsv
g++ -std=c++17 -shared -fPIC -O1 -Iinclude my_kernel/my_tog.cpp -o my_kernel/tog.so
build/bin/Simulator --config configs/tpuv6e.yml \
  --tog_so my_kernel/tog.so --cycle_table my_kernel/tog_cycles.tsv \
  --unit_table my_kernel/tog_units.tsv | tail -n 1
```

```text
[2026-10-12 14:29:13.020] [info] Total execution cycles: 840
```

Change the 600 in `tog_cycles.tsv` to 1000 and the total becomes 1239.

## 2. Asynchronous loads and a barrier

A load can be synchronous or asynchronous.

- A **synchronous** load (`is_async` 0) is finished when its data has arrived. Whatever reads its
  buffer simply waits for it.
- An **asynchronous** load (`is_async` 1) counts as finished once its last request has been sent,
  without waiting for the data, and signals a semaphore with the number of elements it moves. The
  TOG must then call `togsim_memory_barrier` on that semaphore before it uses the data.

`examples/async_barrier/async_barrier_tog.cpp` loads two operands the second way:

```cpp
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, 0, 2, dims, strides, 32, 1, SEM, nullptr, 0, a, 1, 0);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 1, 0, 2, dims, strides, 32, 1, SEM, nullptr, 0, b, 1, 0);
  togsim_memory_barrier(ctx, SEM, 2 * ROWS * COLS);
```

The barrier waits for every load signaled on `SEM` since the last barrier on it, and its last
argument must be the sum of their element counts (here two loads of 64 x 64). After the barrier,
a compute that reads buffer 1 or 2 waits for the barrier, not for the loads.

```bash
examples/run.sh async_barrier
```

The first run of four (the dots are cuts made here):

```text
[trace] [2][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=true ...
[trace] [3][Core 0][INST_ISSUED    ][INST_ID=1] MOVIN (addr_name=tag1 dram=0x0000000000004000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=true ...
[trace] [18][Core 0][ASYNC_DMA_ISSUE][INST_ID=0] MOVIN (ASYNC subgraph_id=0 addr_name=tag0 ...
[trace] [34][Core 0][ASYNC_DMA_ISSUE][INST_ID=1] MOVIN (ASYNC subgraph_id=0 addr_name=tag1 ...
[trace] [35][Core 0][INST_ISSUED    ][INST_ID=2] MEMORY_BAR (addr_name=tag0 ...
[trace] [143][Core 0][DRAM_RESP_DONE ][INST_ID=0] MOVIN (ASYNC subgraph_id=0 addr_name=tag0 ...
[trace] [159][Core 0][DRAM_RESP_DONE ][INST_ID=1] MOVIN (ASYNC subgraph_id=0 addr_name=tag1 ...
[trace] [160][Core 0][INST_FINISHED  ][INST_ID=2] MEMORY_BAR (addr_name=tag0 ...
[trace] [160][Core 0][INST_ISSUED    ][INST_ID=3] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [760][Core 0][INST_FINISHED  ][INST_ID=3] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
...
[info] Total execution cycles: 858
```

The loads have sent their requests at cycles 18 and 34 (`ASYNC_DMA_ISSUE`), so the barrier is
issued at 35. It finishes at 160, one cycle after the data of the second load is there
(`DRAM_RESP_DONE` at 159), and the compute starts in the same cycle.

**Which form to write.** The other three runs rebuild the TOG with `-DSYNC` (both loads
synchronous, no barrier) and with `-DTWO_STEPS` (a first compute that reads only operand A):

| TOG | Asynchronous loads and a barrier | Synchronous loads |
|---|---|---|
| one compute reads both operands | 858 | 858 |
| `-DTWO_STEPS`: a first compute reads only A | 1459, first compute at 160 | 1443, first compute at 144 |

A barrier holds every reader of its loads' buffers until the whole batch has arrived. A
synchronous load releases its readers when its own data has arrived, which is cycle 143 for A.
In these runs the asynchronous form is never the faster one. Use it to replay a kernel that
really issues a batch of loads and then waits for all of them; in a hand-written TOG, synchronous
loads are simpler. A TOG that signals a semaphore and never waits on it, or waits for another
count, is refused (step 6).

## 3. Many work-items, on one core, two and four

A real kernel is a loop over data tiles. `examples/tiles/tiles_tog.cpp` dispatches one work-item
per cell of an M x N grid, in ordinary C++ loops:

```cpp
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  const int64_t M = n_shape_args > 0 ? shape_args[0] : 1, N = n_shape_args > 1 ? shape_args[1] : 1;
  for (int64_t m = 0; m < M; m++)
    for (int64_t n = 0; n < N; n++) {
      int64_t iv[3] = {m, n, N};
      togsim_dispatch(ctx, body, iv, 3, m * N + n);
    }
}
```

Three things are new:

- **Shape arguments.** `shape_args` holds the integers of `tog_shape.txt`, a file beside `tog.so`.
  Here it contains `2 4`, so the kernel dispatches eight work-items. The same `tog.so` serves
  another grid when the file changes. Without the file `shape_args` is null and `n_shape_args` 0.
- **`iv`.** `togsim_dispatch` copies the `n_iv` integers of `iv` and hands the copy to the body
  later. It is the only way to tell a body which work-item it is: the body here uses it to compute
  the element offset of its own data tile.
- **Cores.** The TOG never names a core. TOGSim binds work-items to the cores round-robin, in
  dispatch order. The last argument of `togsim_dispatch` is a key that names the work-item in
  messages; it does not choose a core.

All eight work-items run compute tile 0, so the cycle table still has one row. They also all use
buffer ids 1 and 2: buffer ids are local to a work-item, so these are sixteen separate buffers.

```bash
examples/run.sh tiles
```

The script replays the kernel on tpuv6e as shipped, which has one core, and on copies of it with
one key changed. The first three:

| Machine | `Total execution cycles` | What the trace shows |
|---|---|---|
| tpuv6e (`num_cores: 1`) | 5021 | Computes are issued at 144, 744, 1344, ...: one at a time, 600 cycles apart. |
| `num_cores: 2` | 2635 | Work-items 0, 2, 4, 6 are on core 0 and 1, 3, 5, 7 on core 1. The first computes start at 142 and 156. |
| `num_cores: 4` | 1487 | Two work-items per core. The first computes start at 176 to 198. |

The run with `num_cores: 2`, where `tile=N` is the work-item's index in dispatch order:

```text
[trace] [1][Core 0][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [1][Core 0][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [2][Core 1][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [2][Core 1][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [142][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [156][Core 1][INST_ISSUED    ][INST_ID=4] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=1)
[trace] [742][Core 0][INST_ISSUED    ][INST_ID=7] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=2)
[trace] [756][Core 1][INST_ISSUED    ][INST_ID=10] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=3)
...
[info] Total execution cycles: 2635
```

Two things to take from these runs:

- **More cores are not proportionally faster.** Two cores take 2635 cycles and four take 1487,
  against 5021 for one. All cores load through the same DRAM and interconnect: on four
  cores the first computes start at 176 to 198, against 144 on one core.
- **Do not rely on the order of work-items.** On one core the computes run in the order
  `tile=` 0, 1, 2, 4, 5, 6, 7, 3. A TOG can order instructions inside a work-item, with buffers,
  and nothing across work-items.

The script's last two runs shrink the scratchpad and halve the DRAM channels;
[Machine configuration](machine.md#changing-the-machine) reads them.

## 4. Two kinds of compute, unit rows and the array split

`examples/gemm/gemm_tog.cpp` is one work-item of a matrix multiply C = A x B, in K = 2 steps:

```cpp
  for (int64_t k = 0; k < K; k++) {
    togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, (uint64_t)(k * T), 2, dims, a_strides, 32, 0, 0, nullptr, 0, a, 1, 0);
    togsim_dma(ctx, TOGSIM_DMA_LOAD, 1, (uint64_t)(k * T * T), 2, dims, b_strides, 32, 0, 0, nullptr, 0, b, 1, 0);
    togsim_compute(ctx, 0, 0, 0, nullptr, a_b_acc, 3, acc, 1);
  }
  togsim_compute(ctx, 1, 0, 0, nullptr, acc, 1, out, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, 2, 0, 2, dims, b_strides, 32, 0, 0, out, 1, nullptr, 0, 0);
```

**Two kinds of compute and an accumulator.** Each step loads a data tile of A and one of B and
runs compute tile 0, a multiply-accumulate. That compute lists buffer 3, the accumulator, in both
`read_bufs` and `write_bufs`: it updates the buffer in place, and because the next step reads
what this step wrote, the two steps run one after the other. A last compute, compute tile 1,
turns the accumulator into the output buffer 4, which is stored. The loads here take data tiles
out of larger tensors: A is 64 x 128 elements, so a data tile of A has `strides = {128, 1}` and
step `k` starts at element offset `k * 64`.

Two `tile_id`s mean two rows in the cycle table and rows for both in the unit table:

| File | Contents |
|---|---|
| `tog_cycles.tsv` | `600<TAB>0`<br>`150<TAB>0` |
| `tog_units.tsv` | `0<TAB>TPUV6_MXU<TAB>rows<TAB>1<TAB>512`<br>`0<TAB>Vpu<TAB>cycles<TAB>1<TAB>50`<br>`1<TAB>Vpu<TAB>cycles<TAB>1<TAB>150` |

**Unit rows and the array split.** So far the unit table only fed the utilization lines. It also
changes how long a compute takes. A row says what one hardware unit did during one run of a
compute tile: here compute tile 0 pushes 512 rows through the unit named `TPUV6_MXU`, the systolic
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
| 512 rows, 2 arrays, 128 lanes (as shipped) | 600 - 512 + 256 = 344 | 150 | 1099 |
| `tog_units_128_rows.tsv`: 128 rows | 600 | 150 | 1611 |
| 512 rows, `num_systolic_array_per_core: 1` | 600 | 150 | 1611 |
| 128 rows, `vpu_num_lanes: 64` | 600 - 128 + 64 = 536 | 150 | 1483 |
| `systolic_array_split: none`, no unit table | 600 | 150 | 1611 |

The first run:

```text
[trace] [161][Core 0][INST_ISSUED    ][INST_ID=2] COMP (compute_type=0 compute_cycle=344 overlapping_cycle=0 sa=-1 tile=0)
[trace] [506][Core 0][INST_ISSUED    ][INST_ID=5] COMP (compute_type=0 compute_cycle=344 overlapping_cycle=0 sa=-1 tile=0)
[trace] [851][Core 0][INST_ISSUED    ][INST_ID=6] COMP (compute_type=0 compute_cycle=150 overlapping_cycle=0 sa=-1 tile=0)
[info] Core [0] : TPUV6_MXU utilization(%): 46.59, active_cycles: 512, idle_cycles: 587
[info] Core [0] : Vpu utilization(%): 22.75, active_cycles: 250, idle_cycles: 849
[info] Total execution cycles: 1099
```

All three lines say `tile=0`, the third too although it runs `tile_id` 1: `tile=N` in the log is
the work-item, and this TOG has one. Tell computes apart by `compute_cycle`. 128 rows fit one
array of 128 lanes, so nothing is spread and the compute keeps its 600 cycles; on a machine with
64 lanes the same 128 rows are split again.

## 5. Several kernels: session mode

`--models_list <file>` replays a command stream instead of one TOG. Each `LAUNCH_KERNEL` line
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
[trace] [142][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[info] [Scheduler 0] Enqueued kernel_id: 1, tog_path: examples/session/k1/tog.so, operation: kernel, request_time_cycles: 0
[trace] [948][Core 0][INST_ISSUED    ][INST_ID=4] COMP (compute_type=0 compute_cycle=644 overlapping_cycle=0 sa=-1 tile=0)
[info] Total execution cycles: 1668
# examples/session/two_partitions.gen.yml: configs/tpuv6e.yml with num_cores: 2, num_partition: 2, partition: {core_0: 0, core_1: 1}
+ build/bin/Simulator --config examples/session/two_partitions.gen.yml --models_list examples/session/commands_two_partitions.txt --log_level trace
[info] [Scheduler 0] Enqueued kernel_id: 0, tog_path: examples/session/k0/tog.so, operation: kernel, request_time_cycles: 0
[info] [Scheduler 1] Enqueued kernel_id: 1, tog_path: examples/session/k1/tog.so, operation: kernel, request_time_cycles: 0
[trace] [145][Core 1][INST_ISSUED    ][INST_ID=4] COMP (compute_type=0 compute_cycle=644 overlapping_cycle=0 sa=-1 tile=0)
[trace] [147][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[info] Total execution cycles: 866
```

- **First run, one partition.** Kernel 1 starts after kernel 0 is done, and each kernel's compute
  follows its own tables: 600 cycles and 128 rows for `k0`, which fit one array of tpuv6e, so 600;
  900 cycles and 512 rows for `k1`, so 900 - 512 + 256 = 644.
- **Second run, two partitions.** A copy of tpuv6e with two cores, one in each partition
  (`num_partition: 2` and its `partition` map), and the kernels run side by side: 866 cycles, each
  compute as in the first run.

In a session the total is followed by one more line, `Wall-clock time for simulation`.

## 6. Mistakes: what TOGSim refuses, and what it does not

`examples/mistakes/` holds two valid TOGs that a `-D<NAME>` flag breaks in one way each, and the
script also feeds a valid TOG wrong inputs and wrong machines.

```bash
examples/run.sh mistakes
```

**Refused.** A refused run exits non-zero with a line that names the cause, before anything is
simulated. For example:

```text
+ g++ -std=c++17 -shared -fPIC -O1 -Iinclude -DWRONG_COUNT examples/mistakes/refused_tog.cpp -o examples/mistakes/WRONG_COUNT.so
+ build/bin/Simulator --config configs/tpuv6e.yml --tog_so examples/mistakes/WRONG_COUNT.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv
[error] [TOGSim] wait expects another element count: work-item 0 (key 0) sem 7 signaled 4096 elements, wait expects 2048
exit 1
```

The script prints 21 refusals; the [refusal table](tog-abi.md#what-gets-a-run-refused) lists each with its
cause and its fix.

**Said, but not refused.** Two mistakes still run to the end with exit 0, and TOGSim names them in
one `[warning]` line while it loads the TOG. The valid TOG of `examples/mistakes/silent_tog.cpp`
takes 840 cycles.

| Mistake | Total | What happens | The warning |
|---|---|---|---|
| `TENSOR_PAST_LIST`: an `arg_id` `tog_tensors.txt` has no line for | 819 | The tensor is placed at address 0, on top of tensor 0: `dram=0x0000000000000000` on the store's trace line. | `its DMAs name arg_id 2, and tog_tensors.txt lists 2 tensor(s): those DMAs are addressed from base 0` |
| a unit table whose rows are all for `tile_id` 1, while the TOG computes `tile_id` 0 | 840 | The compute adds nothing to any unit, and the array split leaves it whole: `TPUV6_MXU utilization(%): 0.00`. | `the unit table has no rows for tile ids 0, which its computes name: those computes add nothing to the units` |

```text
+ build/bin/Simulator --config configs/tpuv6e.yml --tog_so examples/mistakes/TENSOR_PAST_LIST.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv --log_level trace
[warning] [TOGSim] TOG examples/mistakes/TENSOR_PAST_LIST.so: its DMAs name arg_id 2, and tog_tensors.txt lists 2 tensor(s): those DMAs are addressed from base 0
```

**Not said at all.** The dangerous mistake is the one nothing reports. It exits 0 and prints a
total; only the number is wrong.

| Mistake (`-D` flag) | Total | What happens | How to notice |
|---|---|---|---|
| `NULL_STRIDES`: `strides` is a null pointer | 785 | The whole DMA becomes one DRAM request. | `stride=[]` on the trace line; 513 responses instead of 1024. |

**Valid, with another meaning.** `CONSUMER_FIRST` calls the store before the compute that writes
its buffer, so the store drains [data already on the chip](tog-abi.md#data-that-is-already-on-the-chip).
It is issued at cycle 3, an `[info]` line names the buffer, and a 16 KB spad refuses the TOG.

```text
+ build/bin/Simulator --config configs/tpuv6e.yml --tog_so examples/mistakes/CONSUMER_FIRST.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv --log_level trace
[info] [TOGSim] TOG examples/mistakes/CONSUMER_FIRST.so: core 0 holds 16384 bytes for the kernel: buffers 2, which its work-items read without having written them
[trace] [2][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=false ...
[trace] [3][Core 0][INST_ISSUED    ][INST_ID=1] MOVOUT (addr_name=tag1 dram=0x0000000000004000 rank=2 elem_bits=32 async=false ...
[trace] [147][Core 0][INST_ISSUED    ][INST_ID=2] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[info] Core [0] : DMA active_cycles: 32, DMA idle_cycles: 716, DRAM BW: 76.000 GB/s (1024 responses)
[info] Total execution cycles: 748
+ build/bin/Simulator --config examples/mistakes/spad_16kb.gen.yml --tog_so examples/mistakes/CONSUMER_FIRST.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv
[error] [TOGSim] TOG examples/mistakes/CONSUMER_FIRST.so: work-item 0 (key 0) needs 16384 bytes in the spad at once beside the 16384 bytes core 0 holds for the kernel (buffers 2, read without having been written), and a core's spad holds 16384 (core_spad_size_kb: 16)
exit 1
```

So when you bring up a new TOG, read its log once at `--log_level trace`: check that it has no
`[warning]` line, that every compute is there with the cycles you expect, that each DMA has the
address and strides you meant, and that the instructions are issued in the order the data flows.
