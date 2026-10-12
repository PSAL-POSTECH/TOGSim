# TOGSim

TOGSim tells you how many cycles a kernel takes on a modeled accelerator. Its input is a
tile-operation-graph (TOG): a small shared object whose function calls are the kernel's memory
transfers, computes and waits. A compiler emits a TOG, or you write one by hand in C++. TOGSim
loads it, replays those calls on a machine you describe in a YAML file (cores with systolic arrays
and a vector unit, DRAM, an interconnect), and reports the total cycles, how busy each unit was and
how much DRAM bandwidth the kernel used.

A TOG moves no data and contains no timing. What one compute costs comes from a table you supply.
The calls are the nodes of the graph, and the scratchpad buffers they read and write are its edges.

```
  a compiler, or you by hand
            |
            v
  tog.so            the TOG: calls only
  tog_tensors.txt   bytes of each tensor argument  --+
  tog_cycles.tsv    cycles of each kind of compute   +-->  build/bin/Simulator  -->  Total execution cycles: N
  tog_units.tsv     what each unit did per compute --+            ^                  per-core unit utilization
                                                                  |                  DMA and DRAM bandwidth
                                                   machine config (configs/*.yml)
```

| You want to | Read |
|---|---|
| run something in five minutes | [Quick start](#quick-start) |
| write a TOG by hand, or emit TOGs from your own frontend | [The model](#the-model), [Write your first TOG](#write-your-first-tog), then [The TOG ABI](#the-tog-abi) |
| explore hardware configurations | [Machine configuration](#machine-configuration), with steps [3](#3-many-work-items-on-one-core-two-and-four) and [4](#4-two-kinds-of-compute-unit-rows-and-the-array-split) |

**Contents**

- [Quick start](#quick-start)
- [The model](#the-model)
- [Write your first TOG](#write-your-first-tog)
  - [1. Load, compute, store](#1-load-compute-store)
  - [2. Asynchronous loads and a barrier](#2-asynchronous-loads-and-a-barrier)
  - [3. Many work-items, on one core, two and four](#3-many-work-items-on-one-core-two-and-four)
  - [4. Two kinds of compute, unit rows and the array split](#4-two-kinds-of-compute-unit-rows-and-the-array-split)
  - [5. Several kernels: session mode](#5-several-kernels-session-mode)
  - [6. Mistakes: what TOGSim refuses, and what it does not](#6-mistakes-what-togsim-refuses-and-what-it-does-not)
- [The TOG ABI](#the-tog-abi)
  - [What a TOG defines and what it calls](#what-a-tog-defines-and-what-it-calls)
  - [How a TOG is replayed](#how-a-tog-is-replayed)
  - [`togsim_dispatch`](#togsim_dispatch) · [`togsim_dma`](#togsim_dma) ·
    [`togsim_compute`](#togsim_compute) · [`togsim_memory_barrier`](#togsim_memory_barrier)
  - [Buffers and the scratchpad](#buffers-and-the-scratchpad)
  - [Indirect DMAs](#indirect-dmas)
  - [Files beside `tog.so`](#files-beside-togso)
  - [Where the numbers in the tables come from](#where-the-numbers-in-the-tables-come-from)
  - [The cycle table](#the-cycle-table)
  - [The unit table and the array split](#the-unit-table-and-the-array-split)
  - [What gets a run refused](#what-gets-a-run-refused)
  - [Checklist for a TOG generator](#checklist-for-a-tog-generator)
- [Running TOGSim](#running-togsim)
- [Reading the output](#reading-the-output)
- [Machine configuration](#machine-configuration)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Testing](#testing)
- [What you may depend on](#what-you-may-depend-on)
- [Glossary](#glossary)
- [Repository layout and license](#repository-layout-and-license)

## Quick start

TOGSim builds on Ubuntu 22.04 with GCC 11, `conan` 1.x and `cmake` 3.16 or newer; CI builds with
conan 1.56.0 and cmake 3.26.4. The build needs network access: conan downloads boost and spdlog,
and cmake fetches yaml-cpp and fmt.

**Not on Ubuntu 22.04, or no toolchain on the host?** Start in a container, and run everything
below inside it:

```bash
docker run --rm -it -v "$PWD":"$PWD" -w "$PWD" ubuntu:22.04 bash
```

The directory is mounted at its own path because the build tree records absolute paths. The
container runs as root, so what it creates belongs to root on the host; before you leave the
container, run `chown -R --reference=.. .` in `TOGSim/` to hand the files to the owner of the
directory you started in.

**1. Install the toolchain**, as root:

```bash
apt-get update
apt-get install -y --no-install-recommends g++ make git python3 python3-pip ca-certificates
python3 -m pip install conan==1.56.0 cmake==3.26.4
```

**2. Clone with the submodules** (BookSim2 and Ramulator2 are built with the simulator):

```bash
git clone --recursive https://github.com/PSAL-POSTECH/TOGSim.git
cd TOGSim
```

**3. Build:**

```bash
conan profile new default --detect --force
conan profile update settings.compiler.libcxx=libstdc++ default
mkdir -p build && cd build
conan install .. --build=missing
cmake ..
make -j"$(nproc)"
cd ..
```

conan prints a `WARNING: GCC OLD ABI COMPATIBILITY` banner and the compile prints about 120
warnings from `extern/booksim`; both are expected. The build has succeeded when
`build/bin/Simulator` exists. From an empty container, steps 1 to 3 take between one and two
minutes on a 64-core machine.

**4. Replay a TOG.** A TOG is compiled with plain `g++` against `include/`, then handed to the
simulator with its two tables:

```bash
g++ -std=c++17 -shared -fPIC -O1 -Iinclude examples/hello/hello_tog.cpp -o examples/hello/tog.so
build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml \
  --tog_so examples/hello/tog.so --cycle_table examples/hello/tog_cycles.tsv \
  --unit_table examples/hello/tog_units.tsv | tail -n 1
```

```text
[2026-10-12 14:29:12.768] [info] Total execution cycles: 1102
```

That line is the result: the kernel took 1102 cycles of the core clock. It is the last line of the
log of a `--tog_so` run, and the whole log of this run is 2,050 lines, so read a log through
`tail` or `grep`.

`configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml` is the machine every example uses; this page
calls it **tpuv6e**. `examples/run.sh` builds and replays the examples of this page and prints the
lines of each log that matter:

```bash
examples/run.sh hello     # one example; with no argument, all six
```

[Step 1](#1-load-compute-store) reads its output line by line.

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
| DMA engine | one DMA at a time, loads ahead of stores; it sends `icnt_injection_ports_per_core` DRAM requests per cycle (16 on tpuv6e) |
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

Six terms carry the rest of this page; the [glossary](#glossary) has the others.

| Term | Meaning |
|---|---|
| TOG | A shared object (`tog.so`) whose calls are one kernel's DMAs, computes and barriers. |
| work-item | One `togsim_dispatch` call: the unit of parallelism. The log calls it a tile: `TILE_SCHEDULED`, `tile=N`. |
| DMA, data tile | A DMA is a transfer between DRAM and a core's scratchpad: a load (`MOVIN` in the log) or a store (`MOVOUT`). The block of a tensor it moves is a data tile. |
| buffer | A region of the scratchpad, named by a small integer id. Buffers are what DMAs fill and drain, and what orders instructions. |
| compute tile, `tile_id` | A kind of compute. Every `togsim_compute` call names one by its `tile_id`, and many calls can name the same one. |
| cycle table, unit table | Per `tile_id`: how many cycles the compute takes, and what each hardware unit did during it. You supply both. |

## Write your first TOG

Each step below is a directory under `examples/` with a TOG source, the files it needs and
`expected.txt`, the output of `examples/run.sh <name>`. The tests replay every example against its
`expected.txt`, and check that each log line and source line quoted in a fenced block of this page
is still a line of those files and that the refusal table lists what the mistakes example prints.
The numbers in the other tables and in prose were copied from those files by hand.

### 1. Load, compute, store

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

[The TOG ABI](#the-tog-abi) has the full reference of each call.

Three small files stand beside the source:

| File | Contents | Meaning |
|---|---|---|
| `tog_tensors.txt` | `16384`<br>`16384` | The TOG has two tensor arguments of 16384 bytes each. |
| `tog_cycles.tsv` | `600<TAB>0` | Compute tile 0 takes 600 cycles. |
| `tog_units.tsv` | `0<TAB>Systolic<TAB>rows<TAB>1<TAB>256`<br>`0<TAB>Vpu<TAB>cycles<TAB>1<TAB>100` | During one run of compute tile 0 the `Systolic` unit took in 256 rows at 1 row per cycle, and the `Vpu` unit was busy for 100 cycles. |

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
+ build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml --tog_so examples/hello/tog.so --cycle_table examples/hello/tog_cycles.tsv --unit_table examples/hello/tog_units.tsv --log_level trace
[trace] [1][Core 0][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [2][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=false ...
[trace] [310][Core 0][INST_FINISHED  ][INST_ID=0] MOVIN (addr_name=tag0 tile=0)
[trace] [311][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [911][Core 0][INST_FINISHED  ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [912][Core 0][INST_ISSUED    ][INST_ID=2] MOVOUT (addr_name=tag1 dram=0x0000000000004000 rank=2 elem_bits=32 async=false ...
[trace] [944][Core 0][INST_FINISHED  ][INST_ID=2] MOVOUT (addr_name=tag1 tile=0)
[info] Total execution cycles: 1102
```

Each trace line is `[cycle][Core c][TAG][INST_ID=n] OPCODE (fields)`:

| Cycle | Line | What happened |
|---|---|---|
| 1 | `TILE_SCHEDULED spad_footprint=32768` | The work-item was handed to core 0. Its footprint is the two buffers, 2 x 16384 bytes. `max_dispatch=2`: this core keeps two work-items in flight. |
| 2 | `INST_ISSUED` `MOVIN` | The load starts. 16384 bytes are 512 DRAM requests of 32 bytes. |
| 310 | `INST_FINISHED` `MOVIN` | The last response arrived. |
| 311 | `INST_ISSUED` `COMP` `compute_cycle=600` | The compute starts, with the 600 cycles of the cycle table. |
| 911 | `INST_FINISHED` `COMP` | 311 + 600. |
| 912 | `INST_ISSUED` `MOVOUT` `dram=0x...4000` | The store starts. Tensor 1 sits at 0x4000, after the 16384 bytes of tensor 0. |
| 944 | `INST_FINISHED` `MOVOUT` | The last write request was sent: 512 requests at 16 per cycle. |
| 1102 | `Total execution cycles` | The simulation ends when DRAM has answered the last write. |

In a trace line `tile=0` is the index of the work-item, not a `tile_id`
([all fields](docs/output-lines.md#trace-lines)). The script's later runs print one instruction's
events, the final statistics ([Reading the output](#reading-the-output)) and a whole log on a
machine without a unit table.

**A kernel of your own.** Copy the example and create the three files. The unit table must be
tab-separated, so `printf` is safer than an editor that turns tabs into spaces:

```bash
mkdir -p my_kernel
cp examples/hello/hello_tog.cpp my_kernel/my_tog.cpp
printf '16384\n16384\n' > my_kernel/tog_tensors.txt
printf '600\t0\n' > my_kernel/tog_cycles.tsv
printf '0\tSystolic\trows\t1\t256\n0\tVpu\tcycles\t1\t100\n' > my_kernel/tog_units.tsv
g++ -std=c++17 -shared -fPIC -O1 -Iinclude my_kernel/my_tog.cpp -o my_kernel/tog.so
build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml \
  --tog_so my_kernel/tog.so --cycle_table my_kernel/tog_cycles.tsv \
  --unit_table my_kernel/tog_units.tsv | tail -n 1
```

```text
[2026-10-12 14:29:13.020] [info] Total execution cycles: 1102
```

Change the 600 in `tog_cycles.tsv` to 1000 and the total becomes 1502.

### 2. Asynchronous loads and a barrier

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
[trace] [34][Core 0][ASYNC_DMA_ISSUE][INST_ID=0] MOVIN (ASYNC subgraph_id=0 addr_name=tag0 ...
[trace] [66][Core 0][ASYNC_DMA_ISSUE][INST_ID=1] MOVIN (ASYNC subgraph_id=0 addr_name=tag1 ...
[trace] [67][Core 0][INST_ISSUED    ][INST_ID=2] MEMORY_BAR (addr_name=tag0 ...
[trace] [310][Core 0][DRAM_RESP_DONE ][INST_ID=0] MOVIN (ASYNC subgraph_id=0 addr_name=tag0 ...
[trace] [450][Core 0][DRAM_RESP_DONE ][INST_ID=1] MOVIN (ASYNC subgraph_id=0 addr_name=tag1 ...
[trace] [451][Core 0][INST_FINISHED  ][INST_ID=2] MEMORY_BAR (addr_name=tag0 ...
[trace] [451][Core 0][INST_ISSUED    ][INST_ID=3] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [1051][Core 0][INST_FINISHED  ][INST_ID=3] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
...
[info] Total execution cycles: 1273
```

The loads have sent their requests at cycles 34 and 66 (`ASYNC_DMA_ISSUE`), so the barrier is
issued at 67. It finishes at 451, one cycle after the data of the second load is there
(`DRAM_RESP_DONE` at 450), and the compute starts in the same cycle.

**Which form to write.** The other three runs rebuild the TOG with `-DSYNC` (both loads
synchronous, no barrier) and with `-DTWO_STEPS` (a first compute that reads only operand A):

| TOG | Asynchronous loads and a barrier | Synchronous loads |
|---|---|---|
| one compute reads both operands | 1273 | 1273 |
| `-DTWO_STEPS`: a first compute reads only A | 1874, first compute at 451 | 1735, first compute at 311 |

A barrier holds every reader of its loads' buffers until the whole batch has arrived. A
synchronous load releases its readers when its own data has arrived, which is cycle 310 for A.
In these runs the asynchronous form is never the faster one. Use it to replay a kernel that
really issues a batch of loads and then waits for all of them; in a hand-written TOG, synchronous
loads are simpler. A TOG that signals a semaphore and never waits on it, or waits for another
count, is refused (step 6).

### 3. Many work-items, on one core, two and four

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
| tpuv6e (`num_cores: 1`) | 5301 | Computes are issued at 311, 911, 1511, ...: one at a time, 600 cycles apart. |
| `num_cores: 2` | 3201 | Work-items 0, 2, 4, 6 are on core 0 and 1, 3, 5, 7 on core 1. The first computes start at 435 and 443. |
| `num_cores: 4` | 1971 | Two work-items per core. The first computes start at 428 to 433. |

The run with `num_cores: 2`, where `tile=N` is the work-item's index in dispatch order:

```text
[trace] [1][Core 0][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [1][Core 0][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [2][Core 1][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [2][Core 1][TILE_SCHEDULED ] spad_footprint=32768 max_dispatch=2
[trace] [435][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[trace] [443][Core 1][INST_ISSUED    ][INST_ID=4] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=1)
[trace] [1035][Core 0][INST_ISSUED    ][INST_ID=7] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=2)
[trace] [1043][Core 1][INST_ISSUED    ][INST_ID=10] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=3)
...
[info] Total execution cycles: 3201
```

Two things to take from these runs:

- **More cores are not proportionally faster.** Two cores take 3201 cycles and four take 1971,
  against 5301 for one. All cores load through the same DRAM and interconnect, and the first
  compute starts later than the 311 of one core.
- **Do not rely on the order of work-items.** On one core the computes run in the order
  `tile=` 0, 1, 2, 4, 5, 6, 7, 3. A TOG can order instructions inside a work-item, with buffers,
  and nothing across work-items.

The script's last two runs shrink the scratchpad and halve the DRAM channels;
[Machine configuration](#changing-the-machine) reads them.

### 4. Two kinds of compute, unit rows and the array split

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
| `tog_units.tsv` | `0<TAB>Systolic<TAB>rows<TAB>1<TAB>512`<br>`0<TAB>Vpu<TAB>cycles<TAB>1<TAB>50`<br>`1<TAB>Vpu<TAB>cycles<TAB>1<TAB>150` |

**Unit rows and the array split.** So far the unit table only fed the utilization lines. It also
changes how long a compute takes. A row says what one hardware unit did during one run of a
compute tile: here compute tile 0 pushes 512 rows through the unit named `Systolic`, the systolic
array, at one row per cycle, so 512 of its 600 cycles are array work; `Vpu`, the vector unit, is
busy for 50 cycles.

A core can have several systolic arrays (`num_systolic_array_per_core`, 2 on tpuv6e), and one
array takes `vpu_num_lanes` rows (256 on tpuv6e) before a compute tile is split. With 512 rows
TOGSim spreads the array work over both arrays, 256 rows each, and the compute takes
600 - 512 + 256 = 344 cycles ([the formula](#the-unit-table-and-the-array-split)).

```bash
examples/run.sh gemm
```

The script runs that, then changes one thing at a time:

| Run | Compute tile 0 | Compute tile 1 | `Total execution cycles` |
|---|---|---|---|
| 512 rows, 2 arrays, 256 lanes (as shipped) | 600 - 512 + 256 = 344 | 150 | 1399 |
| `tog_units_256_rows.tsv`: 256 rows | 600 | 150 | 1910 |
| 512 rows, `num_systolic_array_per_core: 1` | 600 | 150 | 1910 |
| 256 rows, `vpu_num_lanes: 128` | 600 - 256 + 128 = 472 | 150 | 1654 |
| `systolic_array_split: none`, no unit table | 600 | 150 | 1910 |

The first run:

```text
[trace] [336][Core 0][INST_ISSUED    ][INST_ID=2] COMP (compute_type=0 compute_cycle=344 overlapping_cycle=0 sa=-1 tile=0)
[trace] [681][Core 0][INST_ISSUED    ][INST_ID=5] COMP (compute_type=0 compute_cycle=344 overlapping_cycle=0 sa=-1 tile=0)
[trace] [1026][Core 0][INST_ISSUED    ][INST_ID=6] COMP (compute_type=0 compute_cycle=150 overlapping_cycle=0 sa=-1 tile=0)
[info] Core [0] : Systolic utilization(%): 36.60, active_cycles: 512, idle_cycles: 887
[info] Core [0] : Vpu utilization(%): 17.87, active_cycles: 250, idle_cycles: 1149
[info] Total execution cycles: 1399
```

All three lines say `tile=0`, the third too although it runs `tile_id` 1: `tile=N` in the log is
the work-item, and this TOG has one. Tell computes apart by `compute_cycle`. 256 rows fit one
array of 256 lanes, so nothing is spread and the compute keeps its 600 cycles; on a machine with
128 lanes the same 256 rows are split again.

### 5. Several kernels: session mode

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
+ build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml --models_list examples/session/commands.txt --log_level trace
[info] [Scheduler 0] Enqueued kernel_id: 0, tog_path: examples/session/k0/tog.so, operation: kernel, request_time_cycles: 0
[trace] [311][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[info] [Scheduler 0] Enqueued kernel_id: 1, tog_path: examples/session/k1/tog.so, operation: kernel, request_time_cycles: 0
[trace] [1355][Core 0][INST_ISSUED    ][INST_ID=4] COMP (compute_type=0 compute_cycle=644 overlapping_cycle=0 sa=-1 tile=0)
[info] Total execution cycles: 2189
+ build/bin/Simulator --config configs/systolic_ws_128x128_c2_simple_noc_tpuv3_partition.yml --models_list examples/session/commands_two_partitions.txt --log_level trace
[info] [Scheduler 0] Enqueued kernel_id: 0, tog_path: examples/session/k0/tog.so, operation: kernel, request_time_cycles: 0
[info] [Scheduler 1] Enqueued kernel_id: 1, tog_path: examples/session/k1/tog.so, operation: kernel, request_time_cycles: 0
[trace] [160][Core 0][INST_ISSUED    ][INST_ID=1] COMP (compute_type=0 compute_cycle=472 overlapping_cycle=0 sa=-1 tile=0)
[trace] [161][Core 1][INST_ISSUED    ][INST_ID=4] COMP (compute_type=0 compute_cycle=644 overlapping_cycle=0 sa=-1 tile=0)
[info] Total execution cycles: 908
```

- **First run, one partition.** Kernel 1 starts after kernel 0 is done, and each kernel's compute
  follows its own tables: 600 cycles and 256 rows for `k0`, which fit one array of tpuv6e, so 600;
  900 cycles and 512 rows for `k1`, so 900 - 512 + 256 = 644.
- **Second run, two partitions.** The machine has two cores, one in each partition, and the
  kernels run side by side: 908 cycles. Its arrays have 128 lanes, so `k0`'s 256 rows are split
  here, 600 - 256 + 128 = 472, and `k1`'s 512 rows are spread over the two arrays as before.

In a session the total is followed by one more line, `Wall-clock time for simulation`.

### 6. Mistakes: what TOGSim refuses, and what it does not

`examples/mistakes/` holds two valid TOGs that a `-D<NAME>` flag breaks in one way each, and the
script also feeds a valid TOG wrong inputs and wrong machines.

```bash
examples/run.sh mistakes
```

**Refused.** A refused run exits non-zero with a line that names the cause, before anything is
simulated. For example:

```text
+ g++ -std=c++17 -shared -fPIC -O1 -Iinclude -DWRONG_COUNT examples/mistakes/refused_tog.cpp -o examples/mistakes/WRONG_COUNT.so
+ build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml --tog_so examples/mistakes/WRONG_COUNT.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv
[error] [TOGSim] wait expects another element count: work-item 0 (key 0) sem 7 signaled 4096 elements, wait expects 2048
exit 1
```

The script prints 21 refusals; the [refusal table](#what-gets-a-run-refused) lists each with its
cause and its fix.

**Said, but not refused.** Two mistakes still run to the end with exit 0, and TOGSim names them in
one `[warning]` line while it loads the TOG. The valid TOG of `examples/mistakes/silent_tog.cpp`
takes 1102 cycles.

| Mistake | Total | What happens | The warning |
|---|---|---|---|
| `TENSOR_PAST_LIST`: an `arg_id` `tog_tensors.txt` has no line for | 1102 | The tensor is placed at address 0, on top of tensor 0: `dram=0x0000000000000000` on the store's trace line. | `its DMAs name arg_id 2, and tog_tensors.txt lists 2 tensor(s): those DMAs are addressed from base 0` |
| a unit table whose rows are all for `tile_id` 1, while the TOG computes `tile_id` 0 | 1102 | The compute adds nothing to any unit, and the array split leaves it whole: `Systolic utilization(%): 0.00`. | `the unit table has no rows for tile ids 0, which its computes name: those computes add nothing to the units` |

```text
+ build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml --tog_so examples/mistakes/TENSOR_PAST_LIST.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv --log_level trace
[warning] [TOGSim] TOG examples/mistakes/TENSOR_PAST_LIST.so: its DMAs name arg_id 2, and tog_tensors.txt lists 2 tensor(s): those DMAs are addressed from base 0
```

**Not said at all.** The dangerous mistakes are the ones nothing reports. Each of these exits 0
and prints a total; only the number is wrong.

| Mistake (`-D` flag) | Total | What happens | How to notice |
|---|---|---|---|
| `NULL_STRIDES`: `strides` is a null pointer | 970 | The whole DMA becomes one DRAM request. | `stride=[]` on the trace line; 513 responses instead of 1024. |
| `CONSUMER_FIRST`: the store is called before the compute that writes its buffer | 912 | The store depends on nothing and is issued at cycle 3. | The order of the `INST_ISSUED` lines. |

```text
+ build/bin/Simulator --config configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml --tog_so examples/mistakes/CONSUMER_FIRST.so --cycle_table examples/mistakes/tog_cycles.tsv --unit_table examples/mistakes/tog_units.tsv --log_level trace
[trace] [2][Core 0][INST_ISSUED    ][INST_ID=0] MOVIN (addr_name=tag0 dram=0x0000000000000000 rank=2 size=[64,64] stride=[64,1] elem_bits=32 async=false ...
[trace] [3][Core 0][INST_ISSUED    ][INST_ID=1] MOVOUT (addr_name=tag1 dram=0x0000000000004000 rank=2 elem_bits=32 async=false ...
[trace] [311][Core 0][INST_ISSUED    ][INST_ID=2] COMP (compute_type=0 compute_cycle=600 overlapping_cycle=0 sa=-1 tile=0)
[info] Core [0] : DMA active_cycles: 64, DMA idle_cycles: 848, DRAM BW: 125.000 GB/s (1024 responses)
[info] Total execution cycles: 912
```

So when you bring up a new TOG, read its log once at `--log_level trace`: check that it has no
`[warning]` line, that every compute is there with the cycles you expect, that each DMA has the
address and strides you meant, and that the instructions are issued in the order the data flows.

## The TOG ABI

The ABI is one header, `include/togsim_runtime.h`, at version 16 (`TOGSIM_ABI_VERSION`). A TOG is a
C++ translation unit that includes it and is compiled to a shared object with
`g++ -std=c++17 -shared -fPIC -I<TOGSim>/include`.

### What a TOG defines and what it calls

A TOG defines three symbols and gets a fourth from the header:

| Symbol | Who writes it | What it is |
|---|---|---|
| `togsim_kernel(ctx, shape_args, n_shape_args)` | you, `extern "C"` | The entry point. |
| `togsim_spad_buffer_count` | you, `extern "C" const int32_t` | The number of entries in the next array. |
| `togsim_spad_buffer_bytes[]` | you, `extern "C" const int64_t[]` | The byte size of each spad buffer id; 0 for an id that is not a buffer. |
| `togsim_abi_version` | the header | The ABI version the TOG was compiled against. Including the header defines it. |

The `extern "C"` on the two constants is required: without it a C++ `const` has no external linkage
and TOGSim cannot find it.

A TOG calls four functions, which TOGSim provides at load time:

| Call | What it models |
|---|---|
| `togsim_dispatch(ctx, fn, iv, n_iv, key)` | Registers one work-item. Called from `togsim_kernel`. |
| `togsim_dma(ctx, ...)` | One load or store. Called from a work-item body. |
| `togsim_compute(ctx, ...)` | One compute. Called from a work-item body. |
| `togsim_memory_barrier(ctx, sem, expected)` | A wait for asynchronous loads. Called from a work-item body. |

`ctx` is an opaque handle; pass it through.

### How a TOG is replayed

1. TOGSim opens `tog.so`, checks the ABI version, reads the buffer table and calls `togsim_kernel`
   **once**. Each `togsim_dispatch` only registers a work-item (the body function, a copy of `iv`,
   the key) and binds it to a core: round-robin, in dispatch order, over the cores of the
   kernel's partition.
2. TOGSim then calls each work-item's body to record its instructions. It calls each body **three
   times** over the run: twice up front, to find the buffers and to plan the spad, and once more
   when it builds the work-item's instructions.
3. The recorded instructions run on the modeled core as [the model](#the-model) says: each as soon
   as the instructions it depends on are done and the resource it needs is free.

Three rules follow, and none of them is checked for you:

- **A body must be pure.** It may read only `ctx` and its `iv`, and must make the same calls every
  time. A counter, a static variable or I/O in a body happens three times.
- **Call order is not execution order.** An instruction depends only on instructions that were
  called *before* it in the same work-item and wrote a buffer it reads; a load depends on nothing.
  Write the producer before the consumer. Two instructions that share no buffer are not ordered at
  all.
- **Make body functions `static`.** In a session all TOGs are loaded into one process, and a
  second kernel's non-static `body` function resolves to the first kernel's.

### `togsim_dispatch`

```cpp
typedef void (*togsim_tile_fn)(EmitCtx* ctx, int64_t* iv, int32_t n_iv);
```

`togsim_dispatch(ctx, fn, iv, n_iv, key)` registers one work-item. The type of a body is named
`togsim_tile_fn` because the header, like the log, calls a work-item a tile.

| Argument | Meaning |
|---|---|
| `fn` | The body of the work-item. |
| `iv`, `n_iv` | `n_iv` integers, copied at the call; the body receives the copy. May be null with `n_iv` 0. |
| `key` | An integer that names the work-item in error messages and in the file names of indirect index dumps. It does not select a core. |

### `togsim_dma`

```cpp
void togsim_dma(EmitCtx* ctx, int32_t dir, int32_t arg_id,
                uint64_t offset, int32_t ndim, const int64_t* dims,
                const int64_t* strides, int32_t elem_bits,
                int32_t is_async, uint64_t sem,
                const int64_t* read_bufs, int32_t n_read,
                const int64_t* write_bufs, int32_t n_write,
                int32_t indirect);
```

| Argument | Meaning |
|---|---|
| `dir` | `TOGSIM_DMA_LOAD` (0, DRAM to spad, `MOVIN`) or `TOGSIM_DMA_STORE` (1, spad to DRAM, `MOVOUT`). |
| `arg_id` | Which tensor argument: the line of `tog_tensors.txt`, from 0. |
| `offset` | Where the data tile starts inside that tensor, **in elements**. The byte address is the tensor's base plus `offset * (elem_bits / 8)`. |
| `ndim`, `dims` | The data tile's extent in elements, 1 to 4 dimensions. |
| `strides` | The distance in elements between neighbours along each dimension. Always pass it. |
| `elem_bits` | The width of an element in bits: a multiple of 8. The division above is an integer division, so a width below 8 drops the offset. |
| `is_async` | 0, or 1 for an asynchronous load. Pass 0 for a store. |
| `sem` | The semaphore an asynchronous load signals. Ignored otherwise. |
| `read_bufs`, `n_read` | For a store: the buffer it drains first, then any buffer it only depends on. For a load: null, 0. |
| `write_bufs`, `n_write` | For a load: the buffers it fills. For a store: null, 0. |
| `indirect` | 0, or 1 for a gather or scatter (see [indirect DMAs](#indirect-dmas)). |

`dims` and `strides` describe a data tile inside a larger tensor. A T x T data tile that starts at
row `r`, column `c` of a matrix with `W` columns has `offset = r * W + c`, `dims = {T, T}` and
`strides = {W, 1}`. The gemm example loads its data tiles of A, a 64 x 128 matrix, with
`strides = {128, 1}`; the hello data tile is the whole tensor, so its row stride equals its width.

**What a DMA costs.** A DMA is one DRAM request for every distinct request-sized block its
elements touch. A request is 32 bytes on every shipped config (the log's `[Config/DRAM]` line
states it), so the 64 x 64 x 4-byte data tile above is 512 requests. A core's DMA engine sends one
DMA at a time, loads ahead of stores, at `icnt_injection_ports_per_core` requests per cycle. A
synchronous load is finished when its last response arrives. An asynchronous load counts as
finished when its last request is sent. A store is finished when its last request is sent, and
the run still waits for DRAM to answer.

### `togsim_compute`

```cpp
void togsim_compute(EmitCtx* ctx, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write);
```

| Argument | Meaning |
|---|---|
| `tile_id` | The compute tile: the row of the cycle table, from 0, that gives its cycles, and the key of its unit-table rows. A `tile_id` the cycle table has no row for is refused. |
| `compute_type` | Pass 0. The values 1, 2 and 3 are [deprecated](docs/compatibility.md#deprecated-compute_type-1-2-and-3). |
| `ndim`, `dims` | Not used. Pass `0, nullptr`. |
| `read_bufs`, `write_bufs` | The buffers it reads and writes. They create its dependencies. |

A core runs one compute at a time. A compute that lists the same buffer in both `read_bufs` and
`write_bufs` updates it in place, which is how the gemm example chains its accumulate steps.

### `togsim_memory_barrier`

```cpp
void togsim_memory_barrier(EmitCtx* ctx, uint64_t sem, int64_t expected);
```

It waits for the data of every asynchronous load signaled on `sem` since the previous barrier on
`sem` in this work-item, and becomes the writer of their buffers: whatever reads one of them
afterwards waits for the barrier. `expected` must equal the sum of those loads' element counts
(the product of `dims` each). Semaphore numbers are local to a work-item. A second barrier on a
semaphore with no new loads is accepted if it states the same count again.

### Buffers and the scratchpad

Buffer ids do two jobs. They are the edges of the dependency graph, and they are what occupies
the scratchpad.

- **Buffer ids are local to a work-item.** The same id in two work-items is two unrelated
  buffers: no instruction waits for an instruction of another work-item, and each work-item's
  buffers take their own space in the spad.
- State a size in `togsim_spad_buffer_bytes` for every buffer a DMA fills or drains. By convention
  id 0 has size 0 and is not used.
- A work-item's `spad_footprint` (printed on its `TILE_SCHEDULED` line) is the sum of the sizes of
  the distinct buffers it names, counting only buffers that some DMA of the TOG moves. A buffer
  that only computes touch, like the gemm accumulator, is not counted.
- A buffer occupies the spad from the load, or the compute, that first writes it until its last
  reader is issued: a store, or a compute that reads it without writing it. A compute that reads
  and writes the same buffer keeps it alive.
- The machine's `core_spad_size_kb` is the spad of one core. A core keeps two work-items in flight,
  or one if a work-item's footprint is more than half the spad (`max_dispatch` on the
  `TILE_SCHEDULED` line). Without the key the spad is unlimited.
- `core_spad_size_kb` must be at least the largest footprint: the hello TOG, with a footprint of
  32768 bytes, replays in a spad of 32 KB. In a smaller one a run cannot finish
  ([known limitations](#known-limitations)).

### Indirect DMAs

`indirect` 1 marks a gather (load) or scatter (store): each element moves by an extra offset read
from a file. TOGSim looks for
`<directory of tog.so>/runtime/indirect_access/indirect_index_<key>_<n>.raw`, where `<key>` is the
work-item's dispatch key printed as an unsigned 64-bit number and `<n>` counts that work-item's
indirect DMAs from 0. The file holds one native `uint64` per element of the data tile, each an
offset in elements. If the file is absent or has another size the DMA is modeled as a dense data
tile, with one warning (`[TOGSim] indirect dma without its index dump ...`) and a summary line
(`[TOGSim] indirect dmas: X with their index dump, Y without`). An indirect store lists its index
buffer after the buffer it drains, in `read_bufs`. `tests/spad_footprint/indirect_store_tog.cpp` is
a worked example.

### Files beside `tog.so`

TOGSim reads these from the directory of the TOG, by fixed name:

| File | Required | Format | Example |
|---|---|---|---|
| `tog_tensors.txt` | always | One byte size per tensor argument, whitespace-separated, in `arg_id` order. | `16384`<br>`16384` |
| `tog_shape.txt` | no | The integers handed to `togsim_kernel` as `shape_args`. | `2 4` |
| `tog_tensor_shapes.txt` | only if the config sets `tog_tensor_tile` | `rows cols elem_bytes` per tensor argument. | `256 256 4` |
| `runtime/indirect_access/*.raw` | no | Index dumps of indirect DMAs. | |

Tensors are placed in DRAM back to back from address 0 in `arg_id` order, each start rounded up to
16 KiB. An empty `tog_tensors.txt` is fine for a TOG without DMAs; a missing one is refused. A DMA
whose `arg_id` the file does not list still runs, addressed from base 0, and TOGSim says so once
per TOG in a `[warning]` line (step 6).

### Where the numbers in the tables come from

TOGSim does not produce the cycle table or the unit table; it only reads them. Something that
models one core at the level of a single compute has to supply them.

- **In the compiler flow** gem5 does: its model of the core runs each kind of compute tile and
  reports its cycles and, for each hardware unit, what the unit took in. That is also why
  the array unit is called `Systolic`: it is gem5's name for it.
- **By hand**, take the numbers from whatever you trust for your core: an RTL or cycle-accurate
  simulation of one compute tile, or an estimate. Every example on this page uses made-up round
  numbers.

A unit-table row counts work in the unit's own terms. `unit_of_work` names what is counted (rows
pushed through an array, cycles a vector unit was busy), `admitted` is how many of those the unit
took in during one run of the compute tile, and `capacity_per_cycle` is how many it can take in
per cycle. So `admitted / capacity_per_cycle` is the number of cycles the unit was busy.

### The cycle table

Two whitespace-separated integers per compute tile, `cycle overlapping`. The row number, from 0,
is the `tile_id`, so a table of `n` rows holds `tile_id` 0 to `n - 1`. The second column is
deprecated and has no effect on timing: write 0.

```text
600	0
150	0
```

`--cycle_table <file>` names it; in a session it is `tog_cycles.tsv` in the kernel's directory.
There is no default table:

- A run without a cycle table, or with one that cannot be opened, is refused.
- A TOG whose `togsim_compute` names a `tile_id` the table does not hold is refused while it is
  loaded, before anything is simulated.
- Do not give the table a header line or comments. The reader takes numbers in pairs until one
  fails to parse, so a header leaves the table without a row, and the first compute is refused
  with `the cycle table does not hold (it has no line)`.
- A table without a row is a correct table for a TOG that has no compute.

### The unit table and the array split

Exactly five **tab-separated** fields per line, with no header and no comments. Empty lines are
skipped.

```text
tile_id	unit	unit_of_work	capacity_per_cycle	admitted
```

| Field | Meaning |
|---|---|
| `tile_id` | The compute tile the row describes. |
| `unit` | The name of a hardware unit. Names are yours, except that the systolic array must be called `Systolic` (or what the config's `systolic_array_unit` says). |
| `unit_of_work` | What the unit counts: `rows`, `cycles`, ... A name; it appears in messages. The same on every row of a unit. |
| `capacity_per_cycle` | How many of those the unit can take in per cycle. An integer above 0, the same on every row of a unit. |
| `admitted` | How many the unit took in during **one** run of the compute tile. |

At most one row per `tile_id` and unit. A compute whose `tile_id` has no rows at all still runs:
it adds nothing to any unit, the array split leaves it whole, and TOGSim says so once per TOG in a
`[warning]` line (step 6). Rows of a `tile_id` that no compute names are ignored.
`--unit_table <file>` names the table; in a session it is `tog_units.tsv` in the kernel's
directory.

The table does two things.

**Utilization.** A unit's busy cycles are `admitted / capacity_per_cycle`, summed over the
computes that ran. The final statistics print one line per unit:
`Core [0] : Vpu utilization(%): 17.87, active_cycles: 250, idle_cycles: 1149` is 2 x 50 + 150 busy
cycles of 1399. A unit that comes out above 100% ends the run with an error.

**The array split.** With `systolic_array_split: round_robin`, which is the default, a compute
tile whose array rows exceed `vpu_num_lanes` is spread over several arrays and gets shorter. For a
compute tile of `cc` cycles:

```text
B_sys  = Systolic admitted / Systolic capacity_per_cycle          cycles of array work in the compute
N_eff  = min(num_systolic_array_per_core, max(1, ceil(Systolic admitted / vpu_num_lanes)))
cycles = cc                                                       if N_eff is 1
cycles = clamp(ceil(cc - B_sys + max(B_sys / N_eff, every other unit's admitted / capacity)), 1, cc)
```

With the gemm numbers: `cc` 600, 512 rows, 2 arrays, 256 lanes give `N_eff` 2 and
600 - 512 + max(256, 50) = 344. The `Systolic` utilization line counts a split compute as
`B_sys / N_eff`, the busy time of one array.

Because the split is on by default and no shipped config turns it off, **every shipped config
needs a unit table that has `Systolic` rows**. To run without one, set
`systolic_array_split: none` in the machine config (`examples/minimal.yml` does).

### What gets a run refused

`examples/run.sh mistakes` prints every message of this table. Three kinds of message go to
**stderr**, so capture a log with `2>&1`: lines that start with `togsim:`, `what():` lines, and
`Config file ... does not exist!`.

| Message | Exit | Cause | Fix |
|---|---|---|---|
| `togsim: refusing TOG <so>: it was built against TOG ABI 15, TOGSim expects ABI 16; rebuild the TOG` | 1 | The TOG's `togsim_abi_version` is another number: it was compiled against another version of the header. | Recompile it against this tree's `include/`. |
| `togsim: refusing TOG <so>: it embeds no togsim_abi_version (built without togsim_runtime.h); TOGSim expects ABI 16` | 1 | The TOG does not include the header. | Include it instead of declaring the calls yourself. |
| `togsim: dlsym togsim_kernel failed: <so>: undefined symbol: togsim_kernel` | 1 | The TOG defines no `togsim_kernel`. | Define it, `extern "C"`. |
| `togsim: refusing TOG <so>: it states no spad buffer sizes (togsim_spad_buffer_count / togsim_spad_buffer_bytes)` | 1 | The buffer table is missing or not `extern "C"`. | Define both constants. |
| `[TOGSim] spad buffer 2 is DMA'd but the TOG states no size for it (togsim_spad_buffer_bytes has 2 entries)` | 1 | A DMA moves a buffer whose id is past the table or has size 0. | Add its size. |
| `[TOGSim] wait expects another element count: work-item 0 (key 0) sem 7 signaled 4096 elements, wait expects 2048` | 1 | `expected` is not the sum of the loads' element counts. | Pass the sum. |
| `[TOGSim] wait on a semaphore never signaled: work-item 0 (key 0) sem 8 signaled 0 elements, wait expects 4096` | 1 | The barrier names a semaphore no asynchronous load of this work-item used. | Use the loads' `sem`. |
| `[TOGSim] unwaited load batch: work-item 0 (key 0) sem 7 signaled 4096 elements, wait expects none` | 1 | An asynchronous load has no barrier after it. | Add the barrier, or make the load synchronous. |
| `[TOGSim] TOG <so>: work-item 0 (key 0) computes tile id 1, which the cycle table does not hold (it has 1 tile ids, 0 to 0)` | 1 | A `togsim_compute` names a `tile_id` at or beyond the number of rows of the cycle table. | Add the row, or fix the `tile_id`. |
| `[TOGSim] --tog_so needs --cycle_table: the TOG's computes take their cycles from it` | 1 | No `--cycle_table`. | Give the table. |
| `[TOGSim] cycle table <file> cannot be opened: the TOG's computes take their cycles from it` | 1 | The path of the cycle table is wrong. | Fix the path. |
| `[TOGSim] TOG <so>: work-item 0 (key 0) computes tile id 0, which the cycle table does not hold (it has no line)` | 1 | The cycle table has a header line, so no row was read. | Remove the header. |
| `[TOGSim] systolic_array_split (on by default) reads each tile's Systolic rows; give --unit_table, or set systolic_array_split: none` | 1 | No `--unit_table` on a machine with the array split. | Give the table, or turn the split off. |
| `[TOGSim] --unit_table <file> line 1: expected 5 tab-separated fields (...), got 1` | 1 | The line is separated by spaces, or is a comment. | Use tabs and nothing else. |
| `[TOGSim] --unit_table <file> line 1: tile_id 'tile_id' is not a non-negative integer` | 1 | The table has a header line. | Remove it. |
| `[TOGSim] systolic_array_split: the unit table has no array unit Systolic` | 1 | The table has no row for the array unit. | Add a `Systolic` row, or turn the split off. |
| `[TOGSim] <dir>/tog_tensors.txt is absent: the TOG's tensor arguments have no addresses` | 1 | No `tog_tensors.txt` beside `tog.so`. | Create it. |
| `togsim: dlopen failed: valid.so: cannot open shared object file: No such file or directory` | 1 | `--tog_so` has no `/` in it, so the loader searches the library path, not the current directory. | Write `./valid.so`. |
| `[Simulator] simulation wedged at cycle 52837: work remains but nothing is in flight -- the per-core spad (core_spad_size_kb) is too small to hold a kernel's working set. Increase it.` | 1 | `core_spad_size_kb` is smaller than a work-item's `spad_footprint`. | Raise it to at least the footprint. |
| `Config file <path>/configs/ramulator2_configs/HBM3_TPUv6e.yaml does not exist!` | 255 | A shipped config was copied out of `configs/`; its `../configs/...` path no longer resolves. | Make the path absolute ([how](#changing-the-machine)). |
| `what():  Config key num_cores not found` | 134 | The machine config lacks a required key. | Add it ([keys](docs/config-keys.md)). |

A message that starts with `togsim: dlopen failed:` and names an undefined symbol means the TOG
calls a function that neither it nor TOGSim defines.

TOGSim has more refusals than a first TOG runs into. These are not in the example; all exit 1:

| Message contains | Cause |
|---|---|
| `re-wait expects another element count` | A second barrier on a semaphore, with no new loads, states another count than the first. |
| `issued tile is not supported format` | A DMA has `ndim` 0, `ndim` above 4, or null `dims`. |
| `--unit_table <file>: cannot open` | The unit table is not there. |
| `unit <name> disagrees with line <n> on unit_of_work/capacity` | Two rows of one unit differ in `unit_of_work` or `capacity_per_cycle`. |
| `second row for tile <id> unit <name>` | Two rows for the same `tile_id` and unit. |
| `exceeds capacity`, then `[TOGSim] --unit_table: a unit ...` | A unit took in more than its capacity allows in the cycles it had. |
| `--unit_table is keyed by one TOG's tile_ids` | `--unit_table` was given with `--models_list`. |
| `no tog.so next to <tog_path>`, `no tog_units.tsv next to <tog_path>`, `cycle table <dir>/tog_cycles.tsv cannot be opened` | A `LAUNCH_KERNEL` whose directory lacks the file. |
| `unit <name> is not one the first kernel's table has` | In a session, a later kernel's unit table names a new unit. |
| `tog_tensor_tile ...`, `tensor argument <n> is ..., which no tile holds`, `tog_tensors.txt lists <n> tensors, tog_tensor_shapes.txt <m>` | The config sets `tog_tensor_tile` and the layout files do not fit it. |

### Checklist for a TOG generator

If you emit TOGs from another frontend, this is the whole contract:

1. Emit one C++ file that includes `togsim_runtime.h` and defines `togsim_kernel` and the two
   buffer-table constants, all `extern "C"`. Make every body function `static`.
2. In `togsim_kernel`, call `togsim_dispatch` once per work-item, passing the loop indices in `iv`.
3. In each body, emit producers before consumers, pass explicit `strides`, use `compute_type` 0
   and an `elem_bits` that is a multiple of 8, and follow every batch of asynchronous loads with a
   barrier that states their element count. Number buffers per work-item; the same ids can serve
   every work-item.
4. Number the distinct kinds of compute from 0 and use that number as `tile_id`.
5. Compile with `g++ -std=c++17 -shared -fPIC -I<TOGSim>/include`.
6. Write `tog_tensors.txt` (and `tog_shape.txt` if the kernel reads shape arguments) beside the
   `.so`.
7. Write the cycle table, one row per `tile_id` and no header, and the unit table, with at least
   one row per `tile_id` and a `Systolic` row for every compute tile that uses the array.
8. Replay once with `--log_level trace` and check that there is no `[warning]` line, and the
   computes, addresses and issue order.

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
| 1 | A refusal of [the tables above](#what-gets-a-run-refused), an unknown flag, no `--config`, a config that cannot be read, neither `--tog_so` nor `--models_list`, a `LAUNCH_KERNEL` that fails. |
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
with the total. [docs/output-lines.md](docs/output-lines.md) walks through a whole log line by
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
  zero cycles is not counted; it is reported as `skipped inst_count`. All computes of a TOG
  written as this page shows are under `Vector`.
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
`MOVOUT`, `COMP` and `MEMORY_BAR`. [docs/output-lines.md](docs/output-lines.md) explains every
field of a trace line, the periodic and DRAM statistics, why a log is long, and which lines a
script may rely on.

## Machine configuration

A machine config is a flat YAML file. `examples/minimal.yml` has every required key and depends on
no other file:

```yaml
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
```

`configs/` ships 19 machines named `systolic_ws_<lanes>x<lanes>_c<cores>_<interconnect>_<machine>`:
`<lanes>` is `vpu_num_lanes`, `<cores>` is `num_cores`, `<interconnect>` is `simple_noc` or
`booksim`, and `<machine>` selects the clock, the arrays per core and the DRAM file. So
`systolic_ws_256x256_c1_simple_noc_tpuv6e.yml`, tpuv6e on this page, has 256 lanes, one core and
the simple interconnect. [docs/config-keys.md](docs/config-keys.md) lists every key with its
default, and all 19 configs.

### The knobs

| Question | Keys | Effect |
|---|---|---|
| How many cores, and how are kernels spread over them? | `num_cores`; `num_partition`, `partition: {core_<i>: <id>}` | A kernel's work-items go round-robin over the cores of its partition. All cores are in partition 0 unless `partition` says otherwise. |
| How fast is the core clock? | `core_freq_mhz` | Totals are in cycles of this clock. DRAM and the interconnect tick at their own `dram_freq_mhz` and `icnt_freq_mhz`. |
| How many systolic arrays, how wide? | `num_systolic_array_per_core` (default 1), `vpu_num_lanes` | Used by the array split: how many arrays a compute tile's rows can be spread over, and how many rows one array takes before the compute is split. |
| Is the array split on? | `systolic_array_split`: `round_robin` (default) or `none`; `systolic_array_unit` (default `Systolic`) | With `round_robin` a unit table is required and computes get shorter (step 4). |
| How large is the scratchpad? | `core_spad_size_kb` (default 0: unlimited) | Limits the buffers alive on a core, and whether one or two work-items are in flight. |
| Which DRAM model? | `dram_type`: `ramulator2` with `ramulator_config_path`, or `simple` with `dram_latency`; `dram_channels` | Ramulator2 models the device its file describes. `simple` answers every request after a fixed latency, optionally under a bandwidth cap. |
| Is there an L2 cache in front of DRAM? | `l2d_type`: `nocache` (default) or `datacache` with `l2d_config` | A TOG marks no address cacheable. The totals of the hello, tiles, gemm and async_barrier examples on tpuv6e are the same with either value. |
| Which interconnect between cores and DRAM? | `icnt_type`: `simple` with `icnt_latency_cycles`, or `booksim2` with `booksim_config_path`; `icnt_injection_ports_per_core` (default 1) | The ports per core are how many DRAM requests a core can send per cycle. |
| How chatty is the log? | `core_stats_print_period_cycles`, `dram_stats_print_period_cycles` | 0 turns the periodic blocks off. They do not change the timing. |

Keys TOGSim does not read are ignored without a message, so check the spelling of a key that seems
to have no effect.

**The array size is not a key.** The `256x256` in a file name matches that config's
`vpu_num_lanes: 256`; TOGSim reads no other array dimension. What a compute costs comes from the
cycle table, and the array reaches the simulator only through `num_systolic_array_per_core`,
`vpu_num_lanes` and the unit table. To model another array, change those keys and supply the two
tables for it.

### Changing the machine

The shipped configs name their Ramulator2 and BookSim2 files relative to themselves
(`../configs/ramulator2_configs/...`), so a copy made elsewhere exits 255. Make that path absolute
while you change a key:

```bash
g++ -std=c++17 -shared -fPIC -O1 -Iinclude examples/tiles/tiles_tog.cpp -o examples/tiles/tog.so
sed -e "s#: \.\./configs/#: $PWD/configs/#" -e 's/^num_cores: .*/num_cores: 2/' \
  configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml > build/two_cores.yml
build/bin/Simulator --config build/two_cores.yml --tog_so examples/tiles/tog.so \
  --cycle_table examples/tiles/tog_cycles.tsv --unit_table examples/tiles/tog_units.tsv | tail -n 1
```

```text
[2026-10-12 14:29:13.593] [info] Total execution cycles: 3201
```

The `sed` changes a key the base config has. A key it lacks, such as `systolic_array_split` on
every shipped config, is appended instead: `echo 'systolic_array_split: none' >> build/my.yml`.

**Sweeping a knob** is the same command in a loop. The tiles kernel on one, two and four cores:

```bash
for n in 1 2 4; do
  sed -e "s#: \.\./configs/#: $PWD/configs/#" -e "s/^num_cores: .*/num_cores: $n/" \
    configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml > build/cores_$n.yml
  build/bin/Simulator --config build/cores_$n.yml --tog_so examples/tiles/tog.so \
    --cycle_table examples/tiles/tog_cycles.tsv --unit_table examples/tiles/tog_units.tsv \
    | awk -v n=$n '/Total execution cycles/ { print "num_cores", n, $NF }'
done
```

```text
num_cores 1 5301
num_cores 2 3201
num_cores 4 1971
```

Before and after, for the knobs the examples turn (all on tpuv6e, one key changed):

| Example | Key changed | Before | After | What the trace shows |
|---|---|---|---|---|
| tiles (8 work-items) | `num_cores: 1` to `2` | 5301 | 3201 | Four work-items per core. |
| tiles | `num_cores: 1` to `4` | 5301 | 1971 | Two work-items per core. |
| tiles | `core_spad_size_kb: 131072` to `48` | 5301 | 7690 | `max_dispatch=1`: a work-item needs 32768 bytes, more than half the spad, so only one is in flight. |
| tiles | `dram_channels: 32` to `16` | 5301 | 5676 | The first compute starts at 433 instead of 311. |
| gemm (512 rows) | `num_systolic_array_per_core: 2` to `1` | 1399 | 1910 | `compute_cycle=600` instead of 344. |
| gemm (256 rows) | `vpu_num_lanes: 256` to `128` | 1910 | 1654 | `compute_cycle=472` instead of 600. |
| gemm | `systolic_array_split: round_robin` to `none` | 1399 | 1910 | `compute_cycle=600`, and the built-in unit lines. |

`examples/run.sh` makes these copies with its `machine` function and prints one
`# <copy>: <base> with <key>` line for each.

## Troubleshooting

A run that ends with a message: look the message up in the
[refusal tables](#what-gets-a-run-refused). The symptoms below come without one.

| Symptom | Likely cause |
|---|---|
| The log shows only `[TOGSim] the TOG failed to load or replay`, or the run ends with `Aborted` and no reason | The reason went to stderr. Add `2>&1`. |
| The total is too small | A consumer called before its producer, or null `strides` (step 6). |
| A unit's utilization is lower than expected, or a compute is not split | The unit table has no rows for that `tile_id`. Look for the `[warning]` line that says so. |
| A compute takes fewer cycles than the cycle table says | The array split shortened it. `compute_cycle` in the trace is the value after the split. |
| The run never ends | `core_spad_size_kb` is smaller than a work-item's `spad_footprint` ([known limitations](#known-limitations)). |
| The second kernel of a session behaves like the first | Its body function is not `static`. |
| A changed config key has no effect | Misspelled keys are ignored. Compare with the echo at the top of the log. |
| A DMA goes to a wrong address | `offset` is in elements, not bytes; an `arg_id` past `tog_tensors.txt` is placed at address 0, with a `[warning]` line. Read `dram=` on its trace line. |
| The log is thousands of lines | Read it through `tail -n 1` or `grep`. Most of a short log is Ramulator2's dump per DRAM channel, which no key turns off ([why](docs/output-lines.md#periodic-and-final-statistics)). |
| The binary does not start after the build tree was moved | It finds `build/lib/libramulator.so` by an absolute path. Build where you run. |

## Known limitations

Behaviour of the simulator that a user runs into. Each was reproduced with this tree.

- **A spad smaller than a work-item's footprint cannot finish.** If no other work-item is queued
  for that core, TOGSim stops with `simulation wedged at cycle N: ...` and exit 1 (the hello TOG in
  16 KB on tpuv6e, at cycle 52837). If more work-items are queued, the run never ends: the tiles
  example in 16 KB was still running when it was stopped after 20 seconds, with a log of more
  than 70,000 lines on tpuv6e, and on `examples/minimal.yml`, which prints no periodic statistics,
  with nothing after `======Start Simulation=====`.
- **An unreadable `--models_list` file ends in exit 136**, not in a clean refusal: the log says
  `[TOGSim] Failed to open trace file: <path>` and the process dies printing the statistics of a
  run of zero cycles.
- **Null `strides` are not "contiguous".** The header comment says so, but the DMA becomes a
  single DRAM request (step 6).
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

## Testing

```bash
python3 tests/run.py
```

It runs every case against `build/`, prints one `PASS` or `FAIL` line per case, and ends with a
line like `ALL PASS (100 cases in 7.4s)`; it exits 1 if a case failed. The cases need `g++` and
`python3`.
[tests/README.md](tests/README.md) says what each directory pins, how to run some of the cases,
and how to add one.

## What you may depend on

A compiler, a driver or a script may depend on five things, and on nothing else in this
repository: the C ABI of `include/togsim_runtime.h`; the files beside `tog.so` and the two tables;
the command line and exit statuses; the output lines listed in
[docs/output-lines.md](docs/output-lines.md#lines-a-script-may-rely-on); and the machine config
keys listed in [docs/config-keys.md](docs/config-keys.md).
[docs/compatibility.md](docs/compatibility.md) states the contract and how another repository
pins this one.

## Glossary

| Term | Meaning |
|---|---|
| tile-operation-graph (TOG) | A shared object (`tog.so`) whose calls are one kernel's DMAs, computes and barriers. It holds control flow only. |
| kernel | One operation to simulate: one TOG with one entry point, `togsim_kernel`. |
| replay | TOGSim running a TOG's calls on the modeled machine. |
| work-item | One `togsim_dispatch` call: a body function plus its loop indices, bound to one core. The unit of parallelism. The log calls it a tile. |
| body | The function of a work-item; its calls are the work-item's instructions. |
| instruction | One `togsim_dma`, `togsim_compute` or `togsim_memory_barrier` call of a body. |
| compute tile, `tile_id` | A kind of compute: a row of the cycle table and the key of its unit-table rows. |
| data tile | The block of a tensor that one DMA moves, given by `offset`, `dims` and `strides`. |
| tensor argument | An array of the kernel that lives in DRAM. A TOG names it by `arg_id`; `tog_tensors.txt` gives its size. |
| DMA | A transfer between DRAM and a core's scratchpad: a load (`MOVIN` in the log) or a store (`MOVOUT`). |
| scratchpad (spad), buffer | The memory of a core that DMAs fill and drain. A TOG names its regions, the buffers, by small integer ids. |
| semaphore, barrier | An asynchronous load signals a semaphore; a barrier (`MEMORY_BAR`) waits for the data of the loads signaled on it. |
| cycle table | `tile_id` to cycles: what each compute tile costs. You supply it. |
| unit table | What each hardware unit (the systolic array, the vector unit, ...) took in during one run of each compute tile. You supply it. |
| unit, unit of work, admitted | A hardware unit named in the unit table; what it counts (rows, cycles); how many of those it took in. |
| lanes | `vpu_num_lanes`: the rows one systolic array takes before a compute tile is split over several arrays. |
| systolic array split | Spreading a compute tile's array work over the arrays of a core, which shortens the compute. On by default. |
| machine config | The YAML file given to `--config`. tpuv6e is `configs/systolic_ws_256x256_c1_simple_noc_tpuv6e.yml`. |
| partition | A set of cores with its own queue of kernels. A kernel runs on the cores of one partition. |
| session, command stream | A run of several kernels, launched by the lines of the file given to `--models_list`. |
| trace | The per-instruction log of a replay, printed at `--log_level trace`. |
| `max_dispatch` | On a `TILE_SCHEDULED` line: how many work-items that core keeps in flight, 1 or 2. |

## Repository layout and license

| Path | What it holds |
|---|---|
| `include/togsim_runtime.h` | The TOG C ABI. The only header a TOG includes. |
| `include/`, `src/` | The simulator. |
| `configs/` | Machine configs and the Ramulator2 / BookSim2 files they point at. |
| `examples/` | The TOGs of this page, `run.sh`, and a self-contained machine config. |
| `docs/` | Reference: [config keys](docs/config-keys.md), [output lines](docs/output-lines.md), [compatibility](docs/compatibility.md). |
| `extern/booksim`, `extern/ramulator2` | Submodules, built with the simulator. |
| `tests/` | Hand-written TOGs, the cases that replay them, and their runner ([tests/README.md](tests/README.md)). |

MIT license; see [LICENSE](LICENSE). The submodules carry their own terms: `extern/ramulator2` in
its `LICENSE` file, `extern/booksim` in the header of each source file.
