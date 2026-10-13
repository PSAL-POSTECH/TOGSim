# The TOG ABI

The ABI is one header, `include/togsim_runtime.h`, at version 16 (`TOGSIM_ABI_VERSION`). A TOG is a
C++ translation unit that includes it and is compiled to a shared object with
`g++ -std=c++17 -shared -fPIC -I<TOGSim>/include`.

## What a TOG defines and what it calls

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

## How a TOG is replayed

1. TOGSim opens `tog.so`, checks the ABI version, reads the buffer table and calls `togsim_kernel`
   **once**. Each `togsim_dispatch` only registers a work-item (the body function, a copy of `iv`,
   the key) and binds it to a core: round-robin, in dispatch order, over the cores of the
   kernel's partition.
2. TOGSim then calls each work-item's body to record its instructions. It calls each body **three
   times** over the run: twice up front, to find the buffers and to plan the spad, and once more
   when it builds the work-item's instructions.
3. The recorded instructions run on the modeled core as [the model](tutorial.md#the-model) says: each as soon
   as the instructions it depends on are done and the resource it needs is free.

Three rules follow, and none of them is checked for you:

- **A body must be pure.** It may read only `ctx` and its `iv`, and must make the same calls every
  time. A counter, a static variable or I/O in a body happens three times.
- **Call order is not execution order.** An instruction depends only on instructions that were
  called *before* it in the same work-item and wrote a buffer it reads; a load depends on nothing.
  Write the producer before the consumer: otherwise the consumer reads
  [data already on the chip](#data-that-is-already-on-the-chip). Two instructions that share no
  buffer are not ordered at all.
- **Make body functions `static`.** In a session all TOGs are loaded into one process, and a
  second kernel's non-static `body` function resolves to the first kernel's.

## `togsim_dispatch`

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

## `togsim_dma`

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

## `togsim_compute`

```cpp
void togsim_compute(EmitCtx* ctx, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write);
```

| Argument | Meaning |
|---|---|
| `tile_id` | The compute tile: the row of the cycle table, from 0, that gives its cycles, and the key of its unit-table rows. A `tile_id` the cycle table has no row for is refused. |
| `compute_type` | Pass 0. The values 1, 2 and 3 are [deprecated](compatibility.md#deprecated-compute_type-1-2-and-3). |
| `ndim`, `dims` | Not used. Pass `0, nullptr`. |
| `read_bufs`, `write_bufs` | The buffers it reads and writes. They create its dependencies. |

A core runs one compute at a time. A compute that lists the same buffer in both `read_bufs` and
`write_bufs` updates it in place, which is how the gemm example chains its accumulate steps.

## `togsim_memory_barrier`

```cpp
void togsim_memory_barrier(EmitCtx* ctx, uint64_t sem, int64_t expected);
```

It waits for the data of every asynchronous load signaled on `sem` since the previous barrier on
`sem` in this work-item, and becomes the writer of their buffers: whatever reads one of them
afterwards waits for the barrier. `expected` must equal the sum of those loads' element counts
(the product of `dims` each). Semaphore numbers are local to a work-item. A second barrier on a
semaphore with no new loads is accepted if it states the same count again.

## Buffers and the scratchpad

Buffer ids do two jobs. They are the edges of the dependency graph, and they are what occupies
the scratchpad.

- **Buffer ids are local to a work-item.** The same id in two work-items is two unrelated
  buffers: no instruction waits for an instruction of another work-item, and each work-item's
  buffers take their own space in the spad.
- State a size in `togsim_spad_buffer_bytes` for every buffer a DMA fills or drains, and for
  every [held](#data-that-is-already-on-the-chip) buffer. By convention id 0 has size 0 and is
  not used.
- A work-item's `spad_footprint` (printed on its `TILE_SCHEDULED` line) is the sum of the sizes of
  its own distinct buffers, counting only buffers that some DMA of the TOG moves. A buffer that
  only computes touch, like the gemm accumulator, is not counted.
- A work-item's own buffer occupies the spad from the load, or the compute, that first writes it
  until the last of its readers is issued: a store, or a compute that reads it without writing it.
  A compute that reads and writes the same buffer keeps it alive. A buffer that nothing reads
  keeps its bytes until its work-item ends.
- The machine's `core_spad_size_kb` is the spad of one core. A core keeps two work-items in flight
  when two of them fit, else one (`max_dispatch` on the `TILE_SCHEDULED` line). Without the key
  the spad is unlimited.
- `core_spad_size_kb` must hold what a work-item needs **at once**: the most that one of its
  instructions finds in the spad when it is issued, whatever the order of issue. That is the bytes
  the instruction occupies itself (a load its data tile, a compute a buffer it writes first), plus
  the bytes of every buffer that was filled by an instruction it waits for and that it, or an
  instruction that waits for it, reads, and of every buffer nothing reads whose fill does not wait
  for it. An instruction is issued only while that fits, so a work-item that needs more would never end: its TOG is
  [refused](#what-gets-a-run-refused) while it is loaded, before anything is simulated. The hello
  TOG holds its two buffers together at the compute, 32768 bytes: it replays in a spad of 32 KB
  and is refused in a smaller one.
- A work-item that names more bytes than the spad without holding them together is not refused:
  its loads wait for the spad. A fill that its reader does not wait for is not counted. The
  refusal does not cover every run that stops for want of spad: one that depends on the order of
  issue, or on what another work-item holds, is not seen while loading
  ([known limitations](running.md#known-limitations)).

### Data that is already on the chip

A TOG states the transfers that happen. When data is on the chip already, the TOG has no load for
it, and an instruction reads a buffer that nothing in its work-item wrote: a **held buffer**.
TOGSim does not decide that a load can be left out; the emitter of the TOG does.

- A held buffer has the size `togsim_spad_buffer_bytes` states for its id, and a read of it waits
  for nothing.
- The core that runs the work-item holds those bytes once, from the kernel's first work-item there
  until its last one there ends. A later write to the id in the same work-item takes no more.
- A kernel that holds `H` bytes is refused and scheduled as the same kernel without those reads in
  a spad `H` bytes smaller. It has its core to itself: no other kernel's work-items run there
  beside it.
- The spad between kernels is not modelled: a kernel accounts only what it reads as held, while it
  runs. A copy node is a TOG of its own that only loads, launched before the kernels that read the
  data as held.

## Indirect DMAs

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

## Files beside `tog.so`

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
per TOG in a `[warning]` line ([tutorial step 6](tutorial.md#6-mistakes-what-togsim-refuses-and-what-it-does-not)).

## Where the numbers in the tables come from

TOGSim does not produce the cycle table or the unit table; it only reads them. Something that
models one core at the level of a single compute has to supply them.

- **In the compiler flow** gem5 does: its model of the core runs each kind of compute tile and
  reports its cycles and, for each hardware unit, what the unit took in. That is also why
  the array unit is called `SYSTOLIC`, or `TPUV6_MXU` on tpuv6e: they are the names vcix's
  libtpu.so and libtpuv6.so give it.
- **By hand**, take the numbers from whatever you trust for your core: an RTL or cycle-accurate
  simulation of one compute tile, or an estimate. Every example in `examples/` uses made-up round
  numbers.

A unit-table row counts work in the unit's own terms. `unit_of_work` names what is counted (rows
pushed through an array, cycles a vector unit was busy), `admitted` is how many of those the unit
took in during one run of the compute tile, and `capacity_per_cycle` is how many it can take in
per cycle. So `admitted / capacity_per_cycle` is the number of cycles the unit was busy.

## The cycle table

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

## The unit table and the array split

Exactly five **tab-separated** fields per line, with no header and no comments. Empty lines are
skipped.

```text
tile_id	unit	unit_of_work	capacity_per_cycle	admitted
```

| Field | Meaning |
|---|---|
| `tile_id` | The compute tile the row describes. |
| `unit` | The name of a hardware unit. Names are yours, except that the systolic array must be called `SYSTOLIC` (or what the config's `systolic_array_unit` says: `TPUV6_MXU` on tpuv6e). |
| `unit_of_work` | What the unit counts: `rows`, `cycles`, ... A name; it appears in messages. The same on every row of a unit. |
| `capacity_per_cycle` | How many of those the unit can take in per cycle. An integer above 0, the same on every row of a unit. |
| `admitted` | How many the unit took in during **one** run of the compute tile. |

At most one row per `tile_id` and unit. A compute whose `tile_id` has no rows at all still runs:
it adds nothing to any unit, the array split leaves it whole, and TOGSim says so once per TOG in a
`[warning]` line ([tutorial step 6](tutorial.md#6-mistakes-what-togsim-refuses-and-what-it-does-not)). Rows of a `tile_id` that no compute names are ignored.
`--unit_table <file>` names the table; in a session it is `tog_units.tsv` in the kernel's
directory.

The table does two things.

**Utilization.** A unit's busy cycles are `admitted / capacity_per_cycle`, summed over the
computes that ran. The final statistics print one line per unit:
`Core [0] : Vpu utilization(%): 22.75, active_cycles: 250, idle_cycles: 849` is 2 x 50 + 150 busy
cycles of 1099. A unit that comes out above 100% ends the run with an error.

**The array split.** With `systolic_array_split: round_robin`, which is the default, a compute
tile whose array rows exceed `vpu_num_lanes` is spread over several arrays and gets shorter. For a
compute tile of `cc` cycles:

```text
B_sys  = array admitted / array capacity_per_cycle              cycles of array work in the compute
N_eff  = min(num_systolic_array_per_core, max(1, ceil(array admitted / vpu_num_lanes)))
cycles = cc                                                       if N_eff is 1
cycles = clamp(ceil(cc - B_sys + max(B_sys / N_eff, every other unit's admitted / capacity)), 1, cc)
```

Here `array` is the unit `systolic_array_unit` names. With the gemm numbers: `cc` 600, 512 rows, 2
arrays, 128 lanes give `N_eff` 2 and 600 - 512 + max(256, 50) = 344. The array unit's utilization line counts a split compute as
`B_sys / N_eff`, the busy time of one array.

**What a tile's cycles hold.** A unit cannot be busy for longer than the compute runs, so every row
of a compute tile must fit the cycles the tile gets, which are `cycles` above (`cc` without the
split, with `N_eff` 1, or when `cc` is 0):

```text
admitted <= capacity_per_cycle x cycles             every unit but the array
admitted <= capacity_per_cycle x cycles x N_eff     the array unit: each of its N_eff arrays takes its share
```

A TOG with a compute whose `tile_id` has a row above that is
[refused](#what-gets-a-run-refused) while it is loaded, before anything is simulated: 128 rows at
1 per cycle do not fit a tile of 100 cycles on one array, and a tile of 0 cycles admits nothing.
Under the split a tile needs about `cc >= B_sys`: 512 rows fit a `cc` of 512 (the compute then
runs 256 cycles, 256 rows on each of two arrays) and 513 rows do not. The refusal does not depend
on when the compute runs or on `core_stats_print_period_cycles`. Only the `tile_id`s that a
compute names are measured.

Because the split is on by default and no shipped config turns it off, **every shipped config
needs a unit table that has rows of its array unit**. To run without one, set
`systolic_array_split: none` in the machine config (`examples/minimal.yml` does).

## What gets a run refused

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
| `[TOGSim] systolic_array_split (on by default) reads each tile's TPUV6_MXU rows; give --unit_table, or set systolic_array_split: none` | 1 | No `--unit_table` on a machine with the array split. | Give the table, or turn the split off. |
| `[TOGSim] --unit_table <file> line 1: expected 5 tab-separated fields (...), got 1` | 1 | The line is separated by spaces, or is a comment. | Use tabs and nothing else. |
| `[TOGSim] --unit_table <file> line 1: tile_id 'tile_id' is not a non-negative integer` | 1 | The table has a header line. | Remove it. |
| `[TOGSim] systolic_array_split: the unit table has no array unit TPUV6_MXU` | 1 | The table has no row for the array unit. | Add a row of it, or turn the split off. |
| `[TOGSim] <dir>/tog_tensors.txt is absent: the TOG's tensor arguments have no addresses` | 1 | No `tog_tensors.txt` beside `tog.so`. | Create it. |
| `togsim: dlopen failed: valid.so: cannot open shared object file: No such file or directory` | 1 | `--tog_so` has no `/` in it, so the loader searches the library path, not the current directory. | Write `./valid.so`. |
| `[TOGSim] TOG <so>: work-item 0 (key 0) needs 32768 bytes in the spad at once, and a core's spad holds 16384 (core_spad_size_kb: 16)` | 1 | A work-item holds more bytes in the spad [at once](#buffers-and-the-scratchpad) than `core_spad_size_kb`. | Raise it to at least those bytes, or free a buffer before the next is filled. |
| `[TOGSim] TOG <so>: work-item 0 (key 0) needs 16384 bytes in the spad at once beside the 16384 bytes core 0 holds for the kernel (buffers 2, read without having been written), and a core's spad holds 16384 (core_spad_size_kb: 16)` | 1 | The same, beside [held buffers](#data-that-is-already-on-the-chip). | Raise it, or write the producer of those ids before their reader. |
| `Config file <path>/configs/ramulator2_configs/HBM3_TPUv6e.yaml does not exist!` | 255 | A shipped config was copied out of `configs/`; its `../configs/...` path no longer resolves. | Make the path absolute ([how](machine.md#changing-the-machine)). |
| `what():  Config key num_cores not found` | 134 | The machine config lacks a required key. | Add it ([keys](config-keys.md)). |

A message that starts with `togsim: dlopen failed:` and names an undefined symbol means the TOG
calls a function that neither it nor TOGSim defines.

In a session a refusal is the error of the command that met it, `[TOGSim] Error processing command
<n> (type: LAUNCH_KERNEL): ...`. The spad refusal is always the launch's. When the config sets
`core_spad_size_kb` the work-items are built once while the TOG is loaded, so a broken semaphore
contract is the launch's error too; without a spad size it is the error of the `DEVICE_SYNC` that
runs the kernel.

TOGSim has more refusals than a first TOG runs into. These are not in the example; all exit 1:

| Message contains | Cause |
|---|---|
| `re-wait expects another element count` | A second barrier on a semaphore, with no new loads, states another count than the first. |
| `issued tile is not supported format` | A DMA has `ndim` 0, `ndim` above 4, or null `dims`. |
| `[Simulator] simulation wedged at cycle <n>: work remains but nothing is in flight` | A work-item waits for spad that is never freed, in a way the loader does not see ([known limitations](running.md#known-limitations)). |
| `--unit_table <file>: cannot open` | The unit table is not there. |
| `unit <name> disagrees with line <n> on unit_of_work/capacity` | Two rows of one unit differ in `unit_of_work` or `capacity_per_cycle`. |
| `second row for tile <id> unit <name>` | Two rows for the same `tile_id` and unit. |
| `TOG <so>: work-item <n> (key <k>) computes tile id <id>, whose unit table row admits 128 rows to TPUV6_MXU, and the unit takes 100 in the tile's 100 cycles (1 per cycle)` | A row of that compute tile admits more than the unit takes in [the tile's cycles](#the-unit-table-and-the-array-split). With the array split the line ends `(1 per cycle on each of 2 arrays; the array split leaves 256 of the cycle table's 512 cycles)`. |
| `exceeds capacity`, then `[TOGSim] --unit_table: a unit ...` | During the run a unit took in more than its capacity allows in the cycles it had, in one statistics interval or in the whole run: a check on the sum of the computes that ran, which the tiles that fit their cycles do not reach. |
| `--unit_table is keyed by one TOG's tile_ids` | `--unit_table` was given with `--models_list`. |
| `no tog.so next to <tog_path>`, `no tog_units.tsv next to <tog_path>`, `cycle table <dir>/tog_cycles.tsv cannot be opened` | A `LAUNCH_KERNEL` whose directory lacks the file. |
| `unit <name> is not one the first kernel's table has` | In a session, a later kernel's unit table names a new unit. |
| `tog_tensor_tile ...`, `tensor argument <n> is ..., which no tile holds`, `tog_tensors.txt lists <n> tensors, tog_tensor_shapes.txt <m>` | The config sets `tog_tensor_tile` and the layout files do not fit it. |

## Checklist for a TOG generator

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
   one row per `tile_id` and a row of the array unit for every compute tile that uses the array.
8. Replay once with `--log_level trace` and check that there is no `[warning]` line, and the
   computes, addresses and issue order.
