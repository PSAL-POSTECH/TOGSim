# The TOG ABI

`include/togsim_runtime.h`, C ABI 16 (`TOGSIM_ABI_VERSION`), is what TOGSim loads;
[togsim.hpp](tog-api.md) lowers every call to it. This page says what the calls model, the buffer
and scratchpad rules, the files and tables, and the refusals.

togsim.hpp exports the four symbols TOGSim looks up: `togsim_kernel` (the TOG's `entry`),
`togsim_spad_buffer_count` and `togsim_spad_buffer_bytes[]` (from `buffers`; 0 for an id that is
no buffer) and `togsim_abi_version`.

## How TOGSim runs a TOG

1. TOGSim opens `tog.so`, checks the ABI version, reads the buffer table and calls `togsim_kernel`
   **once**. Each `dispatch` only registers a work-item (its function, a copy of its arguments,
   its key) and binds it to a core: round-robin, in dispatch order, over the cores of the
   kernel's partition.
2. TOGSim then calls each work-item's body to record its instructions. It calls each body **three
   times** over the run: twice up front, to find the buffers and to plan the spad, and once more
   when it builds the work-item's instructions.
3. The recorded instructions run on the modeled core as [the model](tutorial.md#the-model) says: each as soon
   as the instructions it depends on are done and the resource it needs is free.

Two rules follow, and nothing checks them:

- **A body must be pure.** It may read only its `Item` and arguments, and must make the same calls
  every time. A counter, a static variable or I/O in a body happens three times.
- **Call order is not execution order.** An instruction depends only on instructions that were
  called *before* it in the same work-item and wrote a buffer it reads; a load depends on nothing.
  Write the producer before the consumer: otherwise the consumer reads
  [data already on the chip](#data-that-is-already-on-the-chip). Two instructions that share no
  buffer are not ordered at all.

## DMAs

A DMA's `offset` is **in elements**: its byte address is the tensor's base plus `offset` times the
element's bytes.

`dims` and `strides` describe a data tile inside a larger tensor. A T x T data tile that starts at
row `r`, column `c` of a matrix with `W` columns has `offset = r * W + c`, `dims = {T, T}` and
`strides = {W, 1}`.

**What a DMA costs.** A DMA is one DRAM request for every distinct request-sized block its
elements touch. A request is 32 bytes on every shipped config (the log's `[Config/DRAM]` line
states it), so a 64 x 64 x 4-byte data tile is 512 requests. A core's DMA engine sends one
DMA at a time, loads ahead of stores, at `icnt_injection_ports_per_core` requests per cycle. A
synchronous load is finished `dma_load_ready_cycles` after its last response arrives. An
asynchronous load counts as finished when its last request is sent, and its barrier waits for its
last response and the same `dma_load_ready_cycles`. A store is finished when its last request is
sent; the end of the run (or a `DEVICE_SYNC`) still waits for DRAM to answer and for
`dma_store_ready_cycles` after its requests.
The engine holds each DMA `dma_load_latency_cycles` (a load) or `dma_store_latency_cycles` (a
store) before its first request, so the DMAs pay it one after another, asynchronous ones too.
`dma_load_ready_cycles` does not hold the engine: it goes on to the next DMA, so loads in flight
together wait theirs at the same time.

## Computes

A compute that lists one buffer in both `.reads` and `.writes` updates it in place, which is how
the gemm example chains its accumulate steps.

## Waits

A `wait` (`MEMORY_BAR` in the log) waits for the data of the async loads that signaled its
semaphore since its last wait, and becomes the writer of their buffers: whatever reads one of them
afterwards waits for it.

## Buffers and the scratchpad

Buffer ids do two jobs. They are the edges of the dependency graph, and they are what occupies
the scratchpad.

- **Buffer ids are local to a work-item.** The same id in two work-items is two unrelated
  buffers: no instruction waits for an instruction of another work-item, and each work-item's
  buffers take their own space in the spad.
- Give every buffer a DMA fills or drains, and every [held](#data-that-is-already-on-the-chip)
  buffer, a size in `buffers`.
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
  TOG holds its two buffers together at the compute, 32768 bytes: it runs in a spad of 32 KB
  and is refused in a smaller one.
- A work-item that names more bytes than the spad without holding them together is not refused:
  its loads wait for the spad, and some such runs stall or never end
  ([known limitations](running.md#known-limitations)).

### Data that is already on the chip

A TOG states the transfers that happen. When data is on the chip already, the TOG has no load for
it, and an instruction reads a buffer that nothing in its work-item wrote: a **held buffer**.
TOGSim does not decide that a load can be left out; the emitter of the TOG does.

- A held buffer has the size `buffers` gives its id, and a read of it waits
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

`Desc.indirect` marks a gather (load) or scatter (store): each element moves by an extra offset read
from a file. TOGSim looks for
`<directory of tog.so>/runtime/indirect_access/indirect_index_<key>_<n>.raw`, where `<key>` is the
work-item's dispatch key printed as an unsigned 64-bit number and `<n>` counts that work-item's
indirect DMAs from 0. The file holds one native `uint64` per element of the data tile, each an
offset in elements. If the file is absent or has another size the DMA is modeled as a dense data
tile, with one warning (`[TOGSim] indirect dma without its index dump ...`) and a summary line
(`[TOGSim] indirect dmas: X with their index dump, Y without`).
`tests/spad_footprint/indirect_store_tog.cpp` is a worked example.

## Files beside `tog.so`

TOGSim reads these from the directory of the TOG, by fixed name:

| File | Required | Format | Example |
|---|---|---|---|
| `tog_tensors.txt` | always | One byte size per tensor argument, whitespace-separated, in `ptr` order. | `16384`<br>`16384` |
| `tog_shape.txt` | no | The kernel's shape arguments, read by `shape<N>`. | `2 4` |
| `tog_tensor_shapes.txt` | only if the config sets `tog_tensor_tile` | `rows cols elem_bytes` per tensor argument. | `256 256 4` |
| `runtime/indirect_access/*.raw` | no | Index dumps of indirect DMAs. | |

Tensors are placed in DRAM back to back from address 0 in `ptr` order, each start rounded up to
16 KiB. An empty `tog_tensors.txt` is fine for a TOG without DMAs; a missing one is refused. A DMA
whose `ptr` the file does not list still runs, addressed from base 0, and TOGSim says so once
per TOG in a `[warning]` line ([tutorial step 6](tutorial.md#6-mistakes-what-togsim-refuses-and-what-it-does-not)).

## Where the numbers in the tables come from

TOGSim does not produce the cycle table or the unit table; it only reads them. Something that
models one core at the level of a single compute has to supply them.

- **In the compiler flow** gem5 does: its model of the core runs each kind of compute tile and
  reports its cycles and, for each hardware unit, what the unit took in. That is also why
  the array unit is called what the config's `systolic_array_unit` says: `TPUV6E_MXU` is the name
  vcix's libtpuv6e.so gives it.
- **By hand**, take the numbers from whatever you trust for your core: an RTL or cycle-accurate
  simulation of one compute tile, or an estimate. Every example in `examples/` uses made-up round
  numbers.

## The cycle table

Two whitespace-separated integers per compute tile, `cycle overlapping`. The row number, from 0,
is the `tile_id`, so a table of `n` rows holds `tile_id` 0 to `n - 1`. The second column is
deprecated and has no effect on timing: write 0.

```text
600	0
150	0
```

`--cycle_table <file>` names it; in a session it is `tog_cycles.tsv` in the kernel's directory.
There is no default table. Give it no header line or comments: the reader takes numbers in pairs
until one fails to parse, so a header leaves it without a row. A table without a row suits a TOG
that has no compute.

## The unit table and the array split

Exactly five **tab-separated** fields per line, with no header and no comments. Empty lines are
skipped.

```text
tile_id	unit	unit_of_work	capacity_per_cycle	admitted
```

| Field | Meaning |
|---|---|
| `tile_id` | The compute tile the row describes. |
| `unit` | The name of a hardware unit. Names are yours, except that the systolic array must be called what the config's `systolic_array_unit` says (`TPUV6E_MXU` on tpuv6e). |
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
`Core [0] : Vpu utilization(%): 9.19, active_cycles: 250, idle_cycles: 2469` is 2 x 50 + 150 busy
cycles of 2719. A unit that comes out above 100% ends the run with an error.

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
runs 256 cycles, 256 rows on each of two arrays) and 513 rows do not.

Because the split is on by default and no shipped config turns it off, **every shipped config
needs a unit table that has rows of its array unit**. To run without one, set
`systolic_array_split: none` in the machine config (`examples/minimal.yml` does).

## What gets a run refused

`examples/run.sh mistakes` prints every message of this table. Three kinds of message go to
**stderr**, so capture a log with `2>&1`: lines that start with `togsim:`, `what():` lines, and
`Config file ... does not exist!`.

| Message | Exit | Cause | Fix |
|---|---|---|---|
| `[TOGSim] spad buffer 2 is DMA'd but the TOG states no size for it (togsim_spad_buffer_bytes has 2 entries)` | 1 | A DMA moves a buffer whose id is past the table or has size 0. | Add its size. |
| `togsim.hpp: no TOG registered: derive the TOG's struct from togsim::Tog<itself>` | 1 | The TOG's struct does not derive from `togsim::Tog<itself>` (generated code: no `register_tog`). | Derive it. |
| `togsim.hpp: a work-item returned with a semaphore whose loads it did not wait` | 1 | A load signaled a semaphore that no `wait` followed. | Wait for it, or make the load synchronous. |
| `togsim.hpp: a wait on a semaphore no load signaled since its last wait` | 1 | A `wait` with no load on its semaphore since the last one. | Wait on the loads' semaphore, once per round of loads. |
| `[TOGSim] TOG <so>: work-item 0 (key 0) computes tile id 1, which the cycle table does not hold (it has 1 tile ids, 0 to 0)` | 1 | A `togsim_compute` names a `tile_id` at or beyond the number of rows of the cycle table. | Add the row, or fix the `tile_id`. |
| `[TOGSim] --tog_so needs --cycle_table: the TOG's computes take their cycles from it` | 1 | No `--cycle_table`. | Give the table. |
| `[TOGSim] cycle table <file> cannot be opened: the TOG's computes take their cycles from it` | 1 | The path of the cycle table is wrong. | Fix the path. |
| `[TOGSim] TOG <so>: work-item 0 (key 0) computes tile id 0, which the cycle table does not hold (it has no line)` | 1 | The cycle table has a header line, so no row was read. | Remove the header. |
| `[TOGSim] systolic_array_split (on by default) reads each tile's TPUV6E_MXU rows; give --unit_table, or set systolic_array_split: none` | 1 | No `--unit_table` on a machine with the array split. | Give the table, or turn the split off. |
| `[TOGSim] --unit_table <file> line 1: expected 5 tab-separated fields (...), got 1` | 1 | The line is separated by spaces, or is a comment. | Use tabs and nothing else. |
| `[TOGSim] --unit_table <file> line 1: tile_id 'tile_id' is not a non-negative integer` | 1 | The table has a header line. | Remove it. |
| `[TOGSim] systolic_array_split: the unit table has no array unit TPUV6E_MXU` | 1 | The first unit table that has rows has none for the array unit. | Add a row of it, or turn the split off. |
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

TOGSim has more refusals than a first TOG runs into; all exit 1. Its raw C ABI checks (version
symbol, buffer table, semaphore counts, `ndim`) are ones a TOG built on togsim.hpp cannot fail
([its refusals](tog-api.md#refusals)), except a `.so` built against another ABI:

| Message contains | Cause |
|---|---|
| `togsim: refusing TOG <so>: it was built against TOG ABI 15, TOGSim expects ABI 16; rebuild the TOG` | The TOG's `togsim_abi_version` is another number: it was compiled against another version of the header. |
| `[Simulator] simulation wedged at cycle <n>: work remains but nothing is in flight` | A work-item waits for spad that is never freed, in a way the loader does not see ([known limitations](running.md#known-limitations)). |
| `--unit_table <file>: cannot open` | The unit table is not there. |
| `unit <name> disagrees with line <n> on unit_of_work/capacity` | Two rows of one unit differ in `unit_of_work` or `capacity_per_cycle`. |
| `second row for tile <id> unit <name>` | Two rows for the same `tile_id` and unit. |
| `TOG <so>: work-item <n> (key <k>) computes tile id <id>, whose unit table row admits 128 rows to TPUV6E_MXU, and the unit takes 100 in the tile's 100 cycles (1 per cycle)` | A row of that compute tile admits more than the unit takes in [the tile's cycles](#the-unit-table-and-the-array-split). With the array split the line ends `(1 per cycle on each of 2 arrays; the array split leaves 256 of the cycle table's 512 cycles)`. |
| `exceeds capacity`, then `[TOGSim] --unit_table: a unit ...` | During the run a unit took in more than its capacity allows in the cycles it had, in one statistics interval or in the whole run: a check on the sum of the computes that ran, which the tiles that fit their cycles do not reach. |
| `--unit_table is keyed by one TOG's tile_ids` | `--unit_table` was given with `--models_list`. |
| `no tog.so next to <tog_path>`, `no tog_units.tsv next to <tog_path>`, `cycle table <dir>/tog_cycles.tsv cannot be opened` | A `LAUNCH_KERNEL` whose directory lacks the file. |
| `tog_tensor_tile ...`, `tensor argument <n> is ..., which no tile holds`, `tog_tensors.txt lists <n> tensors, tog_tensor_shapes.txt <m>` | The config sets `tog_tensor_tile` and the layout files do not fit it. |

## Checklist for a TOG generator

Emit the TOG against togsim.hpp ([generated code](tog-api.md#generated-code)), write
[the files beside it](#files-beside-togso) and both tables, with a row for every
`node.id + node.count * variant` a compute names and an array-unit row for each compute that uses
the array. Run it once with `--log_level trace`: no `[warning]` line, and the computes, addresses
and issue order as intended.
