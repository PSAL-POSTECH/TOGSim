# TOGSim API v1.0.0

Every TOG, hand-written or compiler-emitted, includes `include/togsim.hpp`, which lowers each call to
C ABI 16 of `include/togsim_runtime.h` ([The TOG ABI](tog-abi.md#the-tog-abi)). A TOG is C++20 with
no API macros and no `extern "C"`; build it with
`g++ -std=c++20 -shared -fPIC -I<TOGSim>/include my_tog.cpp -o tog.so`. The header's version
constants are `togsim::API_MAJOR`, `API_MINOR`, `API_PATCH` (1, 0, 0), `togsim::API_VERSION`
(`"1.0.0"`) and `togsim::ABI` (16, checked against `TOGSIM_ABI_VERSION`).

## A TOG

A TOG is one struct that derives from `togsim::Tog<itself>`, with a `static constexpr std::array
buffers{...}` and a `void entry(Kernel&) const`.
[`examples/hello/hello_tog.cpp`](tutorial.md#1-load-compute-store) is a complete one.

- `buffers` lists every spad buffer a call names.
- `entry` runs once and dispatches the work-items. A work-item function is a member
  `void f(Item&) const` or `void f(Item&, const Args&) const`, or, in generated code, a free
  function `void f(Item&)` or `void f(Item&, Args)`. Members run on a fresh `X{}`, so a member
  holds nothing across calls.
- A `Kernel` can only shape and dispatch, and an `Item` can only load, store, compute and wait.
- One TOG per `.so`, in one `.cpp`: the header defines the ABI's symbols, so a second `.cpp` that
  includes it does not link.
- TOGSim calls a work-item body three times, so it must be pure, and its calls are ordered only
  by their buffers ([the rules](tog-abi.md#how-togsim-runs-a-tog)).

## Types

| Type | Fields | What it is |
|---|---|---|
| `Elem` | `.bits`, `.kind` | An element type: `u8 i8 i16 f16 bf16 i32 f32 i64 f64`. The ABI sees only `bits`; `kind` tells f16 from bf16 and i32 from f32. `f32(64, 64)` is the bytes of 64 x 64 of them. |
| `Buf` | `.id`, `.bytes` | A spad buffer: its id, below 1024, and its size. 0 bytes is a dependency edge only. A list of buffers is written `{A, B}`. |
| `Dims` | | One to four extents or strides: `{64, 64}`, or `Dims(64, 64)`, the form generated code builds. |
| `Desc` | `.ptr`, `.type`, `.offset`, `.dims`, `.strides`, `.indirect` | A DMA descriptor: `dims` elements of `type` from kernel argument `ptr` (its line of `tog_tensors.txt`), `strides` elements apart, from `offset` elements in. `indirect`: a gather or scatter. |
| `Node` | `.id`, `.count` | A compute node: node `id` of the kernel's `count`. |
| `Semaphore` | | A semaphore that async loads signal, owned by the `Item` that made it. `Semaphore()` is none. |
| `Args` | your own | A work-item's arguments: a struct of `int64_t` fields only, copied into the ABI's `iv`. |
| `Keyed<Args>` | `.key`, `.args` | A dispatch that names its own key. The key names the work-item in messages and in the file names of its [indirect index dumps](tog-abi.md#indirect-dmas); it does not choose a core. `Keyed<void>` has `.key` only. |
| `Kernel` | | What `entry` gets; `shape` and `dispatch` take it. Not copyable. |
| `Item` | | One run of one work-item body. Not copyable. |

## Calls

| Call | Lowers to | What it does |
|---|---|---|
| `load(w, {.src, .dst, .sem, .after})` | `togsim_dma` | DRAM to spad: `src` into buffer `dst`. With `.sem` (default none: synchronous) it is async and signals that semaphore. `.after` (default `{}`) lists buffers it reads too (an indirect load's index) or must follow. |
| `Semaphore s = load_async(w, {.src, .dst, .after})` | `togsim_dma` | One async load on a new semaphore of its own. |
| `Semaphore s = semaphore(w)` | | A new semaphore, ABI sem id 1, 2, ... in the work-item's creation order. |
| `wait(w, {.sem = s})` | `togsim_memory_barrier` | Waits for every load that signaled `s` since its last wait. |
| `compute(w, {.node, .variant, .reads, .writes})` | `togsim_compute` | One compute at [row](#rows-node-and-variant) `node.id + node.count * variant` of the cycle and unit tables. `.variant` defaults to 0, `.reads` and `.writes` to `{}`. |
| `store(w, {.src, .dst, .also})` | `togsim_dma` | Spad to DRAM: buffer `src` into `dst`. `.also` (default `{}`) lists buffers it reads too (an indirect store's index) or must follow. |
| `dispatch<&X::f>(k)`, `dispatch<&X::f>(k, Args{...})` | `togsim_dispatch` | One work-item, under the next key: 0, 1, ... |
| `dispatch<&X::f>(k, {.key = key, .args = Args{...}})`, `{.key = key}` | `togsim_dispatch` | One work-item under `key`. A kernel uses keyed or counted dispatch, not both. |
| `shape<N>(k, {.kernel, .names, .example, .extents})` | | The kernel's `N` shape arguments from `tog_shape.txt`, else `example`, as a `std::array<int64_t, N>`. The first `.extents` (default `N`) must be at least 1, the rest at least 0. `.kernel` and `.names` name them in messages. |
| `extent({.dim, .at, .tile})` | | What is left of a `tile`-long step at `at` on an axis `dim` long. |
| `short_axis({.axis, .extent, .tile})` | | Bit `axis` of a variant: set when the step is short of its tile. |
| `static const bool tog = register_tog<entry>({A, B})` | | Generated code only, in place of `Tog<X>`: registers the free function `void entry(Kernel&)` and the buffer list. |

## Rows: Node and variant

Variant 0 is the node's ordinary row. Hand-written kernels use the others for short last tiles, one
`short_axis` bit per axis ([kernels](kernels.md#the-kernel-library)); a compiler uses them for
gem5's calls: call `c` of node `t` is row `t + count * c`.

## Async loads

A semaphore stays usable after a wait, and later loads may signal it again. Each wait needs a new
signaling load, a work-item returns only with every signaling load waited, and it makes up to 64
semaphores ([example](tutorial.md#2-asynchronous-loads-and-a-barrier)).

## Generated code

Generated code uses the same types and calls. EmitC has no base classes and no designated
initializers for run-time values, so a compiler registers a free entry function and builds
run-time structs with parenthesized initialization, in field order:

```text
static const togsim::Buf BUF1 = {.id = 1, .bytes = 65536};
static const togsim::Node NODE1 = {.id = 1, .count = 3};
  togsim::Desc v34 = togsim::Desc(0, togsim::f32, v31, v32, v33);
  togsim::Load v35 = togsim::Load(v34, BUF2);
  togsim::load(v1, v35);
  togsim::compute(v1, {.node = NODE1, .variant = 1, .reads = {BUF0, BUF1, BUF2}, .writes = {BUF0, BUF3}});
      togsim::Keyed<WorkItemArgs> v36 = togsim::Keyed<WorkItemArgs>(v35, v20);
      togsim::dispatch<tog_work_item>(v1, v36);
static const bool tog = togsim::register_tog<tog_entry>({BUF0, BUF1, BUF2, BUF3});
```

From PyTorchSim's gemm TOG; its work-item and entry are `static void tog_work_item(togsim::Item&,
WorkItemArgs)` and `static void tog_entry(togsim::Kernel&)`, and its keys are program ids.

## Refusals

At run time a broken rule prints `togsim.hpp: <message>` on stderr (`kernel <name>: ...` from
`shape`) and exits 1 before anything is simulated. Messages that explain themselves are not listed:

| Message | Cause |
|---|---|
| `two TOGs registered in one .so` | Two TOG structs, or a struct and a `register_tog`. |
| `a wait on a semaphore no load signaled since its last wait` | A wait with no new load on its semaphore: a first wait before any load, or a second with none between. |
| `a semaphore this work-item did not make with semaphore() or load_async()` | `Semaphore()`, or a `Semaphore` another work-item made. |
| `a work-item without arguments was dispatched with some`, `a work-item was dispatched with other arguments` | The work-item and its `Args` disagree. |

A rule broken at compile time stops the build; these messages need a word:

| Message | Cause |
|---|---|
| `a TOG struct has a member void entry(togsim::Kernel&) const` | `entry` missing, or with another signature. |
| `non-constant condition for static assertion`, quoting the broken rule's `fail_("...")` line | A `buffers` entry has an id outside 0..1023, a duplicate id or a negative size. |
| `a work-item's arguments are a struct of int64_t fields` | `Args` is not an aggregate, or its size or alignment is not that of `int64_t` fields (an `int32_t` field, say). Field types are not checked: a lone `double` passes. |

TOGSim refuses some runs itself: [what gets a run refused](tog-abi.md#what-gets-a-run-refused).
