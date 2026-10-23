# TOGSim API v1.0.0

Every TOG, hand-written or compiler-emitted, includes `include/togsim.hpp`, which lowers each call to
C ABI 17 of `include/togsim_runtime.h` ([The TOG ABI](tog-abi.md#the-tog-abi)). A TOG is C++20 with
no API macros and no `extern "C"`; build it with
`g++ -std=c++20 -shared -fPIC -I<TOGSim>/include my_tog.cpp -o tog.so`. The header's version
constants are `togsim::API_MAJOR`, `API_MINOR`, `API_PATCH` (1, 0, 0), `togsim::API_VERSION`
(`"1.0.0"`) and `togsim::ABI` (17, checked against `TOGSIM_ABI_VERSION`).

## A TOG

A TOG is one struct that derives from `togsim::Tog<itself>`, with a `static constexpr std::array
buffers{...}`, an optional `static constexpr std::array attributes{...}` and a
`void entry(Kernel&) const`.
[`examples/hello/hello_tog.cpp`](tutorial.md#1-load-compute-store) is a complete one.

- `buffers` lists every spad buffer a call names.
- `attributes` lists every [runtime attribute](#runtime-attributes) a work-item reads.
- `entry` runs once and dispatches the work-items. A work-item function is a member
  `void f(Item&) const` or `void f(Item&, const Args&) const`, or, in generated code, a free
  function `void f(Item&)` or `void f(Item&, Args)`. Members run on a fresh `X{}`, so a member
  holds nothing across calls.
- A `Kernel` can only shape and dispatch. An `Item` can load, store, compute and wait, and read
  the runtime attributes the functional run recorded for its key; nothing else reaches it.
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
| `Desc` | `.ptr`, `.type`, `.offset`, `.dims`, `.strides`, `.gather` | A DMA descriptor: `dims` elements of `type` from kernel argument `ptr` (its line of `tog_tensors.txt`), `strides` elements apart, from `offset` elements in. `gather` (default none) makes it a gather or scatter. |
| `Attribute` | `.id`, `.rank` | A [runtime attribute](#runtime-attributes): an `int64_t` scalar (`rank` 0) or array the functional run recorded per work-item key. Its extents are in each record, not here. |
| `Gather` | `.attribute`, `.at`, `.axis`, `.scale` | Where each element of a gather or scatter moves: by `scale` times an entry of `attribute`'s row `at` (a `Dims` of leading indices, default none); `axis` 0 or above: one entry per position on that dim of the tile, `-1` (the default): one per element. `Gather()` names no attribute: a dense tile. |
| `Node` | `.id`, `.count` | A compute node: node `id` of the kernel's `count`. |
| `Semaphore` | | A semaphore that async loads signal, owned by the `Item` that made it. `Semaphore()` is none. |
| `Args` | your own | A work-item's arguments: a struct of `int64_t` fields only, copied into the ABI's `iv`. |
| `Keyed<Args>` | `.key`, `.args` | A dispatch that names its own key. The key names the work-item in messages and selects its [runtime attribute records](tog-abi.md#runtime-attributes); it does not choose a core. `Keyed<void>` has `.key` only. A TOG with attributes dispatches only so. |
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
| `int64_t v = scalar(w, A)` | `togsim_runtime_attribute` | The value of scalar attribute `A` (rank 0) recorded for `w`'s key. |
| `int64_t v = element(w, A, i, ...)` | `togsim_runtime_attribute` | Element `i, ...` of array attribute `A`: one index per rank, each in `[0, extent)`. |
| `int64_t n = extent(w, A, d)` | `togsim_runtime_attribute` | Extent `d` (`0 <= d < rank`) of `A` as `w`'s record has it. |
| `static const bool tog = register_tog<entry>({A, B})` | | Generated code only, in place of `Tog<X>`: registers the free function `void entry(Kernel&)` and the buffer list. `register_tog<entry>({A, B}, {ATTR0, ATTR1})` registers an attribute list too. |

## Rows: Node and variant

Variant 0 is the node's ordinary row. Hand-written kernels use the others for short last tiles, one
`short_axis` bit per axis ([kernels](kernels.md#the-kernel-library)); a compiler uses them for
gem5's calls: call `c` of node `t` is row `t + count * c`.

## Async loads

A semaphore stays usable after a wait, and later loads may signal it again. Each wait needs a new
signaling load, a work-item returns only with every signaling load waited, and it makes up to 64
semaphores ([example](tutorial.md#2-asynchronous-loads-and-a-barrier)).

## Runtime attributes

A work-item can read values the kernel's functional run computed from data: a DMA offset read from
a tensor, a trip count, a block table, the index tile of a gather. The run records them per
program as int64 **records**, keyed by the work-item's dispatch key and the attribute's id; TOGSim
loads them before it simulates ([files and checks](tog-abi.md#runtime-attributes)).

- Declare each as `Attribute{.id, .rank}` in `attributes` (generated code: the second list of
  `register_tog`). Ids are exactly 0 to n - 1, each once, at most 64; ranks are 0 to 8.
- A TOG with attributes dispatches every work-item with `.key`, the key the run recorded it under;
  a gridless one uses `{.key = 0}`.
- Read them with `scalar`, `element` and `extent`, which are free functions because generated code
  cannot call methods. A record of extent 0 is present and has no element; a loop to its extent
  runs no trip.
- A gather or scatter puts the attribute in `Desc.gather`: element `e` of the tile moves by
  `scale * A[at..., i_axis(e)]` (`axis` >= 0), or by `scale * A[at..., e's indices]` (`axis` -1).
  The row `at` names must have `dims[axis]` entries (resp. the tile's dims). The index buffer the
  gather reads still goes in `Load.after` or `Store.also`, for its dependency.
- Reads are pure: a body that reads a record calls the same calls every time.

The array walk of `tests/runtime_attribute/array_loop_tog.cpp`, a trip count and a block table per
work-item:

```text
static constexpr Attribute N{.id = 0, .rank = 0}, BLOCKS{.id = 1, .rank = 1};
static constexpr std::array attributes{N, BLOCKS};
    const int64_t trips = scalar(w, N);
    for (int64_t j = 0; j < trips; ++j) {
      const int64_t block = element(w, BLOCKS, j);
      load(w, {.src = {.ptr = 0, .type = f32, .offset = block * T * T, .dims = {T, T}, .strides = {T, 1}}, .dst = IN});
```

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
WorkItemArgs)` and `static void tog_entry(togsim::Kernel&)`, and its keys are program ids. A TOG
with runtime attributes adds lines of this form (written for this page, not taken from a compiler's
output):

```text
static const togsim::Attribute ATTR0 = {.id = 0, .rank = 0};
  int64_t v12 = togsim::scalar(v1, ATTR0);
  togsim::Desc v40 = togsim::Desc(1, togsim::f32, v38, v32, v33, togsim::Gather(ATTR1, togsim::Dims(v20), 0, 64));
static const bool tog = togsim::register_tog<tog_entry>({BUF0, BUF1}, {ATTR0, ATTR1});
```

## Refusals

At run time a broken rule prints `togsim.hpp: <message>` on stderr (`kernel <name>: ...` from
`shape`) and exits 1 before anything is simulated. Messages that explain themselves are not listed:

| Message | Cause |
|---|---|
| `two TOGs registered in one .so` | Two TOG structs, or a struct and a `register_tog`. |
| `a wait on a semaphore no load signaled since its last wait` | A wait with no new load on its semaphore: a first wait before any load, or a second with none between. |
| `a semaphore this work-item did not make with semaphore() or load_async()` | `Semaphore()`, or a `Semaphore` another work-item made. |
| `a work-item without arguments was dispatched with some`, `a work-item was dispatched with other arguments` | The work-item and its `Args` disagree. |
| `a TOG with runtime attributes dispatches without .key` | A counted `dispatch` in a TOG that declares attributes. |
| `runtime attribute <id> is not declared` | A read of an id the list does not hold, or of another rank than the list gives it. |
| `scalar(w, a) of attribute <id> of rank <r>` | `scalar` of an array. |
| `element(w, a, i...) of attribute <id>: <k> indices for rank <r>` | Another count of indices than the rank. |
| `element(w, a, i...) of attribute <id>: index <v> on dim <d> outside its extent <x>` | An index past the record's extent. |
| `extent(w, a, d) of attribute <id>: dim <d> outside rank <r>` | |
| `a gather on attribute <id> on axis <axis> of a <n>-dim tile` | `axis` is not -1 nor a dim of the tile. |
| `a gather on attribute <id> of rank <r>: <k> indices at and a row of <m> dims on axis <axis>` | `at` and the row do not make up the rank: the row is 1 dim (`axis` >= 0) or the tile's dims (-1). |
| `a gather on attribute <id> at [<at>]: index <v> on dim <d> outside its extent <x>` | `at` past the record's extent. |
| `a gather on attribute <id> at [<at>]: its row holds <x> entries for <y> tile elements on axis <axis>` | The record's row and the tile disagree (with `axis` -1 both are shapes, `[8, 4]`). |

A rule broken at compile time stops the build; these messages need a word:

| Message | Cause |
|---|---|
| `a TOG struct has a member void entry(togsim::Kernel&) const` | `entry` missing, or with another signature. |
| `non-constant condition for static assertion`, quoting the broken rule's `fail_("...")` line | A `buffers` entry has an id outside 0..1023, a duplicate id or a negative size. |
| `a work-item's arguments are a struct of int64_t fields` | `Args` is not an aggregate, or its size or alignment is not that of `int64_t` fields (an `int32_t` field, say). Field types are not checked: a lone `double` passes. |
| `non-constant condition for static assertion`, quoting `fail_("an attribute id outside 0..63")`, `fail_("attribute ids are not 0..n-1 once each")` or `fail_("an attribute of rank outside 0..8")` | An `attributes` entry breaks a [declaration rule](#runtime-attributes). Through `register_tog` the same rules are checked when the `.so` is loaded, and print `togsim.hpp: <rule>`. |

TOGSim refuses some runs itself: [what gets a run refused](tog-abi.md#what-gets-a-run-refused), and
[a record that is missing or not the TOG's](tog-abi.md#what-togsim-checks).

## ABI 16 to 17

API 1.0.0 with ABI 17 adds `Attribute`, `Gather`, `scalar`, `element`, `extent`, `Tog::attributes`
and the second list of `register_tog`, and removes `Desc.indirect`: a gather or scatter names its
offsets through `Desc.gather` instead of TOGSim reading them from index dumps. A TOG that never named
`.indirect` builds unchanged; any TOG must be rebuilt, as ABI 16 is refused. The removal is
incompatible, so by [the version rules](compatibility.md#version-numbers) the API would move; it
was kept at 1.0.0 by decision, and the ABI number is what refuses an old TOG.
