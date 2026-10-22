# TOGSim API v1.0.0

Every TOG is written against one header, `include/togsim.hpp`: by hand, in `kernels/`, and by a
compiler such as PyTorchSim's. It lowers each call to the C ABI 16 of `include/togsim_runtime.h`
([The TOG ABI](tog-abi.md#the-tog-abi)), which is what TOGSim loads. A TOG is C++20, uses no
macros of the API and writes no `extern "C"`:

```text
g++ -std=c++20 -shared -fPIC -I<TOGSim>/include my_tog.cpp -o tog.so
```

The header states its version as constants: `togsim::API_MAJOR`, `API_MINOR`, `API_PATCH` (1, 0,
0), `togsim::API_VERSION` (`"1.0.0"`) and `togsim::ABI` (16, checked against
`TOGSIM_ABI_VERSION`). Its limits are constants too: `togsim::MAX_SPAD_BUFFERS` (1024 spad buffer
ids) and `togsim::SEMAPHORES` (64 semaphores per work-item).

## A TOG

A TOG is one struct that derives from `togsim::Tog<itself>`. `examples/hello/hello_tog.cpp`:

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

- `buffers` lists every spad buffer. Loading the `.so` fills the ABI's buffer table from it.
- `entry` runs once and dispatches the work-items. A work-item function is a member
  `void f(Item&) const` or `void f(Item&, const Args&) const`, or, in generated code, a free
  function `void f(Item&)` or `void f(Item&, Args)`. Members run on a fresh `X{}`, so a member
  holds nothing across calls.
- A `Kernel` can only dispatch, and an `Item` can only load, store, compute and wait.
- One TOG per `.so`, in one `.cpp`. The header defines `togsim_kernel` and the buffer table, so
  a second `.cpp` that includes it does not link. Every other
  function of the TOG has internal linkage, so two TOGs in one session cannot resolve to each
  other's functions.

A work-item function is still called three times per replay, and its calls are still ordered
only by their buffers ([how a TOG is replayed](tog-abi.md#how-a-tog-is-replayed)).

## Types

| Type | Fields | What it is |
|---|---|---|
| `Elem` | `.bits`, `.kind` | An element type: `u8 i8 i16 f16 bf16 i32 f32 i64 f64`. The ABI sees only `bits`; `kind` tells f16 from bf16 and i32 from f32. `f32(64, 64)` is the bytes of 64 x 64 of them. |
| `Buf` | `.id`, `.bytes` | A spad buffer: its id, below 1024, and its size. 0 bytes is a dependency edge only. A list of buffers is written `{A, B}`. |
| `Dims` | | One to four extents or strides: `{64, 64}`, or `Dims(64, 64)`, the form generated code builds. |
| `Desc` | `.ptr`, `.type`, `.offset`, `.dims`, `.strides`, `.indirect` | The DMA descriptor: kernel argument `ptr` (its line of `tog_tensors.txt`), elements of `type`, `dims` (1 to 4) elements `strides` apart, from `offset` elements in. `indirect` (default false) makes it a gather or scatter. |
| `Node` | `.id`, `.count` | A compute node: node `id` of the kernel's `count`. |
| `Semaphore` | | A semaphore that async loads signal, owned by the `Item` that made it. `Semaphore()` is none. |
| `Args` | your own | A work-item's arguments: a struct of `int64_t` fields only. The header checks it is an aggregate, trivially copyable, with the size and alignment of `int64_t` fields. `dispatch` copies it into the ABI's `iv` array. |
| `Keyed<Args>` | `.key`, `.args` | A dispatch that names its own key. `Keyed<void>` has `.key` only. |
| `Kernel` | | The kernel scope that `entry` gets, made when TOGSim calls `togsim_kernel`. It takes `shape<N>` and `dispatch` only. Made only by the header, not copyable. |
| `Item` | | One replay of one work-item, made anew each time TOGSim runs it (three times per replay). It takes `load`, `load_async`, `semaphore`, `wait`, `compute` and `store`, and at return refuses a semaphore with loads not waited. Made only by the header, not copyable. |

## Calls

| Call | Lowers to | What it does |
|---|---|---|
| `load(w, {.src, .dst, .sem, .after})` | `togsim_dma` | DRAM to spad: `src` into buffer `dst`. With `.sem` (default none: synchronous) it is async and signals that semaphore. `.after` (default `{}`) lists buffers it reads too (an indirect load's index) or must follow. |
| `Semaphore s = load_async(w, {.src, .dst, .after})` | `togsim_dma` | One async load on a new semaphore of its own. |
| `Semaphore s = semaphore(w)` | | A new semaphore, ABI sem id 1, 2, ... in the work-item's creation order. |
| `wait(w, {.sem = s})` | `togsim_memory_barrier` | Waits for every load that signaled `s` since its last wait. The header states their element count. |
| `compute(w, {.node, .variant, .reads, .writes})` | `togsim_compute` | One compute at row `node.id + node.count * variant` of the cycle and unit tables, `compute_type` 0. `.variant` defaults to 0, `.reads` and `.writes` to `{}`. |
| `store(w, {.src, .dst, .also})` | `togsim_dma` | Spad to DRAM: buffer `src` into `dst`. `.also` (default `{}`) lists buffers it reads too (an indirect store's index) or must follow. |
| `dispatch<&X::f>(k)`, `dispatch<&X::f>(k, Args{...})` | `togsim_dispatch` | One work-item, under the next key: 0, 1, ... |
| `dispatch<&X::f>(k, {.key = key, .args = Args{...}})`, `{.key = key}` | `togsim_dispatch` | One work-item under `key`. A kernel uses keyed or counted dispatch, not both. |
| `shape<N>(k, {.kernel, .names, .example, .extents})` | | The kernel's `N` shape arguments from `tog_shape.txt`, else `example`, as a `std::array<int64_t, N>`. The first `.extents` (default `N`) must be at least 1, the rest at least 0. `.kernel` and `.names` name them in messages. |
| `extent({.dim, .at, .tile})` | | What is left of a `tile`-long step at `at` on an axis `dim` long. |
| `short_axis({.axis, .extent, .tile})` | | Bit `axis` of a variant: set when the step is short of its tile. |
| `static const bool tog = register_tog<entry>({A, B})` | | Generated code only, in place of `Tog<X>`: registers the free function `void entry(Kernel&)` and the buffer list. |

The key names a work-item in messages and in the file names of its
[indirect index dumps](tog-abi.md#indirect-dmas). It does not choose a core.

## Rows: Node and variant

A compute names the row `node.id + node.count * variant` of the cycle table and the unit table.
Variant 0 is the node's ordinary row. A hand-written kernel uses the other variants for short
last tiles, one `short_axis` bit per axis ([the kernel library](kernels.md#the-kernel-library)). A
compiler uses them for the calls gem5 measured: call `c` of node `t` is row `t + n * c`.

## Async loads

`examples/async_barrier/async_barrier_tog.cpp` loads two operands on one semaphore:

```cpp
    Semaphore ab = semaphore(w);
    load(w, {.src = LHS, .dst = A, .sem = ab});
    load(w, {.src = RHS, .dst = B, .sem = ab});
    wait(w, {.sem = ab});
```

A semaphore stays usable after a wait: later loads may signal it again, and a later wait waits for
those. A wait needs a load that signaled the semaphore since its last wait, and a work-item returns
only with every such load waited. A work-item makes up to 64 semaphores.

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

These lines are from PyTorchSim's TOG of a gemm. Its work-item and entry are free functions,
`static void tog_work_item(togsim::Item&, WorkItemArgs)` and
`static void tog_entry(togsim::Kernel&)`, and its keys are program ids.

## Refusals

A rule a TOG breaks at run time prints a `togsim.hpp: ...` line on stderr and exits 1, before
TOGSim simulates anything; `shape` prints `kernel <name>: ...` instead (see [kernels](kernels.md)). One that a constant expression breaks fails to compile.

| Message | Cause |
|---|---|
| `no TOG registered: derive the TOG's struct from togsim::Tog<itself>` | No `Tog<>` base, and no `register_tog`. |
| `two TOGs registered in one .so` | Two TOG structs, or a struct and a `register_tog`. |
| `a spad buffer id outside 0..1023`, `two spad buffers with one id`, `a spad buffer with a negative size` | The buffer list. |
| `a buffer with a negative id` | A `Buf` with `.id` below 0 in a load, store or compute. |
| `a work-item returned with a semaphore whose loads it did not wait` | A load signaled a semaphore that no `wait` followed. |
| `a wait on a semaphore no load signaled since its last wait` | A wait with no new load on its semaphore: a first wait before any load, or a second with none between. |
| `a semaphore this work-item did not make with semaphore() or load_async()` | `Semaphore()`, or a `Semaphore` another work-item made. |
| `more than 64 semaphores in one work-item` | Too many semaphores. |
| `load_async makes its own semaphore: give it no .sem` | `load_async` with `.sem`. |
| `a DMA with no dims`, `one to four dims`, `a DMA gives another count of strides than of dims` | The descriptor's `dims` and `strides`. |
| `a DMA's descriptor has no byte-sized element type` | An `Elem` of fewer than 8 bits. |
| `a Node whose id is not below its count`, `a negative variant` | The compute's row. |
| `a kernel mixes dispatches with and without .key` | Keyed and counted dispatch in one kernel. |
| `a work-item without arguments was dispatched with some`, `a work-item was dispatched with other arguments` | The work-item and its `Args` disagree. |

A rule broken at compile time stops the build with one of these messages:

| Message | Cause |
|---|---|
| `a TOG struct has a member void entry(togsim::Kernel&) const` | `entry` missing, or with another signature. |
| `non-constant condition for static assertion`, quoting the broken rule's `fail_("...")` line | A `buffers` entry breaks a buffer-list rule above. |
| `dispatch<f>(k): f takes arguments` | `dispatch<f>(k)` for an `f` that takes `Args`. |
| `dispatch<f>(k, args): f takes this struct` | `args` is not the struct `f` takes. |
| `a work-item's arguments are a struct of int64_t fields` | `Args` is not an aggregate, or its size or alignment is not that of `int64_t` fields (an `int32_t` field, say). Field types are not checked: a lone `double` passes. |
| `togsim.hpp implements TOG ABI 16` | `togsim_runtime.h` states another ABI version. |

TOGSim checks the C contract itself as well ([what gets a run refused](tog-abi.md#what-gets-a-run-refused)).
Tests of that contract (versions, the buffer table, semaphores, the deprecated `compute_type` 1
to 3) are written on the raw C ABI, because the header cannot express its breaks.
