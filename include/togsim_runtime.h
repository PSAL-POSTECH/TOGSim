#pragma once
// togsim_runtime.h -- the C ABI between a compiled, shape-parametric tile-operation-graph (TOG)
// (`.so`, MLIR -> EmitC -> C++) and TOGSim, which simulates it: each call is one modeled
// instruction, and the TOG carries no timing model.

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// TOG ABI version. A TOG's TU including this header defines the weak togsim_abi_version it was
// built against (TOGSim's TUs, TOGSIM_HOST, only declare it); LazyTog::open refuses a TOG that
// lacks it or differs. 17 added the runtime attributes and moved gathers onto them.
#define TOGSIM_ABI_VERSION 17
#ifdef TOGSIM_HOST
extern const int32_t togsim_abi_version;
#else
__attribute__((weak, used, visibility("default")))
extern const int32_t togsim_abi_version = TOGSIM_ABI_VERSION;
#endif

// The TOG also defines, with external linkage, the byte size of every scratchpad buffer id
// its read_bufs/write_bufs name: togsim_spad_buffer_bytes[id], 0 for an id that is no spad buffer
// (a virtual dependency edge), and togsim_spad_buffer_count entries (the array may be absent at
// 0). A buffer is as large as the compiler allocated it; TOGSim does not infer it from the DMAs.
//   const int32_t togsim_spad_buffer_count;
//   const int64_t togsim_spad_buffer_bytes[togsim_spad_buffer_count];

// And the rank of every runtime attribute it reads, by id: ids 0 to togsim_runtime_attribute_count - 1,
// at most TOGSIM_MAX_RUNTIME_ATTRIBUTES (0: the TOG reads none). Both symbols are required.
//   const int32_t togsim_runtime_attribute_count;
//   const int32_t togsim_runtime_attribute_rank[TOGSIM_MAX_RUNTIME_ATTRIBUTES];
#define TOGSIM_MAX_RUNTIME_ATTRIBUTES 64
#define TOGSIM_MAX_ATTRIBUTE_RANK 8

// Opaque per-invocation context owned by TOGSim. Holds the recorded trace of the work-item
// being run and the tile_id->cycle lookup. Never dereferenced by the TOG.
typedef struct EmitCtx EmitCtx;

// Direction for togsim_dma.
typedef enum {
  TOGSIM_DMA_LOAD  = 0,  // DRAM -> SRAM (MOVIN)
  TOGSIM_DMA_STORE = 1,  // SRAM -> DRAM (MOVOUT)
} togsim_dma_dir;

// Emit a DMA. `offset` is an ELEMENT offset into tensor `arg_id`; null
// `strides` => contiguous. `is_async` => it finishes at ISSUE and signals semaphore
// `sem` with its element count; the barrier on `sem` gates consumers. `read_bufs`/
// `write_bufs` name the spad buffers it depends on and fills: a store drains read_bufs[0], and
// an indirect store's index buffer, read for addresses only, follows it. A store's `sem` is ignored (its waits are not lowered).
// `element_offsets` (a gather or scatter): one ELEMENT offset per element of the tile, row-major
// over `dims`, added to that element's address; `n_offsets` is the tile's element count. Null
// and 0: a dense tile.

void togsim_dma(EmitCtx* ctx, int32_t dir, int32_t arg_id,
                uint64_t offset, int32_t ndim, const int64_t* dims,
                const int64_t* strides, int32_t elem_bits,
                int32_t is_async, uint64_t sem,
                const int64_t* read_bufs, int32_t n_read,
                const int64_t* write_bufs, int32_t n_write,
                const int64_t* element_offsets, int64_t n_offsets);

// The record of runtime attribute `id` for the work-item being run, as the functional run left
// it under the work-item's dispatch key: returns its values (row-major), sets *rank and points
// *extents at its rank extents. TOGSim refuses the TOG when it has no such record; it never returns null.
const int64_t* togsim_runtime_attribute(EmitCtx* ctx, int32_t id, int32_t* rank, const int64_t** extents);

// Emit a fixed-size tile compute. Cost comes from the tile_id->cycle table, not from `dims`.
// `compute_type` 0 holds the core for its cycles. DEPRECATED, the typed grouping: 1 matmul and
// 2 preload route it to a systolic array, 3 cross-lane to the cross-lane unit; kept for comparison.
void togsim_compute(EmitCtx* ctx, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write);

// The explicit async-DMA sync: waits every async load signaled on `sem`
// since its last wait, whose element counts must sum to `expected`, and becomes the
// last writer of their buffers. A sync dma blocks to arrival and needs no barrier.
void togsim_memory_barrier(EmitCtx* ctx, uint64_t sem, int64_t expected);

// A parallel work-item body, outlined by the compiler: `iv` holds the
// packed parallel loop indices (e.g. the (m,n) output-tile indices). One uniform
// signature => one general dispatcher serves every kernel. The runtime only reads iv.
typedef void (*togsim_tile_fn)(EmitCtx* ctx, int64_t* iv, int32_t n_iv);

// Dispatch one work-item: round-robin a core, bracket `fn` with
// TILE_BEGIN/TILE_END, and invoke it -- so the work-item scope IS the call. Core
// choice is runtime-owned; the TOG never names num_cores or a core. `key` is
// the work-item's program key in the functional run, under which its runtime attributes are recorded.
void togsim_dispatch(EmitCtx* ctx, togsim_tile_fn fn,
                     int64_t* iv, int32_t n_iv, int64_t key);

// Entry point the loader resolves in the TOG `.so`. `shape_args` carries
// the runtime values for the kernel's symbolic dimensions (in a kernel-specific
// order recorded alongside the cached `.so`); `n_shape_args` is their count.
void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args);

#ifdef __cplusplus
}  // extern "C"
#endif
