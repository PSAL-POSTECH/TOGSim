#pragma once
// togsim_loader.h -- the TOGSim half (not the TOG ABI): `dlopen` a TOG `.so`, run its
// `togsim_kernel`, and record the instructions a run makes as TraceRecs (sec 5.3 / 9.7).
// togsim_trace_bridge.h turns them into a TileGraph.

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "togsim_runtime.h"

namespace togsim {

// One modeled instruction recorded by the runtime callbacks.
struct TraceRec {
  enum Kind { TILE_BEGIN, TILE_END, DMA, COMPUTE, MEMORY_BAR } kind;
  int32_t  core;          // work-item -> core binding (set by togsim_dispatch)
  // DMA / MEMORY_BAR
  int32_t  dir;           // togsim_dma_dir
  int32_t  arg_id;        // tensor
  int32_t  elem_bits;
  int32_t  is_async;
  uint64_t addr;          // resolved DRAM byte address = base[arg_id] + off*bytes
  uint64_t sem;           // DMA/MEMORY_BAR: the semaphore an async load signals / a bar waits
  int64_t  expected;      // MEMORY_BAR: elements the waited semaphore's loads must have moved
  std::vector<int64_t> dims;     // tile extents (DMA)
  std::vector<int64_t> strides;  // tile strides (DMA)
  std::vector<int64_t> read_bufs;   // SRAM buffer ids read  (sec 10 dependency model)
  std::vector<int64_t> write_bufs;  // SRAM buffer ids written
  std::vector<int64_t> element_offsets;  // DMA, a gather or scatter: each element's extra offset (empty: dense)
  // COMPUTE
  uint64_t tile_id;
  int32_t  compute_type;  // 0, or DEPRECATED 1 matmul / 2 preload / 3 cross-lane (TypedGrouping.h)
  int64_t  cycle;         // looked up from the cycle table
  int64_t  overlapping;   // looked up from the cycle table
};

// The TOG is run ON DEMAND, one work-item at a time (materializing the whole
// record stream cost 12.7 GiB on a large 8x8 conv). A tile body reads only ctx and
// iv, so a work-item is (fn, iv, core) -- runnable on its own, whenever needed.
struct WorkItem {
  void* fn = nullptr;           // togsim_tile_fn
  std::vector<int64_t> iv;      // the enclosing parallel loop indices
  int32_t core = 0;             // round-robin binding, fixed when it is registered
  int64_t key = 0;              // its program's key: the records of its runtime attributes (togsim_dispatch)
  int64_t spad = 0;             // the most bytes per lane it holds in the spad at once (togsim_dispatch)
};

// One record of a runtime attribute: what the functional run stored for one (key, id), row-major.
struct AttributeRecord {
  int32_t rank = 0;
  std::vector<int64_t> extents;
  std::vector<int64_t> values;
};
using AttributeRecords = std::map<std::pair<uint64_t, int32_t>, AttributeRecord>;

class LazyTog {
 public:
  LazyTog() = default;
  ~LazyTog();
  LazyTog(const LazyTog&) = delete;
  LazyTog& operator=(const LazyTog&) = delete;

  // dlopen the .so (refused unless its togsim_abi_version is a version TOGSim accepts and it states its runtime
  // attribute table) and run togsim_kernel once: each togsim_dispatch only registers its work-item, so
  // num_items() is known before any record is emitted. Then load the records under `attribute_path` (throws on any
  // record that is missing, malformed or not the TOG's; see docs/tog-abi.md).
  bool open(const char* so_path, const int64_t* shape_args, int32_t n_shape,
            const uint64_t* tensor_base, int32_t n_tensors,
            const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
            const int32_t* partition_cores, int32_t n_partition_cores,
            const std::string& attribute_path);

  size_t num_items() const;
  // How many tile ids the cycle table holds: a togsim_compute may name 0 to num_tile_ids() - 1.
  uint64_t num_tile_ids() const;
  int64_t item_key(size_t i) const;   // work-item i's togsim_dispatch key
  int64_t item_spad(size_t i) const;  // work-item i's togsim_dispatch spad_bytes_per_lane
  // Run work-item `i` and return its record stream (TILE_BEGIN, body,
  // TILE_END). The returned vector is a buffer reused by the next call.
  const std::vector<TraceRec>& run_item(size_t i);

 private:
  struct EmitCtx* _ctx = nullptr;   // opaque; owns the work-item list
  void* _lib = nullptr;
  std::vector<uint64_t> _bases;
  std::vector<int64_t> _cyc, _ovl;
};

}  // namespace togsim
