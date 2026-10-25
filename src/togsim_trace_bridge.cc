// togsim_trace_bridge.cc -- see togsim_trace_bridge.h
#include "togsim_trace_bridge.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Tile.h"
#include "Instruction.h"

namespace {

// `ids` as text, each run of consecutive ids folded: "1, 3 to 5, 9".
template <class T>
std::string id_list(const std::set<T>& ids) {
  std::string out;
  for (auto it = ids.begin(); it != ids.end();) {
    auto last = it;
    for (auto next = std::next(it); next != ids.end() && *next == *last + 1; ++next) last = next;
    out += (out.empty() ? "" : ", ") + std::to_string(*it) + (last == it ? "" : " to " + std::to_string(*last));
    it = std::next(last);
  }
  return out;
}

// Log `line` at warn level unless this run already logged it: a kernel launched again repeats nothing.
void warn_once(const std::string& line) {
  static std::set<std::string> said;
  if (said.insert(line).second) spdlog::warn("{}", line);
}

// `uniq` is a per-DMA-record Core tag key, unique within its work-item; the barrier
// that waits the dma's semaphore carries it among its wait keys.
std::shared_ptr<Instruction> make_dma(const togsim::TraceRec& t, int64_t uniq) {
  Opcode op = (t.dir == 1) ? Opcode::MOVOUT : Opcode::MOVIN;
  std::vector<size_t> tile_size(t.dims.begin(), t.dims.end());
  std::vector<int> tile_stride(t.strides.begin(), t.strides.end());
  std::vector<int64_t> tag_idx{0};
  std::vector<int64_t> tag_stride{1};
  auto inst = std::make_shared<Instruction>(
      op, /*compute_cycle=*/0, /*num_parents=*/0, /*dram_addr=*/t.addr,
      tile_size, tile_stride, (size_t)t.elem_bits, tag_idx, tag_stride,
      /*accum_tag_idx_list=*/std::vector<int64_t>{});
  inst->set_is_async(t.is_async != 0);
  inst->set_element_offsets(t.element_offsets);
  inst->set_addr_name("tag" + std::to_string(uniq), uniq);
  inst->prepare_tag_key();
  return inst;
}

// A MEMORY_BAR waiting the tag keys of every async dma in `dmas` -- the Core's tag
// table releases it at the LAST of their DATA-ready (resp-complete) events, unlike a
// raw DONE edge that an async dma releases at issue-complete. Named after the first.
std::shared_ptr<Instruction> make_mem_bar(const std::vector<std::shared_ptr<Instruction>>& dmas) {
  auto bar = std::make_shared<Instruction>(
      Opcode::MEMORY_BAR, 0, 0, 0,
      std::vector<size_t>{}, std::vector<int>{}, 0,
      std::vector<int64_t>{0}, std::vector<int64_t>{1},
      std::vector<int64_t>{});
  bar->set_addr_name(dmas.front()->get_addr_name(), dmas.front()->get_addr_id());
  bar->prepare_tag_key();
  for (const auto& d : dmas) bar->add_wait_key(d->get_tag_id());
  return bar;
}

// The cycle table's overlapping column is DEPRECATED (kept for the ABI): it reaches the trace log
// only. A core runs one compute at a time (Core::_compute_busy_until), which no overlap shortens.
std::shared_ptr<Instruction> make_compute(const togsim::TraceRec& t) {
  auto inst = std::make_shared<Instruction>(
      Opcode::COMP, /*compute_cycle=*/(cycle_type)t.cycle, /*num_parents=*/0,
      /*dram_addr=*/0, std::vector<size_t>{}, std::vector<int>{}, /*elem_bits=*/0,
      std::vector<int64_t>{}, std::vector<int64_t>{}, std::vector<int64_t>{});
  inst->set_overlapping_cycle((cycle_type)t.overlapping);
  inst->set_compute_type(t.compute_type);  // route to VPU, systolic array or cross-lane unit
  inst->set_tile_id((int64_t)t.tile_id);
  return inst;
}

// The current producers of one SRAM buffer: a write REPLACEs them. DEPRECATED typed grouping: the
// K matmuls of Y += X@W commute, so each JOINs the set (accumulate) and waits only on whoever
// INITIALIZED the accumulator, read from initializer() in O(1), not O(K).
class BufferWriters {
 public:
  void replace(const std::shared_ptr<Instruction>& w) {   // normal write: sole producer
    _all.assign(1, w);
    _initializer = typed::accumulates(w->get_compute_type()) ? nullptr : w;
  }
  void accumulate(const std::shared_ptr<Instruction>& mm) { _all.push_back(mm); }

  const std::vector<std::shared_ptr<Instruction>>& all() const { return _all; }
  const std::shared_ptr<Instruction>& initializer() const { return _initializer; }

 private:
  std::vector<std::shared_ptr<Instruction>> _all;
  std::shared_ptr<Instruction> _initializer;   // the non-MATMUL member of _all, if any
};

// All builder state for one kernel, driven one dispatch tile at a time. Owned by
// the TileGraph's tile source, so it survives between materializations. link() is
// the dependency DAG, feed() the dispatch.
struct BuildState {
  int64_t unit_tile_base = 0;    // see tog_to_tilegraph
  const UnitTable* units = nullptr;
  togsim::LazyTog tog;
  size_t next = 0;               // next work-item to materialize
  size_t spad_per_lane = 0;      // what a lane's spad holds (0: no limit)
  std::shared_ptr<TileSubGraph> sg_out;   // the tile just built

  std::shared_ptr<TileSubGraph> sg;
  std::shared_ptr<Tile> tile;
  std::map<int64_t, BufferWriters> writers;
  // Per work-item semaphore state: the async loads signaled since the last wait, and
  // the element count each semaphore's last wait covered (a re-wait must expect it again).
  struct Signaled { std::vector<std::shared_ptr<Instruction>> dmas; std::vector<int64_t> bufs; int64_t moved = 0; };
  std::map<uint64_t, Signaled> signaled;
  std::map<uint64_t, int64_t> waited;
  int64_t next_tag = 0;
  int cur_tile_group = -1;
  std::string attribute_path;             // the functional run's runtime attribute records
  std::vector<std::shared_ptr<const TensorTiling>> tilings;   // per arg_id; empty = row-major
  size_t item = 0;

  static size_t rec_numel(const togsim::TraceRec& t) {
    size_t numel = 1;
    for (auto d : t.dims) numel *= (size_t)d;
    return numel;
  }

  // ---- index ----------------------------------------------------------------
  // Replay every work-item once. Refused: a dispatch whose spad is negative or more than a lane's spad holds, a
  // compute whose tile id the cycle table does not hold, or whose unit table rows exceed what a unit takes in the
  // tile's cycles. Named and still run: tile ids without unit rows and stray arg_ids (a warning each).
  bool index(const char* so_path, const int64_t* shape_args, int32_t n_shape,
             const uint64_t* tensor_base, int32_t n_tensors,
             const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
             const int32_t* partition_cores, int32_t n_partition_cores) {
    using togsim::TraceRec;
    if (!tog.open(so_path, shape_args, n_shape, tensor_base, n_tensors,
                   cyc, ovl, n_tiles, partition_cores, n_partition_cores, attribute_path))
      return false;
    std::set<uint64_t> unitless, fitted;
    std::set<int64_t> stray_args;
    for (size_t i = 0; i < tog.num_items(); i++) {
      const int64_t spad = tog.item_spad(i);
      if (spad < 0)
        throw std::runtime_error(fmt::format("[TOGSim] TOG {}: work-item {} (key {}) states a negative spad, {} "
                                             "bytes per lane", so_path, i, tog.item_key(i), spad));
      if (spad_per_lane && (uint64_t)spad > spad_per_lane)
        throw std::runtime_error(fmt::format(
            "[TOGSim] TOG {}: work-item {} (key {}) states {} bytes per lane of spad, and a lane's spad holds {} "
            "(vpu_spad_size_kb_per_lane: {})", so_path, i, tog.item_key(i), spad, spad_per_lane, spad_per_lane / 1024));
      for (const TraceRec& t : tog.run_item(i)) {
        if (t.kind == TraceRec::COMPUTE && t.tile_id >= tog.num_tile_ids())
          throw std::runtime_error(fmt::format(
              "[TOGSim] TOG {}: work-item {} (key {}) computes tile id {}, which the cycle table does "
              "not hold ({})", so_path, i, tog.item_key(i), t.tile_id,
              tog.num_tile_ids() ? fmt::format("it has {} tile ids, 0 to {}", tog.num_tile_ids(), tog.num_tile_ids() - 1)
                                 : std::string("it has no line")));
        if (t.kind == TraceRec::COMPUTE && units && !units->has_rows(unit_tile_base + (int64_t)t.tile_id))
          unitless.insert(t.tile_id);
        if (t.kind == TraceRec::COMPUTE && units && fitted.insert(t.tile_id).second) {
          const std::string over = units->overfull(unit_tile_base + (int64_t)t.tile_id, (uint64_t)t.cycle);
          if (!over.empty())
            throw std::runtime_error(fmt::format("[TOGSim] TOG {}: work-item {} (key {}) computes tile id {}, {}",
                                                 so_path, i, tog.item_key(i), t.tile_id, over));
        }
        if (t.kind == TraceRec::COMPUTE && typed::is_typed(t.compute_type))
          typed::Units::warn_deprecated_once(so_path);
        if (t.kind == TraceRec::DMA && (t.arg_id < 0 || t.arg_id >= n_tensors)) stray_args.insert(t.arg_id);
      }
    }
    if (!unitless.empty())
      warn_once(fmt::format("[TOGSim] TOG {}: the unit table has no rows for tile ids {}, which its computes "
                            "name: those computes add nothing to the units", so_path, id_list(unitless)));
    if (!stray_args.empty())
      warn_once(fmt::format("[TOGSim] TOG {}: its DMAs name arg_id {}, and tog_tensors.txt lists {} tensor(s): "
                            "those DMAs are addressed from base 0", so_path, id_list(stray_args),
                            std::max(n_tensors, 0)));
    return true;
  }

  // A TOG that breaks the semaphore contract: names the work-item and the counts
  // (`expected` < 0: no wait follows the semaphore's loads).
  [[noreturn]] void sem_error(const char* what, uint64_t sem, int64_t sum, int64_t expected) const {
    throw std::runtime_error(fmt::format(
        "[TOGSim] {}: work-item {} (key {}) sem {} signaled {} elements, wait expects {}",
        what, item, tog.item_key(item), sem, sum,
        expected < 0 ? std::string("none") : std::to_string(expected)));
  }

  // ---- per-tile close -----------------------------------------------------
  void flush() {
    if (!signaled.empty())
      sem_error("unwaited semaphore", signaled.begin()->first, signaled.begin()->second.moved, -1);
    if (sg && tile) {
      tile->set_spad(tog.item_spad(item));
      sg->add_tile(tile);
      tile->set_owner(sg);
      sg_out = sg;   // hand this tile to the consumer
    }
    sg.reset();
    tile.reset();
    writers.clear();
    waited.clear();
    next_tag = 0;
  }

  // ---- dependency DAG (sec 10.3) ------------------------------------------
  // READ b: depend on all writers(b), ISSUE when both are SA ops else DONE.
  // WRITE b: replace writers(b), except a commutative matmul accumulator, which
  // joins them and waits only the initializer. WAR: resource models.
  bool is_mm_accum(const std::shared_ptr<Instruction>& inst, int64_t b,
                   const std::vector<int64_t>& writes) {
    if (!typed::accumulates(inst->get_compute_type())) return false;
    for (int64_t w : writes) if (w == b) return true;
    return false;
  }
  void link(std::shared_ptr<Instruction> inst,
            const std::vector<int64_t>& reads,
            const std::vector<int64_t>& writes) {
    for (int64_t b : reads) {
      if (is_mm_accum(inst, b, writes)) continue;   // accumulator read -> handled in WRITE (UNION)
      auto it = writers.find(b);
      if (it != writers.end())
        for (auto& w : it->second.all()) {
          // both SA ops -> occupancy (overlap on the SA pipeline); else latency.
          DepEvent on = typed::overlaps(inst->get_compute_type(), w->get_compute_type())
                            ? DepEvent::ISSUE : DepEvent::DONE;
          w->add_dep(inst, on);
        }
    }
    for (int64_t b : writes) {
      auto& w = writers[b];
      if (is_mm_accum(inst, b, writes)) {   // JOIN: commutative, waits only the init
        if (const auto& init = w.initializer()) init->add_dep(inst, DepEvent::DONE);
        w.accumulate(inst);
      } else {                              // REPLACE: a normal output resets the set
        w.replace(inst);
      }
    }
    tile->append_instuction(inst);
  }


  // ---- the record consumer (one record at a time) -------------------------
  // An async load signals its semaphore; a barrier waits every load signaled since the last wait
  // (count must equal `expected`) and replaces their writers. A re-wait adds no edge: the data
  // already passed that barrier, and whatever wrote the buffers since stays their writer.
  void feed(const togsim::TraceRec& t) {
    using togsim::TraceRec;
    if (t.kind == TraceRec::TILE_BEGIN) {
      // togsim_dispatch opened a work-item -> new subgraph (bound to its core) +
      // tile. The scope runs until the matching TILE_END (the dispatch wrapper
      // brackets the tile fn call), not until the next begin.
      flush();
      sg = std::make_shared<TileSubGraph>();
      sg->set_core_id(t.core);
      tile = std::make_shared<Tile>(Tile::Status::INITIALIZED);
      cur_tile_group++;
      return;
    }
    if (t.kind == TraceRec::TILE_END) {
      flush();   // close the work-item explicitly (scope = the tile fn call)
      return;
    }
    if (!tile) return;  // defensive: ops before the first TILE_BEGIN

    if (t.kind == TraceRec::DMA) {
      int64_t uniq = next_tag++;                         // fresh Core tag key per dma record
      auto inst = make_dma(t, uniq);
      inst->set_tile_group(cur_tile_group);
      if (t.arg_id >= 0 && (size_t)t.arg_id < tilings.size()) inst->set_tensor_tiling(tilings[t.arg_id]);
      if (t.dir == 1) {                                  // STORE
        // store reads the result buffer(s) -> link() JOINs all their writers.
        link(inst, t.read_bufs, t.write_bufs);
      } else {                                           // LOAD
        tile->append_instuction(inst);
        if (t.is_async) {
          auto& sem = signaled[t.sem];
          sem.dmas.push_back(inst);
          sem.bufs.insert(sem.bufs.end(), t.write_bufs.begin(), t.write_bufs.end());
          sem.moved += (int64_t)rec_numel(t);
        }
        // No hard WAR edge: a latency WAR edge would force single-buffering and kill the spad overlap.
        for (int64_t b : t.write_bufs) writers[b].replace(inst);
      }
    } else if (t.kind == TraceRec::MEMORY_BAR) {
      auto os = signaled.find(t.sem);
      if (os == signaled.end()) {
        auto w = waited.find(t.sem);
        if (w == waited.end()) sem_error("wait on a semaphore never signaled", t.sem, 0, t.expected);
        if (w->second != t.expected)
          sem_error("re-wait expects another element count", t.sem, w->second, t.expected);
        return;
      }
      Signaled& sem = os->second;
      if (sem.moved != t.expected)
        sem_error("wait expects another element count", t.sem, sem.moved, t.expected);
      auto bar = make_mem_bar(sem.dmas);
      bar->set_tile_group(cur_tile_group);
      for (auto& d : sem.dmas) d->add_dep(bar, DepEvent::DONE);
      tile->append_instuction(bar);
      for (int64_t b : sem.bufs) writers[b].replace(bar);
      waited[t.sem] = sem.moved;
      signaled.erase(os);
    } else if (t.kind == TraceRec::COMPUTE) {
      auto inst = make_compute(t);
      inst->set_tile_id(unit_tile_base + (int64_t)t.tile_id);
      inst->set_tile_group(cur_tile_group);
      link(inst, t.read_bufs, t.write_bufs);
    }
  }

  // ---- materialize exactly one dispatch tile ------------------------------
  // Returns the work-item `next`'s subgraph; nullptr once the TOG is
  // exhausted. Called from the TileGraph's tile source, on demand.
  std::shared_ptr<TileSubGraph> build_one_tile() {
    if (next >= tog.num_items()) return nullptr;
    item = next;
    for (const togsim::TraceRec& t : tog.run_item(next++)) feed(t);
    auto out = std::move(sg_out);
    sg_out.reset();
    return out;
  }

};

}  // namespace

std::unique_ptr<TileGraph> tog_to_tilegraph(
    const char* so_path, const int64_t* shape_args, int32_t n_shape,
    const uint64_t* tensor_base, int32_t n_tensors,
    const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
    const int32_t* partition_cores, int32_t n_partition_cores,
    const std::string& attribute_path, const std::string& name,
    const std::vector<std::shared_ptr<const TensorTiling>>& tilings, int64_t unit_tile_base,
    const UnitTable* units, size_t spad_per_lane) {
  using togsim::TraceRec;
  auto S = std::make_shared<BuildState>();
  S->unit_tile_base = unit_tile_base;
  S->units = units;
  S->attribute_path = attribute_path;
  S->tilings = tilings;
  S->spad_per_lane = spad_per_lane;

  // Index the dispatches (records each work-item's fn/iv/core/spad). Builds no Instruction.
  if (!S->index(so_path, shape_args, n_shape, tensor_base, n_tensors,
                cyc, ovl, n_tiles, partition_cores, n_partition_cores))
    return nullptr;

  auto tg = std::make_unique<TileGraph>(name, name);
  // Empty cache plan (no L2/CMEM persistence) -- the tile source propagates it
  // to each subgraph, and DMA::is_cacheable dereferences it, so it must be a
  // valid (if empty) IntervalTree rather than null.
  tg->init_cache_plan({});
  tg->set_tile_source([S]() { return S->build_one_tile(); });
  return tg;
}
