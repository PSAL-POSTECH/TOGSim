// togsim_trace_bridge.cc -- see togsim_trace_bridge.h
#include "togsim_trace_bridge.h"

#include <algorithm>
#include <filesystem>
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
// that waits the dma's semaphore batch carries it among its wait keys.
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
  inst->set_addr_name("tag" + std::to_string(uniq), uniq);
  inst->prepare_tag_key();
  return inst;
}

// A MEMORY_BAR waiting the tag keys of every async dma in `batch` -- the Core's tag
// table releases it at the LAST of their DATA-ready (resp-complete) events, unlike a
// raw DONE edge that an async dma releases at issue-complete. Named after the first.
std::shared_ptr<Instruction> make_mem_bar(const std::vector<std::shared_ptr<Instruction>>& batch) {
  auto bar = std::make_shared<Instruction>(
      Opcode::MEMORY_BAR, 0, 0, 0,
      std::vector<size_t>{}, std::vector<int>{}, 0,
      std::vector<int64_t>{0}, std::vector<int64_t>{1},
      std::vector<int64_t>{});
  bar->set_addr_name(batch.front()->get_addr_name(), batch.front()->get_addr_id());
  bar->prepare_tag_key();
  for (const auto& d : batch) bar->add_wait_key(d->get_tag_id());
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
// the dependency DAG, sram_effects() the buffer-version rule, feed() the dispatch.
struct BuildState {
  int64_t unit_tile_base = 0;    // see tog_to_tilegraph
  const UnitTable* units = nullptr;
  togsim::LazyTog tog;
  size_t next = 0;               // next work-item to materialize
  size_t spad_bytes = 0;
  std::map<int64_t, size_t> buf_bytes;
  std::shared_ptr<TileSubGraph> sg_out;   // the tile just built

  std::shared_ptr<TileSubGraph> sg;
  std::shared_ptr<Tile> tile;
  std::map<int64_t, BufferWriters> writers;
  // Per work-item semaphore state: the async loads signaled since the last wait, and
  // the element count each semaphore's last wait covered (a re-wait must expect it again).
  struct SemBatch { std::vector<std::shared_ptr<Instruction>> dmas; std::vector<int64_t> bufs; int64_t moved = 0; };
  std::map<uint64_t, SemBatch> open_batch;
  std::map<uint64_t, int64_t> waited;
  int64_t next_tag = 0;
  int cur_tile_group = -1;
  std::string indirect_dir;               // where the functional run left its index dumps
  size_t indirect_found = 0, indirect_missing = 0;
  std::vector<std::shared_ptr<const TensorTiling>> tilings;   // per arg_id; empty = row-major
  std::set<int64_t> cur_tile_bufs;
  size_t cur_tile_footprint = 0;
  std::set<int64_t> cur_touched, cur_held;
  struct CoreHold { std::set<int64_t> ids; std::shared_ptr<SpadHold> hold = std::make_shared<SpadHold>(); };
  std::map<int32_t, CoreHold> core_holds;

  // Buffer versions of the work-item being fed; one counter numbers the versions of every kernel of the process.
  inline static int64_t next_alloc = 0;
  int64_t tile_first_alloc = 0;
  std::map<int64_t, int64_t> cur_alloc;   // buf -> current version id
  std::map<int64_t, bool> open_ver;       // buf -> version still accepting writes
  size_t item = 0;
  bool measuring = false;                  // refuse_overflow's replay, which attaches no index dump

  static size_t rec_numel(const togsim::TraceRec& t) {
    size_t numel = 1;
    for (auto d : t.dims) numel *= (size_t)d;
    return numel;
  }
  static size_t rec_bytes(const togsim::TraceRec& t) {   // single source of the tile footprint
    return rec_numel(t) * (t.elem_bits / 8);
  }
  // The spad buffers whose bytes a DMA moves: what a load fills, the one buffer a store drains
  // (read_bufs[0]); an indirect store's index buffer, read for addresses, is not among them.
  static std::vector<int64_t> moved_bufs(const togsim::TraceRec& t) {
    if (t.dir != 1) return t.write_bufs;
    return t.read_bufs.empty() ? std::vector<int64_t>{} : std::vector<int64_t>{t.read_bufs.front()};
  }

  // The first record of a work-item that names a buffer decides what it is there: written (in place or not), the
  // work-item's own; only read, `held`: data already on the chip, of which the work-item has no version.
  static void first_touch(const togsim::TraceRec& t, std::set<int64_t>& touched, std::set<int64_t>& held) {
    if (t.kind != togsim::TraceRec::DMA && t.kind != togsim::TraceRec::COMPUTE) return;
    for (int64_t b : t.write_bufs) touched.insert(b);
    for (int64_t b : t.read_bufs)
      if (touched.insert(b).second) held.insert(b);
  }

  // Point an indirect dma at its index dump, indirect_index_<key>_<seq>.raw (the vcix
  // tpu model's dma_index_key naming): one uint64 element offset per tile element.
  // A dump that is absent or of another size leaves the dma dense, and is counted.
  void attach_indices(const togsim::TraceRec& t, Instruction& inst) {
    const std::string path = indirect_dir + "/indirect_index_" + std::to_string((uint64_t)t.index_key) +
                             "_" + std::to_string(t.index_seq) + ".raw";
    std::error_code ec;
    const auto bytes = std::filesystem::file_size(path, ec);
    if (!ec && bytes == rec_numel(t) * sizeof(uint64_t)) {
      inst.set_indirect_index_path(path);
      indirect_found++;
      return;
    }
    if (!indirect_missing++)
      spdlog::warn("[TOGSim] indirect dma without its index dump ({}: {}); modeled as a dense tile",
                   path, ec ? "absent" : std::to_string(bytes) + " bytes for " +
                                             std::to_string(rec_numel(t)) + " elements");
  }

  // ---- index + footprint --------------------------------------------------
  // Replay every work-item once. Refused: a buffer the TOG DMAs and does not size, a compute whose tile id the cycle
  // table does not hold, or whose unit table rows exceed what a unit takes in the tile's cycles. Named and still run:
  // tile ids without unit rows and stray arg_ids (a warning each), and the buffers each core holds.
  bool index(const char* so_path, const int64_t* shape_args, int32_t n_shape,
             const uint64_t* tensor_base, int32_t n_tensors,
             const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
             const int32_t* partition_cores, int32_t n_partition_cores) {
    using togsim::TraceRec;
    if (!tog.open(so_path, shape_args, n_shape, tensor_base, n_tensors,
                   cyc, ovl, n_tiles, partition_cores, n_partition_cores))
      return false;
    const std::vector<int64_t>& stated = tog.spad_buffer_bytes();
    std::set<uint64_t> unitless, fitted;
    std::set<int64_t> stray_args;
    for (size_t i = 0; i < tog.num_items(); i++) {
      std::set<int64_t> touched, held;
      CoreHold* core = nullptr;
      for (const TraceRec& t : tog.run_item(i)) {
        if (t.kind == TraceRec::TILE_BEGIN) core = &core_holds[t.core];
        first_touch(t, touched, held);
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
        if (t.kind != TraceRec::DMA) continue;
        if (t.arg_id < 0 || t.arg_id >= n_tensors) stray_args.insert(t.arg_id);
        for (int64_t b : moved_bufs(t)) {
          if (b < 0 || (size_t)b >= stated.size() || stated[b] <= 0)
            throw std::runtime_error(fmt::format(
                "[TOGSim] spad buffer {} is DMA'd but the TOG states no size for it "
                "(togsim_spad_buffer_bytes has {} entries)", b, stated.size()));
          buf_bytes[b] = (size_t)stated[b];
        }
      }
      if (!core) continue;
      core->ids.insert(held.begin(), held.end());
      core->hold->left++;
    }
    for (auto& [id, core] : core_holds) {
      for (int64_t b : core.ids)
        if (b >= 0 && (size_t)b < stated.size() && stated[b] > 0) core.hold->bytes += (size_t)stated[b];
      if (core.ids.empty()) continue;
      if (spad_bytes)
        spdlog::info("[TOGSim] TOG {}: core {} holds {} bytes for the kernel: buffers {}, which its work-items "
                     "read without having written them", so_path, id, core.hold->bytes, id_list(core.ids));
      else
        spdlog::info("[TOGSim] TOG {}: the work-items of core {} read buffers {} without having written them: {} "
                     "bytes on the chip already, which nothing accounts without core_spad_size_kb", so_path, id,
                     id_list(core.ids), core.hold->bytes);
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

  // ---- the SRAM buffer-version rule (THE single source of truth) -----------
  // What `t` does to the spad: `reads` consume (and close) a buffer's current
  // version, `opens` may start a new one. Apply `reads` before `opens`.
  void sram_effects(const togsim::TraceRec& t,
                    std::vector<int64_t>& reads,
                    std::vector<int64_t>& opens) const {
    using togsim::TraceRec;
    reads.clear();
    opens.clear();
    auto has = [](const std::vector<int64_t>& v, int64_t b) {
      return std::find(v.begin(), v.end(), b) != v.end();
    };
    if (t.kind == TraceRec::DMA) {
      if (t.dir == 1) reads = t.read_bufs;    // store drains the spad
      else            opens = t.write_bufs;   // load fills it
    } else if (t.kind == TraceRec::COMPUTE) {
      // An in-place buffer (read AND written: the accumulator, an in-place
      // vector op) is version-transparent -- it neither closes nor opens one.
      for (int64_t b : t.read_bufs)
        if (!has(t.write_bufs, b)) reads.push_back(b);          // consuming reads
      for (int64_t b : t.write_bufs)
        if (!has(t.read_bufs, b) && buf_bytes.count(b))         // fresh outputs; a
          opens.push_back(b);                                   // never-DMA'd buf has
    }                                                           // no size -> untracked
  }

  // A TOG that breaks the semaphore contract: names the work-item and the counts
  // (`expected` < 0: no wait follows the batch).
  [[noreturn]] void sem_error(const char* what, uint64_t sem, int64_t sum, int64_t expected) const {
    throw std::runtime_error(fmt::format(
        "[TOGSim] {}: work-item {} (key {}) sem {} signaled {} elements, wait expects {}",
        what, item, tog.item_key(item), sem, sum,
        expected < 0 ? std::string("none") : std::to_string(expected)));
  }

  // ---- per-tile close -----------------------------------------------------
  void flush() {
    if (!open_batch.empty())
      sem_error("unwaited load batch", open_batch.begin()->first, open_batch.begin()->second.moved, -1);
    if (sg && tile) {
      tile->set_spad_footprint(cur_tile_footprint);   // distinct-buffer resident set (1- vs 2-dispatch)
      tile->set_spad_versions(tile_first_alloc, next_alloc);
      tile->set_spad_hold(core_holds[sg->get_core_id()].hold);
      sg->add_tile(tile);
      tile->set_owner(sg);
      sg_out = sg;   // hand this tile to the consumer
    }
    sg.reset();
    tile.reset();
    writers.clear();
    waited.clear();
    cur_alloc.clear();
    open_ver.clear();
    tile_first_alloc = next_alloc;
    cur_tile_bufs.clear();
    cur_tile_footprint = 0;
    cur_touched.clear();
    cur_held.clear();
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

  // ---- SRAM-capacity tracking (buffer-version allocations, sec 10.4) -------
  // A coarse tile is one version of its buffer, freed once every reader of it is issued, or with its work-item.
  void note_bufs(const std::vector<int64_t>& bufs) {
    for (int64_t b : bufs)   // once per buffer: a K-loop reload reuses the same id
      if (!cur_held.count(b) && cur_tile_bufs.insert(b).second) {
        auto it = buf_bytes.find(b);
        if (it != buf_bytes.end()) cur_tile_footprint += it->second;
      }
  }
  // Tag `inst` with the version record `t` fills and the versions it reads (sram_effects). A held buffer has
  // no version.
  void sram_apply(const togsim::TraceRec& t, const std::shared_ptr<Instruction>& inst) {
    using togsim::TraceRec;
    std::vector<int64_t> reads, opens;
    sram_effects(t, reads, opens);

    for (int64_t b : reads) {                      // consume the current version
      auto it = cur_alloc.find(b);
      if (it == cur_alloc.end()) continue;         // untracked buffer
      inst->add_sram_release(it->second);
      open_ver[b] = false;                         // next write starts a new version
    }
    opens.erase(std::remove_if(opens.begin(), opens.end(), [&](int64_t b) { return cur_held.count(b) != 0; }),
                opens.end());
    for (int64_t b : opens) {
      const bool fresh = !cur_alloc.count(b) || !open_ver[b];   // a read closed it
      if (fresh) {
        cur_alloc[b] = next_alloc++;
        open_ver[b] = true;
      }
      if (t.kind == TraceRec::DMA) {
        // Every fine DMA that fills the buffer carries the version id, so the
        // reloads of a reduction's K-loop share one allocation. Its footprint is
        // derived from the tile it moves, so nothing to set here.
        inst->set_sram_alloc(cur_alloc[b]);
      } else if (fresh) {
        // A compute that freshly produces b opens a version like a load and must
        // carry b's size explicitly (it was never DMA'd in). Further producing
        // writes into an already-open version are the same physical bytes.
        inst->set_sram_alloc(cur_alloc[b]);
        inst->set_sram_footprint(buf_bytes.at(b));
      }
    }
  }

  // ---- the record consumer (one record at a time) -------------------------
  // An async load joins its semaphore's open batch; a barrier waits that whole batch (count
  // must equal `expected`) and replaces the batch's writers. A re-wait adds no edge: the data
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
    first_touch(t, cur_touched, cur_held);

    if (t.kind == TraceRec::DMA) {
      int64_t uniq = next_tag++;                         // fresh Core tag key per dma record
      auto inst = make_dma(t, uniq);
      inst->set_tile_group(cur_tile_group);
      if (t.indirect && !measuring) attach_indices(t, *inst);
      if (t.arg_id >= 0 && (size_t)t.arg_id < tilings.size()) inst->set_tensor_tiling(tilings[t.arg_id]);
      tile->inc_required_sram_size(rec_bytes(t));         // SRAM footprint (ready-tile ordering)
      note_bufs(t.read_bufs); note_bufs(t.write_bufs);   // distinct-buffer footprint for 1- vs 2-dispatch
      if (t.dir == 1) {                                  // STORE
        // store reads the result buffer(s) -> link() JOINs all their writers.
        link(inst, t.read_bufs, t.write_bufs);
      } else {                                           // LOAD
        tile->append_instuction(inst);
        if (t.is_async) {
          auto& batch = open_batch[t.sem];
          batch.dmas.push_back(inst);
          batch.bufs.insert(batch.bufs.end(), t.write_bufs.begin(), t.write_bufs.end());
          batch.moved += (int64_t)rec_numel(t);
        }
        // No hard WAR edge: load-buffer reuse is modeled by the SRAM version /
        // capacity machinery (sram_apply), which caps how many versions coexist. A
        // latency WAR edge would force single-buffering and kill the spad overlap.
        for (int64_t b : t.write_bufs) writers[b].replace(inst);
      }
      sram_apply(t, inst);   // a store frees what it drains; a load occupies the spad
    } else if (t.kind == TraceRec::MEMORY_BAR) {
      auto ob = open_batch.find(t.sem);
      if (ob == open_batch.end()) {
        auto w = waited.find(t.sem);
        if (w == waited.end()) sem_error("wait on a semaphore never signaled", t.sem, 0, t.expected);
        if (w->second != t.expected)
          sem_error("re-wait expects another element count", t.sem, w->second, t.expected);
        return;
      }
      SemBatch& batch = ob->second;
      if (batch.moved != t.expected)
        sem_error("wait expects another element count", t.sem, batch.moved, t.expected);
      auto bar = make_mem_bar(batch.dmas);
      bar->set_tile_group(cur_tile_group);
      for (auto& d : batch.dmas) d->add_dep(bar, DepEvent::DONE);
      tile->append_instuction(bar);
      for (int64_t b : batch.bufs) writers[b].replace(bar);
      waited[t.sem] = batch.moved;
      open_batch.erase(ob);
    } else if (t.kind == TraceRec::COMPUTE) {
      auto inst = make_compute(t);
      inst->set_tile_id(unit_tile_base + (int64_t)t.tile_id);
      inst->set_tile_group(cur_tile_group);
      link(inst, t.read_bufs, t.write_bufs);
      note_bufs(t.read_bufs); note_bufs(t.write_bufs);   // distinct-buffer footprint for 1- vs 2-dispatch
      sram_apply(t, inst);   // consuming reads free their version; fresh outputs open one
    }
  }

  // ---- materialize exactly one dispatch tile ------------------------------
  // Returns the work-item `next`'s subgraph; nullptr once the TOG is
  // exhausted. Called from the TileGraph's tile source, on demand.
  std::shared_ptr<TileSubGraph> build_one_tile() {
    if (next >= tog.num_items()) {
      if (indirect_found + indirect_missing)
        spdlog::info("[TOGSim] indirect dmas: {} with their index dump, {} without",
                     indirect_found, indirect_missing);
      indirect_found = indirect_missing = 0;
      return nullptr;
    }
    item = next;
    for (const togsim::TraceRec& t : tog.run_item(next++)) feed(t);
    auto out = std::move(sg_out);
    sg_out.reset();
    if (out && spad_bytes) {
      const size_t held = core_holds[out->get_core_id()].hold->bytes;
      Tile& built = *out->peek_tile();
      built.set_spad_at_once(spad_demand(built, (spad_bytes - std::min(held, spad_bytes)) / 2));
    }
    return out;
  }

  // The most bytes an instruction of `t` finds in the spad when it is issued: its own, each version filled by
  // what it waits for and read by it or what waits for it, and each version nothing reads whose fill does not
  // wait for it. `enough`: see the first return.
  static size_t spad_demand(Tile& t, size_t enough) {
    const auto& insts = t.get_instructions();
    const size_t n = insts.size();
    size_t filled = 0;
    for (const auto& inst : insts)
      if (inst->get_sram_alloc() >= 0) filled += inst->sram_footprint();
    if (filled <= enough) return filled;
    std::unordered_map<const Instruction*, size_t> at;
    for (size_t i = 0; i < n; i++) at[insts[i].get()] = i;
    std::vector<std::vector<size_t>> waiters(n), waited_for(n);
    std::map<int64_t, std::vector<size_t>> readers;
    for (size_t i = 0; i < n; i++) {
      for (DepEvent e : {DepEvent::ISSUE, DepEvent::DONE})
        for (const auto& c : insts[i]->get_deps(e)) {
          waiters[i].push_back(at.at(c.get()));
          waited_for[at.at(c.get())].push_back(i);
        }
      for (int64_t v : insts[i]->get_sram_release()) readers[v].push_back(i);
    }
    std::vector<size_t> held(n, 0), issued_first(n, 0), after(n, n), between(n, n), walk;
    size_t kept = 0;
    for (size_t a = 0; a < n; a++) {
      if (insts[a]->get_sram_alloc() < 0) continue;
      const size_t bytes = insts[a]->sram_footprint();
      held[a] += bytes;
      const auto read_by = readers.find(insts[a]->get_sram_alloc());
      if (read_by == readers.end()) {
        kept += bytes;
        between[a] = a;
        for (walk.assign(1, a); !walk.empty();) {
          const size_t x = walk.back();
          walk.pop_back();
          issued_first[x] += bytes;
          for (size_t y : waited_for[x])
            if (between[y] != a) { between[y] = a; walk.push_back(y); }
        }
        continue;
      }
      const size_t last = read_by->second.back();
      for (walk.assign(1, a); !walk.empty();) {
        const size_t x = walk.back();
        walk.pop_back();
        for (size_t y : waiters[x])
          if (y <= last && after[y] != a) { after[y] = a; walk.push_back(y); }
      }
      for (size_t r : read_by->second)
        if (after[r] == a && between[r] != a) { between[r] = a; walk.push_back(r); }
      while (!walk.empty()) {
        const size_t x = walk.back();
        walk.pop_back();
        held[x] += bytes;
        for (size_t y : waited_for[x])
          if (after[y] == a && between[y] != a) { between[y] = a; walk.push_back(y); }
      }
    }
    size_t most = 0;
    for (size_t x = 0; x < n; x++) most = std::max(most, held[x] + kept - issued_first[x]);
    return most;
  }

  // Refuse a TOG with a work-item that needs more spad at once, beside its core's hold, than a core has
  // (`spad_bytes`; 0: no limit). Each work-item is built and measured once; the ids handed out are put back.
  void refuse_overflow(const char* so_path) {
    if (!spad_bytes) return;
    const uint64_t first_inst = Instruction::next_global_inst_id();
    const int first_subgraph = TileSubGraph::next_id();
    const int64_t first_version = next_alloc;
    measuring = true;
    for (item = 0; item < tog.num_items(); item++) {
      for (const togsim::TraceRec& t : tog.run_item(item)) feed(t);
      const std::shared_ptr<TileSubGraph> built = std::move(sg_out);
      if (!built) continue;
      const CoreHold& core = core_holds[built->get_core_id()];
      const size_t held = core.hold->bytes;
      const size_t needed = spad_demand(*built->peek_tile(), spad_bytes - std::min(held, spad_bytes));
      if (held + needed <= spad_bytes) continue;
      const std::string beside = held ? fmt::format(
          " beside the {} bytes core {} holds for the kernel (buffers {}, read without having been written)",
          held, built->get_core_id(), id_list(core.ids)) : std::string();
      throw std::runtime_error(fmt::format(
          "[TOGSim] TOG {}: work-item {} (key {}) needs {} bytes in the spad at once{}, and a core's spad "
          "holds {} (core_spad_size_kb: {})", so_path, item, tog.item_key(item), needed, beside, spad_bytes,
          spad_bytes / 1024));
    }
    measuring = false;
    next_alloc = tile_first_alloc = first_version;
    cur_tile_group = -1;
    Instruction::rewind_global_inst_id(first_inst);
    TileSubGraph::rewind_id(first_subgraph);
  }
};

}  // namespace

std::unique_ptr<TileGraph> tog_to_tilegraph(
    const char* so_path, const int64_t* shape_args, int32_t n_shape,
    const uint64_t* tensor_base, int32_t n_tensors,
    const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
    const int32_t* partition_cores, int32_t n_partition_cores,
    const std::string& indirect_dir, const std::string& name,
    const std::vector<std::shared_ptr<const TensorTiling>>& tilings, int64_t unit_tile_base,
    const UnitTable* units, size_t spad_bytes) {
  using togsim::TraceRec;
  auto S = std::make_shared<BuildState>();
  S->unit_tile_base = unit_tile_base;
  S->units = units;
  S->indirect_dir = indirect_dir;
  S->tilings = tilings;
  S->spad_bytes = spad_bytes;

  // Index the dispatches (records each work-item's fn/iv/core) and collect each
  // buffer's spad size. Builds no Instruction.
  if (!S->index(so_path, shape_args, n_shape, tensor_base, n_tensors,
                cyc, ovl, n_tiles, partition_cores, n_partition_cores))
    return nullptr;

  S->refuse_overflow(so_path);

  auto tg = std::make_unique<TileGraph>(name, name);
  // Empty cache plan (no L2/CMEM persistence) -- the tile source propagates it
  // to each subgraph, and DMA::is_cacheable dereferences it, so it must be a
  // valid (if empty) IntervalTree rather than null.
  tg->init_cache_plan({});
  tg->set_tile_source([S]() { return S->build_one_tile(); });
  return tg;
}
