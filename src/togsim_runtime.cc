// togsim_runtime.cc -- the TOG C ABI (togsim_runtime.h) and the loader
// (togsim_loader.h). The TOG's calls each record a TraceRec on the opaque
// EmitCtx, resolving DRAM addresses and per-tile cycles as they go.

#include "togsim_loader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Full definition of the opaque handle from togsim_runtime.h. The TOG holds
// only EmitCtx* and never dereferences it.
struct EmitCtx {
  // inputs supplied by the loader
  const uint64_t* tensor_base = nullptr;
  int32_t         n_tensors = 0;
  const int64_t*  cyc = nullptr;   // tile_id -> cycle
  const int64_t*  ovl = nullptr;   // tile_id -> overlapping_cycle
  int32_t         n_tiles = 0;
  std::vector<int32_t> cores{0};   // the partition's core ids; dispatch round-robins over these
  // mutable run state
  int32_t  rr = 0;            // round-robin cursor into `cores`
  int32_t  cur_core = -1;     // core of the work-item being replayed
  int64_t  cur_key = 0;       // key of the work-item being replayed
  size_t   cur_item = 0;      // its index in `items`
  bool     in_item = false;   // a work-item body is running
  std::vector<togsim::WorkItem> items;   // every dispatch, recorded by togsim_kernel
  std::vector<togsim::TraceRec> trace;   // records of the work-item being replayed
  // the runtime attributes: the TOG's declared count, the records loaded at open, and what messages name
  int32_t  n_attributes = 0;
  togsim::AttributeRecords attributes;
  std::string so_path, attribute_dir;
};

namespace {
// A fresh TraceRec of kind `k` for core `core` (fields zeroed).
inline togsim::TraceRec blank(togsim::TraceRec::Kind k, int32_t core) {
  togsim::TraceRec r{};
  r.kind = k;
  r.core = core;
  return r;
}
inline void emit_rec(EmitCtx* ctx, const togsim::TraceRec& r) {
  ctx->trace.push_back(r);
}

// Refuse the TOG of `ctx`: what every runtime attribute check throws, a launch error in a session.
[[noreturn]] void refuse(const std::string& so_path, const std::string& what) {
  throw std::runtime_error("[TOGSim] TOG " + so_path + ": " + what);
}

// "work-item <n> (key <k>)" of the work-item `ctx` runs.
std::string item_name(const EmitCtx* ctx) {
  return "work-item " + std::to_string(ctx->cur_item) + " (key " + std::to_string((uint64_t)ctx->cur_key) + ")";
}
}  // namespace

extern "C" {

void togsim_dispatch(EmitCtx* ctx, togsim_tile_fn fn, int64_t* iv, int32_t n_iv,
                     int64_t key, int64_t spad_bytes_per_lane) {
  // Register the work-item; LazyTog::run_item runs its body later, on demand.
  // Round-robin over THIS partition's cores only -- a work-item on another
  // partition's core would sit in this partition's scheduler forever.
  togsim::WorkItem w;
  w.fn = (void*)fn;
  w.core = ctx->cores.empty() ? 0 : ctx->cores[ctx->rr++ % (int32_t)ctx->cores.size()];
  if (iv && n_iv > 0) w.iv.assign(iv, iv + n_iv);
  w.key = key;
  w.spad = spad_bytes_per_lane;
  ctx->items.push_back(std::move(w));
}

// An arg_id tog_tensors.txt does not list is addressed from base 0: the bridge names it in a warning when it
// indexes the TOG (togsim_trace_bridge.cc).
void togsim_dma(EmitCtx* ctx, int32_t dir, int32_t arg_id,
                uint64_t offset, int32_t ndim, const int64_t* dims,
                const int64_t* strides, int32_t elem_bits,
                int32_t is_async, uint64_t sem,
                const int64_t* read_bufs, int32_t n_read,
                const int64_t* write_bufs, int32_t n_write,
                const int64_t* element_offsets, int64_t n_offsets) {
  uint64_t base = (arg_id >= 0 && arg_id < ctx->n_tensors)
                      ? ctx->tensor_base[arg_id] : 0;
  uint64_t addr = base + offset * (uint64_t)(elem_bits / 8);
  int64_t elements = 1;
  for (int32_t i = 0; dims && i < ndim; ++i) elements *= dims[i];
  if ((element_offsets || n_offsets) && n_offsets != elements)
    refuse(ctx->so_path, item_name(ctx) + " gives a DMA " + std::to_string(n_offsets) + " element offsets for " +
                             std::to_string(elements) + " elements");
  togsim::TraceRec r = blank(togsim::TraceRec::DMA, ctx->cur_core);
  r.dir = dir; r.arg_id = arg_id; r.elem_bits = elem_bits;
  r.is_async = is_async; r.addr = addr; r.sem = sem;
  if (element_offsets) r.element_offsets.assign(element_offsets, element_offsets + n_offsets);
  if (dims) r.dims.reserve(ndim);
  if (strides) r.strides.reserve(ndim);
  r.read_bufs.reserve(n_read);
  r.write_bufs.reserve(n_write);
  for (int32_t i = 0; i < ndim; ++i) {
    if (dims) r.dims.push_back(dims[i]);
    if (strides) r.strides.push_back(strides[i]);
  }
  for (int32_t i = 0; i < n_read; ++i) r.read_bufs.push_back(read_bufs[i]);
  for (int32_t i = 0; i < n_write; ++i) r.write_bufs.push_back(write_bufs[i]);
  emit_rec(ctx, r);
}

// A tile_id the cycle table does not hold is recorded with no cycles, never looked up: the bridge
// refuses the TOG when it indexes it (togsim_trace_bridge.cc), before anything is simulated.
void togsim_compute(EmitCtx* ctx, uint64_t tile_id, int32_t compute_type,
                    int32_t ndim, const int64_t* dims,
                    const int64_t* read_bufs, int32_t n_read,
                    const int64_t* write_bufs, int32_t n_write) {
  (void)ndim; (void)dims;
  togsim::TraceRec r = blank(togsim::TraceRec::COMPUTE, ctx->cur_core);
  r.tile_id = tile_id;
  r.compute_type = compute_type;
  r.read_bufs.reserve(n_read);
  r.write_bufs.reserve(n_write);
  for (int32_t i = 0; i < n_read; ++i) r.read_bufs.push_back(read_bufs[i]);
  for (int32_t i = 0; i < n_write; ++i) r.write_bufs.push_back(write_bufs[i]);
  const bool held = ctx->n_tiles > 0 && tile_id < (uint64_t)ctx->n_tiles;
  if (held && ctx->cyc) r.cycle = ctx->cyc[tile_id];
  if (held && ctx->ovl) r.overlapping = ctx->ovl[tile_id];
  emit_rec(ctx, r);
}

void togsim_memory_barrier(EmitCtx* ctx, uint64_t sem, int64_t expected) {
  togsim::TraceRec r = blank(togsim::TraceRec::MEMORY_BAR, ctx->cur_core);
  r.sem = sem; r.expected = expected;
  emit_rec(ctx, r);
}

// The record of `id` under the running work-item's key, loaded at open; a read with none refuses the TOG.
const int64_t* togsim_runtime_attribute(EmitCtx* ctx, int32_t id, int32_t* rank, const int64_t** extents) {
  static const int64_t none = 0;
  if (!ctx->in_item) refuse(ctx->so_path, "an attribute read outside a work-item");
  if (id < 0 || id >= ctx->n_attributes)
    refuse(ctx->so_path, item_name(ctx) + " reads attribute " + std::to_string(id) + ", which the TOG does not declare");
  const auto it = ctx->attributes.find({(uint64_t)ctx->cur_key, id});
  if (it == ctx->attributes.end())
    refuse(ctx->so_path, item_name(ctx) + " reads attribute " + std::to_string(id) + ", which " + ctx->attribute_dir +
                             " does not record for that key");
  const togsim::AttributeRecord& r = it->second;
  if (rank) *rank = r.rank;
  if (extents) *extents = r.extents.empty() ? &none : r.extents.data();
  return r.values.empty() ? &none : r.values.data();
}

}  // extern "C"

namespace togsim {

namespace {
namespace fs = std::filesystem;
constexpr uint32_t PROGRAMS_MAGIC = 0x50415354, RECORD_MAGIC = 0x41525354;   // 'TSAP', 'TSRA' little-endian

// A version this TOGSim reads: its own MAJOR, no newer than itself. The integer ABIs 1..17 of before are below 0x10000.
bool accepts_version(uint32_t v) {
  return v >> 16 == TOGSIM_VERSION_MAJOR && v <= (uint32_t)TOGSIM_VERSION;
}

std::string version_name(uint32_t v) {
  if (v < 0x10000) return "TOG ABI " + std::to_string(v);
  return "version " + std::to_string(v >> 16) + "." + std::to_string(v >> 8 & 0xff) + "." + std::to_string(v & 0xff);
}

template <class T>
T word(const std::string& bytes, size_t at) {
  T v;
  std::memcpy(&v, bytes.data() + at, sizeof v);
  return v;
}

std::string bytes_of(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// `text` as a decimal number that fits `T` with no sign, no leading zero and nothing else.
template <class T>
bool decimal(const std::string& text, T& v) {
  if (text.empty() || text.size() > 20 || (text.size() > 1 && text[0] == '0')) return false;
  unsigned long long n = 0;
  for (char c : text) {
    if (c < '0' || c > '9' || __builtin_mul_overflow(n, 10ULL, &n) || __builtin_add_overflow(n, (unsigned)(c - '0'), &n))
      return false;
  }
  v = (T)n;
  return (unsigned long long)v == n;
}

// attr_<key>_<id>.bin -> key, id.
bool record_name(const std::string& name, uint64_t& key, int32_t& id) {
  const std::string head = "attr_", tail = ".bin";
  if (name.size() <= head.size() + tail.size() || name.compare(0, head.size(), head) ||
      name.compare(name.size() - tail.size(), tail.size(), tail))
    return false;
  const std::string middle = name.substr(head.size(), name.size() - head.size() - tail.size());
  const size_t cut = middle.find('_');
  return cut != std::string::npos && decimal(middle.substr(0, cut), key) && decimal(middle.substr(cut + 1), id);
}

// The keys programs.bin of `dir` lists, ascending, each once.
std::set<uint64_t> program_keys(const std::string& so, const std::string& dir) {
  const fs::path file = fs::path(dir) / "programs.bin";
  std::error_code ec;
  if (!fs::is_regular_file(file, ec)) refuse(so, dir + " has no programs.bin: the functional run recorded no program");
  const std::string bytes = bytes_of(file);
  if (bytes.size() < 16 || word<uint32_t>(bytes, 0) != PROGRAMS_MAGIC)
    refuse(so, file.string() + ": not a runtime attribute file");
  if (!accepts_version(word<uint32_t>(bytes, 4)))
    refuse(so, file.string() + " was recorded at " + version_name(word<uint32_t>(bytes, 4)) +
                   ", TOGSim is version " TOGSIM_VERSION_STRING);
  const uint64_t n = word<uint64_t>(bytes, 8);
  if (n > (bytes.size() - 16) / 8 || bytes.size() != 16 + 8 * n)
    refuse(so, file.string() + " is " + std::to_string(bytes.size()) + " bytes, its header says " +
                   (n > bytes.size() / 8 ? "more" : std::to_string(16 + 8 * n)));
  std::set<uint64_t> keys;
  for (uint64_t i = 0; i < n; ++i) {
    const uint64_t key = word<uint64_t>(bytes, 16 + 8 * i);
    if (!keys.empty() && key <= *keys.rbegin())
      refuse(so, file.string() + " lists key " + std::to_string(key) + " out of ascending order or twice");
    keys.insert(key);
  }
  return keys;
}

// The record in `file`, of the attribute file name's key and id, checked against the TOG's `ranks` and `keys`.
AttributeRecord read_record(const std::string& so, const fs::path& file, uint64_t key, int32_t id,
                            const std::vector<int32_t>& ranks, const std::set<uint64_t>& keys) {
  const std::string bytes = bytes_of(file), name = file.string();
  if (bytes.size() < 24 || word<uint32_t>(bytes, 0) != RECORD_MAGIC)
    refuse(so, name + ": not a runtime attribute file");
  if (!accepts_version(word<uint32_t>(bytes, 4)))
    refuse(so, name + " was recorded at " + version_name(word<uint32_t>(bytes, 4)) +
                   ", TOGSim is version " TOGSIM_VERSION_STRING);
  AttributeRecord r;
  const int32_t held_id = word<int32_t>(bytes, 8);
  r.rank = word<int32_t>(bytes, 12);
  const uint64_t held_key = word<uint64_t>(bytes, 16);
  if (held_key != key || held_id != id)
    refuse(so, name + " holds key " + std::to_string(held_key) + " id " + std::to_string(held_id));
  if (id < 0 || id >= (int32_t)ranks.size())
    refuse(so, name + " records attribute " + std::to_string(id) + ", which the TOG does not declare");
  if (r.rank != ranks[id])
    refuse(so, name + " records attribute " + std::to_string(id) + " at rank " + std::to_string(r.rank) +
                   ", the TOG declares rank " + std::to_string(ranks[id]));
  const size_t header = 24 + 8 * (size_t)r.rank;
  if (bytes.size() < header)
    refuse(so, name + " is " + std::to_string(bytes.size()) + " bytes, its header says at least " +
                   std::to_string(header));
  uint64_t values = 1;
  for (int32_t d = 0; d < r.rank; ++d) {
    const int64_t e = word<int64_t>(bytes, 24 + 8 * d);
    if (e < 0) refuse(so, name + " records extent " + std::to_string(e) + " on dim " + std::to_string(d));
    r.extents.push_back(e);
    if (__builtin_mul_overflow(values, (uint64_t)e, &values)) values = UINT64_MAX;
  }
  if (values > (bytes.size() - header) / 8 || bytes.size() != header + 8 * values)
    refuse(so, name + " is " + std::to_string(bytes.size()) + " bytes, its header says " +
                   (values > bytes.size() / 8 ? "more" : std::to_string(header + 8 * values)));
  if (!keys.count(key))
    refuse(so, name + " records key " + std::to_string(key) + ", which programs.bin does not list");
  r.values.resize(values);
  if (values) std::memcpy(r.values.data(), bytes.data() + header, 8 * values);
  return r;
}

// Load the records of `ctx`'s TOG from `path` (docs/tog-abi.md, Runtime attributes): every file checked, the keys
// programs.bin lists equal to the keys dispatched. A TOG that declares none needs no path, and refuses records.
void load_attributes(EmitCtx& ctx, const std::string& path, const std::vector<int32_t>& ranks) {
  const std::string& so = ctx.so_path;
  const bool given = !path.empty() && path != "-";
  std::error_code ec;
  if (ranks.empty() && (!given || !fs::is_directory(path, ec))) return;
  if (!given)
    refuse(so, "it declares " + std::to_string(ranks.size()) +
                   " runtime attribute(s) and was launched with no attribute_path");
  if (!fs::is_directory(path, ec)) refuse(so, "attribute_path " + path + " does not open");
  std::vector<std::string> names;
  for (const auto& entry : fs::directory_iterator(path, ec)) names.push_back(entry.path().filename().string());
  if (ec) refuse(so, "attribute_path " + path + " does not open");
  std::sort(names.begin(), names.end());
  std::vector<std::pair<std::string, std::pair<uint64_t, int32_t>>> records;
  for (const std::string& name : names) {
    uint64_t key;
    int32_t id;
    if (name == "programs.bin") continue;
    if (!record_name(name, key, id) || !fs::is_regular_file(fs::path(path) / name, ec))
      refuse(so, path + "/" + name + " is neither programs.bin nor attr_<key>_<id>.bin");
    records.push_back({name, {key, id}});
  }
  if (ranks.empty()) {
    if (!records.empty())
      refuse(so, "it declares no runtime attribute and " + path + " holds " + std::to_string(records.size()) +
                     " record(s)");
    return;
  }
  const std::set<uint64_t> keys = program_keys(so, path);
  std::set<uint64_t> dispatched;
  for (const WorkItem& w : ctx.items) dispatched.insert((uint64_t)w.key);
  for (uint64_t k : dispatched)
    if (!keys.count(k))
      refuse(so, "work-item key " + std::to_string(k) + " was dispatched and " + path +
                     "/programs.bin does not list it");
  for (uint64_t k : keys)
    if (!dispatched.count(k))
      refuse(so, path + "/programs.bin lists key " + std::to_string(k) + ", which no work-item was dispatched under");
  for (const auto& [name, at] : records)
    ctx.attributes[at] = read_record(so, fs::path(path) / name, at.first, at.second, ranks, keys);
  ctx.attribute_dir = path;
}

void init_ctx(EmitCtx& ctx, const uint64_t* tensor_base, int32_t n_tensors,
              const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
              const int32_t* partition_cores, int32_t n_partition_cores) {
  ctx.tensor_base = tensor_base; ctx.n_tensors = n_tensors;
  ctx.cyc = cyc; ctx.ovl = ovl; ctx.n_tiles = n_tiles;
  ctx.cores.assign(partition_cores, partition_cores + (n_partition_cores > 0 ? n_partition_cores : 0));
  if (ctx.cores.empty()) ctx.cores.push_back(0);
}
}  // namespace

LazyTog::~LazyTog() {
  delete _ctx;   // _lib intentionally left open: the tile fns must stay callable
}

bool LazyTog::open(const char* so_path, const int64_t* shape_args, int32_t n_shape,
                        const uint64_t* tensor_base, int32_t n_tensors,
                        const int64_t* cyc, const int64_t* ovl, int32_t n_tiles,
                        const int32_t* partition_cores, int32_t n_partition_cores,
                        const std::string& attribute_path) {
  // own the tables: the caller's buffers are its locals, and the tile fns are
  // invoked long after it returns.
  if (tensor_base && n_tensors > 0) _bases.assign(tensor_base, tensor_base + n_tensors);
  if (cyc && n_tiles > 0) _cyc.assign(cyc, cyc + n_tiles);
  if (ovl && n_tiles > 0) _ovl.assign(ovl, ovl + n_tiles);

  _ctx = new EmitCtx();
  init_ctx(*_ctx, _bases.empty() ? nullptr : _bases.data(), (int32_t)_bases.size(),
           _cyc.empty() ? nullptr : _cyc.data(), _ovl.empty() ? nullptr : _ovl.data(),
           (int32_t)_cyc.size(), partition_cores, n_partition_cores);
  _ctx->so_path = so_path;

  _lib = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
  if (!_lib) { fprintf(stderr, "togsim: dlopen failed: %s\n", dlerror()); return false; }
  auto abi = (const int32_t*)dlsym(_lib, "togsim_abi_version");
  if (!abi) {
    auto old = (const int32_t*)dlsym(_lib, "togsim_producer_abi_version");
    if (old)
      fprintf(stderr, "togsim: refusing TOG %s: it was built against TOG ABI %d (it embeds "
              "togsim_producer_abi_version, the symbol of ABI 15 and earlier), TOGSim is version %s; "
              "rebuild the TOG\n", so_path, (int)*old, TOGSIM_VERSION_STRING);
    else
      fprintf(stderr, "togsim: refusing TOG %s: it embeds no togsim_abi_version "
              "(built without togsim_runtime.h); TOGSim is version %s\n", so_path, TOGSIM_VERSION_STRING);
    return false;
  }
  if (!togsim::accepts_version((uint32_t)*abi)) {
    fprintf(stderr, "togsim: refusing TOG %s: it was built against %s, TOGSim is version %s; rebuild the TOG\n",
            so_path, togsim::version_name((uint32_t)*abi).c_str(), TOGSIM_VERSION_STRING);
    return false;
  }
  auto n_attrs = (const int32_t*)dlsym(_lib, "togsim_runtime_attribute_count");
  auto ranks = (const int32_t*)dlsym(_lib, "togsim_runtime_attribute_rank");
  if (!n_attrs || !ranks || *n_attrs < 0 || *n_attrs > TOGSIM_MAX_RUNTIME_ATTRIBUTES) {
    fprintf(stderr, "togsim: refusing TOG %s: it states no runtime attribute table "
            "(togsim_runtime_attribute_count / togsim_runtime_attribute_rank)\n", so_path);
    return false;
  }
  const std::vector<int32_t> attribute_ranks(ranks, ranks + *n_attrs);
  _ctx->n_attributes = *n_attrs;
  auto kernel = (void (*)(EmitCtx*, int64_t*, int32_t))dlsym(_lib, "togsim_kernel");
  if (!kernel) { fprintf(stderr, "togsim: dlsym togsim_kernel failed: %s\n", dlerror()); return false; }

  // Running the kernel only walks its outer loop nest: every togsim_dispatch
  // registers a work-item and returns, so no tile body runs and no record is
  // emitted yet. run_item() replays an individual work-item later.
  kernel(_ctx, (int64_t*)shape_args, n_shape);
  load_attributes(*_ctx, attribute_path, attribute_ranks);
  return true;
}

size_t LazyTog::num_items() const { return _ctx->items.size(); }
uint64_t LazyTog::num_tile_ids() const { return _ctx->n_tiles > 0 ? (uint64_t)_ctx->n_tiles : 0; }
int64_t LazyTog::item_key(size_t i) const { return _ctx->items.at(i).key; }
int64_t LazyTog::item_spad(size_t i) const { return _ctx->items.at(i).spad; }

const std::vector<TraceRec>& LazyTog::run_item(size_t i) {
  _ctx->trace.clear();
  if (i >= _ctx->items.size()) return _ctx->trace;
  WorkItem& w = _ctx->items[i];
  _ctx->cur_core = w.core;                     // the binding fixed when registered
  _ctx->cur_key = w.key;
  _ctx->cur_item = i;
  emit_rec(_ctx, blank(TraceRec::TILE_BEGIN, w.core));
  _ctx->in_item = true;
  ((togsim_tile_fn)w.fn)(_ctx, w.iv.empty() ? nullptr : w.iv.data(), (int32_t)w.iv.size());
  _ctx->in_item = false;
  emit_rec(_ctx, blank(TraceRec::TILE_END, w.core));
  return _ctx->trace;   // one work-item's records; overwritten by the next call
}
}  // namespace togsim
