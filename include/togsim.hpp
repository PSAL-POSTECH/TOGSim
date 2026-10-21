#pragma once
// togsim.hpp -- TOGSim API v1.0.0: a TOG as one C++20 struct over the C ABI of togsim_runtime.h, every call with
// one struct of named arguments (docs/tog-api.md). A rule broken at run time prints "togsim.hpp: ..." and exits 1;
// one broken in a constant expression fails to compile.

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <type_traits>
#include <vector>

#include "togsim_runtime.h"

namespace togsim {
inline constexpr int API_MAJOR = 1, API_MINOR = 0, API_PATCH = 0;
inline constexpr char API_VERSION[] = "1.0.0";
inline constexpr int ABI = 16;
inline constexpr int32_t MAX_SPAD_BUFFERS = 1024;
static_assert(TOGSIM_ABI_VERSION == ABI, "togsim.hpp implements TOG ABI 16");
}  // namespace togsim

// The ABI's buffer table, filled from the TOG's buffers when the .so is loaded. Protected: TOGs load RTLD_GLOBAL,
// and this .so's writes must land in its own table.
extern "C" {
__attribute__((visibility("protected"))) int32_t togsim_spad_buffer_count = 0;
__attribute__((visibility("protected"))) int64_t togsim_spad_buffer_bytes[togsim::MAX_SPAD_BUFFERS] = {};
}

namespace togsim {
namespace {

[[noreturn]] inline void fail_(const char* what) {
  std::fprintf(stderr, "togsim.hpp: %s\n", what);
  std::exit(1);
}

// An element type; f32(TM, TK) is the bytes of a TM x TK tile of it. `kind` tells f16 from bf16 and i32 from f32,
// the ABI sees only `bits`.
struct Elem {
  int32_t bits;
  int32_t kind;
  template <class... D>
  constexpr int64_t operator()(D... extents) const {
    return ((int64_t)extents * ... * int64_t{1}) * bits / 8;
  }
};
constexpr Elem u8{8, 0}, i8{8, 1}, i16{16, 2}, f16{16, 3}, bf16{16, 4}, i32{32, 5}, f32{32, 6}, i64{64, 7}, f64{64, 8};

// A spad buffer: its id (its entry in togsim_spad_buffer_bytes) and its bytes; 0 bytes is a dependency edge only.
struct Buf {
  int64_t id;
  int64_t bytes;
};

// One to four extents or strides: {tm, tk}, or Dims(tm, tk).
struct Dims {
  int64_t v[4]{};
  int32_t n = 0;
  constexpr Dims() = default;
  constexpr Dims(std::initializer_list<int64_t> l) {
    if (l.size() < 1 || l.size() > 4) fail_("one to four dims");
    for (int64_t x : l) v[n++] = x;
  }
  template <class... D, class = std::enable_if_t<(sizeof...(D) >= 1 && (std::is_integral_v<D> && ...))>>
  constexpr explicit Dims(D... d) : Dims({(int64_t)d...}) {}
};

// What a DMA moves: `dims` elements of kernel argument `ptr` (its line of tog_tensors.txt), of type `type`,
// `strides` elements apart, from `offset` elements into it. `indirect`: a gather or scatter, each element moved
// by its index (the index buffer goes in Load.after or Store.also).
struct Desc {
  int32_t ptr;
  Elem type;
  int64_t offset;
  Dims dims;
  Dims strides;
  bool indirect = false;
};

// Compute node `id` of `count`: the tile id gem5 names its samples with, and how many there are. A compute takes
// the row `id + count * variant` of the cycle and unit tables.
struct Node {
  uint64_t id;
  uint64_t count;
};

// What is left of a `tile`-long step at `at` on an axis `dim` long.
struct Extent {
  int64_t dim;
  int64_t at;
  int64_t tile;
};
constexpr int64_t extent(const Extent& e) { return e.tile < e.dim - e.at ? e.tile : e.dim - e.at; }

// The variant bit of axis `axis`: set when a step of `extent` is short of its `tile`.
struct ShortAxis {
  int32_t axis;
  int64_t extent;
  int64_t tile;
};
constexpr int64_t short_axis(const ShortAxis& s) { return s.extent < s.tile ? (int64_t)1 << s.axis : 0; }

constexpr int32_t SEMAPHORES = 64;
struct Ops_;

// A semaphore made by semaphore(w) or load_async(...): a handle only its Item signals and waits.
// Semaphore() is no semaphore: a Load with it is synchronous.
class Semaphore {
 public:
  Semaphore() = default;

 private:
  friend class Item;
  friend struct Ops_;
  constexpr explicit Semaphore(int32_t slot) : slot_(slot) {}
  int32_t slot_ = -1;
};

// One replay of a work-item and its semaphores, ABI sem ids 1, 2, ... in creation order. A semaphore a load
// signaled since its last wait must be waited before the work-item returns.
class Item {
 public:
  Item(const Item&) = delete;
  Item& operator=(const Item&) = delete;

 private:
  friend struct Ops_;
  explicit Item(EmitCtx* c) : c_(c) {}
  Semaphore make_() {
    if (made_ == SEMAPHORES) fail_("more than 64 semaphores in one work-item");
    return Semaphore(made_++);
  }
  void check_(Semaphore s) const {
    if (s.slot_ < 0 || s.slot_ >= made_)
      fail_("a semaphore this work-item did not make with semaphore() or load_async()");
  }
  void signal_(Semaphore s, int64_t elements) {
    check_(s);
    signaled_ |= uint64_t{1} << s.slot_;
    sum_[s.slot_] += elements;
  }
  int64_t wait_(Semaphore s) {
    check_(s);
    if (!(signaled_ >> s.slot_ & 1)) fail_("a wait on a semaphore no load signaled since its last wait");
    signaled_ &= ~(uint64_t{1} << s.slot_);
    const int64_t sum = sum_[s.slot_];
    sum_[s.slot_] = 0;
    return sum;
  }
  void end_() const {
    if (signaled_) fail_("a work-item returned with a semaphore whose loads it did not wait");
  }
  EmitCtx* c_;
  int32_t made_ = 0;
  uint64_t signaled_ = 0;
  int64_t sum_[SEMAPHORES]{};
};

// The kernel scope: its shape and how it keys its dispatches (counted 0, 1, ... or each given).
class Kernel {
 public:
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

 private:
  friend struct Ops_;
  Kernel(EmitCtx* c, const int64_t* shape, int32_t n) : c_(c), shape_(shape), n_(n) {}
  EmitCtx* c_;
  const int64_t* shape_;
  int32_t n_;
  int64_t next_ = 0;
  int32_t keyed_ = -1;
};

// The few places that reach the ABI context of an Item or a Kernel.
struct Ops_ {
  static EmitCtx* ctx(Item& w) { return w.c_; }
  static Semaphore make(Item& w) { return w.make_(); }
  static bool none(Semaphore s) { return s.slot_ == -1; }
  static void signal(Item& w, Semaphore s, int64_t e) { w.signal_(s, e); }
  static int64_t wait(Item& w, Semaphore s) { return w.wait_(s); }
  static uint64_t id(Semaphore s) { return (uint64_t)s.slot_ + 1; }
  static EmitCtx* ctx(Kernel& k) { return k.c_; }
  static int32_t n_shape(const Kernel& k) { return k.n_; }
  static int64_t shape(const Kernel& k, int32_t i) { return k.shape_[i]; }
  static int64_t key(Kernel& k, bool keyed, int64_t given) {
    if (k.keyed_ == (keyed ? 0 : 1)) fail_("a kernel mixes dispatches with and without .key");
    k.keyed_ = keyed ? 1 : 0;
    return keyed ? given : k.next_++;
  }
  template <class F>
  static void run_item(EmitCtx* c, F&& f) {
    Item w(c);
    f(w);
    w.end_();
  }
  template <class F>
  static void run_kernel(EmitCtx* c, const int64_t* shape, int32_t n, F&& f) {
    Kernel k(c, shape, n);
    f(k);
  }
};

inline std::vector<int64_t> ids_(const std::vector<Buf>& bufs) {
  std::vector<int64_t> ids;
  for (const Buf& b : bufs) {
    if (b.id < 0) fail_("a buffer with a negative id");
    ids.push_back(b.id);
  }
  return ids;
}

inline void dma_(Item& w, int32_t dir, const Desc& at, int32_t async, uint64_t sem, const std::vector<Buf>& read,
                 const std::vector<Buf>& written) {
  const std::vector<int64_t> reads = ids_(read), writes = ids_(written);
  if (at.dims.n < 1) fail_("a DMA with no dims");
  if (at.strides.n != at.dims.n) fail_("a DMA gives another count of strides than of dims");
  if (at.type.bits < 8 || at.type.bits % 8) fail_("a DMA's descriptor has no byte-sized element type");
  togsim_dma(Ops_::ctx(w), dir, at.ptr, (uint64_t)at.offset, at.dims.n, at.dims.v, at.strides.v, at.type.bits,
             async, sem, reads.empty() ? nullptr : reads.data(), (int32_t)reads.size(),
             writes.empty() ? nullptr : writes.data(), (int32_t)writes.size(), at.indirect ? 1 : 0);
}

inline int64_t elements_(const Dims& d) {
  int64_t e = 1;
  for (int32_t i = 0; i < d.n; ++i) e *= d.v[i];
  return e;
}

// A new semaphore: load(w, {..., .sem = s}) signals it, wait(w, {.sem = s}) waits for those loads.
inline Semaphore semaphore(Item& w) { return Ops_::make(w); }

// DRAM -> spad: `from` into `to`. With `sem` it signals that semaphore and is done once its requests are sent, else
// once its data has arrived. `after`: buffers it must follow (an index it reads, a buffer it must not overtake).
struct Load {
  Desc from;
  Buf to;
  Semaphore sem = Semaphore();
  std::vector<Buf> after = {};
};
inline void load(Item& w, const Load& l) {
  if (Ops_::none(l.sem)) {
    dma_(w, TOGSIM_DMA_LOAD, l.from, 0, 0, l.after, {l.to});
    return;
  }
  Ops_::signal(w, l.sem, elements_(l.from.dims));
  dma_(w, TOGSIM_DMA_LOAD, l.from, 1, Ops_::id(l.sem), l.after, {l.to});
}

// One async load on a semaphore of its own, returned.
inline Semaphore load_async(Item& w, const Load& l) {
  if (!Ops_::none(l.sem)) fail_("load_async makes its own semaphore: give it no .sem");
  Semaphore s = semaphore(w);
  load(w, Load{l.from, l.to, s, l.after});
  return s;
}

// spad -> DRAM: `from` into `to`. `also`: further buffers it reads (an indirect store's index buffer) or follows.
struct Store {
  Buf from;
  Desc to;
  std::vector<Buf> also = {};
};
inline void store(Item& w, const Store& s) {
  std::vector<Buf> read{s.from};
  read.insert(read.end(), s.also.begin(), s.also.end());
  dma_(w, TOGSIM_DMA_STORE, s.to, 0, 0, read, {});
}

// Waits for every load that signaled `sem` since its last wait; later loads may signal it again.
struct Wait {
  Semaphore sem;
};
inline void wait(Item& w, const Wait& x) {
  const int64_t elements = Ops_::wait(w, x.sem);
  togsim_memory_barrier(Ops_::ctx(w), Ops_::id(x.sem), elements);
}

// A compute of `node` at row `node.id + node.count * variant` of the cycle and unit tables (a short last tile, or
// one of gem5's calls); it reads `reads` and writes `writes`.
struct Compute {
  Node node;
  int64_t variant = 0;
  std::vector<Buf> reads = {};
  std::vector<Buf> writes = {};
};
inline void compute(Item& w, const Compute& c) {
  if (c.node.id >= c.node.count) fail_("a Node whose id is not below its count");
  if (c.variant < 0) fail_("a negative variant");
  const uint64_t row = c.node.id + (uint64_t)c.variant * c.node.count;
  const std::vector<int64_t> reads = ids_(c.reads), writes = ids_(c.writes);
  togsim_compute(Ops_::ctx(w), row, 0, 0, nullptr, reads.empty() ? nullptr : reads.data(), (int32_t)reads.size(),
                 writes.empty() ? nullptr : writes.data(), (int32_t)writes.size());
}

// The kernel's shape: tog_shape.txt's N values, else `example`. The first `extents` (default all) must be at least
// 1, the rest at least 0.
template <int N>
struct ShapeOf {
  const char* kernel;
  const char* names;
  std::array<int64_t, N> example;
  int32_t extents = N;
};
template <int N>
std::array<int64_t, N> shape(const Kernel& k, const ShapeOf<N>& s) {
  const int32_t given = Ops_::n_shape(k);
  if (given != 0 && given != N) {
    std::fprintf(stderr, "kernel %s: it takes %d shape arguments (%s), tog_shape.txt gives %d\n", s.kernel, (int)N,
                 s.names, (int)given);
    std::exit(1);
  }
  std::array<int64_t, N> r{};
  for (int32_t i = 0; i < N; ++i) {
    r[i] = given ? Ops_::shape(k, i) : s.example[i];
    const int64_t least = i < s.extents ? 1 : 0;
    if (r[i] < least) {
      std::fprintf(stderr, "kernel %s: shape argument %d of (%s) is %lld, it must be at least %d\n", s.kernel, (int)i,
                   s.names, (long long)r[i], (int)least);
      std::exit(1);
    }
  }
  return r;
}

// A work-item function: a const member of a TOG struct taking its arguments by const&, or a free function
// (generated code) taking them by value; the arguments are a struct of int64_t fields.
template <class F>
struct work_fn_;
template <class K>
struct work_fn_<void (K::*)(Item&) const> {
  using owner = K;
  using args = void;
};
template <class K, class A>
struct work_fn_<void (K::*)(Item&, const A&) const> {
  using owner = K;
  using args = A;
};
template <>
struct work_fn_<void (*)(Item&)> {
  using owner = void;
  using args = void;
};
template <class A>
struct work_fn_<void (*)(Item&, A)> {
  using owner = void;
  using args = A;
};
template <auto F>
using args_of_ = typename work_fn_<decltype(F)>::args;

template <class A>
constexpr bool int64_fields_() {
  return std::is_aggregate_v<A> && std::is_trivially_copyable_v<A> && sizeof(A) % sizeof(int64_t) == 0 &&
         alignof(A) == alignof(int64_t);
}

template <auto F>
void work_item_(EmitCtx* c, int64_t* iv, int32_t n) {
  using W = work_fn_<decltype(F)>;
  Ops_::run_item(c, [&](Item& w) {
    if constexpr (std::is_void_v<typename W::args>) {
      if (n != 0) fail_("a work-item without arguments was dispatched with some");
      if constexpr (std::is_void_v<typename W::owner>) F(w);
      else (typename W::owner{}.*F)(w);
    } else {
      using A = typename W::args;
      if (n != (int32_t)(sizeof(A) / sizeof(int64_t))) fail_("a work-item was dispatched with other arguments");
      A a;
      std::memcpy(&a, iv, sizeof(A));
      if constexpr (std::is_void_v<typename W::owner>) F(w, a);
      else (typename W::owner{}.*F)(w, a);
    }
  });
}

template <auto F, class A>
void dispatch_(Kernel& k, const A* args, bool keyed, int64_t key) {
  int64_t packed[sizeof(A) / sizeof(int64_t) + 1] = {};
  if (args) std::memcpy(packed, args, sizeof(A));
  togsim_dispatch(Ops_::ctx(k), work_item_<F>, args ? packed : nullptr,
                  args ? (int32_t)(sizeof(A) / sizeof(int64_t)) : 0, Ops_::key(k, keyed, key));
}

// A dispatch that names its own key (the dump name of its indirect DMAs): {.key = key, .args = Args{...}}.
template <class A>
struct Keyed {
  int64_t key;
  A args;
};
template <>
struct Keyed<void> {
  int64_t key;
};
template <class T>
inline constexpr bool keyed_ = false;
template <class A>
inline constexpr bool keyed_<Keyed<A>> = true;

// Runs one work-item, F with no arguments, under the kernel's next key: 0, 1, ...
template <auto F>
void dispatch(Kernel& k) {
  static_assert(std::is_void_v<args_of_<F>>, "dispatch<f>(k): f takes arguments");
  dispatch_<F, int64_t>(k, nullptr, false, 0);
}

// Runs one work-item, F with `args` (a struct of int64_t fields), under the kernel's next key.
template <auto F, class A>
  requires(!keyed_<A>)
void dispatch(Kernel& k, const A& args) {
  static_assert(std::is_same_v<args_of_<F>, A>, "dispatch<f>(k, args): f takes this struct");
  static_assert(int64_fields_<A>(), "a work-item's arguments are a struct of int64_t fields");
  dispatch_<F, A>(k, &args, false, 0);
}

// Runs one work-item under `key`: dispatch<f>(k, {.key = key, .args = Args{...}}), or {.key = key} with no args.
template <auto F>
void dispatch(Kernel& k, const Keyed<args_of_<F>>& d) {
  if constexpr (std::is_void_v<args_of_<F>>) {
    dispatch_<F, int64_t>(k, nullptr, true, d.key);
  } else {
    static_assert(int64_fields_<args_of_<F>>(), "a work-item's arguments are a struct of int64_t fields");
    dispatch_<F, args_of_<F>>(k, &d.args, true, d.key);
  }
}

using EntryFn_ = void (*)(EmitCtx*, int64_t*, int32_t);
EntryFn_ entry_ = nullptr;

// A TOG's buffer list: ids from 0 below MAX_SPAD_BUFFERS, each once, no negative size.
constexpr bool buffers_ok(const Buf* bufs, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (bufs[i].id < 0 || bufs[i].id >= MAX_SPAD_BUFFERS) fail_("a spad buffer id outside 0..1023");
    if (bufs[i].bytes < 0) fail_("a spad buffer with a negative size");
    for (std::size_t j = 0; j < i; ++j)
      if (bufs[j].id == bufs[i].id) fail_("two spad buffers with one id");
  }
  return true;
}

inline bool register_(EntryFn_ entry, const Buf* bufs, std::size_t n) {
  if (entry_) fail_("two TOGs registered in one .so");
  entry_ = entry;
  int32_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    togsim_spad_buffer_bytes[bufs[i].id] = bufs[i].bytes;
    if (bufs[i].id + 1 > count) count = (int32_t)bufs[i].id + 1;
  }
  togsim_spad_buffer_count = count;
  return true;
}

template <void (*Entry)(Kernel&)>
void run_free_entry_(EmitCtx* c, int64_t* shape, int32_t n) {
  Ops_::run_kernel(c, shape, n, [](Kernel& k) { Entry(k); });
}

// For generated code, which has no base classes: static const bool tog = register_tog<entry>({Buf{...}, ...});
template <void (*Entry)(Kernel&)>
bool register_tog(std::initializer_list<Buf> bufs) {
  buffers_ok(bufs.begin(), bufs.size());
  return register_(run_free_entry_<Entry>, bufs.begin(), bufs.size());
}

template <class T>
void run_struct_entry_(EmitCtx* c, int64_t* shape, int32_t n) {
  Ops_::run_kernel(c, shape, n, [](Kernel& k) { T{}.entry(k); });
}

template <class T>
bool register_struct_() {
  static_assert(std::is_same_v<decltype(&T::entry), void (T::*)(Kernel&) const>,
                "a TOG struct has a member void entry(togsim::Kernel&) const");
  static_assert(buffers_ok(T::buffers.data(), T::buffers.size()), "the TOG struct's buffers");
  return register_(run_struct_entry_<T>, T::buffers.data(), T::buffers.size());
}

// Derive the TOG's struct from Tog<itself>: loading the .so registers its entry and fills the buffer table from
// its buffers. One TOG per .so, in one .cpp: the header defines the ABI's symbols.
template <class T>
struct Tog {
 private:
  static inline const bool registered_ = register_struct_<T>();
  template <const bool&>
  struct use_ {};
  using registered_use_ = use_<registered_>;
};

}  // namespace
}  // namespace togsim

// The ABI's entry point: the registered TOG's entry.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  if (!togsim::entry_) togsim::fail_("no TOG registered: derive the TOG's struct from togsim::Tog<itself>");
  togsim::entry_(ctx, shape_args, n_shape_args);
}
