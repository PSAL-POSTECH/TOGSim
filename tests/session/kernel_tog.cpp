// One kernel of a session: ITEMS work-items (default two), each a 16 KB load, one compute tile and a 16 KB store.
// -DNO_CALLS: the work-items make no call. -DNO_COMPUTE: they copy, the store draining what the load filled.
#include "togsim.hpp"
using namespace togsim;

struct SessionKernel : Tog<SessionKernel> {
#ifdef ITEMS
  static constexpr int64_t N = ITEMS;
#else
  static constexpr int64_t N = 2;
#endif
  static constexpr int64_t ROWS = 64, COLS = 64;
  static constexpr Buf IN{.id = 1, .bytes = f32(ROWS, COLS)}, OUT{.id = 2, .bytes = f32(ROWS, COLS)};
  static constexpr std::array buffers{IN, OUT};
  static constexpr Node STEP{.id = 0, .count = 1};

  struct At {
    int64_t i;
  };

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < N; i++) dispatch<&SessionKernel::tile>(k, At{.i = i});
  }

  void tile([[maybe_unused]] Item& w, [[maybe_unused]] const At& a) const {
#if !defined(NO_CALLS)
    const int64_t at = a.i * ROWS * COLS;
    load(w, {.from = {.ptr = 0, .type = f32, .offset = at, .dims = {ROWS, COLS}, .strides = {COLS, 1}}, .to = IN});
#if defined(NO_COMPUTE)
    store(w, {.from = IN, .to = {.ptr = 1, .type = f32, .offset = at, .dims = {ROWS, COLS}, .strides = {COLS, 1}}});
#else
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.from = OUT, .to = {.ptr = 1, .type = f32, .offset = at, .dims = {ROWS, COLS}, .strides = {COLS, 1}}});
#endif
#endif
  }
};
