// The smallest TOG that touches DRAM both ways: two work-items (so a two-core machine uses both cores), each a
// 16 KB load, one compute tile and a 16 KB store.
#include "togsim.hpp"
using namespace togsim;

struct LoadComputeStore : Tog<LoadComputeStore> {
  static constexpr int64_t ROWS = 64, COLS = 64;
  static constexpr Buf IN{.id = 1, .bytes = f32(ROWS, COLS)}, OUT{.id = 2, .bytes = f32(ROWS, COLS)};
  static constexpr std::array buffers{IN, OUT};
  static constexpr Node STEP{.id = 0, .count = 1};

  struct At {
    int64_t i;
  };

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) dispatch<&LoadComputeStore::tile>(k, At{.i = i});
  }

  void tile(Item& w, const At& a) const {
    const int64_t at = a.i * ROWS * COLS;
    load(w, {.src = {.ptr = 0, .type = f32, .offset = at, .dims = {ROWS, COLS}, .strides = {COLS, 1}}, .dst = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = {.ptr = 1, .type = f32, .offset = at, .dims = {ROWS, COLS}, .strides = {COLS, 1}}});
  }
};
