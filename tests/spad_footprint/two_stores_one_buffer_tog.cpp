// One 16 KB output computed once and stored to two DRAM tensors: the second store drains the same
// 16 KB, so a work-item's spad footprint is 32 KB (input + output), not 48 KB.
#include "togsim.hpp"
using namespace togsim;

struct TwoStores : Tog<TwoStores> {
  static constexpr int64_t ROWS = 64, COLS = 64;
  static constexpr Buf IN{.id = 1, .bytes = f32(ROWS, COLS)}, OUT{.id = 2, .bytes = f32(ROWS, COLS)};
  static constexpr std::array buffers{IN, OUT};
  static constexpr Node STEP{.id = 0, .count = 1};

  struct At {
    int64_t i;
  };

  static constexpr Desc tile_of(int32_t ptr, int64_t at) {
    return {.ptr = ptr, .type = f32, .offset = at, .dims = {ROWS, COLS}, .strides = {COLS, 1}};
  }

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) dispatch<&TwoStores::tile>(k, At{.i = i});
  }

  void tile(Item& w, const At& a) const {
    const int64_t base = a.i * ROWS * COLS;
    load(w, {.from = tile_of(0, base), .to = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.from = OUT, .to = tile_of(1, base)});
    store(w, {.from = OUT, .to = tile_of(2, base)});
  }
};
