// A 256 B index buffer loaded, a 16 KB output computed and scattered by one indirect store that reads both; the
// scatter moves row r by COLS * ROW[r], ROW the runtime attribute 0 of its key (cases.py records 0s). Each dispatch
// states both buffers, 16640 B over 128 lanes.
#include "togsim.hpp"
using namespace togsim;

struct IndirectStore : Tog<IndirectStore> {
  static constexpr int64_t ROWS = 64, COLS = 64;
  static constexpr Buf OUT{.id = 2}, IDX{.id = 3};
  static constexpr std::array buffers{OUT, IDX};
  static constexpr int64_t SPAD = per_lane(f32(ROWS, COLS) + i32(ROWS), 128);
  static constexpr Attribute ROW{.id = 0, .rank = 1};
  static constexpr std::array attributes{ROW};
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr Desc SCATTER{.ptr = 1, .type = f32, .offset = 0, .dims = {ROWS, COLS}, .strides = {COLS, 1},
                                .gather = {.attribute = ROW, .axis = 0, .scale = COLS}};

  struct At {
    int64_t i;
  };

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) dispatch<&IndirectStore::tile>(k, {.key = i, .spad = SPAD, .args = At{.i = i}});
  }

  void tile(Item& w, const At& a) const {
    load(w, {.src = {.ptr = 0, .type = i32, .offset = a.i * ROWS, .dims = {ROWS}, .strides = {1}}, .dst = IDX});
    compute(w, {.node = STEP, .reads = {IDX}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = SCATTER, .also = {IDX}});
  }
};
