// A 16 KB load buffer filled by four 4 KB pieces, reloaded once after a consuming read, and a 16 KB output drained by
// four 4 KB stores. Each dispatch states both buffers, 32 KB over 128 lanes.
#include "togsim.hpp"
using namespace togsim;

struct PiecewiseFill : Tog<PiecewiseFill> {
  static constexpr int64_t ROWS = 64, COLS = 64, PIECE_ROWS = 16;
  static constexpr Buf IN{.id = 1}, OUT{.id = 2};
  static constexpr std::array buffers{IN, OUT};
  static constexpr int64_t SPAD = per_lane(f32(ROWS, COLS) + f32(ROWS, COLS), 128);
  static constexpr Node STEP{.id = 0, .count = 1};

  struct At {
    int64_t i;
  };

  static constexpr Desc piece(int32_t ptr, int64_t at) {
    return {.ptr = ptr, .type = f32, .offset = at, .dims = {PIECE_ROWS, COLS}, .strides = {COLS, 1}};
  }

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) dispatch<&PiecewiseFill::tile>(k, {.spad = SPAD, .args = At{.i = i}});
  }

  void tile(Item& w, const At& a) const {
    const int64_t base = a.i * ROWS * COLS;
    for (int k = 0; k < 2; k++) {
      for (int64_t p = 0; p < ROWS / PIECE_ROWS; p++)
        load(w, {.src = piece(0, base + p * PIECE_ROWS * COLS), .dst = IN});
      if (k == 0) compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
      else compute(w, {.node = STEP, .reads = {IN, OUT}, .writes = {OUT}});
    }
    for (int64_t p = 0; p < ROWS / PIECE_ROWS; p++)
      store(w, {.src = OUT, .dst = piece(1, base + p * PIECE_ROWS * COLS)});
  }
};
