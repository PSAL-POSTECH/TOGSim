// A 16 KB state loaded once, updated in place and stored on each of 4 trips. Each dispatch states the one buffer,
// 16 KB over 128 lanes.
#include "togsim.hpp"
using namespace togsim;

struct StateStored : Tog<StateStored> {
  static constexpr int64_t ROWS = 64, COLS = 64, TRIPS = 4;
  static constexpr Buf STATE{.id = 1};
  static constexpr std::array buffers{STATE};
  static constexpr int64_t SPAD = per_lane(f32(ROWS, COLS), 128);
  static constexpr Node STEP{.id = 0, .count = 1};

  struct At {
    int64_t i;
  };

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) dispatch<&StateStored::tile>(k, {.spad = SPAD, .args = At{.i = i}});
  }

  void tile(Item& w, const At& a) const {
    load(w, {.src = {.ptr = 0, .type = f32, .offset = a.i * ROWS * COLS, .dims = {ROWS, COLS}, .strides = {COLS, 1}},
             .dst = STATE});
    for (int64_t k = 0; k < TRIPS; k++) {
      compute(w, {.node = STEP, .reads = {STATE}, .writes = {STATE}});
      store(w, {.src = STATE,
                .dst = {.ptr = 1, .type = f32, .offset = (a.i * TRIPS + k) * ROWS * COLS, .dims = {ROWS, COLS},
                       .strides = {COLS, 1}}});
    }
  }
};
