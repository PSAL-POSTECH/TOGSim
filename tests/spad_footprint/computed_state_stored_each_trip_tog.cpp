// A 16 KB output produced by a compute, then updated in place and stored on each of 4 trips: a
// work-item's spad footprint is 32 KB (input + state). Summing the stores made it 80 KB, more
// than the 48 KB spad, and the simulator never finished.
#include "togsim.hpp"
using namespace togsim;

struct ComputedState : Tog<ComputedState> {
  static constexpr int64_t ROWS = 64, COLS = 64, TRIPS = 4;
  static constexpr Buf IN{.id = 1, .bytes = f32(ROWS, COLS)}, STATE{.id = 2, .bytes = f32(ROWS, COLS)};
  static constexpr std::array buffers{IN, STATE};
  static constexpr Node STEP{.id = 0, .count = 1};

  struct At {
    int64_t i;
  };

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) dispatch<&ComputedState::tile>(k, At{.i = i});
  }

  void tile(Item& w, const At& a) const {
    load(w, {.from = {.ptr = 0, .type = f32, .offset = a.i * ROWS * COLS, .dims = {ROWS, COLS}, .strides = {COLS, 1}},
             .to = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {STATE}});
    for (int64_t k = 0; k < TRIPS; k++) {
      compute(w, {.node = STEP, .reads = {STATE}, .writes = {STATE}});
      store(w, {.from = STATE,
                .to = {.ptr = 1, .type = f32, .offset = (a.i * TRIPS + k) * ROWS * COLS, .dims = {ROWS, COLS},
                       .strides = {COLS, 1}}});
    }
  }
};
