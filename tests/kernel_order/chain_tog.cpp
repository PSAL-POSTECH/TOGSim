// Kernel IN of a chain: ITEMS work-items (default three). Work-item i loads 16 KB tile i of tensor argument IN,
// computes and stores tile i + 1 (mod ITEMS) of argument IN + 1, which kernel IN + 1's work-item i + 1 loads. With
// STORE=0 it stores nothing.
#include "togsim.hpp"
using namespace togsim;

struct Chain : Tog<Chain> {
#ifdef IN
  static constexpr int32_t ARG = IN;
#else
  static constexpr int32_t ARG = 0;
#endif
#ifdef ITEMS
  static constexpr int64_t N = ITEMS;
#else
  static constexpr int64_t N = 3;
#endif
#ifdef STORE
  static constexpr bool STORES = STORE;
#else
  static constexpr bool STORES = true;
#endif
  static constexpr int64_t ROWS = 64, COLS = 64;
  static constexpr Buf LOADED{.id = 1}, OUT{.id = 2};
  static constexpr std::array buffers{LOADED, OUT};
  static constexpr int64_t SPAD = per_lane(f32(ROWS, COLS) + f32(ROWS, COLS), 128);
  static constexpr Node STEP{.id = 0, .count = 1};

  struct At {
    int64_t i;
  };

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < N; i++) dispatch<&Chain::tile>(k, {.spad = SPAD, .args = At{.i = i}});
  }

  static constexpr Desc tile_of(int32_t ptr, int64_t i) {
    return {.ptr = ptr, .type = f32, .offset = i * ROWS * COLS, .dims = {ROWS, COLS}, .strides = {COLS, 1}};
  }

  void tile(Item& w, const At& a) const {
    load(w, {.src = tile_of(ARG, a.i), .dst = LOADED});
    compute(w, {.node = STEP, .reads = {LOADED}, .writes = {OUT}});
    if (STORES) store(w, {.src = OUT, .dst = tile_of(ARG + 1, (a.i + 1) % N)});
  }
};
