// A tiled kernel: an M x N grid of work-items, M and N read from tog_shape.txt. Work-item (m, n) loads its own
// 64x64 tile of tensor 0, computes on it (every work-item runs tile 0 of the cycle table) and stores it to tensor 1.
#include "togsim.hpp"
using namespace togsim;

struct Tiles : Tog<Tiles> {
  static constexpr Buf IN{.id = 1, .bytes = f32(64, 64)}, OUT{.id = 2, .bytes = f32(64, 64)};
  static constexpr std::array buffers{IN, OUT};
  static constexpr Node STEP{.id = 0, .count = 1};

  struct Cell {
    int64_t m, n, N;
  };

  // One dispatch per grid cell, in ordinary C++ loops. TOGSim binds the work-items to cores round-robin, in this
  // order, and keys them 0, 1, ...: a key names a work-item, not a core.
  void entry(Kernel& k) const {
    const auto [M, N] = shape<2>(k, {.kernel = "tiles", .names = "M N", .example = {1, 1}});
    for (int64_t m = 0; m < M; m++)
      for (int64_t n = 0; n < N; n++) dispatch<&Tiles::body>(k, Cell{.m = m, .n = n, .N = N});
  }

  void body(Item& w, const Cell& c) const {
    const int64_t at = (c.m * c.N + c.n) * 64 * 64;
    load(w, {.from = {.ptr = 0, .type = f32, .offset = at, .dims = {64, 64}, .strides = {64, 1}}, .to = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.from = OUT, .to = {.ptr = 1, .type = f32, .offset = at, .dims = {64, 64}, .strides = {64, 1}}});
  }
};
