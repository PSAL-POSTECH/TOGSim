// A tiled kernel: an M x N grid of work-items, M and N read from tog_shape.txt. Work-item (m, n) loads its own
// 64x64 tile of tensor 0, computes on it (every work-item runs tile 0 of the cycle table) and stores it to tensor 1.
// Each states its spad: IN and OUT, 2 x 16384 bytes over tpuv6e's 128 lanes.
#include "togsim.hpp"
using namespace togsim;

struct Tiles : Tog<Tiles> {
  static constexpr Buf IN{.id = 1}, OUT{.id = 2};
  static constexpr std::array buffers{IN, OUT};
  static constexpr int64_t SPAD = per_lane(2 * f32(64, 64), 128);
  static constexpr Node STEP{.id = 0, .count = 1};

  struct Cell {
    int64_t m, n, N;
  };

  // One dispatch per grid cell, in ordinary C++ loops. TOGSim binds the work-items to cores round-robin, in this
  // order, and keys them 0, 1, ...: a key names a work-item, not a core.
  void entry(Kernel& k) const {
    const auto [M, N] = shape<2>(k, {.kernel = "tiles", .names = "M N", .example = {1, 1}});
    for (int64_t m = 0; m < M; m++)
      for (int64_t n = 0; n < N; n++) dispatch<&Tiles::body>(k, {.spad = SPAD, .args = Cell{.m = m, .n = n, .N = N}});
  }

  void body(Item& w, const Cell& c) const {
    const int64_t at = (c.m * c.N + c.n) * 64 * 64;
    load(w, {.src = {.ptr = 0, .type = f32, .offset = at, .dims = {64, 64}, .strides = {64, 1}}, .dst = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = {.ptr = 1, .type = f32, .offset = at, .dims = {64, 64}, .strides = {64, 1}}});
  }
};
