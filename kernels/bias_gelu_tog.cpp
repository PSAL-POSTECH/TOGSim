// Bias add then GELU over an M x N tensor (arguments 0 x, 1 bias, 2 out), one TM x TN tile per work-item; a last tile
// short on M or N moves and computes what is left, under its variant. shape_args: M N (example 128 256).
// Build options: -DTM= -DTN= (default 128 64).
#include "togsim.hpp"
using namespace togsim;

struct BiasGelu : Tog<BiasGelu> {
#ifdef TM
  static constexpr int64_t TILE_M = TM;
#else
  static constexpr int64_t TILE_M = 128;
#endif
#ifdef TN
  static constexpr int64_t TILE_N = TN;
#else
  static constexpr int64_t TILE_N = 64;
#endif
  static constexpr Buf BIAS{.id = 0, .bytes = f32(TILE_M, TILE_N)}, X{.id = 1, .bytes = f32(TILE_M, TILE_N)},
                       OUT{.id = 2, .bytes = f32(TILE_M, TILE_N)};
  static constexpr std::array buffers{BIAS, X, OUT};
  static constexpr Node BIAS_GELU{.id = 0, .count = 1};
  static constexpr int32_t AXIS_M = 0, AXIS_N = 1;

  struct TileArgs {
    int64_t m0, n0, M, N;
  };

  // One work-item per tile, in row-major order.
  void entry(Kernel& k) const {
    const auto [M, N] = shape<2>(k, {.kernel = "bias_gelu", .names = "M N", .example = {128, 256}});
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M)
      for (int64_t n0 = 0; n0 < N; n0 += TILE_N)
        dispatch<&BiasGelu::tile>(k, TileArgs{.m0 = m0, .n0 = n0, .M = M, .N = N});
  }

  // The bias row is loaded broadcast down the tile's rows (row stride 0).
  void tile(Item& w, const TileArgs& a) const {
    const int64_t tm = extent({.dim = a.M, .at = a.m0, .tile = TILE_M});
    const int64_t tn = extent({.dim = a.N, .at = a.n0, .tile = TILE_N});
    const int64_t variant = short_axis({.axis = AXIS_M, .extent = tm, .tile = TILE_M}) |
                            short_axis({.axis = AXIS_N, .extent = tn, .tile = TILE_N});
    const int64_t at = a.m0 * a.N + a.n0;
    load(w, {.from = {.ptr = 0, .type = f32, .offset = at, .dims = {tm, tn}, .strides = {a.N, 1}}, .to = X});
    load(w, {.from = {.ptr = 1, .type = f32, .offset = a.n0, .dims = {tm, tn}, .strides = {0, 1}}, .to = BIAS});
    compute(w, {.node = BIAS_GELU, .variant = variant, .reads = {BIAS, X}, .writes = {OUT}});
    store(w, {.from = OUT, .to = {.ptr = 2, .type = f32, .offset = at, .dims = {tm, tn}, .strides = {a.N, 1}}});
  }
};
