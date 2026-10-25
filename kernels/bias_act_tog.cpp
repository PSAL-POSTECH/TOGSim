// Bias add then ReLU over an M x N tensor (arguments 0 x, 1 bias, 2 out), one TM x TN tile per work-item; a last tile
// short on M or N moves and computes what is left, under its variant. shape_args: M N (example 128 256).
// Build options: -DTM= -DTN= (default 128 64).
#include "togsim.hpp"
using namespace togsim;

struct BiasAct : Tog<BiasAct> {
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
#ifdef LANES
  static constexpr int64_t N_LANES = LANES;
#else
  static constexpr int64_t N_LANES = 128;
#endif
  static constexpr Buf BIAS{.id = 0}, X{.id = 1}, OUT{.id = 2};
  static constexpr std::array buffers{BIAS, X, OUT};
  static constexpr int64_t SPAD = per_lane(f32(TILE_M, TILE_N) + f32(TILE_M, TILE_N) + f32(TILE_M, TILE_N), N_LANES);
  static constexpr Node BIAS_ACT{.id = 0, .count = 1};
  static constexpr int32_t AXIS_M = 0, AXIS_N = 1;

  struct TileArgs {
    int64_t m0, n0, M, N;
  };

  // One work-item per tile, in row-major order.
  void entry(Kernel& k) const {
    const auto [M, N] = shape<2>(k, {.kernel = "bias_act", .names = "M N", .example = {128, 256}});
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M)
      for (int64_t n0 = 0; n0 < N; n0 += TILE_N)
        dispatch<&BiasAct::tile>(k, {.spad = SPAD, .args = TileArgs{.m0 = m0, .n0 = n0, .M = M, .N = N}});
  }

  // The bias row is loaded broadcast down the tile's rows (row stride 0).
  void tile(Item& w, const TileArgs& a) const {
    const int64_t tm = extent({.dim = a.M, .at = a.m0, .tile = TILE_M});
    const int64_t tn = extent({.dim = a.N, .at = a.n0, .tile = TILE_N});
    const int64_t variant = short_axis({.axis = AXIS_M, .extent = tm, .tile = TILE_M}) |
                            short_axis({.axis = AXIS_N, .extent = tn, .tile = TILE_N});
    const int64_t at = a.m0 * a.N + a.n0;
    load(w, {.src = {.ptr = 0, .type = f32, .offset = at, .dims = {tm, tn}, .strides = {a.N, 1}}, .dst = X});
    load(w, {.src = {.ptr = 1, .type = f32, .offset = a.n0, .dims = {tm, tn}, .strides = {0, 1}}, .dst = BIAS});
    compute(w, {.node = BIAS_ACT, .variant = variant, .reads = {BIAS, X}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = {.ptr = 2, .type = f32, .offset = at, .dims = {tm, tn}, .strides = {a.N, 1}}});
  }
};
