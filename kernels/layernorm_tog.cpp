// Layer normalization of (x + skip) over the rows of an M x N tensor (arguments 0 x, 1 skip, 2 mean, 3 var, 4 w, 5 b,
// 6 out): a statistics pass, then the normalize pass; a short tile of either has its own variant. shape_args: M N
// (example 128 256). Build options: -DTM= -DTN= -DTM_STATS= (default 64 64 128).
#include <algorithm>

#include "togsim.hpp"
using namespace togsim;

struct Layernorm : Tog<Layernorm> {
#ifdef TM
  static constexpr int64_t TILE_M = TM;
#else
  static constexpr int64_t TILE_M = 64;
#endif
#ifdef TN
  static constexpr int64_t TILE_N = TN;
#else
  static constexpr int64_t TILE_N = 64;
#endif
#ifdef TM_STATS
  static constexpr int64_t TILE_M_STATS = TM_STATS;
#else
  static constexpr int64_t TILE_M_STATS = 128;
#endif
  static constexpr int64_t LARGER = f32(std::max(TILE_M, TILE_M_STATS), TILE_N), TILE = f32(TILE_M, TILE_N);
#ifdef LANES
  static constexpr int64_t N_LANES = LANES;
#else
  static constexpr int64_t N_LANES = 128;
#endif
  static constexpr Buf X{.id = 0}, WEIGHT{.id = 1}, VAR{.id = 2}, MEAN{.id = 3}, SKIP{.id = 4}, BIAS{.id = 5},
                       OUT{.id = 6};
  static constexpr std::array buffers{X, WEIGHT, VAR, MEAN, SKIP, BIAS, OUT};
  static constexpr int64_t SPAD = per_lane(LARGER + TILE + LARGER + LARGER + LARGER + TILE + TILE, N_LANES);
  static constexpr Node REDUCE{.id = 0, .count = 4}, WRITE_MEAN{.id = 1, .count = 4}, WRITE_VAR{.id = 2, .count = 4},
                        NORM{.id = 3, .count = 4};
  static constexpr int32_t AXIS_M = 0, AXIS_N = 1, AXIS_M_STATS = 2;

  struct StatsArgs {
    int64_t m0, M, N;
  };
  struct NormArgs {
    int64_t m0, n0, M, N;
  };

  // The statistics pass over every TM_STATS-row tile, then one work-item per output tile.
  void entry(Kernel& k) const {
    const auto [M, N] = shape<2>(k, {.kernel = "layernorm", .names = "M N", .example = {128, 256}});
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M_STATS)
      dispatch<&Layernorm::stats>(k, {.spad = SPAD, .args = StatsArgs{.m0 = m0, .M = M, .N = N}});
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M)
      for (int64_t n0 = 0; n0 < N; n0 += TILE_N)
        dispatch<&Layernorm::norm>(k, {.spad = SPAD, .args = NormArgs{.m0 = m0, .n0 = n0, .M = M, .N = N}});
  }

  void stats(Item& w, const StatsArgs& a) const {
    const int64_t tm = extent({.dim = a.M, .at = a.m0, .tile = TILE_M_STATS});
    const int64_t m_short = short_axis({.axis = AXIS_M_STATS, .extent = tm, .tile = TILE_M_STATS});
    for (int64_t n0 = 0; n0 < a.N; n0 += TILE_N) {
      const int64_t tn = extent({.dim = a.N, .at = n0, .tile = TILE_N});
      const int64_t variant = m_short | short_axis({.axis = AXIS_N, .extent = tn, .tile = TILE_N});
      const int64_t at = a.m0 * a.N + n0;
      load(w, {.src = {.ptr = 0, .type = f32, .offset = at, .dims = {tm, tn}, .strides = {a.N, 1}}, .dst = X});
      load(w, {.src = {.ptr = 1, .type = f32, .offset = at, .dims = {tm, tn}, .strides = {a.N, 1}}, .dst = SKIP});
      compute(w, {.node = REDUCE, .variant = variant, .reads = {X, SKIP}});
    }
    compute(w, {.node = WRITE_MEAN, .variant = m_short, .writes = {MEAN}});
    store(w, {.src = MEAN, .dst = {.ptr = 2, .type = f32, .offset = a.m0, .dims = {tm}, .strides = {1}}});
    compute(w, {.node = WRITE_VAR, .variant = m_short, .writes = {VAR}});
    store(w, {.src = VAR, .dst = {.ptr = 3, .type = f32, .offset = a.m0, .dims = {tm}, .strides = {1}}});
  }

  // The row statistics are loaded broadcast along the tile's columns, the weights and bias down its rows.
  void norm(Item& w, const NormArgs& a) const {
    const int64_t tm = extent({.dim = a.M, .at = a.m0, .tile = TILE_M});
    const int64_t tn = extent({.dim = a.N, .at = a.n0, .tile = TILE_N});
    const int64_t variant = short_axis({.axis = AXIS_M, .extent = tm, .tile = TILE_M}) |
                            short_axis({.axis = AXIS_N, .extent = tn, .tile = TILE_N});
    const int64_t at = a.m0 * a.N + a.n0;
    const auto tile = [&](int32_t ptr, int64_t offset, Dims strides) {
      return Desc{.ptr = ptr, .type = f32, .offset = offset, .dims = {tm, tn}, .strides = strides};
    };
    load(w, {.src = tile(0, at, {a.N, 1}), .dst = X});
    load(w, {.src = tile(1, at, {a.N, 1}), .dst = SKIP});
    load(w, {.src = tile(2, a.m0, {1, 0}), .dst = MEAN});
    load(w, {.src = tile(3, a.m0, {1, 0}), .dst = VAR});
    load(w, {.src = tile(4, a.n0, {0, 1}), .dst = WEIGHT});
    load(w, {.src = tile(5, a.n0, {0, 1}), .dst = BIAS});
    compute(w, {.node = NORM, .variant = variant, .reads = {X, WEIGHT, VAR, MEAN, SKIP, BIAS}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = tile(6, at, {a.N, 1})});
  }
};
