// Row softmax of an M x N tensor (arguments 0 x, 1 max, 2 sum, 3 out) in three passes: the row max, the row sum of
// exp(x - max), then the quotient; a short tile has its own variant. shape_args: M N (example 128 256).
// Build options: -DTM= -DTN= (default 128 64).
#include "togsim.hpp"
using namespace togsim;

struct Softmax : Tog<Softmax> {
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
  static constexpr int64_t TILE = f32(TILE_M, TILE_N);
  static constexpr Buf SUM{.id = 0, .bytes = TILE}, X{.id = 1, .bytes = TILE}, MAX{.id = 2, .bytes = TILE},
                       OUT{.id = 3, .bytes = TILE};
  static constexpr std::array buffers{SUM, X, MAX, OUT};
  static constexpr Node MAX_REDUCE{.id = 0, .count = 5}, MAX_WRITE{.id = 1, .count = 5},
                        SUM_REDUCE{.id = 2, .count = 5}, SUM_WRITE{.id = 3, .count = 5}, SOFTMAX{.id = 4, .count = 5};
  static constexpr int32_t AXIS_M = 0, AXIS_N = 1;

  struct RowArgs {
    int64_t m0, M, N;
  };
  struct TileArgs {
    int64_t m0, n0, M, N;
  };

  // The max pass over every row tile, then the sum pass, then one work-item per output tile.
  void entry(Kernel& k) const {
    const auto [M, N] = shape<2>(k, {.kernel = "softmax", .names = "M N", .example = {128, 256}});
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M) dispatch<&Softmax::max>(k, RowArgs{.m0 = m0, .M = M, .N = N});
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M) dispatch<&Softmax::sum>(k, RowArgs{.m0 = m0, .M = M, .N = N});
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M)
      for (int64_t n0 = 0; n0 < N; n0 += TILE_N)
        dispatch<&Softmax::quotient>(k, TileArgs{.m0 = m0, .n0 = n0, .M = M, .N = N});
  }

  void max(Item& w, const RowArgs& a) const {
    const int64_t tm = extent({.dim = a.M, .at = a.m0, .tile = TILE_M});
    const int64_t m_short = short_axis({.axis = AXIS_M, .extent = tm, .tile = TILE_M});
    for (int64_t n0 = 0; n0 < a.N; n0 += TILE_N) {
      const int64_t tn = extent({.dim = a.N, .at = n0, .tile = TILE_N});
      const int64_t variant = m_short | short_axis({.axis = AXIS_N, .extent = tn, .tile = TILE_N});
      load(w, {.from = {.ptr = 0, .type = f32, .offset = a.m0 * a.N + n0, .dims = {tm, tn}, .strides = {a.N, 1}},
               .to = X});
      compute(w, {.node = MAX_REDUCE, .variant = variant, .reads = {X}});
    }
    compute(w, {.node = MAX_WRITE, .variant = m_short, .writes = {MAX}});
    store(w, {.from = MAX, .to = {.ptr = 1, .type = f32, .offset = a.m0, .dims = {tm}, .strides = {1}}});
  }

  // The row max is loaded broadcast along the tile's columns.
  void sum(Item& w, const RowArgs& a) const {
    const int64_t tm = extent({.dim = a.M, .at = a.m0, .tile = TILE_M});
    const int64_t m_short = short_axis({.axis = AXIS_M, .extent = tm, .tile = TILE_M});
    for (int64_t n0 = 0; n0 < a.N; n0 += TILE_N) {
      const int64_t tn = extent({.dim = a.N, .at = n0, .tile = TILE_N});
      const int64_t variant = m_short | short_axis({.axis = AXIS_N, .extent = tn, .tile = TILE_N});
      load(w, {.from = {.ptr = 0, .type = f32, .offset = a.m0 * a.N + n0, .dims = {tm, tn}, .strides = {a.N, 1}},
               .to = X});
      load(w, {.from = {.ptr = 1, .type = f32, .offset = a.m0, .dims = {tm, tn}, .strides = {1, 0}}, .to = MAX});
      compute(w, {.node = SUM_REDUCE, .variant = variant, .reads = {X, MAX}});
    }
    compute(w, {.node = SUM_WRITE, .variant = m_short, .writes = {SUM}});
    store(w, {.from = SUM, .to = {.ptr = 2, .type = f32, .offset = a.m0, .dims = {tm}, .strides = {1}}});
  }

  void quotient(Item& w, const TileArgs& a) const {
    const int64_t tm = extent({.dim = a.M, .at = a.m0, .tile = TILE_M});
    const int64_t tn = extent({.dim = a.N, .at = a.n0, .tile = TILE_N});
    const int64_t variant = short_axis({.axis = AXIS_M, .extent = tm, .tile = TILE_M}) |
                            short_axis({.axis = AXIS_N, .extent = tn, .tile = TILE_N});
    const int64_t at = a.m0 * a.N + a.n0;
    load(w, {.from = {.ptr = 0, .type = f32, .offset = at, .dims = {tm, tn}, .strides = {a.N, 1}}, .to = X});
    load(w, {.from = {.ptr = 1, .type = f32, .offset = a.m0, .dims = {tm, tn}, .strides = {1, 0}}, .to = MAX});
    load(w, {.from = {.ptr = 2, .type = f32, .offset = a.m0, .dims = {tm, tn}, .strides = {1, 0}}, .to = SUM});
    compute(w, {.node = SOFTMAX, .variant = variant, .reads = {SUM, X, MAX}, .writes = {OUT}});
    store(w, {.from = OUT, .to = {.ptr = 3, .type = f32, .offset = at, .dims = {tm, tn}, .strides = {a.N, 1}}});
  }
};
