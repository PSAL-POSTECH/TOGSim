// Global average pooling of C channels of HW elements each (arguments 0 x, 1 sum, 2 out): a sum pass over TC x THW
// tiles, then the scale pass over TC_SCALE channels; a short tile of either has its own variant. shape_args: C HW
// (example 256 64). Build options: -DTC= -DTHW= -DTC_SCALE= (default 256 64 128).
#include <algorithm>

#include "togsim.hpp"
using namespace togsim;

struct GlobalAvgpool : Tog<GlobalAvgpool> {
#ifdef TC
  static constexpr int64_t TILE_C = TC;
#else
  static constexpr int64_t TILE_C = 256;
#endif
#ifdef THW
  static constexpr int64_t TILE_HW = THW;
#else
  static constexpr int64_t TILE_HW = 64;
#endif
#ifdef TC_SCALE
  static constexpr int64_t TILE_C_SCALE = TC_SCALE;
#else
  static constexpr int64_t TILE_C_SCALE = 128;
#endif
  static constexpr Buf X{.id = 0, .bytes = f32(TILE_C, TILE_HW)},
                       SUM{.id = 1, .bytes = f32(std::max(TILE_C, TILE_C_SCALE))},
                       OUT{.id = 2, .bytes = f32(TILE_C_SCALE)};
  static constexpr std::array buffers{X, SUM, OUT};
  static constexpr Node REDUCE{.id = 0, .count = 3}, WRITE{.id = 1, .count = 3}, SCALE{.id = 2, .count = 3};
  static constexpr int32_t AXIS_C = 0, AXIS_HW = 1, AXIS_C_SCALE = 2;

  struct SumArgs {
    int64_t c0, C, HW;
  };
  struct ScaleArgs {
    int64_t c0, C;
  };

  // The sum pass over every TC-channel tile, then the scale pass over every TC_SCALE-channel tile.
  void entry(Kernel& k) const {
    const auto [C, HW] = shape<2>(k, {.kernel = "global_avgpool", .names = "C HW", .example = {256, 64}});
    for (int64_t c0 = 0; c0 < C; c0 += TILE_C) dispatch<&GlobalAvgpool::sum>(k, SumArgs{.c0 = c0, .C = C, .HW = HW});
    for (int64_t c0 = 0; c0 < C; c0 += TILE_C_SCALE) dispatch<&GlobalAvgpool::scale>(k, ScaleArgs{.c0 = c0, .C = C});
  }

  void sum(Item& w, const SumArgs& a) const {
    const int64_t tc = extent({.dim = a.C, .at = a.c0, .tile = TILE_C});
    const int64_t c_short = short_axis({.axis = AXIS_C, .extent = tc, .tile = TILE_C});
    for (int64_t hw0 = 0; hw0 < a.HW; hw0 += TILE_HW) {
      const int64_t thw = extent({.dim = a.HW, .at = hw0, .tile = TILE_HW});
      const int64_t variant = c_short | short_axis({.axis = AXIS_HW, .extent = thw, .tile = TILE_HW});
      const Desc x{.ptr = 0, .type = f32, .offset = a.c0 * a.HW + hw0, .dims = {tc, thw}, .strides = {a.HW, 1}};
      load(w, {.src = x, .dst = X});
      compute(w, {.node = REDUCE, .variant = variant, .reads = {X}});
    }
    compute(w, {.node = WRITE, .variant = c_short, .writes = {SUM}});
    store(w, {.src = SUM, .dst = {.ptr = 1, .type = f32, .offset = a.c0, .dims = {tc}, .strides = {1}}});
  }

  void scale(Item& w, const ScaleArgs& a) const {
    const int64_t tc = extent({.dim = a.C, .at = a.c0, .tile = TILE_C_SCALE});
    const int64_t variant = short_axis({.axis = AXIS_C_SCALE, .extent = tc, .tile = TILE_C_SCALE});
    load(w, {.src = {.ptr = 1, .type = f32, .offset = a.c0, .dims = {tc}, .strides = {1}}, .dst = SUM});
    compute(w, {.node = SCALE, .variant = variant, .reads = {SUM}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = {.ptr = 2, .type = f32, .offset = a.c0, .dims = {tc}, .strides = {1}}});
  }
};
