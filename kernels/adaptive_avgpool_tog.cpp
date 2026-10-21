// Average pooling over a 2 x 2 window with stride 2 into a ROWS x COLS output (arguments 0 x, 1 out), one TROW x TCOL
// tile per work-item: four strided loads of the 2 COLS-wide input, one per tap; a short tile has its own variant.
// shape_args: ROWS COLS (example 1024 16). Build options: -DTROW= -DTCOL= (default 256 16).
#include "togsim.hpp"
using namespace togsim;

struct AdaptiveAvgpool : Tog<AdaptiveAvgpool> {
#ifdef TROW
  static constexpr int64_t TILE_ROWS = TROW;
#else
  static constexpr int64_t TILE_ROWS = 256;
#endif
#ifdef TCOL
  static constexpr int64_t TILE_COLS = TCOL;
#else
  static constexpr int64_t TILE_COLS = 16;
#endif
  static constexpr int64_t TILE = f32(TILE_ROWS, TILE_COLS);
  static constexpr Buf TAP3{.id = 0, .bytes = TILE}, TAP0{.id = 1, .bytes = TILE}, TAP1{.id = 2, .bytes = TILE},
                       TAP2{.id = 3, .bytes = TILE}, OUT{.id = 4, .bytes = TILE};
  static constexpr std::array buffers{TAP3, TAP0, TAP1, TAP2, OUT};
  static constexpr Node AVG{.id = 0, .count = 1};
  static constexpr int32_t AXIS_ROWS = 0, AXIS_COLS = 1;

  struct TileArgs {
    int64_t r0, c0, ROWS, COLS;
  };

  // One work-item per output tile, in row-major order.
  void entry(Kernel& k) const {
    const auto [ROWS, COLS] = shape<2>(k, {.kernel = "adaptive_avgpool", .names = "ROWS COLS", .example = {1024, 16}});
    for (int64_t r0 = 0; r0 < ROWS; r0 += TILE_ROWS)
      for (int64_t c0 = 0; c0 < COLS; c0 += TILE_COLS)
        dispatch<&AdaptiveAvgpool::tile>(k, TileArgs{.r0 = r0, .c0 = c0, .ROWS = ROWS, .COLS = COLS});
  }

  void tile(Item& w, const TileArgs& a) const {
    const int64_t trow = extent({.dim = a.ROWS, .at = a.r0, .tile = TILE_ROWS});
    const int64_t tcol = extent({.dim = a.COLS, .at = a.c0, .tile = TILE_COLS});
    const int64_t variant = short_axis({.axis = AXIS_ROWS, .extent = trow, .tile = TILE_ROWS}) |
                            short_axis({.axis = AXIS_COLS, .extent = tcol, .tile = TILE_COLS});
    const int64_t in_w = 2 * a.COLS, base = a.r0 * 2 * in_w + a.c0 * 2;
    const auto tap = [&](int64_t at) {
      return Desc{.ptr = 0, .type = f32, .offset = at, .dims = {trow, tcol}, .strides = {2 * in_w, 2}};
    };
    load(w, {.from = tap(base), .to = TAP0});
    load(w, {.from = tap(base + 1), .to = TAP1});
    load(w, {.from = tap(base + in_w), .to = TAP2});
    load(w, {.from = tap(base + in_w + 1), .to = TAP3});
    compute(w, {.node = AVG, .variant = variant, .reads = {TAP3, TAP0, TAP1, TAP2}, .writes = {OUT}});
    const Desc out{.ptr = 1, .type = f32, .offset = a.r0 * a.COLS + a.c0, .dims = {trow, tcol}, .strides = {a.COLS, 1}};
    store(w, {.from = OUT, .to = out});
  }
};
