// Convolution of a padded I_C x (I_H+P_T+P_B) x (I_W+P_L+P_R) input (argument 0) by O_C x K_H x K_W x I_C weights (1)
// into 2: output (oh, ow) reads row oh*S_H + kh*D_H, column ow*S_W + kw*D_W, a TILE_N x TILE_I_H x TILE_M output tile
// per work-item. shape_args: CONV_NAMES (example 64 16 16 64 16 16 3 3, then 1s). Build options: the tiles and -DSA=.
#include <cstdio>
#include <cstdlib>

#include "togsim.hpp"
using namespace togsim;

struct Conv : Tog<Conv> {
#ifdef TILE_M
  static constexpr int64_t TILE_W = TILE_M;
#else
  static constexpr int64_t TILE_W = 16;
#endif
#ifdef TILE_N
  static constexpr int64_t TILE_OC = TILE_N;
#else
  static constexpr int64_t TILE_OC = 64;
#endif
#ifdef TILE_K
  static constexpr int64_t TILE_IC = TILE_K;
#else
  static constexpr int64_t TILE_IC = 64;
#endif
#ifdef TILE_I_H
  static constexpr int64_t TILE_H = TILE_I_H;
#else
  static constexpr int64_t TILE_H = 8;
#endif
#ifdef SA
  static constexpr int64_t SA_WIDTH = SA;
#else
  static constexpr int64_t SA_WIDTH = 8;
#endif
  static constexpr int64_t IC_PIECES = (TILE_IC + SA_WIDTH - 1) / SA_WIDTH;
  static constexpr int64_t W_PIECES = (TILE_W + SA_WIDTH - 1) / SA_WIDTH;
  static constexpr Buf ACC{.id = 0, .bytes = f32(TILE_OC, TILE_H, TILE_W)},
                       W_TILE{.id = 1, .bytes = f32(IC_PIECES, TILE_OC, SA_WIDTH)},
                       X_TILE{.id = 2, .bytes = f32(TILE_H, W_PIECES, SA_WIDTH, TILE_IC)};
  static constexpr std::array buffers{ACC, W_TILE, X_TILE};
  static constexpr Node ACC_INIT{.id = 0, .count = 2}, MATMUL{.id = 1, .count = 2};
  static constexpr int32_t AXIS_O_C = 0, AXIS_O_H = 1, AXIS_O_W = 2, AXIS_I_C = 3;
  static constexpr const char* CONV_NAMES = "I_C I_H I_W O_C O_H O_W K_H K_W S_H S_W D_H D_W P_T P_L P_B P_R";

  struct TileArgs {
    int64_t oc0, oh0, ow0, I_C, I_H, I_W, O_C, O_H, O_W, K_H, K_W, S_H, S_W, D_H, D_W, P_T, P_L, P_B, P_R;
  };

  // One work-item per output tile: output channels, then rows, then columns.
  void entry(Kernel& k) const {
    const auto [I_C, I_H, I_W, O_C, O_H, O_W, K_H, K_W, S_H, S_W, D_H, D_W, P_T, P_L, P_B, P_R] =
        shape<16>(k, {.kernel = "conv",
                      .names = CONV_NAMES,
                      .example = {64, 16, 16, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1},
                      .extents = 12});
    window_fits("row", O_H, S_H, K_H, D_H, I_H + P_T + P_B);
    window_fits("column", O_W, S_W, K_W, D_W, I_W + P_L + P_R);
    for (int64_t oc0 = 0; oc0 < O_C; oc0 += TILE_OC)
      for (int64_t oh0 = 0; oh0 < O_H; oh0 += TILE_H)
        for (int64_t ow0 = 0; ow0 < O_W; ow0 += TILE_W)
          dispatch<&Conv::tile>(k, TileArgs{.oc0 = oc0, .oh0 = oh0, .ow0 = ow0, .I_C = I_C, .I_H = I_H, .I_W = I_W,
                                            .O_C = O_C, .O_H = O_H, .O_W = O_W, .K_H = K_H, .K_W = K_W, .S_H = S_H,
                                            .S_W = S_W, .D_H = D_H, .D_W = D_W, .P_T = P_T, .P_L = P_L, .P_B = P_B,
                                            .P_R = P_R});
  }

  // Refuse a shape whose last output along `axis` reads past the padded input: its loads would leave the tensor.
  static void window_fits(const char* axis, int64_t out, int64_t stride, int64_t kernel, int64_t dilation,
                          int64_t padded) {
    const int64_t last = (out - 1) * stride + (kernel - 1) * dilation;
    if (last < padded) return;
    std::fprintf(stderr, "kernel conv: output %s %lld reads input %s %lld, and the padded input has %lld %ss\n", axis,
                 (long long)(out - 1), axis, (long long)last, (long long)padded, axis);
    std::exit(1);
  }

  // A kernel position and TILE_K input channels is one compute, fed the weights and the input pixel each output of
  // the tile reads there, S_H rows and S_W columns apart. A short tile loads, in SA-wide pieces, only what is left.
  void tile(Item& w, const TileArgs& a) const {
    const int64_t x_row = (a.I_W + a.P_L + a.P_R) * a.I_C, x_plane = (a.I_H + a.P_T + a.P_B) * x_row;
    const int64_t tn = extent({.dim = a.O_C, .at = a.oc0, .tile = TILE_OC});
    const int64_t th = extent({.dim = a.O_H, .at = a.oh0, .tile = TILE_H});
    const int64_t tw = extent({.dim = a.O_W, .at = a.ow0, .tile = TILE_W});
    const int64_t out = short_axis({.axis = AXIS_O_C, .extent = tn, .tile = TILE_OC}) |
                        short_axis({.axis = AXIS_O_H, .extent = th, .tile = TILE_H}) |
                        short_axis({.axis = AXIS_O_W, .extent = tw, .tile = TILE_W});
    const Dims x_strides{x_plane, a.S_H * x_row, a.S_W * a.I_C, 1};
    const Dims w_strides{a.I_C * a.O_C * a.K_W, a.I_C * a.O_C, a.I_C, 1};

    compute(w, {.node = ACC_INIT, .variant = out, .writes = {ACC}});
    for (int64_t kh = 0; kh < a.K_H; ++kh) {
      for (int64_t kw = 0; kw < a.K_W; ++kw) {
        for (int64_t ic0 = 0; ic0 < a.I_C; ic0 += TILE_IC) {
          const int64_t tk = extent({.dim = a.I_C, .at = ic0, .tile = TILE_IC});
          const int64_t x_off = (a.oh0 * a.S_H + kh * a.D_H) * x_row + (a.ow0 * a.S_W + kw * a.D_W) * a.I_C + ic0;
          const int64_t w_off = (kh * a.K_W + kw) * a.O_C * a.I_C + a.oc0 * a.I_C + ic0;
          Batch x = batch(w), wb = batch(w);
          for (int64_t s = 0; s < tk; s += SA_WIDTH) {
            const int64_t piece = extent({.dim = tk, .at = s, .tile = SA_WIDTH});
            const Desc weights{
                .ptr = 1, .type = f32, .offset = w_off + s, .dims = {1, 1, tn, piece}, .strides = w_strides};
            load(w, {.from = weights, .to = W_TILE, .batch = wb});
          }
          for (int64_t r = 0; r < th; ++r) {
            for (int64_t s = 0; s < tw; s += SA_WIDTH) {
              const int64_t piece = extent({.dim = tw, .at = s, .tile = SA_WIDTH});
              const int64_t at = x_off + r * a.S_H * x_row + s * a.S_W * a.I_C;
              const Desc pixels{.ptr = 0, .type = f32, .offset = at, .dims = {1, 1, piece, tk}, .strides = x_strides};
              load(w, {.from = pixels, .to = X_TILE, .batch = x});
            }
          }
          wait(w, {.batch = wb});
          wait(w, {.batch = x});
          const int64_t variant = out | short_axis({.axis = AXIS_I_C, .extent = tk, .tile = TILE_IC});
          compute(w, {.node = MATMUL, .variant = variant, .reads = {ACC, W_TILE, X_TILE}, .writes = {ACC}});
        }
      }
    }
    const Desc dst{.ptr = 2,
                   .type = f32,
                   .offset = a.oc0 * a.O_H * a.O_W + a.oh0 * a.O_W + a.ow0,
                   .dims = {1, tn, th, tw},
                   .strides = {0, a.O_H * a.O_W, a.O_W, 1}};
    store(w, {.from = ACC, .to = dst});
  }
};
