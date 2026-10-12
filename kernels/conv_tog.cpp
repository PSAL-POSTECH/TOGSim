// Convolution of an I_C x (I_H+P_T+P_B) x (I_W+P_L+P_R) padded input by O_C x K_H x K_W x I_C weights: output (oh, ow)
// reads row oh*S_H + kh*D_H, column ow*S_W + kw*D_W. A work-item is a TILE_N x TILE_I_H x TILE_M output tile (-D, as SA).
// shape_args: I_C I_H I_W O_C O_H O_W K_H K_W S_H S_W D_H D_W P_T P_L P_B P_R (example 64 16 16 64 16 16 3 3, then 1s).
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef TILE_M
#define TILE_M 16
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef TILE_K
#define TILE_K 64
#endif
#ifndef TILE_I_H
#define TILE_I_H 8
#endif
#ifndef SA
#define SA 8
#endif

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;

static const int32_t ARG_X   = 0;
static const int32_t ARG_W   = 1;
static const int32_t ARG_OUT = 2;

static const int64_t BUF_ACC[1] = {0};
static const int64_t BUF_W[1]   = {1};
static const int64_t BUF_X[1]   = {2};

static const int64_t MM_READ[3] = {0, 1, 2};

static const int32_t SYNC  = 0;
static const int32_t ASYNC = 1;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;
static const uint64_t SEM_X    = 1;
static const uint64_t SEM_W    = 2;

static const int32_t CT_SEGMENT = 0;

static const int64_t  NODES        = 2;
static const uint64_t TID_ACC_INIT = 0;
static const uint64_t TID_MATMUL   = 1;

static const int32_t AXIS_O_C = 0;
static const int32_t AXIS_O_H = 1;
static const int32_t AXIS_O_W = 2;
static const int32_t AXIS_I_C = 3;

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {
    TILE_N * TILE_I_H * TILE_M * KERNEL_ELEM_BYTES,
    KERNEL_CEIL(TILE_K, SA) * TILE_N * SA * KERNEL_ELEM_BYTES,
    TILE_I_H * KERNEL_CEIL(TILE_M, SA) * SA * TILE_K * KERNEL_ELEM_BYTES};

#define CONV_NAMES "I_C I_H I_W O_C O_H O_W K_H K_W S_H S_W D_H D_W P_T P_L P_B P_R"
static const int32_t CONV_ARGS = 16;
static const int32_t CONV_EXTENTS = 12;

// One output tile at (oc0, oh0, ow0). A kernel position and TILE_K input channels is one compute, fed the weights
// and the one input pixel each output of the tile reads there: S_H rows and S_W columns apart. A tile short of
// outputs or channels loads, in SA-wide pieces, only what is left, and computes under the tile id of its variant.
static void conv_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t oc0 = iv[0], oh0 = iv[1], ow0 = iv[2];
  const int64_t I_C = iv[3], I_H = iv[4], I_W = iv[5], O_C = iv[6], O_H = iv[7], O_W = iv[8], K_H = iv[9], K_W = iv[10];
  const int64_t S_H = iv[11], S_W = iv[12], D_H = iv[13], D_W = iv[14];
  const int64_t X_H = I_H + iv[15] + iv[17], X_W = I_W + iv[16] + iv[18];
  const int64_t X_ROW = X_W * I_C;
  const int64_t tn = kernel_extent(O_C, oc0, TILE_N), th = kernel_extent(O_H, oh0, TILE_I_H);
  const int64_t tw = kernel_extent(O_W, ow0, TILE_M);
  const int64_t out_short = kernel_short(tn, TILE_N, AXIS_O_C) | kernel_short(th, TILE_I_H, AXIS_O_H) |
                            kernel_short(tw, TILE_M, AXIS_O_W);
  const int64_t TILE_OUT[4]   = {1, tn, th, tw};
  const int64_t STRIDE_X[4]   = {X_H * X_ROW, S_H * X_ROW, S_W * I_C, 1};
  const int64_t STRIDE_W[4]   = {I_C * O_C * K_W, I_C * O_C, I_C, 1};
  const int64_t STRIDE_OUT[4] = {0, O_H * O_W, O_W, 1};

  togsim_compute(ctx, kernel_tile_id(out_short, NODES, TID_ACC_INIT), CT_SEGMENT, 0, nullptr,
                 nullptr, 0, BUF_ACC, 1);

  for (int64_t kh = 0; kh < K_H; ++kh) {
    for (int64_t kw = 0; kw < K_W; ++kw) {
      for (int64_t ic0 = 0; ic0 < I_C; ic0 += TILE_K) {
        const int64_t tk = kernel_extent(I_C, ic0, TILE_K);
        const int64_t x_off = (oh0 * S_H + kh * D_H) * X_ROW + (ow0 * S_W + kw * D_W) * I_C + ic0;
        const int64_t w_off = (kh * K_W + kw) * O_C * I_C + oc0 * I_C + ic0;

        for (int64_t s = 0; s < tk; s += SA) {
          const int64_t TILE_W[4] = {1, 1, tn, kernel_extent(tk, s, SA)};
          togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_W, (uint64_t)(w_off + s),
                     4, TILE_W, STRIDE_W, ELEM_BITS,
                     ASYNC, SEM_W, nullptr, 0, BUF_W, 1, DENSE);
        }
        for (int64_t r = 0; r < th; ++r) {
          for (int64_t s = 0; s < tw; s += SA) {
            const int64_t TILE_X[4] = {1, 1, kernel_extent(tw, s, SA), tk};
            togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X,
                       (uint64_t)(x_off + r * S_H * X_ROW + s * S_W * I_C),
                       4, TILE_X, STRIDE_X, ELEM_BITS,
                       ASYNC, SEM_X, nullptr, 0, BUF_X, 1, DENSE);
          }
        }
        togsim_memory_barrier(ctx, SEM_W, tn * tk);
        togsim_memory_barrier(ctx, SEM_X, th * tw * tk);
        togsim_compute(ctx, kernel_tile_id(out_short | kernel_short(tk, TILE_K, AXIS_I_C), NODES, TID_MATMUL),
                       CT_SEGMENT, 0, nullptr, MM_READ, 3, BUF_ACC, 1);
      }
    }
  }

  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT,
             (uint64_t)(oc0 * O_H * O_W + oh0 * O_W + ow0),
             4, TILE_OUT, STRIDE_OUT, ELEM_BITS,
             SYNC, SEM_NONE, BUF_ACC, 1, nullptr, 0, DENSE);
}

// Refuse a shape whose last output along `axis` reads past the padded input: its loads would leave the tensor.
static void conv_window_fits(const char* axis, int64_t out, int64_t stride, int64_t kernel, int64_t dilation,
                             int64_t padded) {
  const int64_t last = (out - 1) * stride + (kernel - 1) * dilation;
  if (last >= padded) {
    fprintf(stderr, "kernel conv: output %s %lld reads input %s %lld, and the padded input has %lld %ss\n",
            axis, (long long)(out - 1), axis, (long long)last, (long long)padded, axis);
    exit(1);
  }
}

// One work-item per output tile: output channels, then rows, then columns.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[CONV_ARGS] = {64, 16, 16, 64, 16, 16, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1};
  int64_t iv[3 + CONV_ARGS];
  int64_t* shape = iv + 3;
  kernel_shape_from("conv", CONV_NAMES, CONV_ARGS, CONV_EXTENTS, EXAMPLE, shape_args, n, shape);
  const int64_t O_C = shape[3], O_H = shape[4], O_W = shape[5];
  conv_window_fits("row", O_H, shape[8], shape[6], shape[10], shape[1] + shape[12] + shape[14]);
  conv_window_fits("column", O_W, shape[9], shape[7], shape[11], shape[2] + shape[13] + shape[15]);
  int64_t key = 0;
  for (iv[0] = 0; iv[0] < O_C; iv[0] += TILE_N) {
    for (iv[1] = 0; iv[1] < O_H; iv[1] += TILE_I_H) {
      for (iv[2] = 0; iv[2] < O_W; iv[2] += TILE_M) {
        togsim_dispatch(ctx, conv_tile, iv, 3 + CONV_ARGS, key++);
      }
    }
  }
}
