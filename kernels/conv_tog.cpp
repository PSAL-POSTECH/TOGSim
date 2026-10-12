// Stride-1 convolution of an I_C x (I_H+2) x (I_W+2) padded input by O_C x K_H x K_W x I_C weights. A work-item
// is a TILE_N x TILE_I_H x TILE_M output tile; each kernel position and TILE_K input channels is one compute.
// shape_args: I_C I_H I_W O_C O_H O_W K_H K_W (example 64 16 16 64 16 16 3 3). Compile-time: the TILE_* and SA.
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

static const int64_t W_SUBTILES = KERNEL_CEIL(TILE_K, SA);
static const int64_t X_SUBTILES = KERNEL_CEIL(TILE_M, SA);

static const int32_t ARG_X   = 0;
static const int32_t ARG_W   = 1;
static const int32_t ARG_OUT = 2;

static const int64_t BUF_ACC[1] = {0};
static const int64_t BUF_W[1]   = {1};
static const int64_t BUF_X[1]   = {2};

static const int64_t MM_READ[3] = {0, 1, 2};

static const int64_t TILE_X[4]   = {1, 1, SA, TILE_K};
static const int64_t TILE_W[4]   = {1, 1, TILE_N, SA};
static const int64_t TILE_OUT[4] = {1, TILE_N, TILE_I_H, TILE_M};

static const int32_t SYNC  = 0;
static const int32_t ASYNC = 1;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;
static const uint64_t SEM_X    = 1;
static const uint64_t SEM_W    = 2;

static const int32_t CT_SEGMENT = 0;

static const uint64_t TID_ACC_INIT = 0;
static const uint64_t TID_MATMUL   = 1;

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {
    TILE_N * TILE_I_H * TILE_M * KERNEL_ELEM_BYTES,
    KERNEL_CEIL(TILE_K, SA) * TILE_N * SA * KERNEL_ELEM_BYTES,
    TILE_I_H * KERNEL_CEIL(TILE_M, SA) * SA * TILE_K * KERNEL_ELEM_BYTES};

static void conv_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t oc0 = iv[0], oh0 = iv[1], ow0 = iv[2];
  const int64_t I_C = iv[3], I_H = iv[4], I_W = iv[5], O_C = iv[6], O_H = iv[7], O_W = iv[8], K_H = iv[9], K_W = iv[10];
  const int64_t STRIDE_X[4]   = {(I_H + 2) * (I_W + 2) * I_C, (I_W + 2) * I_C, I_C, 1};
  const int64_t STRIDE_W[4]   = {I_C * O_C * K_W, I_C * O_C, I_C, 1};
  const int64_t STRIDE_OUT[4] = {0, O_H * O_W, O_W, 1};
  const int64_t X_ROW = (I_W + 2) * I_C;

  togsim_compute(ctx, TID_ACC_INIT, CT_SEGMENT, 0, nullptr,
                 nullptr, 0, BUF_ACC, 1);

  for (int64_t kh = 0; kh < K_H; ++kh) {
    for (int64_t kw = 0; kw < K_W; ++kw) {
      for (int64_t ic0 = 0; ic0 < I_C; ic0 += TILE_K) {
        const int64_t x_off = (oh0 + kh) * X_ROW + (ow0 + kw) * I_C + ic0;
        const int64_t w_off = (kh * K_W + kw) * O_C * I_C + oc0 * I_C + ic0;

        for (int64_t s = 0; s < W_SUBTILES; ++s) {
          togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_W, (uint64_t)(w_off + s * SA),
                     4, TILE_W, STRIDE_W, ELEM_BITS,
                     ASYNC, SEM_W, nullptr, 0, BUF_W, 1, DENSE);
        }
        for (int64_t r = 0; r < TILE_I_H; ++r) {
          for (int64_t s = 0; s < X_SUBTILES; ++s) {
            togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X,
                       (uint64_t)(x_off + r * X_ROW + s * SA * I_C),
                       4, TILE_X, STRIDE_X, ELEM_BITS,
                       ASYNC, SEM_X, nullptr, 0, BUF_X, 1, DENSE);
          }
        }
        togsim_memory_barrier(ctx, SEM_W, W_SUBTILES * TILE_N * SA);
        togsim_memory_barrier(ctx, SEM_X, TILE_I_H * X_SUBTILES * SA * TILE_K);
        togsim_compute(ctx, TID_MATMUL, CT_SEGMENT, 0, nullptr,
                       MM_READ, 3, BUF_ACC, 1);
      }
    }
  }

  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT,
             (uint64_t)(oc0 * O_H * O_W + oh0 * O_W + ow0),
             4, TILE_OUT, STRIDE_OUT, ELEM_BITS,
             SYNC, SEM_NONE, BUF_ACC, 1, nullptr, 0, DENSE);
}

// One work-item per output tile: output channels, then rows, then columns.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[8] = {64, 16, 16, 64, 16, 16, 3, 3};
  int64_t shape[8];
  kernel_shape("conv", "I_C I_H I_W O_C O_H O_W K_H K_W", 8, EXAMPLE, shape_args, n, shape);
  const int64_t O_C = shape[3], O_H = shape[4], O_W = shape[5];
  int64_t key = 0;
  for (int64_t oc = 0; oc < O_C; oc += TILE_N) {
    for (int64_t oh = 0; oh < O_H; oh += TILE_I_H) {
      for (int64_t ow = 0; ow < O_W; ow += TILE_M) {
        int64_t iv[11] = {oc, oh, ow, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7]};
        togsim_dispatch(ctx, conv_tile, iv, 11, key++);
      }
    }
  }
}
