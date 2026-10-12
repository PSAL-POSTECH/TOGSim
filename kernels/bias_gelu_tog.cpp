// Bias add then GELU over an M x N tensor, one TM x TN tile per work-item: load x, load the bias, compute, store.
// shape_args: M N (example 128 256). The tile is compile-time (-DTM= -DTN=): the spad buffer sizes state it.
// A last tile short on M or N moves and computes what is left, under the tile id of its variant.
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef TM
#define TM 128
#endif
#ifndef TN
#define TN 64
#endif

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;

static const int32_t ARG_X    = 0;
static const int32_t ARG_BIAS = 1;
static const int32_t ARG_OUT  = 2;

static const int64_t BUF_BIAS[1] = {0};
static const int64_t BUF_X[1]    = {1};
static const int64_t BUF_OUT[1]  = {2};

static const int64_t VEC_READ[2] = {0, 1};

static const int64_t STRIDE_BCAST[2] = {0, 1};

static const int32_t SYNC = 0;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;

static const int32_t CT_VECTOR = 0;

static const int64_t  NODES = 1;
static const uint64_t TID_BIAS_GELU = 0;

static const int32_t AXIS_M = 0;
static const int32_t AXIS_N = 1;

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {
    TM * TN * KERNEL_ELEM_BYTES, TM * TN * KERNEL_ELEM_BYTES, TM * TN * KERNEL_ELEM_BYTES};

static void bias_gelu_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], n0 = iv[1], M = iv[2], N = iv[3];
  const int64_t tm = kernel_extent(M, m0, TM), tn = kernel_extent(N, n0, TN);
  const int64_t variant = kernel_short(tm, TM, AXIS_M) | kernel_short(tn, TN, AXIS_N);
  const int64_t TILE[2] = {tm, tn};
  const int64_t STRIDE_2D[2] = {N, 1};
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_X, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_BIAS, (uint64_t)n0,
             2, TILE, STRIDE_BCAST, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_BIAS, 1, DENSE);

  togsim_compute(ctx, kernel_tile_id(variant, NODES, TID_BIAS_GELU), CT_VECTOR, 0, nullptr,
                 VEC_READ, 2, BUF_OUT, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, SEM_NONE, BUF_OUT, 1, nullptr, 0, DENSE);
}

// One work-item per tile, in row-major order.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[2] = {128, 256};
  int64_t shape[2];
  kernel_shape("bias_gelu", "M N", 2, EXAMPLE, shape_args, n, shape);
  const int64_t M = shape[0], N = shape[1];
  int64_t key = 0;
  for (int64_t m0 = 0; m0 < M; m0 += TM) {
    for (int64_t n0 = 0; n0 < N; n0 += TN) {
      int64_t iv[4] = {m0, n0, M, N};
      togsim_dispatch(ctx, bias_gelu_tile, iv, 4, key++);
    }
  }
}
