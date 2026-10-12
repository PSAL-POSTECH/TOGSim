// Row softmax of an M x N tensor in three passes: the row max, the row sum of exp(x - max), then the quotient.
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

static const int32_t ARG_X   = 0;
static const int32_t ARG_MAX = 1;
static const int32_t ARG_SUM = 2;
static const int32_t ARG_OUT = 3;

static const int64_t BUF_SUM[1] = {0};
static const int64_t BUF_X[1]   = {1};
static const int64_t BUF_MAX[1] = {2};
static const int64_t BUF_OUT[1] = {3};

static const int64_t READ_X_MAX[2]     = {1, 2};
static const int64_t READ_X_MAX_SUM[3] = {0, 1, 2};

static const int64_t STRIDE_BCAST[2] = {1, 0};
static const int64_t STRIDE_1D[1]    = {1};

static const int32_t SYNC = 0;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_MAX_REDUCE = 0;
static const uint64_t TID_MAX_WRITE  = 1;
static const uint64_t TID_SUM_REDUCE = 2;
static const uint64_t TID_SUM_WRITE  = 3;
static const uint64_t TID_SOFTMAX    = 4;
static const int64_t  NODES          = 5;

static const int32_t AXIS_M = 0;
static const int32_t AXIS_N = 1;

extern "C" const int32_t togsim_spad_buffer_count = 4;
extern "C" const int64_t togsim_spad_buffer_bytes[4] = {
    TM * TN * KERNEL_ELEM_BYTES, TM * TN * KERNEL_ELEM_BYTES, TM * TN * KERNEL_ELEM_BYTES,
    TM * TN * KERNEL_ELEM_BYTES};

static void max_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], M = iv[1], N = iv[2];
  const int64_t tm = kernel_extent(M, m0, TM);
  const int64_t m_short = kernel_short(tm, TM, AXIS_M);
  const int64_t TILE_ROW[1] = {tm};
  const int64_t STRIDE_2D[2] = {N, 1};

  for (int64_t n0 = 0; n0 < N; n0 += TN) {
    const int64_t tn = kernel_extent(N, n0, TN);
    const int64_t variant = m_short | kernel_short(tn, TN, AXIS_N);
    const int64_t TILE[2] = {tm, tn};
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
               2, TILE, STRIDE_2D, ELEM_BITS,
               SYNC, SEM_NONE, nullptr, 0, BUF_X, 1, DENSE);
    togsim_compute(ctx, kernel_tile_id(variant, NODES, TID_MAX_REDUCE), CT_VECTOR, 0, nullptr,
                   BUF_X, 1, nullptr, 0);
  }

  togsim_compute(ctx, kernel_tile_id(m_short, NODES, TID_MAX_WRITE), CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_MAX, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_MAX, (uint64_t)m0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, SEM_NONE, BUF_MAX, 1, nullptr, 0, DENSE);
}

static void sum_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], M = iv[1], N = iv[2];
  const int64_t tm = kernel_extent(M, m0, TM);
  const int64_t m_short = kernel_short(tm, TM, AXIS_M);
  const int64_t TILE_ROW[1] = {tm};
  const int64_t STRIDE_2D[2] = {N, 1};

  for (int64_t n0 = 0; n0 < N; n0 += TN) {
    const int64_t tn = kernel_extent(N, n0, TN);
    const int64_t variant = m_short | kernel_short(tn, TN, AXIS_N);
    const int64_t TILE[2] = {tm, tn};
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
               2, TILE, STRIDE_2D, ELEM_BITS,
               SYNC, SEM_NONE, nullptr, 0, BUF_X, 1, DENSE);
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_MAX, (uint64_t)m0,
               2, TILE, STRIDE_BCAST, ELEM_BITS,
               SYNC, SEM_NONE, nullptr, 0, BUF_MAX, 1, DENSE);
    togsim_compute(ctx, kernel_tile_id(variant, NODES, TID_SUM_REDUCE), CT_VECTOR, 0, nullptr,
                   READ_X_MAX, 2, nullptr, 0);
  }

  togsim_compute(ctx, kernel_tile_id(m_short, NODES, TID_SUM_WRITE), CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_SUM, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_SUM, (uint64_t)m0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, SEM_NONE, BUF_SUM, 1, nullptr, 0, DENSE);
}

static void softmax_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], n0 = iv[1], M = iv[2], N = iv[3];
  const int64_t tm = kernel_extent(M, m0, TM), tn = kernel_extent(N, n0, TN);
  const int64_t variant = kernel_short(tm, TM, AXIS_M) | kernel_short(tn, TN, AXIS_N);
  const int64_t TILE[2] = {tm, tn};
  const int64_t STRIDE_2D[2] = {N, 1};
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_X, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_MAX, (uint64_t)m0,
             2, TILE, STRIDE_BCAST, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_MAX, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_SUM, (uint64_t)m0,
             2, TILE, STRIDE_BCAST, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_SUM, 1, DENSE);

  togsim_compute(ctx, kernel_tile_id(variant, NODES, TID_SOFTMAX), CT_VECTOR, 0, nullptr,
                 READ_X_MAX_SUM, 3, BUF_OUT, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, SEM_NONE, BUF_OUT, 1, nullptr, 0, DENSE);
}

// The max pass over every row tile, then the sum pass, then one work-item per output tile.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[2] = {128, 256};
  int64_t shape[2];
  kernel_shape("softmax", "M N", 2, EXAMPLE, shape_args, n, shape);
  const int64_t M = shape[0], N = shape[1];
  int64_t key = 0;

  for (int64_t m0 = 0; m0 < M; m0 += TM) {
    int64_t iv[3] = {m0, M, N};
    togsim_dispatch(ctx, max_tile, iv, 3, key++);
  }
  for (int64_t m0 = 0; m0 < M; m0 += TM) {
    int64_t iv[3] = {m0, M, N};
    togsim_dispatch(ctx, sum_tile, iv, 3, key++);
  }
  for (int64_t m0 = 0; m0 < M; m0 += TM) {
    for (int64_t n0 = 0; n0 < N; n0 += TN) {
      int64_t iv[4] = {m0, n0, M, N};
      togsim_dispatch(ctx, softmax_tile, iv, 4, key++);
    }
  }
}
