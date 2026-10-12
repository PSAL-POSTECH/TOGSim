// Layer normalization of (x + skip) over the rows of an M x N tensor: a statistics pass, then the normalize pass.
// shape_args: M N (example 128 256). The tiles are compile-time (-DTM= -DTN= -DTM_STATS=): the spad buffer sizes
// state them.
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef TM
#define TM 64
#endif
#ifndef TN
#define TN 64
#endif
#ifndef TM_STATS
#define TM_STATS 128
#endif

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;

static const int32_t ARG_X    = 0;
static const int32_t ARG_SKIP = 1;
static const int32_t ARG_MEAN = 2;
static const int32_t ARG_VAR  = 3;
static const int32_t ARG_W    = 4;
static const int32_t ARG_B    = 5;
static const int32_t ARG_OUT  = 6;

static const int64_t BUF_X[1]    = {0};
static const int64_t BUF_W[1]    = {1};
static const int64_t BUF_VAR[1]  = {2};
static const int64_t BUF_MEAN[1] = {3};
static const int64_t BUF_SKIP[1] = {4};
static const int64_t BUF_B[1]    = {5};
static const int64_t BUF_OUT[1]  = {6};

static const int64_t READ_X_SKIP[2] = {0, 4};
static const int64_t READ_ALL[6]    = {0, 1, 2, 3, 4, 5};

static const int64_t TILE[2]       = {TM, TN};
static const int64_t TILE_STATS[2] = {TM_STATS, TN};
static const int64_t TILE_ROW[1]   = {TM_STATS};

static const int64_t STRIDE_1D[1]  = {1};
static const int64_t STRIDE_ROW[2] = {1, 0};
static const int64_t STRIDE_COL[2] = {0, 1};

static const int32_t SYNC = 0;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_REDUCE     = 0;
static const uint64_t TID_WRITE_MEAN = 1;
static const uint64_t TID_WRITE_VAR  = 2;
static const uint64_t TID_NORM       = 3;

#define LARGER_TILE_BYTES (KERNEL_MAX(TM, TM_STATS) * TN * KERNEL_ELEM_BYTES)
extern "C" const int32_t togsim_spad_buffer_count = 7;
extern "C" const int64_t togsim_spad_buffer_bytes[7] = {
    LARGER_TILE_BYTES, TM * TN * KERNEL_ELEM_BYTES, LARGER_TILE_BYTES, LARGER_TILE_BYTES,
    LARGER_TILE_BYTES, TM * TN * KERNEL_ELEM_BYTES, TM * TN * KERNEL_ELEM_BYTES};

static void stats_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], N = iv[1];
  const int64_t STRIDE_2D[2] = {N, 1};

  for (int64_t n0 = 0; n0 < N; n0 += TN) {
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
               2, TILE_STATS, STRIDE_2D, ELEM_BITS,
               SYNC, SEM_NONE, nullptr, 0, BUF_X, 1, DENSE);
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_SKIP, (uint64_t)(m0 * N + n0),
               2, TILE_STATS, STRIDE_2D, ELEM_BITS,
               SYNC, SEM_NONE, nullptr, 0, BUF_SKIP, 1, DENSE);
    togsim_compute(ctx, TID_REDUCE, CT_VECTOR, 0, nullptr,
                   READ_X_SKIP, 2, nullptr, 0);
  }

  togsim_compute(ctx, TID_WRITE_MEAN, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_MEAN, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_MEAN, (uint64_t)m0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, SEM_NONE, BUF_MEAN, 1, nullptr, 0, DENSE);
  togsim_compute(ctx, TID_WRITE_VAR, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_VAR, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_VAR, (uint64_t)m0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, SEM_NONE, BUF_VAR, 1, nullptr, 0, DENSE);
}

static void norm_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], n0 = iv[1], N = iv[2];
  const int64_t STRIDE_2D[2] = {N, 1};
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_X, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_SKIP, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_SKIP, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_MEAN, (uint64_t)m0,
             2, TILE, STRIDE_ROW, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_MEAN, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_VAR, (uint64_t)m0,
             2, TILE, STRIDE_ROW, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_VAR, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_W, (uint64_t)n0,
             2, TILE, STRIDE_COL, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_W, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_B, (uint64_t)n0,
             2, TILE, STRIDE_COL, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_B, 1, DENSE);

  togsim_compute(ctx, TID_NORM, CT_VECTOR, 0, nullptr,
                 READ_ALL, 6, BUF_OUT, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(m0 * N + n0),
             2, TILE, STRIDE_2D, ELEM_BITS,
             SYNC, SEM_NONE, BUF_OUT, 1, nullptr, 0, DENSE);
}

// The statistics pass over every TM_STATS-row tile, then one work-item per output tile.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[2] = {128, 256};
  int64_t shape[2];
  kernel_shape("layernorm", "M N", 2, EXAMPLE, shape_args, n, shape);
  const int64_t M = shape[0], N = shape[1];
  int64_t key = 0;

  for (int64_t m0 = 0; m0 < M; m0 += TM_STATS) {
    int64_t iv[2] = {m0, N};
    togsim_dispatch(ctx, stats_tile, iv, 2, key++);
  }
  for (int64_t m0 = 0; m0 < M; m0 += TM) {
    for (int64_t n0 = 0; n0 < N; n0 += TN) {
      int64_t iv[3] = {m0, n0, N};
      togsim_dispatch(ctx, norm_tile, iv, 3, key++);
    }
  }
}
