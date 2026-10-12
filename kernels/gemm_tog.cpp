// C = A x B for an M x K and a K x N matrix, a TM x TN output tile per work-item and one compute per TM x TK x TN
// block of the K loop. A last tile short on M, N or K moves and computes what is left, under its variant's tile id.
// shape_args: M K N (example 128 256 256). The tile is compile-time (-DTM= -DTN= -DTK=): the spad sizes state it.
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef TM
#define TM 32
#endif
#ifndef TN
#define TN 256
#endif
#ifndef TK
#define TK 64
#endif

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;

static const int32_t ARG_A = 0;
static const int32_t ARG_B = 1;
static const int32_t ARG_C = 2;

static const int64_t BUF_C[1] = {0};
static const int64_t BUF_B[1] = {1};
static const int64_t BUF_A[1] = {2};

static const int64_t MM_READ[3] = {0, 1, 2};

static const int32_t SYNC  = 0;
static const int32_t ASYNC = 1;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;
static const uint64_t SEM_A    = 1;
static const uint64_t SEM_B    = 2;

static const int32_t CT_SEGMENT = 0;

static const int64_t  NODES        = 2;
static const uint64_t TID_ACC_INIT = 0;
static const uint64_t TID_MATMUL   = 1;

static const int32_t AXIS_M = 0;
static const int32_t AXIS_N = 1;
static const int32_t AXIS_K = 2;

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {
    TM * TN * KERNEL_ELEM_BYTES, TK * TN * KERNEL_ELEM_BYTES, TM * TK * KERNEL_ELEM_BYTES};

static void gemm_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t m0 = iv[0], n0 = iv[1], M = iv[2], K = iv[3], N = iv[4];
  const int64_t tm = kernel_extent(M, m0, TM), tn = kernel_extent(N, n0, TN);
  const int64_t out_short = kernel_short(tm, TM, AXIS_M) | kernel_short(tn, TN, AXIS_N);
  const int64_t TILE_C[2] = {tm, tn};
  const int64_t STRIDE_A[2] = {K, 1};
  const int64_t STRIDE_BC[2] = {N, 1};
  togsim_compute(ctx, kernel_tile_id(out_short, NODES, TID_ACC_INIT), CT_SEGMENT, 0, nullptr,
                 nullptr, 0, BUF_C, 1);

  for (int64_t k0 = 0; k0 < K; k0 += TK) {
    const int64_t tk = kernel_extent(K, k0, TK);
    const int64_t TILE_A[2] = {tm, tk};
    const int64_t TILE_B[2] = {tk, tn};
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_A, (uint64_t)(m0 * K + k0),
               2, TILE_A, STRIDE_A, ELEM_BITS,
               ASYNC, SEM_A, nullptr, 0, BUF_A, 1, DENSE);
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_B, (uint64_t)(k0 * N + n0),
               2, TILE_B, STRIDE_BC, ELEM_BITS,
               ASYNC, SEM_B, nullptr, 0, BUF_B, 1, DENSE);
    togsim_memory_barrier(ctx, SEM_B, tk * tn);
    togsim_memory_barrier(ctx, SEM_A, tm * tk);
    togsim_compute(ctx, kernel_tile_id(out_short | kernel_short(tk, TK, AXIS_K), NODES, TID_MATMUL), CT_SEGMENT,
                   0, nullptr, MM_READ, 3, BUF_C, 1);
  }

  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_C, (uint64_t)(m0 * N + n0),
             2, TILE_C, STRIDE_BC, ELEM_BITS,
             SYNC, SEM_NONE, BUF_C, 1, nullptr, 0, DENSE);
}

// One work-item per output tile, in row-major order.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[3] = {128, 256, 256};
  int64_t shape[3];
  kernel_shape("gemm", "M K N", 3, EXAMPLE, shape_args, n, shape);
  const int64_t M = shape[0], K = shape[1], N = shape[2];
  int64_t key = 0;
  for (int64_t m0 = 0; m0 < M; m0 += TM) {
    for (int64_t n0 = 0; n0 < N; n0 += TN) {
      int64_t iv[5] = {m0, n0, M, K, N};
      togsim_dispatch(ctx, gemm_tile, iv, 5, key++);
    }
  }
}
