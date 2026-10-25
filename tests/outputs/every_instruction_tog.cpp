// A TOG that makes TOGSim print every kind of line it has for the segment grouping: per work-item an async load
// and its barrier, a sync load, a compute, a store, and a scatter whose element offsets are all 0.
// Two work-items; compute_type 0 only.
#include "togsim_runtime.h"
#define SPAD_PER_LANE (((ROWS * COLS * 4) + (ROWS * COLS * 4) + (ROWS * COLS * 4) + (ROWS * 4) + 127) / 128)

#define ROWS 64
#define COLS 64

static const int64_t scattered[ROWS * COLS] = {};

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t d[2] = {ROWS, COLS}, st[2] = {COLS, 1}, idx_d[1] = {ROWS}, idx_st[1] = {1};
  const int64_t a[1] = {1}, b[1] = {2}, both[2] = {1, 2}, out[1] = {3}, idx[1] = {4}, scatter[2] = {3, 4};
  const uint64_t at = (uint64_t)iv[0] * ROWS * COLS;
  togsim_dma(ctx, 0, 0, at, 2, d, st, 32, 1, 5, nullptr, 0, a, 1, nullptr, 0);
  togsim_dma(ctx, 0, 1, at, 2, d, st, 32, 0, 0, nullptr, 0, b, 1, nullptr, 0);
  togsim_dma(ctx, 0, 2, (uint64_t)iv[0] * ROWS, 1, idx_d, idx_st, 32, 0, 0, nullptr, 0, idx, 1, nullptr, 0);
  togsim_memory_barrier(ctx, 5, ROWS * COLS);
  togsim_compute(ctx, 0, 0, 0, nullptr, both, 2, out, 1);
  togsim_dma(ctx, 1, 3, at, 2, d, st, 32, 0, 0, out, 1, nullptr, 0, nullptr, 0);
  togsim_dma(ctx, 1, 4, 0, 2, d, st, 32, 0, 0, scatter, 2, nullptr, 0, scattered, ROWS * COLS);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < 2; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, tile, iv, 1, i, SPAD_PER_LANE);
  }
}

extern "C" const int32_t togsim_runtime_attribute_count = 0;
extern "C" const int32_t togsim_runtime_attribute_rank[1] = {0};
