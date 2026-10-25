// One work-item: a 16 KB load, one compute that names tile id TILE_ID (1 unless defined) and a 16 KB store.
// compute_type 0: the segment grouping. NO_COMPUTE: the load is stored as it is, and no tile id is named.
#include "togsim_runtime.h"
#define SPAD_PER_LANE (((ROWS * COLS * 4) + (ROWS * COLS * 4) + 127) / 128)

#ifndef TILE_ID
#define TILE_ID 1
#endif
#define ROWS 64
#define COLS 64

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t d[2] = {ROWS, COLS}, st[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  togsim_dma(ctx, 0, 0, 0, 2, d, st, 32, 0, 0, nullptr, 0, in, 1, nullptr, 0);
#ifdef NO_COMPUTE
  togsim_dma(ctx, 1, 1, 0, 2, d, st, 32, 0, 0, in, 1, nullptr, 0, nullptr, 0);
#else
  togsim_compute(ctx, (uint64_t)(TILE_ID), 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, 1, 1, 0, 2, d, st, 32, 0, 0, out, 1, nullptr, 0, nullptr, 0);
#endif
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  int64_t iv[1] = {0};
  togsim_dispatch(ctx, tile, iv, 1, 7, SPAD_PER_LANE);
}

extern "C" const int32_t togsim_runtime_attribute_count = 0;
extern "C" const int32_t togsim_runtime_attribute_rank[1] = {0};
