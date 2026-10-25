// Four work-items, each a 16 KB load, one compute and a 16 KB store; work-item i names tile id TILE_IDS[i]
// ({0, 0, 1, 1} unless defined). The load names arg_id LOAD_ARG (0), the store STORE_ARG (1).
// compute_type 0: the segment grouping.
#include "togsim_runtime.h"
#define SPAD_PER_LANE (((ROWS * COLS * 4) + (ROWS * COLS * 4) + 127) / 128)

#ifndef TILE_IDS
#define TILE_IDS {0, 0, 1, 1}
#endif
#ifndef LOAD_ARG
#define LOAD_ARG 0
#endif
#ifndef STORE_ARG
#define STORE_ARG 1
#endif
#define ROWS 64
#define COLS 64

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  static const uint64_t tile_ids[4] = TILE_IDS;
  const int64_t d[2] = {ROWS, COLS}, st[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  const uint64_t at = (uint64_t)iv[0] * ROWS * COLS;
  togsim_dma(ctx, 0, LOAD_ARG, at, 2, d, st, 32, 0, 0, nullptr, 0, in, 1, nullptr, 0);
  togsim_compute(ctx, tile_ids[iv[0]], 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, 1, STORE_ARG, at, 2, d, st, 32, 0, 0, out, 1, nullptr, 0, nullptr, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < 4; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, tile, iv, 1, i, SPAD_PER_LANE);
  }
}

extern "C" const int32_t togsim_runtime_attribute_count = 0;
extern "C" const int32_t togsim_runtime_attribute_rank[1] = {0};
