// One kernel of a session: two work-items (so a two-core partition uses both cores), each a 16 KB load, one
// compute tile and a 16 KB store. compute_type 0: the segment grouping.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t d[2] = {ROWS, COLS}, st[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  const uint64_t at = (uint64_t)iv[0] * ROWS * COLS;
  togsim_dma(ctx, 0, 0, at, 2, d, st, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, 1, 1, at, 2, d, st, 32, 0, 0, out, 1, nullptr, 0, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < 2; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, tile, iv, 1, i);
  }
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
