// One work-item: a load of FIRST_BYTES into buffer 1 and a compute that reads it, then, if SECOND_BYTES is not 0, a
// load of SECOND_BYTES into buffer 2 and a compute that reads it. Each buffer has one version, numbered from 0.
#include "togsim_runtime.h"

#ifndef FIRST_BYTES
#define FIRST_BYTES 8192
#endif
#ifndef SECOND_BYTES
#define SECOND_BYTES 0
#endif

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t first[1] = {FIRST_BYTES}, second[1] = {SECOND_BYTES}, a[1] = {1}, b[1] = {2};
  togsim_dma(ctx, 0, 0, 0, 1, first, nullptr, 8, 0, 0, nullptr, 0, a, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, a, 1, nullptr, 0);
  if (SECOND_BYTES) {
    togsim_dma(ctx, 0, 0, FIRST_BYTES, 1, second, nullptr, 8, 0, 0, nullptr, 0, b, 1, 0);
    togsim_compute(ctx, 0, 0, 0, nullptr, b, 1, nullptr, 0);
  }
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  int64_t iv[1] = {0};
  togsim_dispatch(ctx, tile, iv, 1, 0);
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, FIRST_BYTES, SECOND_BYTES ? SECOND_BYTES : 1};
