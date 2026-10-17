// Kernel IN of a chain: ITEMS work-items (default three). Work-item i loads 16 KB tile i of tensor argument IN,
// computes and stores tile i + 1 (mod ITEMS) of argument IN + 1, which kernel IN + 1's work-item i + 1 loads. With
// STORE=0 it stores nothing.
#include "togsim_runtime.h"

#ifndef IN
#define IN 0
#endif
#ifndef ITEMS
#define ITEMS 3
#endif
#ifndef STORE
#define STORE 1
#endif

#define ROWS 64
#define COLS 64

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t d[2] = {ROWS, COLS}, st[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  const uint64_t at = (uint64_t)iv[0] * ROWS * COLS, to = (uint64_t)((iv[0] + 1) % ITEMS) * ROWS * COLS;
  togsim_dma(ctx, 0, IN, at, 2, d, st, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
  if (STORE) togsim_dma(ctx, 1, IN + 1, to, 2, d, st, 32, 0, 0, out, 1, nullptr, 0, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < ITEMS; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, tile, iv, 1, i);
  }
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
