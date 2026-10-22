// ITEMS work-items (key 10, 11, ...), each loading a 16 KB buffer and one of BIG_BYTES. One compute reads both into
// a 16 KB output that is stored: the three are in the spad at once, 32 KB + BIG_BYTES. -DAPART: a compute consumes
// each input by itself and nothing is written, so no two are held together; with -DLEFT_OPEN the last work-item
// then loads the 16 KB buffer once more and nothing reads it; with -DREFILL the 16 KB buffer is loaded twice before
// its compute, which waits for the second load only. -DSMALL_FIRST: work-item 0 loads no big.
#include "togsim_runtime.h"

#ifndef ITEMS
#define ITEMS 2
#endif
#ifndef BIG_BYTES
#define BIG_BYTES 16384
#endif
#define SMALL_BYTES 16384

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t small[1] = {SMALL_BYTES}, big[1] = {BIG_BYTES};
  const int64_t a[1] = {1}, b[1] = {2}, in[2] = {1, 2}, out[1] = {3};
  const uint64_t base = (uint64_t)iv[0] * BIG_BYTES;
#if defined(APART)
  togsim_dma(ctx, 0, 0, base, 1, small, nullptr, 8, 0, 0, nullptr, 0, a, 1, nullptr, 0);
#if defined(REFILL)
  togsim_dma(ctx, 0, 0, base, 1, small, nullptr, 8, 0, 0, nullptr, 0, a, 1, nullptr, 0);
#endif
  togsim_compute(ctx, 0, 0, 0, nullptr, a, 1, nullptr, 0);
  togsim_dma(ctx, 0, 1, base, 1, big, nullptr, 8, 0, 0, nullptr, 0, b, 1, nullptr, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, b, 1, nullptr, 0);
#if defined(LEFT_OPEN)
  if (iv[0] == ITEMS - 1) togsim_dma(ctx, 0, 0, base, 1, small, nullptr, 8, 0, 0, nullptr, 0, a, 1, nullptr, 0);
#endif
#else
  bool with_big = true;
#if defined(SMALL_FIRST)
  with_big = iv[0] != 0;
#endif
  togsim_dma(ctx, 0, 0, base, 1, small, nullptr, 8, 1, 1, nullptr, 0, a, 1, nullptr, 0);
  if (with_big) togsim_dma(ctx, 0, 1, base, 1, big, nullptr, 8, 1, 2, nullptr, 0, b, 1, nullptr, 0);
  togsim_memory_barrier(ctx, 1, SMALL_BYTES);
  if (with_big) togsim_memory_barrier(ctx, 2, BIG_BYTES);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, with_big ? 2 : 1, out, 1);
  togsim_dma(ctx, 1, 2, base, 1, small, nullptr, 8, 0, 0, out, 1, nullptr, 0, nullptr, 0);
#endif
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < ITEMS; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, tile, iv, 1, 10 + i);
  }
}

extern "C" const int32_t togsim_spad_buffer_count = 4;
extern "C" const int64_t togsim_spad_buffer_bytes[4] = {0, SMALL_BYTES, BIG_BYTES, SMALL_BYTES};
extern "C" const int32_t togsim_runtime_attribute_count = 0;
extern "C" const int32_t togsim_runtime_attribute_rank[1] = {0};
