// ITEMS work-items (key 10, 11, ...), each loading W_BYTES of weights into buffer 1 (tensor 0, the same elements in
// every work-item) and A_BYTES of input into buffer 2 (tensor 1, its own elements); one compute reads both into
// buffer 3, as large as the input, which is stored to tensor 2. The defines below change what happens to buffer 1.
#include "togsim_runtime.h"

#ifndef ITEMS
#define ITEMS 4
#endif
#ifndef W_BYTES
#define W_BYTES 65536
#endif
#ifndef A_BYTES
#define A_BYTES 4096
#endif
#ifndef LAST_A_BYTES
#define LAST_A_BYTES A_BYTES
#endif
#ifndef TAIL_BYTES
#define TAIL_BYTES 0
#endif
#ifndef SIDE_BYTES
#define SIDE_BYTES 0
#endif
#define MIB8 (8u << 20)
#define TAIL_OR_1 (TAIL_BYTES ? TAIL_BYTES : 1)
#define SIDE_OR_1 (SIDE_BYTES ? SIDE_BYTES : 1)

// -DMOVING: own weights per work-item. -DSYNC: a synchronous weights load. -DSTORED: the output goes to tensor 0.
// -DWRITTEN: the compute also writes buffer 1. -DOTHER: tensor 1 is loaded into buffer 1 after the store. -DTWICE: the
// weights twice. -DTAIL_BYTES=n: n bytes into buffer 4, computed with 3 into 5. -DSIDE_BYTES=n: own n bytes into 6, computed.
static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t a_bytes = iv[0] == ITEMS - 1 ? LAST_A_BYTES : A_BYTES;
  const int64_t w_dims[1] = {W_BYTES}, a_dims[1] = {a_bytes}, tail_dims[1] = {TAIL_OR_1};
  const int64_t w[1] = {1}, a[1] = {2}, in[2] = {1, 2}, out[1] = {3}, out_w[2] = {3, 1}, tail[1] = {4};
  const int64_t out_tail[2] = {3, 4}, last[1] = {5}, side[1] = {6}, side_dims[1] = {SIDE_OR_1};
  uint64_t w_at = 0;
  int32_t w_async = 1, w_loads = 1;
#if defined(MOVING)
  w_at = (uint64_t)iv[0] * W_BYTES;
#endif
#if defined(SYNC)
  w_async = 0;
#endif
#if defined(TWICE)
  w_loads = 2;
#endif
  for (int32_t i = 0; i < w_loads; i++)
    togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, w_at, 1, w_dims, nullptr, 8, w_async, 1, nullptr, 0, w, 1, 0);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 1, (uint64_t)iv[0] * A_BYTES, 1, a_dims, nullptr, 8, 1, 2, nullptr, 0, a, 1, 0);
  if (w_async) togsim_memory_barrier(ctx, 1, (int64_t)w_loads * W_BYTES);
  togsim_memory_barrier(ctx, 2, a_bytes);
#if defined(WRITTEN)
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 2, out_w, 2);
#else
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 2, out, 1);
#endif
  if (TAIL_BYTES) {
    togsim_dma(ctx, TOGSIM_DMA_LOAD, 1, 0, 1, tail_dims, nullptr, 8, 1, 3, nullptr, 0, tail, 1, 0);
    togsim_memory_barrier(ctx, 3, TAIL_BYTES);
    togsim_compute(ctx, 0, 0, 0, nullptr, out_tail, 2, last, 1);
  }
#if defined(STORED)
  const int32_t to = 0;
#else
  const int32_t to = 2;
#endif
  togsim_dma(ctx, TOGSIM_DMA_STORE, to, (uint64_t)iv[0] * A_BYTES, 1, a_dims, nullptr, 8, 0, 0, TAIL_BYTES ? last : out,
             1, nullptr, 0, 0);
  if (SIDE_BYTES) {
    togsim_dma(ctx, TOGSIM_DMA_LOAD, 1, MIB8 + (uint64_t)iv[0] * SIDE_BYTES, 1, side_dims, nullptr, 8, 1, 5, nullptr, 0,
               side, 1, 0);
    togsim_memory_barrier(ctx, 5, SIDE_BYTES);
    togsim_compute(ctx, 0, 0, 0, nullptr, side, 1, nullptr, 0);
  }
#if defined(OTHER)
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 1, 0, 1, w_dims, nullptr, 8, 1, 4, nullptr, 0, w, 1, 0);
  togsim_memory_barrier(ctx, 4, W_BYTES);
  togsim_compute(ctx, 0, 0, 0, nullptr, w, 1, nullptr, 0);
#endif
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < ITEMS; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, tile, iv, 1, 10 + i);
  }
}

extern "C" const int32_t togsim_spad_buffer_count = 7;
extern "C" const int64_t togsim_spad_buffer_bytes[7] = {0,         W_BYTES, LAST_A_BYTES, LAST_A_BYTES,
                                                        TAIL_OR_1, A_BYTES, SIDE_OR_1};
