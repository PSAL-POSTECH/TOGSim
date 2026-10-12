// Two asynchronous loads and the barrier that waits for both: operand tiles from tensors 0 and 1 go to spad
// buffers 1 and 2, a compute reads both and writes buffer 3, and the store drains buffer 3 to tensor 2.
// -DSYNC makes the loads synchronous, with no barrier; -DTWO_STEPS adds a first compute that reads buffer 1 only.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64
#define SEM 7

// An async load counts as finished once its requests are sent, and signals SEM with its element count; the
// barrier waits for the data of every load signaled on SEM and must state the sum of their counts.
static void body(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[2] = {ROWS, COLS}, strides[2] = {COLS, 1};
  const int64_t a[1] = {1}, b[1] = {2}, ab[2] = {1, 2}, out[1] = {3};
#if defined(SYNC)
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, 0, 2, dims, strides, 32, 0, 0, nullptr, 0, a, 1, 0);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 1, 0, 2, dims, strides, 32, 0, 0, nullptr, 0, b, 1, 0);
#else
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, 0, 2, dims, strides, 32, 1, SEM, nullptr, 0, a, 1, 0);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 1, 0, 2, dims, strides, 32, 1, SEM, nullptr, 0, b, 1, 0);
  togsim_memory_barrier(ctx, SEM, 2 * ROWS * COLS);
#endif
#if defined(TWO_STEPS)
  const int64_t partial[1] = {4}, partial_b[2] = {4, 2};
  togsim_compute(ctx, 0, 0, 0, nullptr, a, 1, partial, 1);
  togsim_compute(ctx, 0, 0, 0, nullptr, partial_b, 2, out, 1);
#else
  togsim_compute(ctx, 0, 0, 0, nullptr, ab, 2, out, 1);
#endif
  togsim_dma(ctx, TOGSIM_DMA_STORE, 2, 0, 2, dims, strides, 32, 0, 0, out, 1, nullptr, 0, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  togsim_dispatch(ctx, body, nullptr, 0, 0);
}

// Buffer 4 is the first compute's result under -DTWO_STEPS; no DMA moves it.
extern "C" const int32_t togsim_spad_buffer_count = 5;
extern "C" const int64_t togsim_spad_buffer_bytes[5] = {0, ROWS * COLS * 4, ROWS * COLS * 4, ROWS * COLS * 4,
                                                        ROWS * COLS * 4};
