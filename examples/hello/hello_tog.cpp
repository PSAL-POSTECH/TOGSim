// The smallest TOG: one work-item loads a 64x64 tile of 32-bit elements from tensor 0 into spad buffer 1,
// computes buffer 2 from it (tile 0 of the cycle table), and stores buffer 2 to tensor 1.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64

// The body of the work-item: each call is one modeled instruction, and the buffer ids order them.
static void body(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[2] = {ROWS, COLS}, strides[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, 0, 2, dims, strides, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, 1, 0, 2, dims, strides, 32, 0, 0, out, 1, nullptr, 0, 0);
}

// The entry point, called once when the TOG is loaded: it registers the one work-item.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  togsim_dispatch(ctx, body, nullptr, 0, 0);
}

// The byte size of each spad buffer id named above; id 0 is not a buffer.
extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
