// A valid TOG (a load, a compute, a store) that one -D<NAME> breaks in a way TOGSim does NOT refuse: the run
// exits 0. TENSOR_PAST_LIST is said in a warning; NULL_STRIDES and CONSUMER_FIRST are not said at all.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64

// TENSOR_PAST_LIST names an arg_id tog_tensors.txt lacks, NULL_STRIDES passes no strides, CONSUMER_FIRST calls
// the store before the compute that feeds it.
static void body(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[2] = {ROWS, COLS}, strides[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
#if defined(NULL_STRIDES)
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, 0, 2, dims, nullptr, 32, 0, 0, nullptr, 0, in, 1, 0);
#else
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, 0, 2, dims, strides, 32, 0, 0, nullptr, 0, in, 1, 0);
#endif
#if defined(CONSUMER_FIRST)
  togsim_dma(ctx, TOGSIM_DMA_STORE, 1, 0, 2, dims, strides, 32, 0, 0, out, 1, nullptr, 0, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
#elif defined(TENSOR_PAST_LIST)
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, 2, 0, 2, dims, strides, 32, 0, 0, out, 1, nullptr, 0, 0);
#else
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, 1, 0, 2, dims, strides, 32, 0, 0, out, 1, nullptr, 0, 0);
#endif
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  togsim_dispatch(ctx, body, nullptr, 0, 0);
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
