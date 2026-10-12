// A GEMM-like tile in the segment form (compute_type 0 only): C = A x B over K reduction steps. Each step loads a
// 64x64 tile of A and of B and runs tile 0, the multiply-accumulate, in place on the accumulator (buffer 3);
// tile 1, an epilogue, then turns the accumulator into the output (buffer 4), which is stored.
#include "togsim_runtime.h"

#define T 64
#define K 2

// A is T x (K*T) and B is (K*T) x T elements, so step k's tiles sit at element offsets k*T and k*T*T, and a
// tile's rows are one row of its matrix apart. Reading and writing buffer 3 chains the computes one after another.
static void body(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[2] = {T, T}, a_strides[2] = {K * T, 1}, b_strides[2] = {T, 1};
  const int64_t a[1] = {1}, b[1] = {2}, acc[1] = {3}, a_b_acc[3] = {1, 2, 3}, out[1] = {4};
  for (int64_t k = 0; k < K; k++) {
    togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, (uint64_t)(k * T), 2, dims, a_strides, 32, 0, 0, nullptr, 0, a, 1, 0);
    togsim_dma(ctx, TOGSIM_DMA_LOAD, 1, (uint64_t)(k * T * T), 2, dims, b_strides, 32, 0, 0, nullptr, 0, b, 1, 0);
    togsim_compute(ctx, 0, 0, 0, nullptr, a_b_acc, 3, acc, 1);
  }
  togsim_compute(ctx, 1, 0, 0, nullptr, acc, 1, out, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, 2, 0, 2, dims, b_strides, 32, 0, 0, out, 1, nullptr, 0, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  togsim_dispatch(ctx, body, nullptr, 0, 0);
}

extern "C" const int32_t togsim_spad_buffer_count = 5;
extern "C" const int64_t togsim_spad_buffer_bytes[5] = {0, T * T * 4, T * T * 4, T * T * 4, T * T * 4};
