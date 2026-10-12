// A tiled kernel: an M x N grid of work-items, M and N read from tog_shape.txt. Work-item (m, n) loads its own
// 64x64 tile of tensor 0, computes on it (every work-item runs tile 0 of the cycle table) and stores it to tensor 1.
#include "togsim_runtime.h"

#define ROWS 64
#define COLS 64

// iv = {m, n, N}, the copy togsim_dispatch took: all a body may read besides ctx.
static void body(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t dims[2] = {ROWS, COLS}, strides[2] = {COLS, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  const uint64_t offset = (uint64_t)(iv[0] * iv[2] + iv[1]) * ROWS * COLS;
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, offset, 2, dims, strides, 32, 0, 0, nullptr, 0, in, 1, 0);
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, 1, offset, 2, dims, strides, 32, 0, 0, out, 1, nullptr, 0, 0);
}

// One dispatch per grid cell, in ordinary C++ loops. TOGSim binds the work-items to cores round-robin, in this
// order; the last argument is a key that names the work-item, not a core.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  const int64_t M = n_shape_args > 0 ? shape_args[0] : 1, N = n_shape_args > 1 ? shape_args[1] : 1;
  for (int64_t m = 0; m < M; m++)
    for (int64_t n = 0; n < N; n++) {
      int64_t iv[3] = {m, n, N};
      togsim_dispatch(ctx, body, iv, 3, m * N + n);
    }
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ROWS * COLS * 4, ROWS * COLS * 4};
