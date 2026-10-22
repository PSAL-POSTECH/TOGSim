// DEPRECATED typed grouping (compute_type 1 and 2). One work-item: 16 KB loads of buffers 1 and 2, a preload in place
// on buffer 1, and a matmul that reads both. The matmul overlaps the preload: it waits for the preload's issue, not
// its finish, and through it for the load of buffer 1, so both loads are in the spad when it is issued: 32 KB.
#include "togsim_runtime.h"

#define BYTES 16384

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[1] = {BYTES}, w[1] = {1}, y[1] = {2}, both[2] = {1, 2};
  togsim_dma(ctx, 0, 0, 0, 1, dims, nullptr, 8, 0, 0, nullptr, 0, w, 1, nullptr, 0);
  togsim_dma(ctx, 0, 1, 0, 1, dims, nullptr, 8, 0, 0, nullptr, 0, y, 1, nullptr, 0);
  togsim_compute(ctx, 0, 2, 0, nullptr, w, 1, w, 1);
  togsim_compute(ctx, 0, 1, 0, nullptr, both, 2, nullptr, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  int64_t iv[1] = {0};
  togsim_dispatch(ctx, tile, iv, 1, 0);
}

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, BYTES, BYTES};
extern "C" const int32_t togsim_runtime_attribute_count = 0;
extern "C" const int32_t togsim_runtime_attribute_rank[1] = {0};
