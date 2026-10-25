// DEPRECATED typed grouping (compute_type 3). Load A on sem 3, wait, A = f(A) in place on the cross-lane
// unit (tile 0, long), wait sem 3 again, then a vector compute (tile 1, short) reads A: it must follow f,
// not the barrier. -DWRONG_COUNT re-waits expecting another count.
#include "togsim_runtime.h"
#define SPAD_PER_LANE (((8 * 16 * 4) + (8 * 16 * 4) + 127) / 128)

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[2] = {8, 16}, st[2] = {16, 1};
  const int64_t a[1] = {1}, out[1] = {2};
  togsim_dma(ctx, 0, 0, 0, 2, dims, st, 32, 1, 3, nullptr, 0, a, 1, nullptr, 0);
  togsim_memory_barrier(ctx, 3, 128);
  togsim_compute(ctx, 0, 3, 0, nullptr, a, 1, a, 1);
#if defined(WRONG_COUNT)
  togsim_memory_barrier(ctx, 3, 64);
#else
  togsim_memory_barrier(ctx, 3, 128);
#endif
  togsim_compute(ctx, 1, 0, 0, nullptr, a, 1, out, 1);
  togsim_dma(ctx, 1, 1, 0, 2, dims, st, 32, 0, 0, out, 1, nullptr, 0, nullptr, 0);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  int64_t iv[1] = {0};
  togsim_dispatch(ctx, tile, iv, 1, 0, SPAD_PER_LANE);
}

extern "C" const int32_t togsim_runtime_attribute_count = 0;
extern "C" const int32_t togsim_runtime_attribute_rank[1] = {0};
