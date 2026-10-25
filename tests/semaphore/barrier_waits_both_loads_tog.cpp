// Two async loads signal one semaphore; the first is strided across DRAM rows so the second,
// issued later, arrives first. One barrier waits both, one compute reads the buffer.
// -DWRONG_COUNT, -DUNSIGNALED_SEM or -DNO_WAIT breaks the contract: the barrier's count, its semaphore, or no barrier.
#include "togsim_runtime.h"
#define SPAD_PER_LANE ((((ROWS + 1) * 16 * 4) + (16 * 4) + 127) / 128)

#define ROWS 256
#define ROW_STRIDE 65536

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t big[2] = {ROWS, 16}, small[2] = {1, 16};
  const int64_t big_st[2] = {ROW_STRIDE, 1}, small_st[2] = {16, 1};
  const int64_t buf[1] = {1}, out[1] = {2};
  togsim_dma(ctx, 0, 0, 0, 2, big, big_st, 32, 1, 7, nullptr, 0, buf, 1, nullptr, 0);
  togsim_dma(ctx, 0, 1, 0, 2, small, small_st, 32, 1, 7, nullptr, 0, buf, 1, nullptr, 0);
#if defined(WRONG_COUNT)
  togsim_memory_barrier(ctx, 7, 16);
#elif defined(UNSIGNALED_SEM)
  togsim_memory_barrier(ctx, 8, 16);
#elif !defined(NO_WAIT)
  togsim_memory_barrier(ctx, 7, ROWS * 16 + 16);
#endif
  togsim_compute(ctx, 0, 0, 0, nullptr, buf, 1, out, 1);
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  int64_t iv[1] = {0};
  togsim_dispatch(ctx, tile, iv, 1, 0, SPAD_PER_LANE);
}

extern "C" const int32_t togsim_runtime_attribute_count = 0;
extern "C" const int32_t togsim_runtime_attribute_rank[1] = {0};
