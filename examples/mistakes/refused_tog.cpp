// A valid TOG (an async load, its barrier, a compute, a store) that one -D<NAME> breaks in one way, to show each
// refusal: OTHER_ABI, NO_VERSION, NO_KERNEL, NO_BUFFER_TABLE, UNSIZED_BUFFER, WRONG_COUNT,
// UNSIGNALED_SEM, NO_WAIT, TILE_PAST_TABLE.
#if defined(OTHER_ABI) || defined(NO_VERSION)
#define TOGSIM_HOST
#endif
#include "togsim_runtime.h"

// TOGSIM_HOST makes the header only declare the version symbol, so each of these states its own: another
// version (OTHER_ABI) or none at all (NO_VERSION).
#if defined(OTHER_ABI)
extern "C" const int32_t togsim_abi_version = TOGSIM_ABI_VERSION - 1;
#endif

#define ELEMS (64 * 64)
#define SEM 7

// WRONG_COUNT waits for half of what the load signaled, UNSIGNALED_SEM waits on another semaphore, NO_WAIT
// leaves the async load without a barrier, TILE_PAST_TABLE names a tile_id the one-line cycle table lacks.
static void body(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  const int64_t dims[2] = {64, 64}, strides[2] = {64, 1};
  const int64_t in[1] = {1}, out[1] = {2};
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, 0, 2, dims, strides, 32, 1, SEM, nullptr, 0, in, 1, 0);
#if defined(WRONG_COUNT)
  togsim_memory_barrier(ctx, SEM, ELEMS / 2);
#elif defined(UNSIGNALED_SEM)
  togsim_memory_barrier(ctx, SEM + 1, ELEMS);
#elif !defined(NO_WAIT)
  togsim_memory_barrier(ctx, SEM, ELEMS);
#endif
#if defined(TILE_PAST_TABLE)
  togsim_compute(ctx, 1, 0, 0, nullptr, in, 1, out, 1);
#else
  togsim_compute(ctx, 0, 0, 0, nullptr, in, 1, out, 1);
#endif
  togsim_dma(ctx, TOGSIM_DMA_STORE, 1, 0, 2, dims, strides, 32, 0, 0, out, 1, nullptr, 0, 0);
}

// NO_KERNEL gives the entry point a name TOGSim does not look for.
#if defined(NO_KERNEL)
extern "C" void my_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
#else
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
#endif
  (void)shape_args; (void)n_shape_args;
  togsim_dispatch(ctx, body, nullptr, 0, 0);
}

// UNSIZED_BUFFER states no size for buffer 2, which the store drains; NO_BUFFER_TABLE states none at all.
#if defined(UNSIZED_BUFFER)
extern "C" const int32_t togsim_spad_buffer_count = 2;
extern "C" const int64_t togsim_spad_buffer_bytes[2] = {0, ELEMS * 4};
#elif !defined(NO_BUFFER_TABLE)
extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {0, ELEMS * 4, ELEMS * 4};
#endif
