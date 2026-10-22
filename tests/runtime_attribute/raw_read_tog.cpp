// The raw C ABI, which togsim.hpp never writes so: one scalar attribute, read in togsim_kernel (-DIN_KERNEL, outside
// any work-item), or by a work-item at an id the TOG does not declare (-DUNDECLARED), or a DMA that gives another
// count of element offsets than it has elements (-DSHORT_OFFSETS).
#include "togsim_runtime.h"

extern "C" const int32_t togsim_spad_buffer_count = 2;
extern "C" const int64_t togsim_spad_buffer_bytes[2] = {0, 64};
extern "C" const int32_t togsim_runtime_attribute_count = 1;
extern "C" const int32_t togsim_runtime_attribute_rank[1] = {0};

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)iv; (void)n_iv;
  int32_t rank;
  const int64_t* extents;
#if defined(UNDECLARED)
  togsim_runtime_attribute(ctx, 1, &rank, &extents);
#endif
  const int64_t dims[1] = {16}, strides[1] = {1}, out[1] = {1}, offsets[15] = {};
  const int64_t at = togsim_runtime_attribute(ctx, 0, &rank, &extents)[0];
#if defined(SHORT_OFFSETS)
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, (uint64_t)at, 1, dims, strides, 32, 0, 0, nullptr, 0, out, 1, offsets, 15);
#else
  togsim_dma(ctx, TOGSIM_DMA_LOAD, 0, (uint64_t)at, 1, dims, strides, 32, 0, 0, nullptr, 0, out, 1, nullptr, 0);
#endif
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
#if defined(IN_KERNEL)
  int32_t rank;
  const int64_t* extents;
  togsim_runtime_attribute(ctx, 0, &rank, &extents);
#endif
  togsim_dispatch(ctx, tile, nullptr, 0, 0);
}
