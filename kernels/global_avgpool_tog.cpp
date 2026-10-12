// Global average pooling of C channels of HW elements each: a sum pass over TC x THW tiles, then the scale pass.
// shape_args: C HW (example 256 64). The tiles are compile-time (-DTC= -DTHW= -DTC_SCALE=): the spad buffer sizes
// state them.
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef TC
#define TC 256
#endif
#ifndef THW
#define THW 64
#endif
#ifndef TC_SCALE
#define TC_SCALE 128
#endif

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;

static const int32_t ARG_X   = 0;
static const int32_t ARG_SUM = 1;
static const int32_t ARG_OUT = 2;

static const int64_t BUF_X[1]   = {0};
static const int64_t BUF_SUM[1] = {1};
static const int64_t BUF_OUT[1] = {2};

static const int64_t TILE[2]       = {TC, THW};
static const int64_t TILE_ROW[1]   = {TC};
static const int64_t TILE_SCALE[1] = {TC_SCALE};

static const int64_t STRIDE_1D[1] = {1};

static const int32_t SYNC = 0;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_REDUCE = 0;
static const uint64_t TID_WRITE  = 1;
static const uint64_t TID_SCALE  = 2;

extern "C" const int32_t togsim_spad_buffer_count = 3;
extern "C" const int64_t togsim_spad_buffer_bytes[3] = {
    TC * THW * KERNEL_ELEM_BYTES, KERNEL_MAX(TC, TC_SCALE) * KERNEL_ELEM_BYTES, TC_SCALE * KERNEL_ELEM_BYTES};

static void sum_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t c0 = iv[0], HW = iv[1];
  const int64_t STRIDE_2D[2] = {HW, 1};

  for (int64_t hw0 = 0; hw0 < HW; hw0 += THW) {
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(c0 * HW + hw0),
               2, TILE, STRIDE_2D, ELEM_BITS,
               SYNC, SEM_NONE, nullptr, 0, BUF_X, 1, DENSE);
    togsim_compute(ctx, TID_REDUCE, CT_VECTOR, 0, nullptr,
                   BUF_X, 1, nullptr, 0);
  }

  togsim_compute(ctx, TID_WRITE, CT_VECTOR, 0, nullptr,
                 nullptr, 0, BUF_SUM, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_SUM, (uint64_t)c0,
             1, TILE_ROW, STRIDE_1D, ELEM_BITS,
             SYNC, SEM_NONE, BUF_SUM, 1, nullptr, 0, DENSE);
}

static void scale_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t c0 = iv[0];
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_SUM, (uint64_t)c0,
             1, TILE_SCALE, STRIDE_1D, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_SUM, 1, DENSE);

  togsim_compute(ctx, TID_SCALE, CT_VECTOR, 0, nullptr,
                 BUF_SUM, 1, BUF_OUT, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)c0,
             1, TILE_SCALE, STRIDE_1D, ELEM_BITS,
             SYNC, SEM_NONE, BUF_OUT, 1, nullptr, 0, DENSE);
}

// The sum pass over every TC-channel tile, then the scale pass over every TC_SCALE-channel tile.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[2] = {256, 64};
  int64_t shape[2];
  kernel_shape("global_avgpool", "C HW", 2, EXAMPLE, shape_args, n, shape);
  const int64_t C = shape[0], HW = shape[1];
  int64_t key = 0;

  for (int64_t c0 = 0; c0 < C; c0 += TC) {
    int64_t iv[2] = {c0, HW};
    togsim_dispatch(ctx, sum_tile, iv, 2, key++);
  }
  for (int64_t c0 = 0; c0 < C; c0 += TC_SCALE) {
    int64_t iv[1] = {c0};
    togsim_dispatch(ctx, scale_tile, iv, 1, key++);
  }
}
