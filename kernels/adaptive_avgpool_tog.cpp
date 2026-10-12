// Average pooling over a 2 x 2 window with stride 2, into a ROWS x COLS output, one TROW x TCOL tile per work-item:
// the four window taps are four strided loads of the 2 COLS-wide input. shape_args: ROWS COLS (example 1024 16).
// The tile is compile-time (-DTROW= -DTCOL=): the spad buffer sizes state it.
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef TROW
#define TROW 256
#endif
#ifndef TCOL
#define TCOL 16
#endif

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;

static const int32_t ARG_X   = 0;
static const int32_t ARG_OUT = 1;

static const int64_t BUF_W3[1]  = {0};
static const int64_t BUF_W0[1]  = {1};
static const int64_t BUF_W1[1]  = {2};
static const int64_t BUF_W2[1]  = {3};
static const int64_t BUF_OUT[1] = {4};

static const int64_t VEC_READ[4] = {0, 1, 2, 3};

static const int64_t TILE[2] = {TROW, TCOL};

static const int32_t SYNC = 0;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;

static const int32_t CT_VECTOR = 0;

static const uint64_t TID_AVG = 0;

#define TILE_BYTES (TROW * TCOL * KERNEL_ELEM_BYTES)
extern "C" const int32_t togsim_spad_buffer_count = 5;
extern "C" const int64_t togsim_spad_buffer_bytes[5] = {TILE_BYTES, TILE_BYTES, TILE_BYTES, TILE_BYTES, TILE_BYTES};

static void avgpool_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t r0 = iv[0], c0 = iv[1], COLS = iv[2];
  const int64_t IN_W = 2 * COLS, IN_ROW_STRIDE = 2 * IN_W;
  const int64_t STRIDE_IN[2]  = {IN_ROW_STRIDE, 2};
  const int64_t STRIDE_OUT[2] = {COLS, 1};
  const int64_t base = r0 * IN_ROW_STRIDE + c0 * 2;
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)base,
             2, TILE, STRIDE_IN, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_W0, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(base + 1),
             2, TILE, STRIDE_IN, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_W1, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(base + IN_W),
             2, TILE, STRIDE_IN, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_W2, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_X, (uint64_t)(base + IN_W + 1),
             2, TILE, STRIDE_IN, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_W3, 1, DENSE);
  togsim_compute(ctx, TID_AVG, CT_VECTOR, 0, nullptr,
                 VEC_READ, 4, BUF_OUT, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(r0 * COLS + c0),
             2, TILE, STRIDE_OUT, ELEM_BITS,
             SYNC, SEM_NONE, BUF_OUT, 1, nullptr, 0, DENSE);
}

// One work-item per output tile, in row-major order.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[2] = {1024, 16};
  int64_t shape[2];
  kernel_shape("adaptive_avgpool", "ROWS COLS", 2, EXAMPLE, shape_args, n, shape);
  const int64_t ROWS = shape[0], COLS = shape[1];
  int64_t key = 0;
  for (int64_t r0 = 0; r0 < ROWS; r0 += TROW) {
    for (int64_t c0 = 0; c0 < COLS; c0 += TCOL) {
      int64_t iv[3] = {r0, c0, COLS};
      togsim_dispatch(ctx, avgpool_tile, iv, 3, key++);
    }
  }
}
