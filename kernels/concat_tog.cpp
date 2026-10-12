// Concatenation of an A_ROWS x D and a B_ROWS x D tensor along the rows, by DMA alone: each TROW x TD tile is
// loaded and stored where it lands. shape_args: A_ROWS B_ROWS D (example 128 128 256). The tile is compile-time
// (-DTROW= -DTD=): the spad buffer sizes state it.
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef TROW
#define TROW 128
#endif
#ifndef TD
#define TD 16
#endif

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;

static const int32_t ARG_A   = 0;
static const int32_t ARG_B   = 1;
static const int32_t ARG_OUT = 2;

static const int64_t BUF_A[1] = {0};
static const int64_t BUF_B[1] = {1};

static const int64_t TILE[2] = {TROW, TD};

static const int32_t SYNC = 0;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;

extern "C" const int32_t togsim_spad_buffer_count = 2;
extern "C" const int64_t togsim_spad_buffer_bytes[2] = {TROW * TD * KERNEL_ELEM_BYTES, TROW * TD * KERNEL_ELEM_BYTES};

static void concat_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t d0 = iv[0], A_ROWS = iv[1], B_ROWS = iv[2], D = iv[3];
  const int64_t STRIDE[2] = {D, 1};

  for (int64_t r = 0; r < A_ROWS; r += TROW) {
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_A, (uint64_t)(r * D + d0),
               2, TILE, STRIDE, ELEM_BITS,
               SYNC, SEM_NONE, nullptr, 0, BUF_A, 1, DENSE);
    togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(r * D + d0),
               2, TILE, STRIDE, ELEM_BITS,
               SYNC, SEM_NONE, BUF_A, 1, nullptr, 0, DENSE);
  }

  for (int64_t r = 0; r < B_ROWS; r += TROW) {
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_B, (uint64_t)(r * D + d0),
               2, TILE, STRIDE, ELEM_BITS,
               SYNC, SEM_NONE, nullptr, 0, BUF_B, 1, DENSE);
    togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)((r + A_ROWS) * D + d0),
               2, TILE, STRIDE, ELEM_BITS,
               SYNC, SEM_NONE, BUF_B, 1, nullptr, 0, DENSE);
  }
}

// One work-item per TD-wide column tile; each walks the rows of both inputs.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[3] = {128, 128, 256};
  int64_t shape[3];
  kernel_shape("concat", "A_ROWS B_ROWS D", 3, EXAMPLE, shape_args, n, shape);
  const int64_t A_ROWS = shape[0], B_ROWS = shape[1], D = shape[2];
  int64_t key = 0;
  for (int64_t d0 = 0; d0 < D; d0 += TD) {
    int64_t iv[4] = {d0, A_ROWS, B_ROWS, D};
    togsim_dispatch(ctx, concat_tile, iv, 4, key++);
  }
}
