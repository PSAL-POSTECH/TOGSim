// Embedding lookup with its layer normalization, SEQ token ids into SEQ x DIM, one TSEQ x TDIM tile per work-item.
// shape_args: SEQ DIM (example 128 256). Compile-time: -DTSEQ= -DTDIM= (the spad sizes); a short last tile has its
// own tile id. The table load is modeled dense, as the original was: it depends on the ids but is not an indirect dma.
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef TSEQ
#define TSEQ 128
#endif
#ifndef TDIM
#define TDIM 64
#endif

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;
static const int32_t IDX_BITS  = 64;

static const int32_t ARG_IDS   = 0;
static const int32_t ARG_TABLE = 1;
static const int32_t ARG_OUT   = 2;

static const int64_t BUF_IDS[1] = {0};
static const int64_t BUF_ROW[1] = {1};

static const int64_t VEC_READ[2] = {0, 1};

static const int64_t STRIDE_IDS[2] = {1, 0};
static const int64_t STRIDE_ROW[2] = {0, 1};

static const int32_t SYNC = 0;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;

static const int32_t CT_VECTOR = 0;

static const int64_t  NODES = 1;
static const uint64_t TID_GATHER = 0;

static const int32_t AXIS_SEQ = 0;
static const int32_t AXIS_DIM = 1;

extern "C" const int32_t togsim_spad_buffer_count = 2;
extern "C" const int64_t togsim_spad_buffer_bytes[2] = {TSEQ * TDIM * (IDX_BITS / 8), TSEQ * TDIM * KERNEL_ELEM_BYTES};

static void embed_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t s0 = iv[0], d0 = iv[1], SEQ = iv[2], DIM = iv[3];
  const int64_t tseq = kernel_extent(SEQ, s0, TSEQ), tdim = kernel_extent(DIM, d0, TDIM);
  const int64_t variant = kernel_short(tseq, TSEQ, AXIS_SEQ) | kernel_short(tdim, TDIM, AXIS_DIM);
  const int64_t TILE[2] = {tseq, tdim};
  const int64_t STRIDE_OUT[2] = {DIM, 1};
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_IDS, (uint64_t)s0,
             2, TILE, STRIDE_IDS, IDX_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_IDS, 1, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_TABLE, (uint64_t)d0,
             2, TILE, STRIDE_ROW, ELEM_BITS,
             SYNC, SEM_NONE, BUF_IDS, 1, BUF_ROW, 1, DENSE);

  togsim_compute(ctx, kernel_tile_id(variant, NODES, TID_GATHER), CT_VECTOR, 0, nullptr,
                 VEC_READ, 2, nullptr, 0);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT, (uint64_t)(s0 * DIM + d0),
             2, TILE, STRIDE_OUT, ELEM_BITS,
             SYNC, SEM_NONE, BUF_ROW, 1, nullptr, 0, DENSE);
}

// One work-item per tile, in row-major order.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[2] = {128, 256};
  int64_t shape[2];
  kernel_shape("embed_layernorm", "SEQ DIM", 2, EXAMPLE, shape_args, n, shape);
  const int64_t SEQ = shape[0], DIM = shape[1];
  int64_t key = 0;
  for (int64_t s0 = 0; s0 < SEQ; s0 += TSEQ) {
    for (int64_t d0 = 0; d0 < DIM; d0 += TDIM) {
      int64_t iv[4] = {s0, d0, SEQ, DIM};
      togsim_dispatch(ctx, embed_tile, iv, 4, key++);
    }
  }
}
