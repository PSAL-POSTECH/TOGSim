// Split TOKENS rows of fused QKV into the query and, behind PAST cached tokens, the key and value caches, TTOK
// rows per work-item; a short last tile has its own tile id. shape_args: TOKENS PAST (example 128 128). The row
// widths are compile-time with the tile (-DTTOK= -DHIDDEN= -DNUM_HEADS= -DNUM_KV_HEADS=): the spad sizes state them.
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef TTOK
#define TTOK 32
#endif
#ifndef HIDDEN
#define HIDDEN 256
#endif
#ifndef NUM_HEADS
#define NUM_HEADS 8
#endif
#ifndef NUM_KV_HEADS
#define NUM_KV_HEADS 8
#endif

static const int64_t CACHE_DIM = HIDDEN / NUM_HEADS * NUM_KV_HEADS;
static const int64_t QKV_WIDTH = HIDDEN + 2 * CACHE_DIM;

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;

static const int32_t ARG_QKV   = 0;
static const int32_t ARG_QUERY = 1;
static const int32_t ARG_KEY   = 2;
static const int32_t ARG_VALUE = 3;

static const int64_t BUF_QKV[1]   = {0};
static const int64_t BUF_QUERY[1] = {1};
static const int64_t BUF_KEY[1]   = {2};
static const int64_t BUF_VALUE[1] = {3};

static const int64_t SPLIT_WRITE[3] = {1, 2, 3};

static const int64_t STRIDE_QKV[2]   = {QKV_WIDTH, 1};
static const int64_t STRIDE_QUERY[2] = {HIDDEN, 1};
static const int64_t STRIDE_CACHE[2] = {CACHE_DIM, 1};

static const int32_t SYNC = 0;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;

static const int32_t CT_VECTOR = 0;

static const int64_t  NODES = 1;
static const uint64_t TID_SPLIT = 0;

static const int32_t AXIS_TOKENS = 0;

extern "C" const int32_t togsim_spad_buffer_count = 4;
extern "C" const int64_t togsim_spad_buffer_bytes[4] = {
    TTOK * QKV_WIDTH * KERNEL_ELEM_BYTES, TTOK * HIDDEN * KERNEL_ELEM_BYTES,
    TTOK * CACHE_DIM * KERNEL_ELEM_BYTES, TTOK * CACHE_DIM * KERNEL_ELEM_BYTES};

static void split_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t t0 = iv[0], TOKENS = iv[1], PAST = iv[2];
  const int64_t ttok = kernel_extent(TOKENS, t0, TTOK);
  const int64_t variant = kernel_short(ttok, TTOK, AXIS_TOKENS);
  const int64_t TILE_QKV[2]   = {ttok, QKV_WIDTH};
  const int64_t TILE_QUERY[2] = {ttok, HIDDEN};
  const int64_t TILE_CACHE[2] = {ttok, CACHE_DIM};

  togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_QKV, (uint64_t)(t0 * QKV_WIDTH),
             2, TILE_QKV, STRIDE_QKV, ELEM_BITS,
             SYNC, SEM_NONE, nullptr, 0, BUF_QKV, 1, DENSE);

  togsim_compute(ctx, kernel_tile_id(variant, NODES, TID_SPLIT), CT_VECTOR, 0, nullptr,
                 BUF_QKV, 1, SPLIT_WRITE, 3);

  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_QUERY, (uint64_t)(t0 * HIDDEN),
             2, TILE_QUERY, STRIDE_QUERY, ELEM_BITS,
             SYNC, SEM_NONE, BUF_QUERY, 1, nullptr, 0, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_KEY,
             (uint64_t)((PAST + t0) * CACHE_DIM),
             2, TILE_CACHE, STRIDE_CACHE, ELEM_BITS,
             SYNC, SEM_NONE, BUF_KEY, 1, nullptr, 0, DENSE);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_VALUE,
             (uint64_t)((PAST + t0) * CACHE_DIM),
             2, TILE_CACHE, STRIDE_CACHE, ELEM_BITS,
             SYNC, SEM_NONE, BUF_VALUE, 1, nullptr, 0, DENSE);
}

// One work-item per TTOK-row tile.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[2] = {128, 128};
  int64_t shape[2];
  kernel_shape("kvcache_concat", "TOKENS PAST", 2, EXAMPLE, shape_args, n, shape);
  const int64_t TOKENS = shape[0], PAST = shape[1];
  int64_t key = 0;
  for (int64_t t0 = 0; t0 < TOKENS; t0 += TTOK) {
    int64_t iv[3] = {t0, TOKENS, PAST};
    togsim_dispatch(ctx, split_tile, iv, 3, key++);
  }
}
