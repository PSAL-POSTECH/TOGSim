// Scaled dot-product attention over SEQ tokens with online softmax: a work-item is Q_LEN queries of one KV head,
// and each KV_BLOCK of keys is one QK compute, the six softmax stages, then one SV compute, per query head of it.
// shape_args: HEADS KV_HEADS SEQ (example 8 8 128). Compile-time: -DQ_LEN= -DKV_BLOCK= -DDHEAD= (the spad sizes).
#include "togsim_runtime.h"

#include "kernel_shape.h"

#ifndef Q_LEN
#define Q_LEN 8
#endif
#ifndef KV_BLOCK
#define KV_BLOCK 64
#endif
#ifndef DHEAD
#define DHEAD 64
#endif

static const int32_t ELEM_BITS = KERNEL_ELEM_BITS;

static const int32_t ARG_Q   = 0;
static const int32_t ARG_K   = 1;
static const int32_t ARG_V   = 2;
static const int32_t ARG_OUT = 3;

static const int64_t BUF_K[1]     = {0};
static const int64_t BUF_Q[1]     = {1};
static const int64_t BUF_V[1]     = {2};
static const int64_t BUF_LOGIT[1] = {3};
static const int64_t BUF_STAT[1]  = {4};
static const int64_t BUF_ACC[1]   = {5};

static const int64_t QK_READ[2]    = {0, 1};
static const int64_t SV_READ[3]    = {2, 3, 5};
static const int64_t STAT_READ[2]  = {3, 4};
static const int64_t LOGIT_READ[2] = {3, 4};
static const int64_t ACC_READ[2]   = {4, 5};

static const int64_t TILE_Q[2]   = {Q_LEN, DHEAD};
static const int64_t TILE_KV[2]  = {KV_BLOCK, DHEAD};
static const int64_t TILE_OUT[2] = {Q_LEN, DHEAD};

static const int64_t STRIDE_Q[2]   = {DHEAD, 1};
static const int64_t STRIDE_KV[2]  = {DHEAD, 1};
static const int64_t STRIDE_OUT[2] = {DHEAD, 1};

static const int32_t SYNC  = 0;
static const int32_t ASYNC = 1;
static const int32_t DENSE = 0;

static const uint64_t SEM_NONE = 0;
static const uint64_t SEM_K    = 1;
static const uint64_t SEM_Q    = 2;
static const uint64_t SEM_V    = 3;

static const int32_t CT_SEGMENT = 0;

static const uint64_t TID_QK      = 0;
static const uint64_t TID_ROWMAX  = 1;
static const uint64_t TID_SUB     = 2;
static const uint64_t TID_EXP     = 3;
static const uint64_t TID_ROWSUM  = 4;
static const uint64_t TID_MAC     = 5;
static const uint64_t TID_RESCALE = 6;
static const uint64_t TID_SV      = 7;
static const uint64_t TID_NORM    = 8;

extern "C" const int32_t togsim_spad_buffer_count = 6;
extern "C" const int64_t togsim_spad_buffer_bytes[6] = {
    KV_BLOCK * DHEAD * KERNEL_ELEM_BYTES, Q_LEN * DHEAD * KERNEL_ELEM_BYTES, KV_BLOCK * DHEAD * KERNEL_ELEM_BYTES,
    Q_LEN * KV_BLOCK * KERNEL_ELEM_BYTES, Q_LEN * KERNEL_ELEM_BYTES, Q_LEN * DHEAD * KERNEL_ELEM_BYTES};

static void attention_tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  const int64_t h = iv[0], q0 = iv[1], HEADS_PER_KV = iv[2], SEQ = iv[3];

  for (int64_t kv0 = 0; kv0 < SEQ; kv0 += KV_BLOCK) {
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_K,
               (uint64_t)(h * SEQ * DHEAD + kv0 * DHEAD),
               2, TILE_KV, STRIDE_KV, ELEM_BITS,
               ASYNC, SEM_K, nullptr, 0, BUF_K, 1, DENSE);
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_Q,
               (uint64_t)(h * SEQ * DHEAD + q0 * DHEAD),
               2, TILE_Q, STRIDE_Q, ELEM_BITS,
               ASYNC, SEM_Q, nullptr, 0, BUF_Q, 1, DENSE);
    togsim_dma(ctx, TOGSIM_DMA_LOAD, ARG_V,
               (uint64_t)(h * SEQ * DHEAD + kv0 * DHEAD),
               2, TILE_KV, STRIDE_KV, ELEM_BITS,
               ASYNC, SEM_V, nullptr, 0, BUF_V, 1, DENSE);

    togsim_memory_barrier(ctx, SEM_K, KV_BLOCK * DHEAD);
    togsim_memory_barrier(ctx, SEM_Q, Q_LEN * DHEAD);
    for (int64_t hh = 0; hh < HEADS_PER_KV; ++hh) {
      togsim_compute(ctx, TID_QK, CT_SEGMENT, 0, nullptr,
                     QK_READ, 2, BUF_LOGIT, 1);
    }

    togsim_compute(ctx, TID_ROWMAX, CT_SEGMENT, 0, nullptr,
                   BUF_LOGIT, 1, BUF_STAT, 1);
    togsim_compute(ctx, TID_SUB, CT_SEGMENT, 0, nullptr,
                   STAT_READ, 2, BUF_LOGIT, 1);
    togsim_compute(ctx, TID_EXP, CT_SEGMENT, 0, nullptr,
                   BUF_LOGIT, 1, BUF_LOGIT, 1);
    togsim_compute(ctx, TID_ROWSUM, CT_SEGMENT, 0, nullptr,
                   BUF_LOGIT, 1, BUF_STAT, 1);
    togsim_compute(ctx, TID_MAC, CT_SEGMENT, 0, nullptr,
                   LOGIT_READ, 2, BUF_STAT, 1);
    togsim_compute(ctx, TID_RESCALE, CT_SEGMENT, 0, nullptr,
                   ACC_READ, 2, BUF_ACC, 1);

    togsim_memory_barrier(ctx, SEM_V, KV_BLOCK * DHEAD);
    for (int64_t hh = 0; hh < HEADS_PER_KV; ++hh) {
      togsim_compute(ctx, TID_SV, CT_SEGMENT, 0, nullptr,
                     SV_READ, 3, BUF_ACC, 1);
    }
  }

  togsim_compute(ctx, TID_NORM, CT_SEGMENT, 0, nullptr,
                 ACC_READ, 2, BUF_ACC, 1);
  togsim_dma(ctx, TOGSIM_DMA_STORE, ARG_OUT,
             (uint64_t)(h * SEQ * DHEAD + q0 * DHEAD),
             2, TILE_OUT, STRIDE_OUT, ELEM_BITS,
             SYNC, SEM_NONE, BUF_ACC, 1, nullptr, 0, DENSE);
}

// One work-item per KV head and Q_LEN-query tile.
extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n) {
  static const int64_t EXAMPLE[3] = {8, 8, 128};
  int64_t shape[3];
  kernel_shape("attention", "HEADS KV_HEADS SEQ", 3, EXAMPLE, shape_args, n, shape);
  const int64_t HEADS = shape[0], KV_HEADS = shape[1], SEQ = shape[2];
  int64_t key = 0;
  for (int64_t h = 0; h < KV_HEADS; ++h) {
    for (int64_t q0 = 0; q0 < SEQ; q0 += Q_LEN) {
      int64_t iv[4] = {h, q0, HEADS / KV_HEADS, SEQ};
      togsim_dispatch(ctx, attention_tile, iv, 4, key++);
    }
  }
}
