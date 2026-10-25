// ITEMS work-items (key 10, 11, ...) whose calls the case writes: -DBODY is one work-item's statements, of L(b) a load
// of buffer b, S(b) a store that drains it, C({reads}, {writes}) a compute of tile 0; `i` is the work-item's number.
// -DSIZES lists the bytes each DMA of buffer 0, 1, ... moves; -DSPAD is each dispatch's spad per lane (it may read i).
#include <initializer_list>

#include "togsim_runtime.h"

#ifndef ITEMS
#define ITEMS 1
#endif

static const int64_t moved[] = {SIZES};
extern "C" const int32_t togsim_runtime_attribute_count = 0;
extern "C" const int32_t togsim_runtime_attribute_rank[1] = {0};

static void tile(EmitCtx* ctx, int64_t* iv, int32_t n_iv) {
  (void)n_iv;
  typedef std::initializer_list<int64_t> Bufs;
  const int64_t i = iv[0];
  uint64_t at = (uint64_t)i << 24;
  auto dma = [&](int32_t dir, int64_t b, Bufs reads, Bufs writes) {
    const int64_t bytes[1] = {moved[b]}, contiguous[1] = {1};
    togsim_dma(ctx, dir, dir, at, 1, bytes, contiguous, 8, 0, 0, reads.begin(), (int32_t)reads.size(), writes.begin(),
               (int32_t)writes.size(), nullptr, 0);
    at += (uint64_t)bytes[0];
  };
  auto L = [&](int64_t b) { dma(TOGSIM_DMA_LOAD, b, {}, {b}); };
  auto S = [&](int64_t b) { dma(TOGSIM_DMA_STORE, b, {b}, {}); };
  auto C = [&](Bufs reads, Bufs writes) {
    togsim_compute(ctx, 0, 0, 0, nullptr, reads.begin(), (int32_t)reads.size(), writes.begin(),
                   (int32_t)writes.size());
  };
  (void)L; (void)S; (void)C; (void)i;
  BODY;
}

extern "C" void togsim_kernel(EmitCtx* ctx, int64_t* shape_args, int32_t n_shape_args) {
  (void)shape_args; (void)n_shape_args;
  for (int64_t i = 0; i < ITEMS; i++) {
    int64_t iv[1] = {i};
    togsim_dispatch(ctx, tile, iv, 1, 10 + i, SPAD);
  }
}
