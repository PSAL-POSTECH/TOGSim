// Two async loads, A and B, on one semaphore that one wait covers, then a compute that reads both and a store.
// -DSYNC makes the loads synchronous, with no wait; -DTWO_STEPS first computes PARTIAL from A alone.
#include "togsim.hpp"
using namespace togsim;

struct AsyncBarrier : Tog<AsyncBarrier> {
  static constexpr Buf A{.id = 1, .bytes = f32(64, 64)}, B{.id = 2, .bytes = f32(64, 64)},
                       OUT{.id = 3, .bytes = f32(64, 64)}, PARTIAL{.id = 4, .bytes = f32(64, 64)};
  static constexpr std::array buffers{A, B, OUT, PARTIAL};
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr Desc LHS{.ptr = 0, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
  static constexpr Desc RHS{.ptr = 1, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
  static constexpr Desc DST{.ptr = 2, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};

  void entry(Kernel& k) const { dispatch<&AsyncBarrier::body>(k); }

  // An async load is done once its requests are sent; the wait is done once the data of every load that signaled
  // its semaphore has arrived.
  void body(Item& w) const {
#if defined(SYNC)
    load(w, {.src = LHS, .dst = A});
    load(w, {.src = RHS, .dst = B});
#else
    Semaphore ab = semaphore(w);
    load(w, {.src = LHS, .dst = A, .sem = ab});
    load(w, {.src = RHS, .dst = B, .sem = ab});
    wait(w, {.sem = ab});
#endif
#if defined(TWO_STEPS)
    compute(w, {.node = STEP, .reads = {A}, .writes = {PARTIAL}});
    compute(w, {.node = STEP, .reads = {PARTIAL, B}, .writes = {OUT}});
#else
    compute(w, {.node = STEP, .reads = {A, B}, .writes = {OUT}});
#endif
    store(w, {.src = OUT, .dst = DST});
  }
};
