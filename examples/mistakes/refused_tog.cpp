// A valid TOG (an async load, its wait, a compute, a store) that one -D<NAME> breaks. togsim.hpp refuses NOT_A_TOG
// (no Tog<> base), NO_WAIT and EMPTY_WAIT (a wait on a semaphore no load signaled since its last wait); TOGSim refuses
// SPAD_PAST_LANE (a .spad one byte more than tpuv6e's 1024 KB per lane) and TILE_PAST_TABLE (variant 1: a row the
// one-line cycle table lacks).
#include "togsim.hpp"
using namespace togsim;

#if defined(NOT_A_TOG)
struct Refused {
#else
struct Refused : Tog<Refused> {
#endif
  static constexpr Buf IN{.id = 1}, OUT{.id = 2};
  static constexpr std::array buffers{IN, OUT};
#if defined(SPAD_PAST_LANE)
  static constexpr int64_t SPAD = 1024 * 1024 + 1;
#else
  static constexpr int64_t SPAD = per_lane(2 * f32(64, 64), 128);
#endif
#if defined(TILE_PAST_TABLE)
  static constexpr int64_t VARIANT = 1;
#else
  static constexpr int64_t VARIANT = 0;
#endif
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr Desc SRC{.ptr = 0, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
  static constexpr Desc DST{.ptr = 1, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};

  void entry(Kernel& k) const { dispatch<&Refused::body>(k, {.spad = SPAD}); }

  void body(Item& w) const {
#if defined(EMPTY_WAIT)
    Semaphore none = semaphore(w);
    wait(w, {.sem = none});
#endif
    Semaphore in = load_async(w, {.src = SRC, .dst = IN});
#if !defined(NO_WAIT)
    wait(w, {.sem = in});
#endif
    compute(w, {.node = STEP, .variant = VARIANT, .reads = {IN}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = DST});
  }
};
