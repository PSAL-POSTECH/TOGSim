// A valid TOG (an async load, its wait, a compute, a store) that one -D<NAME> breaks. togsim.hpp refuses NOT_A_TOG
// (no Tog<> base), NO_WAIT and EMPTY_WAIT (a wait on a semaphore no load signaled since its last wait); TOGSim refuses
// UNSIZED_BUFFER (OUT not in buffers) and TILE_PAST_TABLE (variant 1: a row the one-line cycle table lacks).
#include "togsim.hpp"
using namespace togsim;

#if defined(NOT_A_TOG)
struct Refused {
#else
struct Refused : Tog<Refused> {
#endif
  static constexpr Buf IN{.id = 1, .bytes = f32(64, 64)}, OUT{.id = 2, .bytes = f32(64, 64)};
#if defined(UNSIZED_BUFFER)
  static constexpr std::array buffers{IN};
#else
  static constexpr std::array buffers{IN, OUT};
#endif
#if defined(TILE_PAST_TABLE)
  static constexpr int64_t VARIANT = 1;
#else
  static constexpr int64_t VARIANT = 0;
#endif
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr Desc SRC{.ptr = 0, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
  static constexpr Desc DST{.ptr = 1, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};

  void entry(Kernel& k) const { dispatch<&Refused::body>(k); }

  void body(Item& w) const {
#if defined(EMPTY_WAIT)
    Semaphore none = semaphore(w);
    wait(w, {.sem = none});
#endif
    Semaphore in = load_async(w, {.from = SRC, .to = IN});
#if !defined(NO_WAIT)
    wait(w, {.sem = in});
#endif
    compute(w, {.node = STEP, .variant = VARIANT, .reads = {IN}, .writes = {OUT}});
    store(w, {.from = OUT, .to = DST});
  }
};
