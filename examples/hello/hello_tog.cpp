// The smallest TOG: one work-item loads a 64x64 f32 tile of tensor 0 into IN, computes OUT from it (tile 0 of the
// cycle table), and stores OUT to tensor 1. Each call is one modeled instruction; the buffers order them. Its
// dispatch states its spad: IN and OUT, 2 x 16384 bytes over tpuv6e's 128 lanes.
#include "togsim.hpp"
using namespace togsim;

struct Hello : Tog<Hello> {
  static constexpr Buf IN{.id = 1}, OUT{.id = 2};
  static constexpr std::array buffers{IN, OUT};
  static constexpr int64_t SPAD = per_lane(2 * f32(64, 64), 128);
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr Desc SRC{.ptr = 0, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
  static constexpr Desc DST{.ptr = 1, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};

  void entry(Kernel& k) const { dispatch<&Hello::body>(k, {.spad = SPAD}); }

  void body(Item& w) const {
    load(w, {.src = SRC, .dst = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = DST});
  }
};
