// The smallest TOG: one work-item loads a 64x64 f32 tile of tensor 0 into IN, computes OUT from it (tile 0 of the
// cycle table), and stores OUT to tensor 1. Each call is one modeled instruction; the buffers order them.
#include "togsim.hpp"
using namespace togsim;

struct Hello : Tog<Hello> {
  static constexpr Buf IN{.id = 1, .bytes = f32(64, 64)}, OUT{.id = 2, .bytes = f32(64, 64)};
  static constexpr std::array buffers{IN, OUT};
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr Desc SRC{.ptr = 0, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
  static constexpr Desc DST{.ptr = 1, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};

  void entry(Kernel& k) const { dispatch<&Hello::body>(k); }

  void body(Item& w) const {
    load(w, {.from = SRC, .to = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.from = OUT, .to = DST});
  }
};
