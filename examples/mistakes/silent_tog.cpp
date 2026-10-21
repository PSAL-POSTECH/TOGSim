// A valid TOG (a load, a compute, a store) that one -D<NAME> breaks in a way TOGSim does NOT refuse: the run exits
// 0. TENSOR_PAST_LIST stores to a tensor tog_tensors.txt lacks, which a warning names. CONSUMER_FIRST stores OUT
// before the compute that writes it: a store of data already on the chip, which an info line names.
#include "togsim.hpp"
using namespace togsim;

struct Silent : Tog<Silent> {
  static constexpr Buf IN{.id = 1, .bytes = f32(64, 64)}, OUT{.id = 2, .bytes = f32(64, 64)};
  static constexpr std::array buffers{IN, OUT};
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr Desc SRC{.ptr = 0, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
#if defined(TENSOR_PAST_LIST)
  static constexpr Desc DST{.ptr = 2, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
#else
  static constexpr Desc DST{.ptr = 1, .type = f32, .offset = 0, .dims = {64, 64}, .strides = {64, 1}};
#endif

  void entry(Kernel& k) const { dispatch<&Silent::body>(k); }

  void body(Item& w) const {
    load(w, {.from = SRC, .to = IN});
#if defined(CONSUMER_FIRST)
    store(w, {.from = OUT, .to = DST});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
#else
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.from = OUT, .to = DST});
#endif
  }
};
