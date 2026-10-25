// A paged walk: work-item key k loads N(k) 16 x 16 f32 blocks of tensor 0, trip j's block at BLOCKS(k)[j] * 256
// elements, and stores each to its own tile of tensor 1. N is a scalar attribute, BLOCKS an array whose extent the
// record holds; -DBY_EXTENT takes the trip count from that extent. -D<BROKEN> misreads BLOCKS as cases.py names.
#include "togsim.hpp"
using namespace togsim;

struct ArrayLoop : Tog<ArrayLoop> {
  static constexpr int64_t T = 16;
  static constexpr Buf IN{.id = 1}, OUT{.id = 2};
  static constexpr std::array buffers{IN, OUT};
  static constexpr int64_t SPAD = per_lane(f32(T, T) + f32(T, T), 128);
  static constexpr Attribute N{.id = 0, .rank = 0}, BLOCKS{.id = 1, .rank = 1};
  static constexpr std::array attributes{N, BLOCKS};
  static constexpr Node STEP{.id = 0, .count = 1};

  struct At {
    int64_t i;
  };

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) dispatch<&ArrayLoop::tile>(k, {.key = i, .spad = SPAD, .args = At{.i = i}});
  }

  void tile(Item& w, const At& a) const {
#if defined(BY_EXTENT)
    const int64_t trips = extent(w, BLOCKS, 0);
#elif defined(SCALAR_OF_ARRAY)
    const int64_t trips = scalar(w, BLOCKS);
#elif defined(EXTENT_DIM)
    const int64_t trips = extent(w, BLOCKS, 1);
#else
    const int64_t trips = scalar(w, N);
#endif
    for (int64_t j = 0; j < trips; ++j) {
#if defined(TWO_INDICES)
      const int64_t block = element(w, BLOCKS, j, j);
#elif defined(PAST_EXTENT)
      const int64_t block = element(w, BLOCKS, j + 1);
#else
      const int64_t block = element(w, BLOCKS, j);
#endif
      load(w, {.src = {.ptr = 0, .type = f32, .offset = block * T * T, .dims = {T, T}, .strides = {T, 1}}, .dst = IN});
      compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
      store(w, {.src = OUT,
                .dst = {.ptr = 1, .type = f32, .offset = (a.i * 8 + j) * T * T, .dims = {T, T}, .strides = {T, 1}}});
    }
  }
};
