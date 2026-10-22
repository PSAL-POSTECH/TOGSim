// Two work-items, keys 0 and 1 << 21 (program (0, 1)): each loads a 16 x 16 f32 tile of tensor 0 from the element
// offset its scalar attribute OFF records, computes, and stores it to its own tile of tensor 1. -D<BROKEN> breaks it
// as cases.py names; -DNO_ATTRIBUTES declares none and loads from 0.
#include "togsim.hpp"
using namespace togsim;

struct ScalarOffset : Tog<ScalarOffset> {
  static constexpr int64_t T = 16;
  static constexpr Buf IN{.id = 1, .bytes = f32(T, T)}, OUT{.id = 2, .bytes = f32(T, T)};
  static constexpr std::array buffers{IN, OUT};
  static constexpr Attribute OFF{.id = 0, .rank = 0};
#if !defined(NO_ATTRIBUTES)
  static constexpr std::array attributes{OFF};
#endif
  static constexpr Node STEP{.id = 0, .count = 1};
  static constexpr int64_t KEYS[2] = {0, int64_t{1} << 21};

  struct At {
    int64_t i;
  };

  void entry(Kernel& k) const {
    for (int64_t i = 0; i < 2; i++) {
#if defined(COUNTED)
      dispatch<&ScalarOffset::tile>(k, At{.i = i});
#else
      dispatch<&ScalarOffset::tile>(k, {.key = KEYS[i], .args = At{.i = i}});
#endif
    }
  }

  void tile(Item& w, const At& a) const {
#if defined(NO_ATTRIBUTES)
    const int64_t off = 0;
#elif defined(UNDECLARED)
    const int64_t off = scalar(w, Attribute{.id = 1, .rank = 0});
#elif defined(RANK_DIFFERS)
    const int64_t off = element(w, Attribute{.id = 0, .rank = 1}, 0);
#else
    const int64_t off = scalar(w, OFF);
#endif
    load(w, {.src = {.ptr = 0, .type = f32, .offset = off, .dims = {T, T}, .strides = {T, 1}}, .dst = IN});
    compute(w, {.node = STEP, .reads = {IN}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = {.ptr = 1, .type = f32, .offset = a.i * T * T, .dims = {T, T}, .strides = {T, 1}}});
  }
};
