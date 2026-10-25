// One work-item (key 0): two trips of an 8 x 16 f32 row gather of tensor 0, row r of trip t moved by
// 16 * ROWS[t][r] elements (axis 0, the trip as `at`), then a 4 x 8 f32 element gather of tensor 1, element (i, j)
// moved by ELEMS[i][j] (axis -1), each computed and stored densely to tensor 2. -D<BROKEN> as cases.py names.
#include "togsim.hpp"
using namespace togsim;

struct Gathers : Tog<Gathers> {
  static constexpr Buf ROWBUF{.id = 1}, ELEMBUF{.id = 2}, OUT{.id = 3};
  static constexpr std::array buffers{ROWBUF, ELEMBUF, OUT};
  static constexpr int64_t SPAD = per_lane(f32(8, 16) + f32(4, 8) + f32(8, 16), 128);
  static constexpr Attribute ROWS{.id = 0, .rank = 2}, ELEMS{.id = 1, .rank = 2};
  static constexpr std::array attributes{ROWS, ELEMS};
  static constexpr Node STEP{.id = 0, .count = 1};

  void entry(Kernel& k) const { dispatch<&Gathers::tile>(k, {.key = 0, .spad = SPAD}); }

  void tile(Item& w) const {
    for (int64_t t = 0; t < 2; ++t) {
#if defined(AT_PAST_EXTENT)
      const Gather rows{.attribute = ROWS, .at = Dims(t + 2), .axis = 0, .scale = 16};
#elif defined(AXIS_PAST_TILE)
      const Gather rows{.attribute = ROWS, .at = Dims(t), .axis = 2, .scale = 16};
#elif defined(NO_AT)
      const Gather rows{.attribute = ROWS, .axis = 0, .scale = 16};
#else
      const Gather rows{.attribute = ROWS, .at = Dims(t), .axis = 0, .scale = 16};
#endif
      load(w, {.src = {.ptr = 0, .type = f32, .offset = 0, .dims = {8, 16}, .strides = {16, 1}, .gather = rows},
               .dst = ROWBUF});
      compute(w, {.node = STEP, .reads = {ROWBUF}, .writes = {OUT}});
      store(w, {.src = OUT, .dst = {.ptr = 2, .type = f32, .offset = t * 128, .dims = {8, 16}, .strides = {16, 1}}});
    }
    load(w, {.src = {.ptr = 1, .type = f32, .offset = 0, .dims = {4, 8}, .strides = {8, 1},
                     .gather = {.attribute = ELEMS, .axis = -1}},
             .dst = ELEMBUF});
    compute(w, {.node = STEP, .reads = {ELEMBUF}, .writes = {OUT}});
    store(w, {.src = OUT, .dst = {.ptr = 2, .type = f32, .offset = 256, .dims = {8, 16}, .strides = {16, 1}}});
  }
};
