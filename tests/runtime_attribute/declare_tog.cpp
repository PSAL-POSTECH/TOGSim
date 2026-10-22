// A TOG whose attribute list breaks a declaration rule, by -D: BAD_ID (id 64), GAP (ids 0 and 2), TWICE (id 0 twice),
// BAD_RANK (rank 9); -DGENERATED states it through register_tog, as generated code does, so the rule is checked when
// the .so is loaded, not when it is compiled. Without a -D it is valid and reads nothing.
#include "togsim.hpp"
using namespace togsim;

#if defined(BAD_ID)
#define ATTRIBUTES {.id = 64, .rank = 0}
#elif defined(GAP)
#define ATTRIBUTES {.id = 0, .rank = 0}, Attribute{.id = 2, .rank = 0}
#elif defined(TWICE)
#define ATTRIBUTES {.id = 0, .rank = 0}, Attribute{.id = 0, .rank = 1}
#elif defined(BAD_RANK)
#define ATTRIBUTES {.id = 0, .rank = 9}
#else
#define ATTRIBUTES {.id = 0, .rank = 0}
#endif

static constexpr Buf OUT{.id = 1, .bytes = 64};
static constexpr Node STEP{.id = 0, .count = 1};

#if defined(GENERATED)
static void work_item(Item& w) { compute(w, {.node = STEP, .writes = {OUT}}); }
static void tog_entry(Kernel& k) { dispatch<work_item>(k, {.key = 0}); }
static const bool tog = register_tog<tog_entry>({OUT}, {Attribute ATTRIBUTES});
#else
struct Declared : Tog<Declared> {
  static constexpr std::array buffers{OUT};
  static constexpr std::array attributes{Attribute ATTRIBUTES};
  void entry(Kernel& k) const { dispatch<&Declared::tile>(k, {.key = 0}); }
  void tile(Item& w) const { compute(w, {.node = STEP, .writes = {OUT}}); }
};
#endif
