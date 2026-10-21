// Flatten moves no data and computes nothing: the kernel dispatches no work-item and names no spad buffer.
// It takes no shape arguments.
#include "togsim.hpp"
using namespace togsim;

struct Flatten : Tog<Flatten> {
  static constexpr std::array<Buf, 0> buffers{};

  // Nothing to dispatch.
  void entry(Kernel& k) const { shape<0>(k, {.kernel = "flatten", .names = "", .example = {}}); }
};
