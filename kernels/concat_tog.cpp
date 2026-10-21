// Concatenation of an A_ROWS x D and a B_ROWS x D tensor (arguments 0, 1) along the rows into argument 2, by DMA
// alone: each TROW x TD tile is loaded and stored where it lands, a short last tile as what is left.
// shape_args: A_ROWS B_ROWS D (example 128 128 256). Build options: -DTROW= -DTD= (default 128 16).
#include "togsim.hpp"
using namespace togsim;

struct Concat : Tog<Concat> {
#ifdef TROW
  static constexpr int64_t TILE_ROWS = TROW;
#else
  static constexpr int64_t TILE_ROWS = 128;
#endif
#ifdef TD
  static constexpr int64_t TILE_D = TD;
#else
  static constexpr int64_t TILE_D = 16;
#endif
  static constexpr Buf A{.id = 0, .bytes = f32(TILE_ROWS, TILE_D)}, B{.id = 1, .bytes = f32(TILE_ROWS, TILE_D)};
  static constexpr std::array buffers{A, B};

  struct TileArgs {
    int64_t d0, A_ROWS, B_ROWS, D;
  };

  // One work-item per TD-wide column tile; each walks the rows of both inputs.
  void entry(Kernel& k) const {
    const auto [A_ROWS, B_ROWS, D] =
        shape<3>(k, {.kernel = "concat", .names = "A_ROWS B_ROWS D", .example = {128, 128, 256}});
    for (int64_t d0 = 0; d0 < D; d0 += TILE_D)
      dispatch<&Concat::tile>(k, TileArgs{.d0 = d0, .A_ROWS = A_ROWS, .B_ROWS = B_ROWS, .D = D});
  }

  void tile(Item& w, const TileArgs& a) const {
    const int64_t td = extent({.dim = a.D, .at = a.d0, .tile = TILE_D});
    const auto rows = [&](int32_t ptr, int64_t r, int64_t n) {
      return Desc{.ptr = ptr, .type = f32, .offset = r * a.D + a.d0, .dims = {n, td}, .strides = {a.D, 1}};
    };
    for (int64_t r = 0; r < a.A_ROWS; r += TILE_ROWS) {
      const int64_t tr = extent({.dim = a.A_ROWS, .at = r, .tile = TILE_ROWS});
      load(w, {.from = rows(0, r, tr), .to = A});
      store(w, {.from = A, .to = rows(2, r, tr)});
    }
    for (int64_t r = 0; r < a.B_ROWS; r += TILE_ROWS) {
      const int64_t tr = extent({.dim = a.B_ROWS, .at = r, .tile = TILE_ROWS});
      load(w, {.from = rows(1, r, tr), .to = B});
      store(w, {.from = B, .to = rows(2, r + a.A_ROWS, tr)});
    }
  }
};
