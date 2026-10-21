// Embedding lookup with its layer normalization, SEQ i64 token ids (argument 0) into SEQ x DIM rows of the table
// (argument 1) and out (argument 2), one TSEQ x TDIM tile per work-item; a short tile has its own variant. The table
// load waits for the ids but is dense. shape_args: SEQ DIM (example 128 256). Build options: -DTSEQ= -DTDIM=.
#include "togsim.hpp"
using namespace togsim;

struct EmbedLayernorm : Tog<EmbedLayernorm> {
#ifdef TSEQ
  static constexpr int64_t TILE_SEQ = TSEQ;
#else
  static constexpr int64_t TILE_SEQ = 128;
#endif
#ifdef TDIM
  static constexpr int64_t TILE_DIM = TDIM;
#else
  static constexpr int64_t TILE_DIM = 64;
#endif
  static constexpr Buf IDS{.id = 0, .bytes = i64(TILE_SEQ, TILE_DIM)}, ROW{.id = 1, .bytes = f32(TILE_SEQ, TILE_DIM)};
  static constexpr std::array buffers{IDS, ROW};
  static constexpr Node GATHER{.id = 0, .count = 1};
  static constexpr int32_t AXIS_SEQ = 0, AXIS_DIM = 1;

  struct TileArgs {
    int64_t s0, d0, SEQ, DIM;
  };

  // One work-item per tile, in row-major order.
  void entry(Kernel& k) const {
    const auto [SEQ, DIM] = shape<2>(k, {.kernel = "embed_layernorm", .names = "SEQ DIM", .example = {128, 256}});
    for (int64_t s0 = 0; s0 < SEQ; s0 += TILE_SEQ)
      for (int64_t d0 = 0; d0 < DIM; d0 += TILE_DIM)
        dispatch<&EmbedLayernorm::tile>(k, TileArgs{.s0 = s0, .d0 = d0, .SEQ = SEQ, .DIM = DIM});
  }

  void tile(Item& w, const TileArgs& a) const {
    const int64_t tseq = extent({.dim = a.SEQ, .at = a.s0, .tile = TILE_SEQ});
    const int64_t tdim = extent({.dim = a.DIM, .at = a.d0, .tile = TILE_DIM});
    const int64_t variant = short_axis({.axis = AXIS_SEQ, .extent = tseq, .tile = TILE_SEQ}) |
                            short_axis({.axis = AXIS_DIM, .extent = tdim, .tile = TILE_DIM});
    const Desc ids{.ptr = 0, .type = i64, .offset = a.s0, .dims = {tseq, tdim}, .strides = {1, 0}};
    const Desc table{.ptr = 1, .type = f32, .offset = a.d0, .dims = {tseq, tdim}, .strides = {0, 1}};
    const Desc out{.ptr = 2, .type = f32, .offset = a.s0 * a.DIM + a.d0, .dims = {tseq, tdim}, .strides = {a.DIM, 1}};
    load(w, {.src = ids, .dst = IDS});
    load(w, {.src = table, .dst = ROW, .after = {IDS}});
    compute(w, {.node = GATHER, .variant = variant, .reads = {IDS, ROW}});
    store(w, {.src = ROW, .dst = out});
  }
};
