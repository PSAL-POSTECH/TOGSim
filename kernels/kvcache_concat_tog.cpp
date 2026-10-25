// Split TOKENS rows of fused QKV (argument 0) into the query (1) and, behind PAST cached tokens, the key and value
// caches (2, 3), TTOK rows per work-item; a short tile has its own variant. shape_args: TOKENS PAST (example 128 128).
// Build options: -DTTOK= -DHIDDEN= -DNUM_HEADS= -DNUM_KV_HEADS= (default 32 256 8 8).
#include "togsim.hpp"
using namespace togsim;

struct KvcacheConcat : Tog<KvcacheConcat> {
#ifdef TTOK
  static constexpr int64_t TILE_TOK = TTOK;
#else
  static constexpr int64_t TILE_TOK = 32;
#endif
#ifdef HIDDEN
  static constexpr int64_t HIDDEN_SIZE = HIDDEN;
#else
  static constexpr int64_t HIDDEN_SIZE = 256;
#endif
#ifdef NUM_HEADS
  static constexpr int64_t HEADS = NUM_HEADS;
#else
  static constexpr int64_t HEADS = 8;
#endif
#ifdef NUM_KV_HEADS
  static constexpr int64_t KV_HEADS = NUM_KV_HEADS;
#else
  static constexpr int64_t KV_HEADS = 8;
#endif
  static constexpr int64_t CACHE_DIM = HIDDEN_SIZE / HEADS * KV_HEADS, QKV_WIDTH = HIDDEN_SIZE + 2 * CACHE_DIM;
#ifdef LANES
  static constexpr int64_t N_LANES = LANES;
#else
  static constexpr int64_t N_LANES = 128;
#endif
  static constexpr Buf QKV{.id = 0}, QUERY{.id = 1}, KEY{.id = 2}, VALUE{.id = 3};
  static constexpr std::array buffers{QKV, QUERY, KEY, VALUE};
  static constexpr int64_t SPAD = per_lane(f32(TILE_TOK, QKV_WIDTH) + f32(TILE_TOK, HIDDEN_SIZE) +
                                           f32(TILE_TOK, CACHE_DIM) + f32(TILE_TOK, CACHE_DIM), N_LANES);
  static constexpr Node SPLIT{.id = 0, .count = 1};
  static constexpr int32_t AXIS_TOKENS = 0;

  struct TileArgs {
    int64_t t0, TOKENS, PAST;
  };

  // One work-item per TTOK-row tile.
  void entry(Kernel& k) const {
    const auto [TOKENS, PAST] =
        shape<2>(k, {.kernel = "kvcache_concat", .names = "TOKENS PAST", .example = {128, 128}});
    for (int64_t t0 = 0; t0 < TOKENS; t0 += TILE_TOK)
      dispatch<&KvcacheConcat::tile>(k, {.spad = SPAD, .args = TileArgs{.t0 = t0, .TOKENS = TOKENS, .PAST = PAST}});
  }

  void tile(Item& w, const TileArgs& a) const {
    const int64_t ttok = extent({.dim = a.TOKENS, .at = a.t0, .tile = TILE_TOK});
    const int64_t variant = short_axis({.axis = AXIS_TOKENS, .extent = ttok, .tile = TILE_TOK});
    const int64_t cached = (a.PAST + a.t0) * CACHE_DIM;
    const Desc qkv{
        .ptr = 0, .type = f32, .offset = a.t0 * QKV_WIDTH, .dims = {ttok, QKV_WIDTH}, .strides = {QKV_WIDTH, 1}};
    const Desc query{
        .ptr = 1, .type = f32, .offset = a.t0 * HIDDEN_SIZE, .dims = {ttok, HIDDEN_SIZE}, .strides = {HIDDEN_SIZE, 1}};
    const Desc key{.ptr = 2, .type = f32, .offset = cached, .dims = {ttok, CACHE_DIM}, .strides = {CACHE_DIM, 1}};
    const Desc value{.ptr = 3, .type = f32, .offset = cached, .dims = {ttok, CACHE_DIM}, .strides = {CACHE_DIM, 1}};
    load(w, {.src = qkv, .dst = QKV});
    compute(w, {.node = SPLIT, .variant = variant, .reads = {QKV}, .writes = {QUERY, KEY, VALUE}});
    store(w, {.src = QUERY, .dst = query});
    store(w, {.src = KEY, .dst = key});
    store(w, {.src = VALUE, .dst = value});
  }
};
