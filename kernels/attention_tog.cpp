// Attention with online softmax (arguments 0 Q, 1 K, 2 V, 3 OUT): a work-item is Q_LEN queries of one KV head, each
// KV_BLOCK of keys one QK, six softmax stages and one SV per query head; a short tile has its own variant. shape_args:
// HEADS KV_HEADS SEQ (example 8 8 128). Build options: -DQ_LEN= -DKV_BLOCK= -DDHEAD= (default 8 64 64).
#include "togsim.hpp"
using namespace togsim;

struct Attention : Tog<Attention> {
#ifdef Q_LEN
  static constexpr int64_t TILE_Q = Q_LEN;
#else
  static constexpr int64_t TILE_Q = 8;
#endif
#ifdef KV_BLOCK
  static constexpr int64_t TILE_KV = KV_BLOCK;
#else
  static constexpr int64_t TILE_KV = 64;
#endif
#ifdef DHEAD
  static constexpr int64_t D = DHEAD;
#else
  static constexpr int64_t D = 64;
#endif
  static constexpr Buf K_TILE{.id = 0, .bytes = f32(TILE_KV, D)}, Q_TILE{.id = 1, .bytes = f32(TILE_Q, D)},
                       V_TILE{.id = 2, .bytes = f32(TILE_KV, D)}, LOGIT{.id = 3, .bytes = f32(TILE_Q, TILE_KV)},
                       STAT{.id = 4, .bytes = f32(TILE_Q)}, ACC{.id = 5, .bytes = f32(TILE_Q, D)};
  static constexpr std::array buffers{K_TILE, Q_TILE, V_TILE, LOGIT, STAT, ACC};
  static constexpr uint64_t NODES = 9;
  static constexpr Node QK{.id = 0, .count = NODES}, ROWMAX{.id = 1, .count = NODES}, SUB{.id = 2, .count = NODES},
                        EXP{.id = 3, .count = NODES}, ROWSUM{.id = 4, .count = NODES}, MAC{.id = 5, .count = NODES},
                        RESCALE{.id = 6, .count = NODES}, SV{.id = 7, .count = NODES}, NORM{.id = 8, .count = NODES};
  static constexpr int32_t AXIS_Q = 0, AXIS_KV = 1;

  struct TileArgs {
    int64_t h, q0, heads_per_kv, seq;
  };

  // One work-item per KV head and TILE_Q-query tile.
  void entry(Kernel& k) const {
    const auto [HEADS, KV_HEADS, SEQ] =
        shape<3>(k, {.kernel = "attention", .names = "HEADS KV_HEADS SEQ", .example = {8, 8, 128}});
    for (int64_t h = 0; h < KV_HEADS; ++h)
      for (int64_t q0 = 0; q0 < SEQ; q0 += TILE_Q)
        dispatch<&Attention::tile>(k, TileArgs{.h = h, .q0 = q0, .heads_per_kv = HEADS / KV_HEADS, .seq = SEQ});
  }

  void tile(Item& w, const TileArgs& a) const {
    const int64_t tq = extent({.dim = a.seq, .at = a.q0, .tile = TILE_Q});
    const int64_t q_short = short_axis({.axis = AXIS_Q, .extent = tq, .tile = TILE_Q});
    const int64_t head = a.h * a.seq * D;

    const Semaphore k = semaphore(w), q = semaphore(w), vv = semaphore(w);
    for (int64_t kv0 = 0; kv0 < a.seq; kv0 += TILE_KV) {
      const int64_t tkv = extent({.dim = a.seq, .at = kv0, .tile = TILE_KV});
      const int64_t v = q_short | short_axis({.axis = AXIS_KV, .extent = tkv, .tile = TILE_KV});
      const Desc keys{.ptr = 1, .type = f32, .offset = head + kv0 * D, .dims = {tkv, D}, .strides = {D, 1}};
      const Desc queries{.ptr = 0, .type = f32, .offset = head + a.q0 * D, .dims = {tq, D}, .strides = {D, 1}};
      const Desc values{.ptr = 2, .type = f32, .offset = head + kv0 * D, .dims = {tkv, D}, .strides = {D, 1}};
      load(w, {.src = keys, .dst = K_TILE, .sem = k});
      load(w, {.src = queries, .dst = Q_TILE, .sem = q});
      load(w, {.src = values, .dst = V_TILE, .sem = vv});

      wait(w, {.sem = k});
      wait(w, {.sem = q});
      for (int64_t hh = 0; hh < a.heads_per_kv; ++hh)
        compute(w, {.node = QK, .variant = v, .reads = {K_TILE, Q_TILE}, .writes = {LOGIT}});
      compute(w, {.node = ROWMAX, .variant = v, .reads = {LOGIT}, .writes = {STAT}});
      compute(w, {.node = SUB, .variant = v, .reads = {LOGIT, STAT}, .writes = {LOGIT}});
      compute(w, {.node = EXP, .variant = v, .reads = {LOGIT}, .writes = {LOGIT}});
      compute(w, {.node = ROWSUM, .variant = v, .reads = {LOGIT}, .writes = {STAT}});
      compute(w, {.node = MAC, .variant = v, .reads = {LOGIT, STAT}, .writes = {STAT}});
      compute(w, {.node = RESCALE, .variant = v, .reads = {STAT, ACC}, .writes = {ACC}});

      wait(w, {.sem = vv});
      for (int64_t hh = 0; hh < a.heads_per_kv; ++hh)
        compute(w, {.node = SV, .variant = v, .reads = {V_TILE, LOGIT, ACC}, .writes = {ACC}});
    }

    compute(w, {.node = NORM, .variant = q_short, .reads = {STAT, ACC}, .writes = {ACC}});
    const Desc out{.ptr = 3, .type = f32, .offset = head + a.q0 * D, .dims = {tq, D}, .strides = {D, 1}};
    store(w, {.src = ACC, .dst = out});
  }
};
