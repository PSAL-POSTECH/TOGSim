// C = A x B for an M x K and a K x N matrix (arguments 0, 1, 2), a TM x TN output tile per work-item and one compute
// per TM x TK x TN block of the K loop; a last tile short on M, N or K moves and computes what is left, under its
// variant. shape_args: M K N (example 128 256 256). Build options: -DTM= -DTN= -DTK= (default 32 256 64).
#include "togsim.hpp"
using namespace togsim;

struct Gemm : Tog<Gemm> {
#ifdef TM
  static constexpr int64_t TILE_M = TM;
#else
  static constexpr int64_t TILE_M = 32;
#endif
#ifdef TN
  static constexpr int64_t TILE_N = TN;
#else
  static constexpr int64_t TILE_N = 256;
#endif
#ifdef TK
  static constexpr int64_t TILE_K = TK;
#else
  static constexpr int64_t TILE_K = 64;
#endif
  static constexpr Buf ACC{.id = 0, .bytes = f32(TILE_M, TILE_N)}, B_TILE{.id = 1, .bytes = f32(TILE_K, TILE_N)},
                       A_TILE{.id = 2, .bytes = f32(TILE_M, TILE_K)};
  static constexpr std::array buffers{ACC, B_TILE, A_TILE};
  static constexpr Node ACC_INIT{.id = 0, .count = 2}, MATMUL{.id = 1, .count = 2};
  static constexpr int32_t AXIS_M = 0, AXIS_N = 1, AXIS_K = 2;

  struct TileArgs {
    int64_t m0, n0, M, K, N;
  };

  // One work-item per output tile, in row-major order.
  void entry(Kernel& k) const {
    const auto [M, K, N] = shape<3>(k, {.kernel = "gemm", .names = "M K N", .example = {128, 256, 256}});
    for (int64_t m0 = 0; m0 < M; m0 += TILE_M)
      for (int64_t n0 = 0; n0 < N; n0 += TILE_N)
        dispatch<&Gemm::tile>(k, TileArgs{.m0 = m0, .n0 = n0, .M = M, .K = K, .N = N});
  }

  void tile(Item& w, const TileArgs& a) const {
    const int64_t tm = extent({.dim = a.M, .at = a.m0, .tile = TILE_M});
    const int64_t tn = extent({.dim = a.N, .at = a.n0, .tile = TILE_N});
    const int64_t out = short_axis({.axis = AXIS_M, .extent = tm, .tile = TILE_M}) |
                        short_axis({.axis = AXIS_N, .extent = tn, .tile = TILE_N});

    const Semaphore a_ready = semaphore(w), b_ready = semaphore(w);
    compute(w, {.node = ACC_INIT, .variant = out, .writes = {ACC}});
    for (int64_t k0 = 0; k0 < a.K; k0 += TILE_K) {
      const int64_t tk = extent({.dim = a.K, .at = k0, .tile = TILE_K});
      const Desc lhs{.ptr = 0, .type = f32, .offset = a.m0 * a.K + k0, .dims = {tm, tk}, .strides = {a.K, 1}};
      const Desc rhs{.ptr = 1, .type = f32, .offset = k0 * a.N + a.n0, .dims = {tk, tn}, .strides = {a.N, 1}};
      load(w, {.src = lhs, .dst = A_TILE, .sem = a_ready});
      load(w, {.src = rhs, .dst = B_TILE, .sem = b_ready});
      wait(w, {.sem = b_ready});
      wait(w, {.sem = a_ready});
      const int64_t variant = out | short_axis({.axis = AXIS_K, .extent = tk, .tile = TILE_K});
      compute(w, {.node = MATMUL, .variant = variant, .reads = {ACC, B_TILE, A_TILE}, .writes = {ACC}});
    }
    const Desc dst{.ptr = 2, .type = f32, .offset = a.m0 * a.N + a.n0, .dims = {tm, tn}, .strides = {a.N, 1}};
    store(w, {.src = ACC, .dst = dst});
  }
};
