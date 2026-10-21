// A GEMM-like tile: C = A x B over K reduction steps. Each step loads a 64x64 tile of A and of B and runs MAC in
// place on ACC; EPILOGUE then turns ACC into OUT, which is stored. MAC is tile 0 of the cycle table, EPILOGUE 1.
#include "togsim.hpp"
using namespace togsim;

struct Gemm : Tog<Gemm> {
  static constexpr int64_t T = 64, K = 2;
  static constexpr Buf A{.id = 1, .bytes = f32(T, T)}, B{.id = 2, .bytes = f32(T, T)}, ACC{.id = 3, .bytes = f32(T, T)},
                       OUT{.id = 4, .bytes = f32(T, T)};
  static constexpr std::array buffers{A, B, ACC, OUT};
  static constexpr Node MAC{.id = 0, .count = 2}, EPILOGUE{.id = 1, .count = 2};

  void entry(Kernel& k) const { dispatch<&Gemm::body>(k); }

  // A is T x (K*T) and B is (K*T) x T, so step k's tiles start at elements k*T and k*T*T. Reading and writing ACC
  // chains the MACs one after another.
  void body(Item& w) const {
    for (int64_t k = 0; k < K; k++) {
      load(w, {.from = {.ptr = 0, .type = f32, .offset = k * T, .dims = {T, T}, .strides = {K * T, 1}}, .to = A});
      load(w, {.from = {.ptr = 1, .type = f32, .offset = k * T * T, .dims = {T, T}, .strides = {T, 1}}, .to = B});
      compute(w, {.node = MAC, .reads = {A, B, ACC}, .writes = {ACC}});
    }
    compute(w, {.node = EPILOGUE, .reads = {ACC}, .writes = {OUT}});
    store(w, {.from = OUT, .to = {.ptr = 2, .type = f32, .offset = 0, .dims = {T, T}, .strides = {T, 1}}});
  }
};
