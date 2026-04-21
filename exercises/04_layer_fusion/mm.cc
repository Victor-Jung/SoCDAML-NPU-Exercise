//===- mm.cc -------------------------------------------------------*- C++ -*-===//
//
// Kernels for the layer-fusion exercise (npu2, int16).
//
// Exports:
//   matmul_i16_i16       — vectorized matmul (tiled data layout)
//   zero_i16             — vectorized zero
//   relu_i16             — vectorized ReLU  (out[i] = max(in[i], 0))
//   relu_scalar_i16      — scalar ReLU      (reference)
//   relu_fused_scalar_i16 — TODO (Phase 2): student implements scalar ReLU
//   relu_fused_i16        — TODO (Phase 2): student implements vectorized ReLU
//
// Tile dimensions: m=64, k=64, n=64
// Vectorized intrinsic: aie::mmul<r=4, s=4, t=8>  (npu2 int16)
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <aie_api/aie.hpp>

constexpr unsigned m = 64;
constexpr unsigned k = 64;
constexpr unsigned n = 64;

// AIE microkernel intrinsic dimensions for npu2 int16
constexpr unsigned r = 4;
constexpr unsigned s = 4;
constexpr unsigned t = 8;

using MMUL = aie::mmul<r, s, t, int16, int16>;

extern "C" {

// ─── Vectorized matmul ──────────────────────────────────────────────────────
//
// Expects TILED data layout (delivered by the DMA transform):
//   A: (m/r)×(k/s) sub-tiles of r×s = 4×4 elements each
//   B: (k/s)×(n/t) sub-tiles of s×t = 4×8 elements each
//   C: (m/r)×(n/t) sub-tiles of r×t = 4×8 elements each
//
// Uses aie::mmul<4,4,8> with 2×2 spatial unrolling.
// Accumulates: C += A × B.
//
void matmul_i16_i16(const int16 *__restrict A,
                    const int16 *__restrict B,
                    int16 *__restrict C) {
  event0();

  for (unsigned row = 0; row < m / r; row += 2) {
    for (unsigned col = 0; col < n / t; col += 2) {

      const int16 *__restrict pA0 =
          A + ((row + 0) * (k / s) + 0) * MMUL::size_A;
      const int16 *__restrict pA1 =
          A + ((row + 1) * (k / s) + 0) * MMUL::size_A;
      const int16 *__restrict pB0 =
          B + (0 * (n / t) + (col + 0)) * MMUL::size_B;
      const int16 *__restrict pB1 =
          B + (0 * (n / t) + (col + 1)) * MMUL::size_B;

      MMUL C00(aie::load_v<MMUL::size_C>(
          C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C));
      MMUL C01(aie::load_v<MMUL::size_C>(
          C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C));
      MMUL C10(aie::load_v<MMUL::size_C>(
          C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C));
      MMUL C11(aie::load_v<MMUL::size_C>(
          C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C));

      for (unsigned i = 0; i < k / s; i += 1) {
        aie::vector<int16, MMUL::size_A> a0 = aie::load_v<MMUL::size_A>(pA0);
        aie::vector<int16, MMUL::size_A> a1 = aie::load_v<MMUL::size_A>(pA1);
        aie::vector<int16, MMUL::size_B> b0 = aie::load_v<MMUL::size_B>(pB0);
        aie::vector<int16, MMUL::size_B> b1 = aie::load_v<MMUL::size_B>(pB1);

        C00.mac(a0, b0);
        C01.mac(a0, b1);
        C10.mac(a1, b0);
        C11.mac(a1, b1);

        pA0 += MMUL::size_A;
        pA1 += MMUL::size_A;
        pB0 += (n / t) * MMUL::size_B;
        pB1 += (n / t) * MMUL::size_B;
      }

      aie::store_v(C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C,
                   C00.template to_vector<int16>());
      aie::store_v(C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C,
                   C01.template to_vector<int16>());
      aie::store_v(C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C,
                   C10.template to_vector<int16>());
      aie::store_v(C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C,
                   C11.template to_vector<int16>());
    }
  }

  event1();
}

// ─── Vectorized zero ────────────────────────────────────────────────────────
void zero_i16(int16 *__restrict c) {
  constexpr int vec_factor = 32;
  aie::vector<int16, vec_factor> zeros = aie::zeros<int16, vec_factor>();
  for (unsigned i = 0; i < m * n; i += vec_factor) {
    aie::store_v(c + i, zeros);
  }
}

// ─── Vectorized ReLU ────────────────────────────────────────────────────────
//
// out[i] = max(in[i], 0)  for all i in [0, m*n).
// Uses 32-wide int16 vectors for efficient element-wise max.
//
void relu_i16(const int16 *__restrict in, int16 *__restrict out) {
  event0();
  constexpr int vec_factor = 32;
  aie::vector<int16, vec_factor> zeros = aie::zeros<int16, vec_factor>();
  for (unsigned i = 0; i < m * n; i += vec_factor) {
    aie::vector<int16, vec_factor> v = aie::load_v<vec_factor>(in + i);
    aie::store_v(out + i, aie::max(v, zeros));
  }
  event1();
}

// ─── Scalar ReLU (reference) ────────────────────────────────────────────────
void relu_scalar_i16(const int16 *__restrict in, int16 *__restrict out) {
  event0();
  for (unsigned i = 0; i < m * n; i++) {
    out[i] = (in[i] < 0) ? (int16)0 : in[i];
  }
  event1();
}

// ─── TODO (Phase 2): Student implements scalar ReLU ─────────────────────────
//
// Implement out[i] = max(in[i], 0) for all i in [0, m*n)
// using a simple scalar loop.
//
void relu_fused_scalar_i16(const int16 *__restrict in,
                           int16 *__restrict out) {
  // ┌──────────────────────────────────────────────────────┐
  // │  YOUR CODE HERE                                      │
  // │  Hint: loop over m*n elements, compare each to 0.    │
  // └──────────────────────────────────────────────────────┘
}

// ─── TODO (Phase 2): Student implements vectorized ReLU ─────────────────────
//
// Implement out[i] = max(in[i], 0) for all i in [0, m*n)
// using AIE vector intrinsics for efficient execution.
//
// Hint: look at how zero_i16 uses aie::zeros and aie::store_v.
//       The aie::max() intrinsic computes the element-wise maximum
//       of two vectors.
//
void relu_fused_i16(const int16 *__restrict in,
                    int16 *__restrict out) {
  // ┌──────────────────────────────────────────────────────┐
  // │  YOUR CODE HERE                                      │
  // │  Hint: use aie::load_v, aie::max, aie::store_v      │
  // │  with a vector of zeros.                             │
  // └──────────────────────────────────────────────────────┘
}

} // extern "C"
