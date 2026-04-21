//===- mm.cc -------------------------------------------------------*- C++ -*-===//
//
// Matrix multiplication kernels for the AIE (npu2, int16).
//
// Exports:
//   matmul_scalar_i16_i16  — naive triple-loop matmul   (row-major data)
//   matmul_i16_i16         — vectorized aie::mmul matmul (tiled data)
//   zero_scalar_i16        — scalar zero
//   zero_i16               — vectorized zero
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
constexpr unsigned r = 4;   // A-tile rows, C-tile rows
constexpr unsigned s = 4;   // A-tile cols, B-tile rows
constexpr unsigned t = 8;   // B-tile cols, C-tile cols

using MMUL = aie::mmul<r, s, t, int16, int16>;

extern "C" {

// ─── Scalar matmul ──────────────────────────────────────────────────────────
//
// Expects ROW-MAJOR A[m×k], B[k×n], C[m×n].
// Accumulates: C += A × B.
// No vectorization — simple triple nested loop.
//
void matmul_scalar_i16_i16(const int16 *__restrict A,
                           const int16 *__restrict B,
                           int16 *__restrict C) {
  event0();
  for (unsigned row = 0; row < m; row++) {
    for (unsigned col = 0; col < n; col++) {
      int16 acc = C[row * n + col];
      for (unsigned i = 0; i < k; i++) {
        acc += A[row * k + i] * B[i * n + col];
      }
      C[row * n + col] = acc;
    }
  }
  event1();
}

// ─── Vectorized matmul ──────────────────────────────────────────────────────
//
// Expects TILED data layout (delivered by the DMA transform):
//   A: (m/r)×(k/s) sub-tiles of r×s = 4×4 elements each
//   B: (k/s)×(n/t) sub-tiles of s×t = 4×8 elements each
//   C: (m/r)×(n/t) sub-tiles of r×t = 4×8 elements each
//
// Uses aie::mmul<4,4,8> with 2×2 spatial unrolling on the output tiles.
// Accumulates: C += A × B.
//
void matmul_i16_i16(const int16 *__restrict A,
                    const int16 *__restrict B,
                    int16 *__restrict C) {
  event0();

  for (unsigned row = 0; row < m / r; row += 2) {
    for (unsigned col = 0; col < n / t; col += 2) {

      // Pointers to two adjacent A-tile rows and B-tile columns
      const int16 *__restrict pA0 =
          A + ((row + 0) * (k / s) + 0) * MMUL::size_A;
      const int16 *__restrict pA1 =
          A + ((row + 1) * (k / s) + 0) * MMUL::size_A;
      const int16 *__restrict pB0 =
          B + (0 * (n / t) + (col + 0)) * MMUL::size_B;
      const int16 *__restrict pB1 =
          B + (0 * (n / t) + (col + 1)) * MMUL::size_B;

      // Load existing C values (for accumulation across K-blocks)
      MMUL C00(aie::load_v<MMUL::size_C>(
          C + ((row + 0) * (n / t) + (col + 0)) * MMUL::size_C));
      MMUL C01(aie::load_v<MMUL::size_C>(
          C + ((row + 0) * (n / t) + (col + 1)) * MMUL::size_C));
      MMUL C10(aie::load_v<MMUL::size_C>(
          C + ((row + 1) * (n / t) + (col + 0)) * MMUL::size_C));
      MMUL C11(aie::load_v<MMUL::size_C>(
          C + ((row + 1) * (n / t) + (col + 1)) * MMUL::size_C));

      // Iterate over K dimension in s-wide steps
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
  constexpr int vec_factor = 32; // 32 × int16 = 64 bytes
  aie::vector<int16, vec_factor> zeros = aie::zeros<int16, vec_factor>();
  for (unsigned i = 0; i < m * n; i += vec_factor) {
    aie::store_v(c + i, zeros);
  }
}

// ─── Scalar zero ────────────────────────────────────────────────────────────
void zero_scalar_i16(int16 *__restrict c) {
  for (unsigned i = 0; i < m * n; i++) {
    c[i] = 0;
  }
}

} // extern "C"
