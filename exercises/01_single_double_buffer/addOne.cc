//===- addOne.cc ------------------------------------------------*- C++ -*-===//
//
// Element-wise add-one kernel for the AIE.
// Adds 1 to every element of the input buffer using vectorized operations.
//
// C signature:  void addOneLine(int32_t *in, int32_t *out, int32_t lineWidth)
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C" {

void addOneLine(int32_t *restrict in, int32_t *restrict out,
                int32_t lineWidth) {
  event0();

  constexpr int vec_factor = 16; // 16 × int32 = 64 bytes per iteration
  aie::vector<int32_t, vec_factor> ones =
      aie::broadcast<int32_t, vec_factor>((int32_t)1);

  int32_t *restrict pIn = in;
  int32_t *restrict pOut = out;

  for (int i = 0; i < lineWidth; i += vec_factor) {
    aie::vector<int32_t, vec_factor> data = aie::load_v<vec_factor>(pIn);
    pIn += vec_factor;
    aie::vector<int32_t, vec_factor> result = aie::add(data, ones);
    aie::store_v(pOut, result);
    pOut += vec_factor;
  }

  event1();
}

} // extern "C"
