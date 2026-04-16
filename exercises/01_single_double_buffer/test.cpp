//===- test.cpp -- C++ host code for add-one exercise -----------*- C++ -*-===//
//
// Verifies the add-one kernel: output[i] must equal input[i] + 1.
// Comparison is done at the int32 level (matching the kernel's data type).
//
// Build:  cmake -DTARGET_NAME=add_one -DIN1_SIZE=65536 -DOUT_SIZE=65536
//===----------------------------------------------------------------------===//

#include "xrt_test_wrapper.h"
#include <cstdint>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE_IN1 = std::int32_t;
using DATATYPE_OUT = std::int32_t;
#endif

void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn1[i] = static_cast<DATATYPE_IN1>(i);
}

void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE * sizeof(DATATYPE_OUT));
}

int verify_add_one(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut, int SIZE,
                   int verbosity) {
  int errors = 0;
  for (int i = 0; i < SIZE; i++) {
    DATATYPE_OUT expected = bufIn1[i] + 1;
    if (bufOut[i] != expected) {
      if (verbosity >= 1)
        std::cout << "Error at index " << i << ": " << bufOut[i]
                  << " != " << expected << std::endl;
      errors++;
    }
  }
  return errors;
}

int main(int argc, const char *argv[]) {
  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE_IN1);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE_OUT);

  args myargs = parse_args(argc, argv);

  return setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT, initialize_bufIn1,
                           initialize_bufOut, verify_add_one>(
      IN1_VOLUME, OUT_VOLUME, myargs);
}
