# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Reuse mlir-aie's common.cmake for XRT discovery and test_utils
set(MLIR_AIE_DIR "/scratch/jungvi/mlir-aie" CACHE STRING "Path to mlir-aie repository")
include(${MLIR_AIE_DIR}/programming_guide/common.cmake)

find_program(WSL NAMES powershell.exe)
if(NOT WSL)
    if(NOT DEFINED CMAKE_C_COMPILER)
        set(CMAKE_C_COMPILER gcc-13)
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER g++-13)
    endif()
endif()

set(TARGET_NAME add_one CACHE STRING "Target to be built")
set(IN1_SIZE 65536 CACHE STRING "Input buffer size in bytes")
set(OUT_SIZE 65536 CACHE STRING "Output buffer size in bytes")

project(${TARGET_NAME})

add_executable(${TARGET_NAME} test.cpp)

target_compile_definitions(${TARGET_NAME} PRIVATE
    IN1_SIZE=${IN1_SIZE}
    OUT_SIZE=${OUT_SIZE}
)

target_include_directories(${TARGET_NAME} PUBLIC ${XRT_INC_DIR})
target_link_directories(${TARGET_NAME} PUBLIC ${XRT_LIB_DIR})
target_link_libraries(${TARGET_NAME} PUBLIC xrt_coreutil)
target_link_test_utils(${TARGET_NAME})
