# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

# HIP GPU backend (ADR-0017), built only when SPUME_ENABLE_HIP=ON. The kernel
# TUs compile as HIP (hipcc/amdclang); the portable core is unaffected. Note:
# the spume-warnings interface (-Wpedantic -Werror) is deliberately NOT applied
# here -- HIP's <<<>>> launch syntax is not ISO C++ and -Wpedantic rejects it.

add_library(spume-gpu STATIC gpu_spmv.hip.cpp)
set_source_files_properties(gpu_spmv.hip.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(spume-gpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(spume-gpu PUBLIC spume::core)
set_target_properties(spume-gpu PROPERTIES
    HIP_ARCHITECTURES "${SPUME_HIP_ARCH}"
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(spume-gpu PRIVATE -Wall -Wextra)

add_executable(spume-gpu-spmv-check spmv_check.cpp)
set_source_files_properties(spmv_check.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(spume-gpu-spmv-check PROPERTIES HIP_ARCHITECTURES "${SPUME_HIP_ARCH}")
target_link_libraries(spume-gpu-spmv-check PRIVATE spume-gpu spume::core)
target_compile_options(spume-gpu-spmv-check PRIVATE -Wall -Wextra)

# Single-shot driver for `rocprofv3 --pmc` hardware-counter profiling (not a
# test -- run under the profiler by hand; see README).
add_executable(spume-gpu-spmv-pmc spmv_pmc.cpp)
set_source_files_properties(spmv_pmc.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(spume-gpu-spmv-pmc PROPERTIES HIP_ARCHITECTURES "${SPUME_HIP_ARCH}")
target_link_libraries(spume-gpu-spmv-pmc PRIVATE spume-gpu spume::core)
target_compile_options(spume-gpu-spmv-pmc PRIVATE -Wall -Wextra)

if(SPUME_BUILD_TESTS)
    add_test(NAME gpu-spmv-check COMMAND spume-gpu-spmv-check)
    set_tests_properties(gpu-spmv-check PROPERTIES LABELS gpu)
endif()
