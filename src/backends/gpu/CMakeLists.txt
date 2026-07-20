# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

# HIP GPU backend (ADR-0017), built only when SPUME_ENABLE_HIP=ON. The kernel
# TUs compile as HIP (hipcc/amdclang); the portable core is unaffected. Note:
# the spume-warnings interface (-Wpedantic -Werror) is deliberately NOT applied
# here -- HIP's <<<>>> launch syntax is not ISO C++ and -Wpedantic rejects it.

add_library(spume-gpu STATIC
    gpu_spmv.hip.cpp gpu_cheb.hip.cpp gpu_transfer.hip.cpp gpu_vcycle.hip.cpp gpu_reduce.hip.cpp gpu_fcg.hip.cpp)
set_source_files_properties(
    gpu_spmv.hip.cpp gpu_cheb.hip.cpp gpu_transfer.hip.cpp gpu_vcycle.hip.cpp gpu_reduce.hip.cpp gpu_fcg.hip.cpp
    PROPERTIES LANGUAGE HIP)
target_include_directories(spume-gpu PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(spume-gpu PUBLIC spume::core)
set_target_properties(spume-gpu PROPERTIES
    HIP_ARCHITECTURES "${SPUME_HIP_ARCH}"
    POSITION_INDEPENDENT_CODE ON)
target_compile_options(spume-gpu PRIVATE -Wall -Wextra)

# Verify-then-bench gates (one per kernel), compiled as HIP.
foreach(check spmv cheb transfer vcycle reduce fcg kcycle)
    add_executable(spume-gpu-${check}-check ${check}_check.cpp)
    set_source_files_properties(${check}_check.cpp PROPERTIES LANGUAGE HIP)
    set_target_properties(spume-gpu-${check}-check PROPERTIES HIP_ARCHITECTURES "${SPUME_HIP_ARCH}")
    target_link_libraries(spume-gpu-${check}-check PRIVATE spume-gpu spume::core)
    target_compile_options(spume-gpu-${check}-check PRIVATE -Wall -Wextra)
endforeach()

# Single-shot driver for `rocprofv3 --pmc` hardware-counter profiling (not a
# test -- run under the profiler by hand; see README).
add_executable(spume-gpu-spmv-pmc spmv_pmc.cpp)
set_source_files_properties(spmv_pmc.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(spume-gpu-spmv-pmc PROPERTIES HIP_ARCHITECTURES "${SPUME_HIP_ARCH}")
target_link_libraries(spume-gpu-spmv-pmc PRIVATE spume-gpu spume::core)
target_compile_options(spume-gpu-spmv-pmc PRIVATE -Wall -Wextra)

if(SPUME_BUILD_TESTS)
    add_test(NAME gpu-spmv-check COMMAND spume-gpu-spmv-check)
    add_test(NAME gpu-cheb-check COMMAND spume-gpu-cheb-check)
    add_test(NAME gpu-transfer-check COMMAND spume-gpu-transfer-check)
    add_test(NAME gpu-vcycle-check COMMAND spume-gpu-vcycle-check)
    add_test(NAME gpu-reduce-check COMMAND spume-gpu-reduce-check)
    add_test(NAME gpu-fcg-check COMMAND spume-gpu-fcg-check)
    add_test(NAME gpu-kcycle-check COMMAND spume-gpu-kcycle-check)
    set_tests_properties(gpu-spmv-check gpu-cheb-check gpu-transfer-check gpu-vcycle-check gpu-reduce-check gpu-fcg-check gpu-kcycle-check
        PROPERTIES LABELS gpu)
endif()
