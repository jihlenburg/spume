# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

cmake_minimum_required(VERSION 3.25)

project(spume VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SPUME_MARCH_NATIVE "Tune generated code for the build machine" OFF)
option(SPUME_BUILD_TESTS "Build the SPUME test suite" ON)
option(SPUME_BUILD_BENCH "Build the SPUME bench harness" ON)
option(SPUME_WITH_OPENFOAM "Enable contract tests against a built OpenFOAM tree" OFF)
option(SPUME_ENABLE_HIP "Build the HIP GPU backend (requires ROCm, ADR-0017)" OFF)
set(SPUME_HIP_ARCH "gfx1151" CACHE STRING "GPU architecture(s) for the HIP backend")

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Warnings-as-errors for all SPUME-owned code (vendored code is exempt by
# construction: it never links against this interface target).
add_library(spume-warnings INTERFACE)
target_compile_options(spume-warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Werror)

add_library(spume-arch INTERFACE)
if(SPUME_MARCH_NATIVE)
    target_compile_options(spume-arch INTERFACE -march=native)
endif()

# Enable testing before any subdirectory registers tests (add_test only takes
# effect after enable_testing()).
if(SPUME_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(src/core)
add_subdirectory(src/bridge)

if(SPUME_ENABLE_HIP)
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 20)
    add_subdirectory(src/backends/gpu)
endif()

if(SPUME_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(SPUME_BUILD_BENCH)
    add_subdirectory(bench)
endif()
