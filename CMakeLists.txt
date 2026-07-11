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

find_package(OpenMP REQUIRED COMPONENTS CXX)

# Warnings-as-errors for all SPUME-owned code (vendored code is exempt by
# construction: it never links against this interface target).
add_library(spume-warnings INTERFACE)
target_compile_options(spume-warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Werror)

add_library(spume-arch INTERFACE)
if(SPUME_MARCH_NATIVE)
    target_compile_options(spume-arch INTERFACE -march=native)
endif()

add_subdirectory(src/core)

if(SPUME_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SPUME_BUILD_BENCH)
    add_subdirectory(bench)
endif()
