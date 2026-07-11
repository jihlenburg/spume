# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

add_library(spume-core STATIC
    formats.cpp
    poisson.cpp
    reduce.cpp
    sell.cpp
    solver.cpp
    version.cpp
)
add_library(spume::core ALIAS spume-core)

# Public include root is src/, so consumers write #include "core/....hpp".
target_include_directories(spume-core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(spume-core
    PUBLIC OpenMP::OpenMP_CXX
    PRIVATE spume-warnings spume-arch)
