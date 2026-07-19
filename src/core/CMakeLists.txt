# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

add_library(spume-core STATIC
    amg.cpp
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

# The M2 solver library (wmake, shared object) links this static lib, so its
# objects must be position-independent.
set_target_properties(spume-core PROPERTIES POSITION_INDEPENDENT_CODE ON)
