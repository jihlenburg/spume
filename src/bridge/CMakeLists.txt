# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

add_library(spume-bridge STATIC
    ldu_to_sell.cpp
)
add_library(spume::bridge ALIAS spume-bridge)

# Public include root is src/, so consumers write #include "bridge/....hpp".
target_include_directories(spume-bridge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(spume-bridge
    PUBLIC spume::core
    PRIVATE spume-warnings spume-arch)
