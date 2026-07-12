# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

add_executable(spume-bench main.cpp)
target_link_libraries(spume-bench PRIVATE spume::core spume-warnings spume-arch)
