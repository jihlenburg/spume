# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

spume_add_test(spume-regression-mixed-precision regression test_mixed_precision.cpp)
spume_add_test(spume-regression-determinism regression test_determinism.cpp)
target_include_directories(spume-regression-mixed-precision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../support)
