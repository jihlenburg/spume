# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

spume_add_test(spume-unit-sanity unit test_sanity.cpp)
spume_add_test(spume-unit-reduce unit test_reduce.cpp)
spume_add_test(spume-unit-formats unit test_formats.cpp)
spume_add_test(spume-unit-poisson unit test_poisson.cpp)
spume_add_test(spume-unit-equilibrate unit test_equilibrate.cpp)
spume_add_test(spume-unit-solvers unit test_solvers.cpp)
spume_add_test(spume-unit-ldu-to-sell unit test_ldu_to_sell.cpp)
target_link_libraries(spume-unit-ldu-to-sell PRIVATE spume-bridge)
spume_add_test(spume-unit-fused-spmv unit test_fused_spmv.cpp)
spume_add_test(spume-unit-amg unit test_amg.cpp)
spume_add_test(spume-unit-reorder unit test_reorder.cpp)
