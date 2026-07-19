# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

spume_add_test(spume-regression-mixed-precision regression test_mixed_precision.cpp)
spume_add_test(spume-regression-determinism regression test_determinism.cpp)
target_include_directories(spume-regression-mixed-precision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../support)

# The equivalence comparator is pure Python (no OpenFOAM); its self-test runs
# in ordinary CI so the oracle itself stays trusted.
add_test(NAME spume-regression-equivalence-selftest
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_check_equivalence.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(spume-regression-equivalence-selftest PROPERTIES LABELS regression)

# The bitwise leaf-vs-stock equivalence gate needs a built OpenFOAM tree
# (vendored by default, or SPUME_OPENFOAM_DIR). Regular CI does not build
# OpenFOAM, so it is opt-in via -DSPUME_WITH_OPENFOAM=ON and runs in the
# nightly OpenFOAM job alongside the contract suite.
if(SPUME_WITH_OPENFOAM)
    add_test(NAME spume-equivalence-pimple
        COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/run_equivalence_pimple.sh)
    set_tests_properties(spume-equivalence-pimple PROPERTIES
        LABELS equivalence
        TIMEOUT 3600)

    # The runtime-selectable spumePCG solver equals the reference PCG within the
    # rounding-reorder class (ADR-0002). Stage 0 (passthrough) is bitwise; the
    # gate uses the looser reorder-tolerance so it also holds for Stage 1's real
    # M0 CG. Builds libspumeFoamSolvers via wmake on first run.
    add_test(NAME spume-equivalence-spumepcg
        COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/run_spumepcg_equivalence.sh)
    set_tests_properties(spume-equivalence-spumepcg PROPERTIES
        LABELS equivalence
        TIMEOUT 3600)

    # Mixed precision (ADR-0002): spumePCG with the FP32 Chebyshev preconditioner
    # under the FP64 outer CG must stay within the same equivalence class — the
    # FP32 preconditioner steers but never defines the answer.
    add_test(NAME spume-equivalence-spumepcg-fp32
        COMMAND ${CMAKE_COMMAND} -E env SPUME_PRECOND=chebyshevFP32
                sh ${CMAKE_CURRENT_SOURCE_DIR}/run_spumepcg_equivalence.sh)
    set_tests_properties(spume-equivalence-spumepcg-fp32 PROPERTIES
        LABELS equivalence
        TIMEOUT 3600)
endif()
