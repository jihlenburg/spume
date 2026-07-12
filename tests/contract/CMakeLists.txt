# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

# Contract tests need a built OpenFOAM tree (vendored by default, or the
# tree named by SPUME_OPENFOAM_DIR — the nightly canary points that at
# upstream develop). Regular CI does not build OpenFOAM, so these are
# opt-in via -DSPUME_WITH_OPENFOAM=ON.
if(SPUME_WITH_OPENFOAM)
    add_test(NAME spume-contract-cavity
        COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/run_contract_cavity.sh)
    set_tests_properties(spume-contract-cavity PROPERTIES
        LABELS contract
        TIMEOUT 1800)
endif()
