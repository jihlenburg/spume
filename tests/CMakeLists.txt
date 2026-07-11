# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg

# doctest is vendored under tests/thirdparty (see the README there); it is
# included as a SYSTEM directory so its internals are exempt from
# warnings-as-errors.
add_library(spume-test-main STATIC support/doctest_main.cpp)
target_include_directories(spume-test-main SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)
target_link_libraries(spume-test-main PUBLIC spume::core)

# Anchor the fixture directory here: the function body expands variables at
# the call site, which may be a subdirectory.
set(SPUME_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data)

function(spume_add_test name label)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE spume-test-main spume-warnings spume-arch)
    target_compile_definitions(${name} PRIVATE SPUME_TEST_DATA_DIR="${SPUME_TEST_DATA_DIR}")
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES LABELS ${label})
endfunction()

add_subdirectory(unit)
add_subdirectory(regression)
