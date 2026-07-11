// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <cstring>
#include <omp.h>

#include "core/version.hpp"

TEST_CASE("version string is set") {
    CHECK(std::strcmp(spume::version(), "0.1.0") == 0);
}

TEST_CASE("OpenMP runtime is available") {
    CHECK(omp_get_max_threads() >= 1);
}
