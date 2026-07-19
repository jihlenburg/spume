// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <vector>

#include "core/formats.hpp"
#include "core/fused_spmv.hpp"
#include "core/sell.hpp"
#include "core/spmv.hpp"
#include "core/vecops.hpp"

// checkasm-style contract (ADR-0016): the fused kernel r += alpha*(A x) must be
// BIT-FOR-BIT equal to the unfused reference spmv(a,x,t); axpy(alpha,t,r), for
// every precision and dispatch path. A specialized backend later replaces the
// fused path and is pinned to this same equality.

namespace {

// A small banded, width-varying matrix (5 chunks of 8 rows) as CSR. Values are
// arbitrary — the equality holds regardless, since fused and unfused do the
// identical floating-point operations in the identical order.
spume::Csr make_banded_csr(int n) {
    spume::Coo coo;
    coo.nrows = n;
    coo.ncols = n;
    for (int i = 0; i < n; ++i) {
        coo.row.push_back(i);
        coo.col.push_back(i);
        coo.val.push_back(4.0 + 0.1 * i);
        if (i > 0) {
            coo.row.push_back(i);
            coo.col.push_back(i - 1);
            coo.val.push_back(-1.0 - 0.01 * i);
        }
        if (i + 1 < n) {
            coo.row.push_back(i);
            coo.col.push_back(i + 1);
            coo.val.push_back(-1.0 + 0.02 * i);
        }
        // widen some rows so chunk widths vary (exercises the padded lanes)
        if (i % 3 == 0 && i + 2 < n) {
            coo.row.push_back(i);
            coo.col.push_back(i + 2);
            coo.val.push_back(0.25);
        }
    }
    return spume::coo_to_csr(coo);
}

template<typename T>
void check_fused_equals_unfused(const spume::Sell<T>& a, T alpha, spume::Dispatch d) {
    const auto n = static_cast<std::size_t>(a.nrows);
    std::vector<T> x(static_cast<std::size_t>(a.ncols));
    std::vector<T> r0(n);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<T>(0.5 + 0.25 * static_cast<double>(i % 7));
    }
    for (std::size_t i = 0; i < n; ++i) {
        r0[i] = static_cast<T>(-1.3 + 0.11 * static_cast<double>(i % 5));
    }

    // unfused reference: t = A x ; r = r0 ; r += alpha t
    std::vector<T> t(n);
    spume::spmv(a, std::span<const T>(x), std::span<T>(t), d);
    std::vector<T> r_ref = r0;
    spume::axpy(alpha, std::span<const T>(t), std::span<T>(r_ref), d);

    // fused: r = r0 ; r += alpha (A x)
    std::vector<T> r_fused = r0;
    spume::spmv_axpy(a, alpha, std::span<const T>(x), std::span<T>(r_fused), d);

    for (std::size_t i = 0; i < n; ++i) {
        // bit-for-bit, not approximate: identical ops in identical order
        CHECK(r_fused[i] == r_ref[i]);
    }
}

} // namespace

TEST_CASE("spmv_axpy is bit-identical to spmv+axpy (FP64)") {
    const spume::Csr csr = make_banded_csr(40);
    const spume::Sell<double> a = spume::sell_from_csr(csr);
    for (spume::Dispatch d : {spume::Dispatch::reference, spume::Dispatch::openmp}) {
        check_fused_equals_unfused<double>(a, -1.0, d);
        check_fused_equals_unfused<double>(a, 0.75, d);
    }
}

TEST_CASE("spmv_axpy is bit-identical to spmv+axpy (FP32)") {
    const spume::Csr csr = make_banded_csr(40);
    const spume::Sell<float> a = spume::detail::sell_from_csr_as<float>(csr);
    for (spume::Dispatch d : {spume::Dispatch::reference, spume::Dispatch::openmp}) {
        check_fused_equals_unfused<float>(a, -1.0F, d);
        check_fused_equals_unfused<float>(a, 0.75F, d);
    }
}
