// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <vector>

#include "bridge/ldu_to_sell.hpp"
#include "core/sell.hpp"
#include "core/spmv.hpp"

namespace {

std::vector<double> spmv_vec(const spume::Sell<double>& a, const std::vector<double>& x) {
    std::vector<double> y(x.size(), 0.0);
    spume::spmv(a, std::span<const double>(x), std::span<double>(y));
    return y;
}

} // namespace

// A 3-cell 1-D graph: face 0 joins cells (0,1), face 1 joins cells (1,2).
// OpenFOAM LDU: upper[f] = A[lowerAddr[f]][upperAddr[f]],
//               lower[f] = A[upperAddr[f]][lowerAddr[f]], diag[i] = A[i][i].
TEST_CASE("assemble_sell builds a symmetric tridiagonal system") {
    const std::vector<int> lowerAddr{0, 1};
    const std::vector<int> upperAddr{1, 2};
    const std::vector<double> diag{2.0, 2.0, 2.0};
    const std::vector<double> upper{-1.0, -1.0};

    // Symmetric: empty `lower` means lower == upper.
    const spume::Sell<double> a = spume::assemble_sell(
        lowerAddr, upperAddr, diag, upper, /*lower=*/{}, /*nCells=*/3);

    CHECK(a.nrows == 3);
    CHECK(a.ncols == 3);

    //  [ 2 -1  0 ] [1]   [ 0 ]
    //  [-1  2 -1 ] [2] = [ 0 ]
    //  [ 0 -1  2 ] [3]   [ 4 ]
    const std::vector<double> y = spmv_vec(a, {1.0, 2.0, 3.0});
    CHECK(y[0] == doctest::Approx(0.0));
    CHECK(y[1] == doctest::Approx(0.0));
    CHECK(y[2] == doctest::Approx(4.0));
}

// Asymmetric coefficients pin which of lower/upper lands below vs above the
// diagonal — a sign/transpose error cannot pass this.
TEST_CASE("assemble_sell places lower and upper on the correct sides") {
    const std::vector<int> lowerAddr{0, 1};
    const std::vector<int> upperAddr{1, 2};
    const std::vector<double> diag{2.0, 2.0, 2.0};
    const std::vector<double> upper{-1.0, -1.0};  // A[0][1], A[1][2]
    const std::vector<double> lower{-3.0, -4.0};  // A[1][0], A[2][1]

    const spume::Sell<double> a =
        spume::assemble_sell(lowerAddr, upperAddr, diag, upper, lower, 3);

    //  [ 2 -1  0 ] [1]   [  0 ]
    //  [-3  2 -1 ] [2] = [ -2 ]
    //  [ 0 -4  2 ] [3]   [ -2 ]
    const std::vector<double> y = spmv_vec(a, {1.0, 2.0, 3.0});
    CHECK(y[0] == doctest::Approx(0.0));
    CHECK(y[1] == doctest::Approx(-2.0));
    CHECK(y[2] == doctest::Approx(-2.0));
}

// assemble_csr must build the identical operator (it feeds make_eq_operator<T>
// for the FP32 preconditioner). Verified through the same SpMV path.
TEST_CASE("assemble_csr builds the same operator as assemble_sell") {
    const std::vector<int> lowerAddr{0, 1};
    const std::vector<int> upperAddr{1, 2};
    const std::vector<double> diag{2.0, 2.0, 2.0};
    const std::vector<double> upper{-1.0, -1.0};

    const spume::Csr c =
        spume::assemble_csr(lowerAddr, upperAddr, diag, upper, /*lower=*/{}, 3);

    CHECK(c.nrows == 3);
    CHECK(c.nnz() == 7);  // 3 diagonal + 2*2 off-diagonal, no duplicates

    const spume::Sell<double> a = spume::sell_from_csr(c);
    const std::vector<double> y = spmv_vec(a, {1.0, 2.0, 3.0});
    CHECK(y[0] == doctest::Approx(0.0));
    CHECK(y[1] == doctest::Approx(0.0));
    CHECK(y[2] == doctest::Approx(4.0));
}

TEST_CASE("assemble_sell rejects mismatched array sizes") {
    const std::vector<int> l1{0};
    const std::vector<int> u2{1, 2};      // upperAddr longer than lowerAddr
    const std::vector<int> u1{1};
    const std::vector<double> d2{1.0, 1.0};
    const std::vector<double> d3{1.0, 1.0, 1.0};
    const std::vector<double> up1{1.0};
    const std::vector<double> none{};

    // upperAddr size != lowerAddr size
    CHECK_THROWS_AS(spume::assemble_sell(l1, u2, d2, up1, none, 2), std::invalid_argument);
    // diag size != nCells
    CHECK_THROWS_AS(spume::assemble_sell(l1, u1, d3, up1, none, 2), std::invalid_argument);
}
