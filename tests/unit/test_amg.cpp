// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#include <doctest/doctest.h>

#include <vector>

#include "core/amg.hpp"
#include "core/formats.hpp"

namespace {

// 1-D Poisson (tridiagonal 2,-1) of size n as CSR — a clean SPD test operator
// whose constant vector is NOT in the null space (Dirichlet ends), but whose
// interior rows sum to zero, enough to exercise the Galerkin contraction.
spume::Csr poisson1d(int n) {
    spume::Coo coo;
    coo.nrows = n;
    coo.ncols = n;
    for (int i = 0; i < n; ++i) {
        coo.row.push_back(i); coo.col.push_back(i); coo.val.push_back(2.0);
        if (i > 0)     { coo.row.push_back(i); coo.col.push_back(i - 1); coo.val.push_back(-1.0); }
        if (i + 1 < n) { coo.row.push_back(i); coo.col.push_back(i + 1); coo.val.push_back(-1.0); }
    }
    return spume::coo_to_csr(coo);
}

std::vector<double> dense(const spume::Csr& a) {
    std::vector<double> d(static_cast<std::size_t>(a.nrows) * static_cast<std::size_t>(a.ncols), 0.0);
    for (spume::index_t r = 0; r < a.nrows; ++r) {
        for (auto k = a.rowptr[static_cast<std::size_t>(r)]; k < a.rowptr[static_cast<std::size_t>(r) + 1]; ++k) {
            d[static_cast<std::size_t>(r) * static_cast<std::size_t>(a.ncols) +
              static_cast<std::size_t>(a.col[static_cast<std::size_t>(k)])] += a.val[static_cast<std::size_t>(k)];
        }
    }
    return d;
}

} // namespace

TEST_CASE("aggregate assigns every unknown to exactly one aggregate and coarsens") {
    const spume::Csr a = poisson1d(32);
    const spume::Aggregation agg = spume::aggregate(a);

    CHECK(agg.agg.size() == 32);
    CHECK(agg.ncoarse > 0);
    CHECK(agg.ncoarse < 32); // it must actually coarsen
    for (spume::index_t g : agg.agg) {
        CHECK(g >= 0);
        CHECK(g < agg.ncoarse);
    }
}

TEST_CASE("galerkin equals the dense P^T A P for the piecewise-constant P") {
    const spume::Csr a = poisson1d(24);
    const spume::Aggregation agg = spume::aggregate(a);
    const spume::Csr ac = spume::galerkin(a, agg);

    const int n = a.nrows;
    const int nc = agg.ncoarse;
    CHECK(ac.nrows == nc);

    // Build dense P (n x nc), then dense A_ref = P^T A P.
    const std::vector<double> A = dense(a);
    std::vector<double> P(static_cast<std::size_t>(n) * static_cast<std::size_t>(nc), 0.0);
    for (int i = 0; i < n; ++i) {
        P[static_cast<std::size_t>(i) * static_cast<std::size_t>(nc) +
          static_cast<std::size_t>(agg.agg[static_cast<std::size_t>(i)])] = 1.0;
    }
    // AP = A * P  (n x nc)
    std::vector<double> AP(static_cast<std::size_t>(n) * static_cast<std::size_t>(nc), 0.0);
    for (int i = 0; i < n; ++i)
        for (int J = 0; J < nc; ++J) {
            double s = 0.0;
            for (int j = 0; j < n; ++j)
                s += A[static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)] *
                     P[static_cast<std::size_t>(j) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(J)];
            AP[static_cast<std::size_t>(i) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(J)] = s;
        }
    // Aref = P^T * AP  (nc x nc)
    std::vector<double> Aref(static_cast<std::size_t>(nc) * static_cast<std::size_t>(nc), 0.0);
    for (int I = 0; I < nc; ++I)
        for (int J = 0; J < nc; ++J) {
            double s = 0.0;
            for (int i = 0; i < n; ++i)
                s += P[static_cast<std::size_t>(i) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(I)] *
                     AP[static_cast<std::size_t>(i) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(J)];
            Aref[static_cast<std::size_t>(I) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(J)] = s;
        }

    // Galerkin CSR densified must equal Aref exactly (integer-weighted sums).
    const std::vector<double> AcD = dense(ac);
    for (int I = 0; I < nc; ++I)
        for (int J = 0; J < nc; ++J) {
            CHECK(AcD[static_cast<std::size_t>(I) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(J)] ==
                  doctest::Approx(Aref[static_cast<std::size_t>(I) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(J)]));
        }
}

TEST_CASE("galerkin coarse operator is symmetric for a symmetric input") {
    const spume::Csr a = poisson1d(40);
    const spume::Aggregation agg = spume::aggregate(a);
    const spume::Csr ac = spume::galerkin(a, agg);
    const std::vector<double> d = dense(ac);
    const int nc = ac.nrows;
    for (int I = 0; I < nc; ++I)
        for (int J = 0; J < nc; ++J)
            CHECK(d[static_cast<std::size_t>(I) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(J)] ==
                  doctest::Approx(d[static_cast<std::size_t>(J) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(I)]));
}
