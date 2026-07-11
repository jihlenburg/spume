// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "core/formats.hpp"
#include "core/sell.hpp"
#include "core/types.hpp"

namespace spume {

// Symmetrically equilibrated operator: a_eq = S A S with S = diag(s),
// s_i = 1 / sqrt(a_ii). For SPD input, a_eq is SPD with unit diagonal.
//
// Numerics policy (hard rule): diagonal equilibration is REQUIRED before any
// precision demotion. make_eq_operator<float> is therefore the only public
// route to FP32 coefficients: scaling happens in FP64 on the CSR values, and
// narrowing to T is the final store into SELL storage.
template<typename T>
struct EqOperator {
    Sell<T> a;                 // S A S, unit diagonal, values stored as T
    std::vector<double> scale; // s_i in FP64
    double lambda_max_bound;   // Gershgorin upper bound on lambda_max(S A S)
};

// Throws std::invalid_argument if any diagonal entry is missing or <= 0.
template<typename T>
EqOperator<T> make_eq_operator(const Csr& a) {
    if (a.nrows != a.ncols) {
        throw std::invalid_argument("make_eq_operator: matrix must be square");
    }
    const auto n = static_cast<std::size_t>(a.nrows);

    std::vector<double> s(n);
    for (std::size_t r = 0; r < n; ++r) {
        double diag = 0.0;
        bool found = false;
        for (auto k = a.rowptr[r]; k < a.rowptr[r + 1]; ++k) {
            if (a.col[static_cast<std::size_t>(k)] == static_cast<index_t>(r)) {
                diag = a.val[static_cast<std::size_t>(k)];
                found = true;
                break;
            }
        }
        if (!found || !(diag > 0.0)) {
            throw std::invalid_argument("make_eq_operator: nonpositive or missing diagonal");
        }
        s[r] = 1.0 / std::sqrt(diag);
    }

    // Scale in FP64 on a CSR copy; compute the Gershgorin bound on the
    // equilibrated values before any narrowing.
    Csr eq = a;
    double bound = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
        double abs_row_sum = 0.0;
        for (auto k = eq.rowptr[r]; k < eq.rowptr[r + 1]; ++k) {
            const auto kk = static_cast<std::size_t>(k);
            eq.val[kk] = s[r] * eq.val[kk] * s[static_cast<std::size_t>(eq.col[kk])];
            abs_row_sum += std::abs(eq.val[kk]);
        }
        bound = std::max(bound, abs_row_sum);
    }

    EqOperator<T> out;
    out.a = detail::sell_from_csr_as<T>(eq); // narrowing (if any) happens here
    out.scale = std::move(s);
    out.lambda_max_bound = bound;
    return out;
}

} // namespace spume
