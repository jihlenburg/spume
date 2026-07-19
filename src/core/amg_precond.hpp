// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "core/amg.hpp"
#include "core/equilibrate.hpp"
#include "core/formats.hpp"
#include "core/fused_spmv.hpp"
#include "core/precond.hpp"
#include "core/sell.hpp"
#include "core/solver.hpp"
#include "core/spmv.hpp"

// Two-level algebraic-multigrid preconditioner — the V-cycle that assembles the
// FP32-GAMG-under-FP64-Krylov engine (roadmap M2, ADR-0002):
//
//   z = 0
//   z += S(r - A z)                  pre-smooth   (Chebyshev; FP32 when T=float)
//   res = r - A z ; rc = P^T res     restrict
//   A_c ec = rc                      coarse solve (FP64 CG on the Galerkin op)
//   z += P ec                        prolong
//   z += S(r - A z)                  post-smooth
//
// The smoother interior runs at precision T (float => the ADR-0002 mixed path,
// with mandatory diagonal equilibration via make_eq_operator<T>); the fine
// operator, the residual, and the coarse solve stay FP64. The final answer is
// defined entirely by the flexible FP64 outer Krylov this preconditions — the
// V-cycle only accelerates it, so it can be as approximate as it likes. A later
// slice recurses to many levels and applies RCM (core/reorder.hpp) per level;
// this MVP is two levels and takes the caller's ordering as given.

namespace spume {

template<typename T>
class TwoLevelPrecond final : public Preconditioner {
public:
    explicit TwoLevelPrecond(const Csr& fine, ChebyshevOptions smoother_opt = {},
                             double coarse_tol = 1e-2, int coarse_max_iter = 500)
        : agg_(aggregate(fine)),
          af_(sell_from_csr(fine)),
          ac_(sell_from_csr(galerkin(fine, agg_))),
          smoother_(make_eq_operator<T>(fine), smoother_opt),
          coarse_tol_(coarse_tol),
          coarse_max_iter_(coarse_max_iter) {
        res_.resize(static_cast<std::size_t>(af_.nrows));
        az_.resize(static_cast<std::size_t>(af_.nrows));
        sm_.resize(static_cast<std::size_t>(af_.nrows));
        rc_.resize(static_cast<std::size_t>(ac_.nrows));
        ec_.resize(static_cast<std::size_t>(ac_.nrows));
    }

    void apply(std::span<const double> r, std::span<double> z) const override {
        const std::size_t n = r.size();

        // pre-smooth: z starts at 0, so z = S(r)
        smoother_.apply(r, z);

        // res = r - A z
        residual(r, z);
        // restrict rc = P^T res
        std::fill(rc_.begin(), rc_.end(), 0.0);
        for (index_t i = 0; i < af_.nrows; ++i) {
            rc_[static_cast<std::size_t>(agg_.agg[static_cast<std::size_t>(i)])] +=
                res_[static_cast<std::size_t>(i)];
        }
        // coarse solve A_c ec = rc
        std::fill(ec_.begin(), ec_.end(), 0.0);
        SolveOptions co;
        co.tol = coarse_tol_;
        co.max_iter = coarse_max_iter_;
        cg(ac_, std::span<const double>(rc_), std::span<double>(ec_), co);
        // prolong z += P ec
        for (index_t i = 0; i < af_.nrows; ++i) {
            z[static_cast<std::size_t>(i)] +=
                ec_[static_cast<std::size_t>(agg_.agg[static_cast<std::size_t>(i)])];
        }

        // post-smooth: z += S(r - A z)
        residual(r, z);
        smoother_.apply(std::span<const double>(res_), std::span<double>(sm_));
        for (std::size_t i = 0; i < n; ++i) {
            z[i] += sm_[i];
        }
    }

private:
    // res_ = r - A z
    void residual(std::span<const double> r, std::span<const double> z) const {
        spmv(af_, z, std::span<double>(az_));
        for (std::size_t i = 0; i < r.size(); ++i) {
            res_[i] = r[i] - az_[i];
        }
    }

    Aggregation agg_;
    Sell<double> af_; // fine operator (FP64): residual + coarse-grid transfer
    Sell<double> ac_; // coarse Galerkin operator (FP64): CG coarse solve
    ChebyshevPrecond<T> smoother_;
    double coarse_tol_;
    int coarse_max_iter_;
    mutable std::vector<double> res_, az_, sm_, rc_, ec_;
};

} // namespace spume
