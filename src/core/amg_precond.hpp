// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "core/amg.hpp"
#include "core/equilibrate.hpp"
#include "core/formats.hpp"
#include "core/fused_spmv.hpp"
#include "core/precond.hpp"
#include "core/sell.hpp"
#include "core/solver.hpp"
#include "core/spmv.hpp"

// Algebraic-multigrid preconditioner — the FP32-GAMG-under-FP64-Krylov engine
// (roadmap M2, ADR-0002). A recursive V-cycle over a hierarchy built by repeated
// aggregation + Galerkin coarsening down to a small coarsest operator:
//
//   per level:  z += S(r - A z)      pre-smooth   (Chebyshev; FP32 when T=float)
//               res = r - A z ; rc = P^T res       restrict
//               recurse on the coarse level        (V-cycle)
//               z += P ec ; z += S(r - A z)         prolong + post-smooth
//   coarsest:   A_c z = r via FP64 CG
//
// The smoother interior runs at precision T (float => the ADR-0002 mixed path,
// diagonal-equilibrated via make_eq_operator<T>); the operators, residuals, and
// the coarsest solve stay FP64. The final answer is defined entirely by the
// flexible FP64 outer Krylov this preconditions, so the cycle can be as
// approximate as it likes — it only accelerates. Hierarchy setup is amortised
// (mesh static). RCM per-level reordering (core/reorder.hpp) rides on top in a
// follow-up; here coarse levels inherit locality from a well-ordered fine level.

namespace spume {

template<typename T>
class AmgPrecond final : public Preconditioner {
public:
    // Self-coarsening: build the hierarchy with SPUME's own greedy aggregation.
    explicit AmgPrecond(const Csr& fine, ChebyshevOptions smoother_opt = {},
                        index_t coarse_size = 200, int max_levels = 20,
                        double coarse_tol = 1e-2, int coarse_max_iter = 500)
        : coarse_tol_(coarse_tol), coarse_max_iter_(coarse_max_iter) {
        std::vector<Aggregation> aggs;
        Csr cur = fine;
        while (static_cast<int>(aggs.size()) + 1 < max_levels &&
               cur.nrows > coarse_size) {
            Aggregation agg = aggregate(cur);
            if (agg.ncoarse >= cur.nrows) {
                break; // aggregation made no progress; stop coarsening
            }
            Csr coarse = galerkin(cur, agg);
            aggs.push_back(std::move(agg));
            cur = std::move(coarse);
        }
        build(fine, aggs, smoother_opt);
    }

    // External hierarchy: reuse a coarsening built elsewhere — e.g. OpenFOAM's
    // cached, high-quality GAMGAgglomeration (the amortised, faceAreaPair
    // hierarchy). aggs[k] maps level-k cells to level-(k+1) cells; the coarse
    // operators are the Galerkin products on that structure. This is the
    // "reuse the trunk, own the kernels" path (ADR-0001).
    AmgPrecond(const Csr& fine, const std::vector<Aggregation>& aggs,
               ChebyshevOptions smoother_opt = {},
               double coarse_tol = 1e-2, int coarse_max_iter = 500)
        : coarse_tol_(coarse_tol), coarse_max_iter_(coarse_max_iter) {
        build(fine, aggs, smoother_opt);
    }

    void apply(std::span<const double> r, std::span<double> z) const override {
        vcycle(0, r, z);
    }

    int num_levels() const { return static_cast<int>(levels_.size()) + 1; }

private:
    struct Level {
        Sell<double> a;
        std::optional<ChebyshevPrecond<T>> smoother;
        Aggregation agg;
        mutable std::vector<double> res, az, sm, rc, ec;
    };

    void vcycle(std::size_t lvl, std::span<const double> r, std::span<double> z) const {
        if (lvl == levels_.size()) {
            // coarsest level: FP64 CG (initial guess 0)
            std::fill(z.begin(), z.end(), 0.0);
            SolveOptions co;
            co.tol = coarse_tol_;
            co.max_iter = coarse_max_iter_;
            cg(coarsest_, r, z, co);
            return;
        }
        const Level& lev = levels_[lvl];

        // pre-smooth: z <- S(r)  (equivalent to one smoothing step from z = 0)
        lev.smoother->apply(r, z);

        // res = r - A z ; restrict rc = P^T res
        residual(lev, r, z);
        std::fill(lev.rc.begin(), lev.rc.end(), 0.0);
        for (index_t i = 0; i < lev.a.nrows; ++i) {
            lev.rc[agg_of(lev, i)] += lev.res[static_cast<std::size_t>(i)];
        }

        // coarse-grid correction: solve on the next level, ec starts 0
        std::fill(lev.ec.begin(), lev.ec.end(), 0.0);
        vcycle(lvl + 1, std::span<const double>(lev.rc), std::span<double>(lev.ec));

        // prolong: z += P ec
        for (index_t i = 0; i < lev.a.nrows; ++i) {
            z[static_cast<std::size_t>(i)] += lev.ec[agg_of(lev, i)];
        }

        // post-smooth: z += S(r - A z)
        residual(lev, r, z);
        lev.smoother->apply(std::span<const double>(lev.res), std::span<double>(lev.sm));
        for (std::size_t i = 0; i < r.size(); ++i) {
            z[i] += lev.sm[i];
        }
    }

    static std::size_t agg_of(const Level& lev, index_t i) {
        return static_cast<std::size_t>(lev.agg.agg[static_cast<std::size_t>(i)]);
    }

    void residual(const Level& lev, std::span<const double> r,
                  std::span<const double> z) const {
        spmv(lev.a, z, std::span<double>(lev.az));
        for (std::size_t i = 0; i < r.size(); ++i) {
            lev.res[i] = r[i] - lev.az[i];
        }
    }

    // Assemble the level operators + smoothers from a fine operator and an
    // ordered list of aggregations (each level's Galerkin product on the next).
    void build(const Csr& fine, const std::vector<Aggregation>& aggs,
               ChebyshevOptions smoother_opt) {
        Csr cur = fine;
        for (const Aggregation& agg : aggs) {
            // guard a malformed external hierarchy: stop and treat cur as coarsest
            if (static_cast<std::size_t>(cur.nrows) != agg.agg.size()) {
                break;
            }
            Level lev;
            lev.a = sell_from_csr(cur);
            lev.smoother.emplace(make_eq_operator<T>(cur), smoother_opt);
            const auto n = static_cast<std::size_t>(cur.nrows);
            const auto nc = static_cast<std::size_t>(agg.ncoarse);
            lev.res.resize(n);
            lev.az.resize(n);
            lev.sm.resize(n);
            lev.rc.resize(nc);
            lev.ec.resize(nc);
            Csr coarse = galerkin(cur, agg);
            lev.agg = agg;
            levels_.push_back(std::move(lev));
            cur = std::move(coarse);
        }
        coarsest_ = sell_from_csr(cur);
    }

    std::vector<Level> levels_; // finest .. down to one above coarsest
    Sell<double> coarsest_;
    double coarse_tol_;
    int coarse_max_iter_;
};

} // namespace spume
