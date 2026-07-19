// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <algorithm>
#include <cmath>
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
                        double coarse_tol = 1e-2, int coarse_max_iter = 500,
                        Dispatch dispatch = Dispatch::reference,
                        bool kcycle = false, int kcycle_max_levels = 5,
                        Reduction reduction = Reduction::standard)
        : coarse_tol_(coarse_tol), coarse_max_iter_(coarse_max_iter),
          dispatch_(dispatch), reduction_(reduction), kcycle_(kcycle),
          kcycle_max_levels_(kcycle_max_levels) {
        std::vector<Aggregation> aggs;
        std::vector<Csr> coarse_ops; // retained so build() need not recompute
        Csr cur = fine;
        while (static_cast<int>(aggs.size()) + 1 < max_levels &&
               cur.nrows > coarse_size) {
            Aggregation agg = aggregate(cur);
            if (agg.ncoarse >= cur.nrows) {
                break; // aggregation made no progress; stop coarsening
            }
            Csr coarse = galerkin(cur, agg);
            aggs.push_back(std::move(agg));
            coarse_ops.push_back(coarse);
            cur = std::move(coarse);
        }
        // Hand build() the Galerkin products we just formed: recomputing them
        // there (a coo_to_csr sort per level) is ~83% of per-solve setup, and
        // this is the amortised path (roadmap M2). The external-hierarchy ctor
        // has no such precomputed operators and lets build() form them.
        build(fine, aggs, smoother_opt, &coarse_ops);
    }

    // External hierarchy: reuse a coarsening built elsewhere — e.g. OpenFOAM's
    // cached, high-quality GAMGAgglomeration (the amortised, faceAreaPair
    // hierarchy). aggs[k] maps level-k cells to level-(k+1) cells; the coarse
    // operators are the Galerkin products on that structure. This is the
    // "reuse the trunk, own the kernels" path (ADR-0001).
    AmgPrecond(const Csr& fine, const std::vector<Aggregation>& aggs,
               ChebyshevOptions smoother_opt = {},
               double coarse_tol = 1e-2, int coarse_max_iter = 500,
               Dispatch dispatch = Dispatch::reference,
               bool kcycle = false, int kcycle_max_levels = 5,
               Reduction reduction = Reduction::standard)
        : coarse_tol_(coarse_tol), coarse_max_iter_(coarse_max_iter),
          dispatch_(dispatch), reduction_(reduction), kcycle_(kcycle),
          kcycle_max_levels_(kcycle_max_levels) {
        build(fine, aggs, smoother_opt);
    }

    void apply(std::span<const double> r, std::span<double> z) const override {
        coarse_solve(0, r, z);
    }

    int num_levels() const { return static_cast<int>(levels_.size()) + 1; }

private:
    struct Level {
        Sell<double> a;
        std::optional<ChebyshevPrecond<T>> smoother;
        Aggregation agg;
        mutable std::vector<double> res, az, sm, rc, ec;
        mutable std::vector<double> kc, kv, kd, kw, kr; // K-cycle workspace
    };

    // One multigrid cycle at level lvl: pre-smooth, restrict, coarse-correct,
    // prolong, post-smooth. The coarse correction is handled by coarse_solve,
    // which is a single recursive cycle (V-cycle) or a short Krylov acceleration
    // over recursive cycles (K-cycle).
    void cycle(std::size_t lvl, std::span<const double> r, std::span<double> z) const {
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
        coarse_solve(lvl + 1, std::span<const double>(lev.rc), std::span<double>(lev.ec));

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

    // Approximately solve A_lvl x = b. On the coarsest level this is the FP64 CG
    // solve. Above it, kcycle_ selects the coarse-correction strength:
    //   V-cycle : one recursive cycle (x = cycle(b)).
    //   K-cycle : project the correction onto span{c, d}, where c = cycle(b) and
    //             d = cycle(b - alpha A c) (Notay's aggregation-AMG K-cycle,
    //             Galerkin/Ritz over the two directions). A residual test skips
    //             the second direction — and its whole recursive subtree — when
    //             the first already cuts the residual by >=4x, which keeps the
    //             K-cycle near-linear instead of exponential in the level count.
    // The transfer operators are unsmoothed aggregation, for which a plain
    // V-cycle degrades with problem size; the Krylov wrap is what recovers a
    // mesh-independent, GAMG-class convergence rate.
    void coarse_solve(std::size_t lvl, std::span<const double> b,
                      std::span<double> x) const {
        if (lvl == levels_.size()) {
            std::fill(x.begin(), x.end(), 0.0);
            SolveOptions co;
            co.tol = coarse_tol_;
            co.max_iter = coarse_max_iter_;
            co.dispatch = dispatch_;
            co.reduction = reduction_; // preserve cross-thread bitwise reproducibility
            cg(coarsest_, b, x, co);
            return;
        }
        // K-cycle on a window of coarse levels [1 .. kcycle_max_levels_]:
        //  - Skip the finest level (lvl 0): the outer flexible-CG already
        //    Krylov-accelerates the finest grid, so K-accelerating it is
        //    redundant AND doubles passes over the largest (bandwidth-bound)
        //    arrays — the dominant per-iteration cost. Level 0 stays a plain
        //    V-cycle; its coarse correction (lvl 1+) is where K kicks in.
        //  - Cap the depth: the coarse-space (weak-approximation) deficiency of
        //    unsmoothed aggregation is worst at the finest coarse levels; deeper
        //    levels recover with a plain V-cycle. The cap bounds the recursive
        //    fan-out to ~2^kcycle_max_levels_ instead of 2^numLevels — bounded
        //    vs runaway on a deep (pairwise) hierarchy.
        if (!kcycle_ || lvl == 0 || static_cast<int>(lvl) > kcycle_max_levels_) {
            cycle(lvl, b, x);
            return;
        }
        const Level& lev = levels_[lvl];
        const std::size_t n = b.size();

        // c = M b (one recursive cycle), v = A c
        cycle(lvl, b, std::span<double>(lev.kc));
        spmv(lev.a, std::span<const double>(lev.kc), std::span<double>(lev.kv), dispatch_);
        const double rho1 = ddot(lev.kc, b);
        const double sig1 = ddot(lev.kc, lev.kv);
        if (!(sig1 > 0.0)) { // not SPD-usable here: take the plain cycle result
            std::copy(lev.kc.begin(), lev.kc.end(), x.begin());
            return;
        }
        const double alpha = rho1 / sig1;

        // r1 = b - alpha v ; the residual test decides on a second direction.
        double nb = 0.0, nr = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            lev.kr[i] = b[i] - alpha * lev.kv[i];
            nb += b[i] * b[i];
            nr += lev.kr[i] * lev.kr[i];
        }
        if (nr <= 0.0625 * nb) { // ||r1|| <= 0.25 ||b|| : one term is enough
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = alpha * lev.kc[i];
            }
            return;
        }

        // second direction d = M r1, w = A d ; Galerkin projection onto {c, d}.
        cycle(lvl, std::span<const double>(lev.kr), std::span<double>(lev.kd));
        spmv(lev.a, std::span<const double>(lev.kd), std::span<double>(lev.kw), dispatch_);
        const double gam = ddot(lev.kc, lev.kw); // <c, A d> = <d, A c>
        const double bet = ddot(lev.kd, lev.kw); // <d, A d>
        const double rho2 = ddot(lev.kd, b);     // <d, b>
        const double det = sig1 * bet - gam * gam;
        if (!(std::abs(det) > 0.0)) { // directions dependent: fall back to one
            for (std::size_t i = 0; i < n; ++i) {
                x[i] = alpha * lev.kc[i];
            }
            return;
        }
        const double a1 = (rho1 * bet - gam * rho2) / det;
        const double a2 = (sig1 * rho2 - gam * rho1) / det;
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = a1 * lev.kc[i] + a2 * lev.kd[i];
        }
    }

    static double ddot(std::span<const double> a, std::span<const double> b) {
        double s = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            s += a[i] * b[i];
        }
        return s;
    }

    static std::size_t agg_of(const Level& lev, index_t i) {
        return static_cast<std::size_t>(lev.agg.agg[static_cast<std::size_t>(i)]);
    }

    void residual(const Level& lev, std::span<const double> r,
                  std::span<const double> z) const {
        spmv(lev.a, z, std::span<double>(lev.az), dispatch_);
        for (std::size_t i = 0; i < r.size(); ++i) {
            lev.res[i] = r[i] - lev.az[i];
        }
    }

    // Assemble the level operators + smoothers from a fine operator and an
    // ordered list of aggregations (each level's Galerkin product on the next).
    // `precomputed`, when non-null, supplies the Galerkin coarse operator for
    // each level (precomputed[k] is the level-(k+1) operator), so build() can
    // move it in instead of recomputing galerkin(cur, agg). The self-coarsening
    // ctor forms these during its descent; the external-hierarchy ctor passes
    // null and build() forms them here.
    void build(const Csr& fine, const std::vector<Aggregation>& aggs,
               ChebyshevOptions smoother_opt,
               std::vector<Csr>* precomputed = nullptr) {
        Csr cur = fine;
        for (std::size_t k = 0; k < aggs.size(); ++k) {
            const Aggregation& agg = aggs[k];
            // guard a malformed external hierarchy: stop and treat cur as coarsest
            if (static_cast<std::size_t>(cur.nrows) != agg.agg.size()) {
                break;
            }
            Level lev;
            lev.a = sell_from_csr(cur);
            lev.smoother.emplace(make_eq_operator<T>(cur), smoother_opt, dispatch_);
            const auto n = static_cast<std::size_t>(cur.nrows);
            const auto nc = static_cast<std::size_t>(agg.ncoarse);
            lev.res.resize(n);
            lev.az.resize(n);
            lev.sm.resize(n);
            lev.rc.resize(nc);
            lev.ec.resize(nc);
            lev.kc.resize(n);
            lev.kv.resize(n);
            lev.kd.resize(n);
            lev.kw.resize(n);
            lev.kr.resize(n);
            Csr coarse = (precomputed != nullptr) ? std::move((*precomputed)[k])
                                                  : galerkin(cur, agg);
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
    Dispatch dispatch_ = Dispatch::reference;
    // Reduction mode for the coarsest-level FP64 CG. Threaded through so a
    // deterministic outer solve stays bitwise-reproducible across thread counts
    // (numerics policy) — the coarse CG is the one reduction inside the cycle.
    Reduction reduction_ = Reduction::standard;
    bool kcycle_ = false;
    int kcycle_max_levels_ = 5; // K-accelerate the top few coarse levels
};

} // namespace spume
