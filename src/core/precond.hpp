// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include <cassert>
#include <span>
#include <vector>

#include "core/equilibrate.hpp"
#include "core/fused_spmv.hpp"
#include "core/spmv.hpp"
#include "core/types.hpp"
#include "core/vecops.hpp"

namespace spume {

// Preconditioner interface: z ~= M^{-1} r, SPD (possibly mildly varying due
// to reduced-precision arithmetic — which is why the outer Krylov method is
// flexible). apply() is not reentrant: implementations may use internal
// workspace. Parallelism lives inside apply(), driven by Dispatch.
class Preconditioner {
public:
    virtual ~Preconditioner() = default;
    virtual void apply(std::span<const double> r, std::span<double> z) const = 0;
};

class IdentityPrecond final : public Preconditioner {
public:
    void apply(std::span<const double> r, std::span<double> z) const override {
        assert(r.size() == z.size());
        for (std::size_t i = 0; i < r.size(); ++i) {
            z[i] = r[i];
        }
    }
};

// Jacobi through the equilibrated form: M^{-1} = S I S with the inner
// identity evaluated at precision T,
//
//     z_i = s_i * double(T(s_i * r_i)),
//
// so for T = double this is exactly D^{-1}, and for T = float the residual
// makes one round trip through FP32 — the minimal member of the
// reduced-precision preconditioner family. Elementwise, hence bitwise
// reproducible for any thread count.
template<typename T>
class JacobiPrecond final : public Preconditioner {
public:
    JacobiPrecond(const EqOperator<T>& op, Dispatch dispatch = Dispatch::reference)
        : scale_(op.scale), dispatch_(dispatch) {}

    void apply(std::span<const double> r, std::span<double> z) const override {
        assert(r.size() == scale_.size() && z.size() == scale_.size());
        const auto n = static_cast<std::int64_t>(scale_.size());
        if (dispatch_ == Dispatch::openmp) {
#pragma omp parallel for schedule(static)
            for (std::int64_t i = 0; i < n; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                z[ii] = scale_[ii] * static_cast<double>(static_cast<T>(scale_[ii] * r[ii]));
            }
        } else {
            for (std::int64_t i = 0; i < n; ++i) {
                const auto ii = static_cast<std::size_t>(i);
                z[ii] = scale_[ii] * static_cast<double>(static_cast<T>(scale_[ii] * r[ii]));
            }
        }
    }

private:
    std::vector<double> scale_;
    Dispatch dispatch_;
};

struct ChebyshevOptions {
    int steps = 5;     // Chebyshev corrections; steps-1 inner SpMVs per apply
    double eta = 30.0; // targeted spectrum [lambda_max/eta, lambda_max]
};

// Chebyshev semi-iteration on the equilibrated system, entirely at
// precision T:
//
//     z = S p_k(S A S) S r,
//
// where p_k is the Chebyshev polynomial approximation of the inverse on
// [lambda_max_bound/eta, lambda_max_bound] (Saad, "Iterative Methods for
// Sparse Linear Systems", 2nd ed., Alg. 12.1 recurrence). Since the residual
// polynomial q_k satisfies 0 < q_k(l) < 1 below the interval and
// |q_k(l)| < 1 on it, p_k(l) = (1 - q_k(l))/l > 0 on (0, lambda_max]:
// the preconditioner stays SPD even when eta underestimates the condition
// number. All inner work (SpMV + elementwise updates) is bitwise
// reproducible across thread counts; combined with deterministic outer
// reductions this makes full solves bitwise reproducible.
//
// For T = float this is the traffic workhorse: steps-1 SpMVs on FP32
// values + FP32 vectors instead of FP64.
template<typename T>
class ChebyshevPrecond final : public Preconditioner {
public:
    ChebyshevPrecond(EqOperator<T> op, ChebyshevOptions opt = {},
                     Dispatch dispatch = Dispatch::reference)
        : op_(std::move(op)), opt_(opt), dispatch_(dispatch) {
        const auto n = static_cast<std::size_t>(op_.a.nrows);
        wr_.resize(n);
        wx_.resize(n);
        wd_.resize(n);
    }

    void apply(std::span<const double> r, std::span<double> z) const override {
        const auto n = static_cast<std::size_t>(op_.a.nrows);
        assert(r.size() == n && z.size() == n);

        const double lmax = op_.lambda_max_bound;
        const double lmin = lmax / opt_.eta;
        const double theta = 0.5 * (lmax + lmin);
        const double delta = 0.5 * (lmax - lmin);
        const double sigma1 = theta / delta;

        // wr = T(S r): scaling in FP64, one narrowing store (policy-clean —
        // the operand being demoted is the equilibrated residual).
        for (std::size_t i = 0; i < n; ++i) {
            wr_[i] = static_cast<T>(op_.scale[i] * r[i]);
        }

        // d_0 = r_0 / theta, x_0 = 0.
        const T inv_theta = static_cast<T>(1.0 / theta);
        for (std::size_t i = 0; i < n; ++i) {
            wd_[i] = wr_[i] * inv_theta;
            wx_[i] = T{0};
        }

        double rho = 1.0 / sigma1;
        for (int k = 0; k < opt_.steps; ++k) {
            // x_{k+1} = x_k + d_k
            axpy(T{1}, std::span<const T>(wd_), std::span<T>(wx_), dispatch_);
            if (k + 1 == opt_.steps) {
                break; // last correction applied; r/d updates would be dead
            }
            // r_{k+1} = r_k - A d_k (fused: A d_k is never materialised, so two
            // vector passes per step disappear — see core/fused_spmv.hpp)
            spmv_axpy(op_.a, T{-1}, std::span<const T>(wd_), std::span<T>(wr_), dispatch_);
            // d_{k+1} = rho_{k+1} rho_k d_k + (2 rho_{k+1}/delta) r_{k+1}
            const double rho_new = 1.0 / (2.0 * sigma1 - rho);
            axpby(static_cast<T>(2.0 * rho_new / delta), std::span<const T>(wr_),
                  static_cast<T>(rho_new * rho), std::span<T>(wd_), dispatch_);
            rho = rho_new;
        }

        // z = S x_m, widened back to FP64.
        for (std::size_t i = 0; i < n; ++i) {
            z[i] = op_.scale[i] * static_cast<double>(wx_[i]);
        }
    }

    const EqOperator<T>& op() const { return op_; }

private:
    EqOperator<T> op_;
    ChebyshevOptions opt_;
    Dispatch dispatch_;
    mutable std::vector<T> wr_, wx_, wd_;
};

} // namespace spume
