// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

// SPUME Milestone 0 bench harness.
//
//   spume-bench stream [--mb M] [--reps R]      STREAM-style roofline probe
//   spume-bench spmv   [--nx N --ny N --nz N]   SpMV GB/s vs bytes model
//   spume-bench solve  [--nx N --ny N --nz N]   solver traffic comparison
//   spume-bench all                             all of the above, in order
//
// Methodology (per AGENTS.md performance policy):
//  - warm-up runs before every timed region; best-of-reps reported,
//  - no I/O inside timed regions,
//  - all "bytes moved" figures are the documented traffic MODEL from
//    sell.hpp / the per-iteration counts below — model, not counter,
//    measurements; the roofline reference is the measured STREAM triad.
//
// STREAM convention: copy/scale count 2 accesses per element, add/triad 3;
// write-allocate traffic is not counted (standard STREAM counting).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <omp.h>

#include "core/equilibrate.hpp"
#include "core/poisson.hpp"
#include "core/precond.hpp"
#include "core/reduce.hpp"
#include "core/solver.hpp"
#include "core/spmv.hpp"

namespace {

struct Args {
    std::string mode = "all";
    long mb = 256;
    int reps = 10;
    spume::index_t nx = 0, ny = 0, nz = 0; // 0 = mode default
    double tol = 1e-10;
};

Args parse(int argc, char** argv) {
    Args a;
    if (argc > 1 && argv[1][0] != '-') {
        a.mode = argv[1];
    }
    for (int i = 1; i + 1 < argc; ++i) {
        auto is = [&](const char* k) { return std::strcmp(argv[i], k) == 0; };
        if (is("--mb")) {
            a.mb = std::atol(argv[i + 1]);
        } else if (is("--reps")) {
            a.reps = std::atoi(argv[i + 1]);
        } else if (is("--nx")) {
            a.nx = std::atoi(argv[i + 1]);
        } else if (is("--ny")) {
            a.ny = std::atoi(argv[i + 1]);
        } else if (is("--nz")) {
            a.nz = std::atoi(argv[i + 1]);
        } else if (is("--tol")) {
            a.tol = std::atof(argv[i + 1]);
        }
    }
    return a;
}

// Best wall time of `reps` runs of fn(), after `warm` warm-ups.
template<typename F>
double best_time(int warm, int reps, F&& fn) {
    for (int i = 0; i < warm; ++i) {
        fn();
    }
    double best = 1e300;
    for (int i = 0; i < reps; ++i) {
        const double t0 = omp_get_wtime();
        fn();
        best = std::min(best, omp_get_wtime() - t0);
    }
    return best;
}

double g_sink = 0.0; // defeats dead-code elimination across timed kernels

double run_stream(long mb, int reps) {
    const auto n = static_cast<std::size_t>(mb) * 1024 * 1024 / 8;
    std::vector<double> a(n), b(n), c(n);
    const double s = 3.0;

// First-touch initialization with the same schedule as the kernels.
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        const auto ii = static_cast<std::size_t>(i);
        a[ii] = 1.0;
        b[ii] = 2.0;
        c[ii] = 0.5;
    }

    struct Kernel {
        const char* name;
        double bytes_per_elem;
    };
    const Kernel kernels[] = {{"copy", 16.0}, {"scale", 16.0}, {"add", 24.0}, {"triad", 24.0}};
    double triad = 0.0;

    std::printf("== stream: 3 x %ld MiB, best of %d ==\n", mb, reps);
    for (const auto& k : kernels) {
        auto body = [&] {
#pragma omp parallel for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
                const auto ii = static_cast<std::size_t>(i);
                if (k.name[0] == 'c') {
                    c[ii] = a[ii];
                } else if (k.name[0] == 's') {
                    b[ii] = s * c[ii];
                } else if (k.name[1] == 'd') {
                    c[ii] = a[ii] + b[ii];
                } else {
                    a[ii] = b[ii] + s * c[ii];
                }
            }
        };
        const double t = best_time(2, reps, body);
        const double gbps = k.bytes_per_elem * static_cast<double>(n) / t / 1e9;
        std::printf("  %-6s %8.2f GB/s\n", k.name, gbps);
        if (k.name[0] == 't') {
            triad = gbps;
        }
        g_sink += a[n / 2] + b[n / 3] + c[n / 4];
    }
    return triad;
}

std::vector<double> random_vec(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> v(n);
    for (auto& x : v) {
        x = u(rng);
    }
    return v;
}

void run_spmv(spume::index_t nx, spume::index_t ny, spume::index_t nz, int reps, double peak) {
    const auto csr = spume::coo_to_csr(spume::gen::poisson7(nx, ny, nz));
    const auto a64 = spume::sell_from_csr(csr);
    const auto op32 = spume::make_eq_operator<float>(csr);
    const auto n = static_cast<std::size_t>(a64.nrows);

    std::printf("== spmv: poisson7 %dx%dx%d, n=%zu, nnz=%lld, padding %.3f, best of %d ==\n", nx,
                ny, nz, n, static_cast<long long>(a64.nnz), a64.padding_ratio(), reps);

    const auto x64 = random_vec(n, 1);
    std::vector<double> y64(n);
    const double t64 = best_time(3, reps, [&] {
        spume::spmv(a64, std::span<const double>(x64), std::span<double>(y64),
                    spume::Dispatch::openmp);
    });
    g_sink += y64[n / 2];

    std::vector<float> x32(n), y32(n);
    for (std::size_t i = 0; i < n; ++i) {
        x32[i] = static_cast<float>(x64[i]);
    }
    const double t32 = best_time(3, reps, [&] {
        spume::spmv(op32.a, std::span<const float>(x32), std::span<float>(y32),
                    spume::Dispatch::openmp);
    });
    g_sink += y32[n / 2];

    const double gb64 = a64.spmv_bytes() / t64 / 1e9;
    const double gb32 = op32.a.spmv_bytes() / t32 / 1e9;
    std::printf("  fp64        %8.2f GB/s  (%5.1f%% of triad)  %.3f ms\n", gb64,
                peak > 0 ? 100.0 * gb64 / peak : 0.0, t64 * 1e3);
    std::printf("  fp32 (eq)   %8.2f GB/s  (%5.1f%% of triad)  %.3f ms\n", gb32,
                peak > 0 ? 100.0 * gb32 / peak : 0.0, t32 * 1e3);
    std::printf("  model bytes fp64 %.1f MB, fp32 %.1f MB (ratio %.3f); time ratio %.3f\n",
                a64.spmv_bytes() / 1e6, op32.a.spmv_bytes() / 1e6,
                op32.a.spmv_bytes() / a64.spmv_bytes(), t32 / t64);
}

// Per-iteration traffic models (FP64 vectors are 8n bytes; every read or
// write of a full vector counts once — same optimistic-streaming convention
// as the SpMV model in sell.hpp):
//   CG:  spmv + dot(p,q) 2v + axpy(x) 3v + axpy(r) 3v + dot(r,r) 1v
//        + aypx(p) 3v                                   = spmv + 12 v
//   FCG: spmv + dots (p,q)+(r,r)+(r,z)+(z,q) 7v + axpy x,r 6v + aypx 3v
//        = spmv + 16 v  + preconditioner apply
//   Chebyshev apply at precision T (m = steps, w = sizeof(T) n bytes):
//        scale-in 8n + w; init 3w; m corrections 3w each;
//        (m-1) inner spmv(T) + (m-1)*(3w + 3w); scale-out w + 8n
double cheb_apply_bytes(double spmv_bytes_t, double n, double sizeof_t, int steps) {
    const double w = sizeof_t * n;
    return (steps - 1) * spmv_bytes_t + 16.0 * n + w * (4.0 + 3.0 * steps + 6.0 * (steps - 1));
}

void solve_report(const char* name, const spume::SolveResult& r, double seconds,
                  double bytes_per_iter, double cg_seconds, const spume::Sell<double>& a,
                  std::span<const double> b, std::span<const double> x) {
    // Honest verification: recompute the true residual outside the timer.
    std::vector<double> res(b.size());
    spume::spmv(a, x, std::span<double>(res), spume::Dispatch::openmp);
    for (std::size_t i = 0; i < res.size(); ++i) {
        res[i] = b[i] - res[i];
    }
    const double relres =
        spume::nrm2(res, spume::Dispatch::openmp) / spume::nrm2(b, spume::Dispatch::openmp);
    const double gb = bytes_per_iter * r.iterations / 1e9;
    std::printf("  %-10s %5d it  %8.3f s  %8.2f GB (model)  %6.2f GB/s  x%.2f  relres %.1e\n", name,
                r.iterations, seconds, gb, gb / seconds, cg_seconds / seconds, relres);
}

void run_solve(spume::index_t nx, spume::index_t ny, spume::index_t nz, double tol) {
    const auto csr = spume::coo_to_csr(spume::gen::poisson7(nx, ny, nz));
    const auto a = spume::sell_from_csr(csr);
    const auto n = static_cast<std::size_t>(a.nrows);
    const auto b = random_vec(n, 7);
    const double vn = 8.0 * static_cast<double>(n);

    spume::SolveOptions opt;
    opt.tol = tol;
    opt.max_iter = 20000;
    opt.dispatch = spume::Dispatch::openmp;

    std::printf("== solve: poisson7 %dx%dx%d, n=%zu, tol=%.0e ==\n", nx, ny, nz, n, tol);
    std::printf("  (traffic = documented per-iteration model x iterations)\n");

    // Plain FP64 CG (reference solver).
    std::vector<double> x_cg(n, 0.0);
    double t0 = omp_get_wtime();
    const auto r_cg = spume::cg(a, b, x_cg, opt);
    const double t_cg = omp_get_wtime() - t0;
    const double cg_iter_bytes = a.spmv_bytes() + 12.0 * vn;
    solve_report("cg64", r_cg, t_cg, cg_iter_bytes, t_cg, a, b, x_cg);

    const spume::ChebyshevOptions copt{5, 30.0};
    const int m = copt.steps;

    // FCG + FP64 Chebyshev (same algorithm, full-precision preconditioner).
    {
        const auto op64 = spume::make_eq_operator<double>(csr);
        const double bytes = a.spmv_bytes() + 16.0 * vn +
                             cheb_apply_bytes(op64.a.spmv_bytes(), static_cast<double>(n), 8.0, m);
        spume::ChebyshevPrecond<double> pc(op64, copt, opt.dispatch);
        std::vector<double> x(n, 0.0);
        t0 = omp_get_wtime();
        const auto r = spume::fcg(a, pc, b, x, opt);
        solve_report("fcg-cheb64", r, omp_get_wtime() - t0, bytes, t_cg, a, b, x);
    }

    // FCG + FP32 Chebyshev (the Milestone 0 claim).
    {
        const auto op32 = spume::make_eq_operator<float>(csr);
        const double bytes = a.spmv_bytes() + 16.0 * vn +
                             cheb_apply_bytes(op32.a.spmv_bytes(), static_cast<double>(n), 4.0, m);
        spume::ChebyshevPrecond<float> pc(op32, copt, opt.dispatch);
        std::vector<double> x(n, 0.0);
        t0 = omp_get_wtime();
        const auto r = spume::fcg(a, pc, b, x, opt);
        solve_report("fcg-cheb32", r, omp_get_wtime() - t0, bytes, t_cg, a, b, x);
    }
}

} // namespace

int main(int argc, char** argv) {
    const Args args = parse(argc, argv);
    std::printf("spume-bench: %d OpenMP threads\n", omp_get_max_threads());

    double peak = 0.0;
    if (args.mode == "stream" || args.mode == "all") {
        peak = run_stream(args.mb, args.reps);
    }
    if (args.mode == "spmv" || args.mode == "all") {
        const spume::index_t d = args.nx > 0 ? args.nx : 128;
        run_spmv(d, args.ny > 0 ? args.ny : d, args.nz > 0 ? args.nz : d, args.reps, peak);
    }
    if (args.mode == "solve" || args.mode == "all") {
        const spume::index_t d = args.nx > 0 ? args.nx : 96;
        run_solve(d, args.ny > 0 ? args.ny : d, args.nz > 0 ? args.nz : d, args.tol);
    }

    if (g_sink == 12345.6789) { // never true; forces materialization
        std::printf("%f\n", g_sink);
    }
    return 0;
}
