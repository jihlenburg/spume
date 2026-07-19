// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

// SPUME Milestone 0 bench harness.
//
//   spume-bench stream [--mb M] [--reps R]      STREAM-style roofline probe
//   spume-bench spmv   [--nx N --ny N --nz N]   SpMV GB/s vs bytes model
//   spume-bench solve  [--nx N --ny N --nz N]   solver traffic comparison
//   spume-bench checkasm [--nx N ...]           verify+bench dispatch variants
//                                               (+ hardware counters if allowed)
//   spume-bench all                             all of the above, in order
//
// Methodology (per AGENTS.md performance policy and ADR-0013):
//  - warm-up runs before every timed region; no I/O inside timed regions,
//  - all "bytes moved" figures are the documented traffic MODEL from
//    sell.hpp / the per-iteration counts below; the roofline reference is
//    the measured STREAM triad (spmv/solve modes self-calibrate with a
//    short stream run when invoked standalone),
//  - on non-lab machines the harness quantifies its own noise instead of
//    disclaiming it (ADR-0013): steal-time bracketing per section,
//    median + CV across reps (flagged when CV > 5%), an LLC working-set
//    guard, interleaved A/B reps for precision comparisons, and a
//    triad-calibrated prediction error as the counter substitute.
//
// STREAM convention: copy/scale count 2 accesses per element, add/triad 3;
// write-allocate traffic is not counted (standard STREAM counting).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "core/equilibrate.hpp"
#include "core/poisson.hpp"
#include "core/precond.hpp"
#include "core/reduce.hpp"
#include "core/solver.hpp"
#include "core/spmv.hpp"

#include "checkasm.hpp"
#include "perf_counters.hpp"

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

// ---------------------------------------------------------------------------
// Noise instrumentation (ADR-0013)
// ---------------------------------------------------------------------------

struct Timing {
    double best = 0.0;
    double median = 0.0;
    double cv_pct = 0.0; // stddev/mean of the rep times, percent
};

Timing summarize(std::vector<double> t) {
    Timing r;
    if (t.empty()) {
        return r;
    }
    std::sort(t.begin(), t.end());
    r.best = t.front();
    r.median = t[t.size() / 2];
    double mean = 0.0;
    for (double v : t) {
        mean += v;
    }
    mean /= static_cast<double>(t.size());
    double var = 0.0;
    for (double v : t) {
        var += (v - mean) * (v - mean);
    }
    var /= static_cast<double>(t.size());
    r.cv_pct = mean > 0.0 ? 100.0 * std::sqrt(var) / mean : 0.0;
    return r;
}

const char* cv_flag(const Timing& t) {
    return t.cv_pct > 5.0 ? "  UNSTABLE (cv > 5%)" : "";
}

template<typename F>
Timing time_reps(int warm, int reps, F&& fn) {
    for (int i = 0; i < warm; ++i) {
        fn();
    }
    std::vector<double> t;
    t.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        const double t0 = omp_get_wtime();
        fn();
        t.push_back(omp_get_wtime() - t0);
    }
    return summarize(std::move(t));
}

// Alternate single reps of two kernels so slow drift (thermal, neighbors)
// hits both equally: the A/B ratio is drift-immune even when absolute GB/s
// wobbles. Primary evidence for precision comparisons on noisy machines.
template<typename FA, typename FB>
std::pair<Timing, Timing> time_interleaved(int warm, int reps, FA&& fa, FB&& fb) {
    for (int i = 0; i < warm; ++i) {
        fa();
        fb();
    }
    std::vector<double> ta, tb;
    ta.reserve(static_cast<std::size_t>(reps));
    tb.reserve(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        double t0 = omp_get_wtime();
        fa();
        ta.push_back(omp_get_wtime() - t0);
        t0 = omp_get_wtime();
        fb();
        tb.push_back(omp_get_wtime() - t0);
    }
    return {summarize(std::move(ta)), summarize(std::move(tb))};
}

// Aggregate CPU times from the first line of /proc/stat (USER_HZ units).
// steal (8th field) counts time the hypervisor ran someone else while this
// guest was runnable — direct evidence of neighbor interference.
struct CpuStat {
    double total = 0.0;
    double steal = 0.0;
    bool ok = false;
};

CpuStat read_cpu_stat() {
    std::ifstream f("/proc/stat");
    std::string tag;
    CpuStat s;
    if (!f || !(f >> tag) || tag != "cpu") {
        return s;
    }
    double v = 0.0;
    int idx = 0;
    while (f >> v) {
        s.total += v;
        if (++idx == 8) {
            s.steal = v;
        }
    }
    s.ok = idx >= 8;
    return s;
}

// Brackets a timed section; pct() reports the steal share of all CPU time
// that elapsed since construction (-1 when /proc/stat is unavailable).
class StealGauge {
public:
    double pct() const {
        const CpuStat t1 = read_cpu_stat();
        if (!t0_.ok || !t1.ok || t1.total <= t0_.total) {
            return -1.0;
        }
        return 100.0 * (t1.steal - t0_.steal) / (t1.total - t0_.total);
    }

private:
    CpuStat t0_ = read_cpu_stat();
};

void print_steal(const StealGauge& g) {
    const double p = g.pct();
    if (p < 0.0) {
        std::printf("  steal during section: n/a\n");
    } else {
        std::printf("  steal during section: %.2f%%%s\n", p,
                    p > 1.0 ? "  CONTENDED (steal > 1%)" : "");
    }
}

// Largest data/unified cache reported by sysfs, in bytes (0 = unknown).
double llc_bytes() {
    double best = 0.0;
    for (int i = 0; i < 10; ++i) {
        const std::string base =
            "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(i) + "/";
        std::ifstream ts(base + "type"), ss(base + "size");
        std::string type, size;
        if (!(ts >> type) || !(ss >> size) || size.empty()) {
            continue;
        }
        if (type != "Unified" && type != "Data") {
            continue;
        }
        double mult = 1.0;
        const char suf = size.back();
        if (suf == 'K') {
            mult = 1024.0;
        } else if (suf == 'M') {
            mult = 1024.0 * 1024.0;
        } else if (suf == 'G') {
            mult = 1024.0 * 1024.0 * 1024.0;
        }
        if (mult > 1.0) {
            size.pop_back();
        }
        try {
            best = std::max(best, std::stod(size) * mult);
        } catch (...) { // unparseable sysfs entry: ignore
        }
    }
    return best;
}

// A streamed working set that fits in cache produces GB/s numbers that
// flatter the kernel; warn below 2x LLC so it cannot happen silently.
void print_llc_guard(const char* what, double ws_bytes) {
    const double llc = llc_bytes();
    if (llc <= 0.0) {
        std::printf("  [%s working set %.0f MB; LLC unknown]\n", what, ws_bytes / 1e6);
        return;
    }
    const double ratio = ws_bytes / llc;
    std::printf("  [%s working set %.0f MB = %.1fx LLC (%.0f MiB)%s]\n", what, ws_bytes / 1e6,
                ratio, llc / (1024.0 * 1024.0),
                ratio < 2.0 ? "  WARNING: may be cache-resident" : "");
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

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

    std::printf("== stream: 3 x %ld MiB, %d reps ==\n", mb, reps);
    print_llc_guard("stream", 3.0 * 8.0 * static_cast<double>(n));
    const StealGauge gauge;
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
        const Timing t = time_reps(2, reps, body);
        const double gbps = k.bytes_per_elem * static_cast<double>(n) / t.best / 1e9;
        std::printf("  %-6s %8.2f GB/s  (cv %.1f%%)%s\n", k.name, gbps, t.cv_pct, cv_flag(t));
        if (k.name[0] == 't') {
            triad = gbps;
        }
        g_sink += a[n / 2] + b[n / 3] + c[n / 4];
    }
    print_steal(gauge);
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

    std::printf("== spmv: poisson7 %dx%dx%d, n=%zu, nnz=%lld, padding %.3f, %d interleaved "
                "reps ==\n",
                nx, ny, nz, n, static_cast<long long>(a64.nnz), a64.padding_ratio(), reps);
    print_llc_guard("fp64", a64.spmv_bytes());
    print_llc_guard("fp32", op32.a.spmv_bytes());

    const auto x64 = random_vec(n, 1);
    std::vector<double> y64(n);
    std::vector<float> x32(n), y32(n);
    for (std::size_t i = 0; i < n; ++i) {
        x32[i] = static_cast<float>(x64[i]);
    }

    const StealGauge gauge;
    const auto [t64, t32] = time_interleaved(
        2, reps,
        [&] {
            spume::spmv(a64, std::span<const double>(x64), std::span<double>(y64),
                        spume::Dispatch::openmp);
        },
        [&] {
            spume::spmv(op32.a, std::span<const float>(x32), std::span<float>(y32),
                        spume::Dispatch::openmp);
        });
    g_sink += y64[n / 2] + y32[n / 2];

    const double gb64 = a64.spmv_bytes() / t64.best / 1e9;
    const double gb32 = op32.a.spmv_bytes() / t32.best / 1e9;
    std::printf("  fp64        %8.2f GB/s  (%5.1f%% of triad, cv %.1f%%)  %.3f ms%s\n", gb64,
                peak > 0 ? 100.0 * gb64 / peak : 0.0, t64.cv_pct, t64.median * 1e3, cv_flag(t64));
    std::printf("  fp32 (eq)   %8.2f GB/s  (%5.1f%% of triad, cv %.1f%%)  %.3f ms%s\n", gb32,
                peak > 0 ? 100.0 * gb32 / peak : 0.0, t32.cv_pct, t32.median * 1e3, cv_flag(t32));
    std::printf("  model bytes fp64 %.1f MB, fp32 %.1f MB (ratio %.3f); median time ratio %.3f\n",
                a64.spmv_bytes() / 1e6, op32.a.spmv_bytes() / 1e6,
                op32.a.spmv_bytes() / a64.spmv_bytes(), t32.median / t64.median);
    print_steal(gauge);
}

// checkasm mode: verify each SpMV dispatch variant against the portable
// reference, then time it — the FFmpeg/checkasm discipline (ADR-0016). The
// OpenMP SELL kernel accumulates each chunk in the SAME fixed lane order as
// the reference, so it must match BITWISE — this is the machine check for the
// invariant spumePCG relies on ("openmp is bit-identical to reference, pure
// parallelisation"). Where the kernel denies hardware counters this prints the
// honest "unavailable" line and leaves the timing model to stand.
void run_checkasm(spume::index_t nx, spume::index_t ny, spume::index_t nz, int reps) {
    const auto csr = spume::coo_to_csr(spume::gen::poisson7(nx, ny, nz));
    const auto a = spume::sell_from_csr(csr);
    const auto n = static_cast<std::size_t>(a.nrows);
    const auto x = random_vec(n, 11);
    std::vector<double> y_ref(n), y_omp(n), y_refv(n);

    std::printf("== checkasm: poisson7 %dx%dx%d, n=%zu, nnz=%lld ==\n", nx, ny, nz, n,
                static_cast<long long>(a.nnz));

    // Randomized multi-seed verification (FFmpeg/checkasm discipline, ADR-0016):
    // a reordering bug can be sensitive to particular sign/cancellation patterns,
    // so agreement is proven across several inputs before the single-input timing
    // below — one draw could miss it.
    bool multiseed_ok = true;
    for (unsigned seed : {11u, 101u, 1009u, 7919u, 50021u}) {
        const auto xs = random_vec(n, seed);
        spume::spmv(a, std::span<const double>(xs), std::span<double>(y_ref),
                    spume::Dispatch::reference);
        spume::spmv(a, std::span<const double>(xs), std::span<double>(y_omp),
                    spume::Dispatch::openmp);
        std::string reason;
        if (!spume::checkasm::exact(y_ref, y_omp)(reason)) {
            multiseed_ok = false;
            std::printf("  multi-seed verify FAIL (seed %u): %s\n", seed, reason.c_str());
            break;
        }
    }
    if (multiseed_ok) {
        std::printf("  multi-seed verify: openmp bitwise-identical to reference over 5 seeds\n");
    }
    g_sink += y_omp[n / 2];

    // Time the reference too (as the first variant), so the speedup below is a
    // real reference-vs-openmp ratio, not just openmp in isolation.
    spume::checkasm::Case c{"spmv", n, reps, 5};
    c.reference([&] {
        spume::spmv(a, std::span<const double>(x), std::span<double>(y_ref),
                    spume::Dispatch::reference);
    });
    c.variant(
        "reference",
        [&] {
            spume::spmv(a, std::span<const double>(x), std::span<double>(y_refv),
                        spume::Dispatch::reference);
        },
        spume::checkasm::exact(y_ref, y_refv));
    // Same-precision, fixed-order reordering -> must match bitwise.
    c.variant(
        "openmp",
        [&] {
            spume::spmv(a, std::span<const double>(x), std::span<double>(y_omp),
                        spume::Dispatch::openmp);
        },
        spume::checkasm::exact(y_ref, y_omp));
    const bool ok = c.report() && multiseed_ok;
    g_sink += y_ref[n / 2] + y_omp[n / 2] + y_refv[n / 2];

    // Optional hardware-counter view of the OpenMP kernel (degrades honestly).
    std::vector<spume::bench::CounterSpec> specs = {
        spume::bench::cpu_cycles(), spume::bench::instructions(),
        spume::bench::llc_read_misses(), spume::bench::dtlb_load_misses()};
    spume::bench::CounterGroup g(specs);
    if (g.available()) {
        g.start();
        for (int i = 0; i < reps; ++i) {
            spume::spmv(a, std::span<const double>(x), std::span<double>(y_omp),
                        spume::Dispatch::openmp);
        }
        g.stop();
        g_sink += y_omp[0];
        std::printf("  hardware counters (openmp spmv x%d, exclude_kernel):\n", reps);
        std::uint64_t cyc = 0, llc = 0;
        for (const auto& r : g.read()) {
            std::printf("    %-18s %14llu%s\n", r.name.c_str(),
                        static_cast<unsigned long long>(r.value), r.valid ? "" : "  (n/a)");
            if (r.valid && r.name == "cpu_cycles") cyc = r.value;
            if (r.valid && r.name == "llc_read_misses") llc = r.value;
        }
        if (llc > 0) {
            // 64 B per LLC miss is one DRAM cache line on this platform; the
            // model says spmv moves a.spmv_bytes() per pass, so a miss count
            // far below bytes/64 means the working set was cache-resident.
            const double measured_gb = 64.0 * static_cast<double>(llc) / 1e9;
            const double model_gb = a.spmv_bytes() * reps / 1e9;
            std::printf("    -> DRAM (64B*LLC-miss) %.2f GB vs model %.2f GB (%.0f%% of model traffic)\n",
                        measured_gb, model_gb, 100.0 * measured_gb / model_gb);
        }
        if (cyc > 0) {
            std::printf("    -> %.2f cycles/nonzero over %d passes\n",
                        static_cast<double>(cyc) /
                            (static_cast<double>(a.nnz) * reps),
                        reps);
        }
    } else {
        std::printf("  hardware counters unavailable (%s) — timing model stands (ADR-0013)\n",
                    g.why().c_str());
    }
    if (!ok) {
        std::printf("  checkasm FAILED — a dispatch variant left the correctness class\n");
    }
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
                  double bytes_per_iter, double peak, const StealGauge& gauge,
                  const spume::Sell<double>& a, std::span<const double> b,
                  std::span<const double> x) {
    // Honest verification: recompute the true residual outside the timer.
    std::vector<double> res(b.size());
    spume::spmv(a, x, std::span<double>(res), spume::Dispatch::openmp);
    for (std::size_t i = 0; i < res.size(); ++i) {
        res[i] = b[i] - res[i];
    }
    const double relres =
        spume::nrm2(res, spume::Dispatch::openmp) / spume::nrm2(b, spume::Dispatch::openmp);
    const double gb = bytes_per_iter * r.iterations / 1e9;

    // Triad-calibrated prediction: the counter substitute (ADR-0013). If the
    // model were exact and the solver ran at triad bandwidth, the solve
    // would take gb/peak seconds; the signed error says how far reality is
    // from that — cache residency shows up as negative error.
    char pred[64] = "pred n/a";
    if (peak > 0.0) {
        std::snprintf(pred, sizeof(pred), "pred %7.2f s (err %+5.1f%%)", gb / peak,
                      100.0 * (seconds - gb / peak) / (gb / peak));
    }
    const double steal = gauge.pct();
    char steal_s[32] = "steal n/a";
    if (steal >= 0.0) {
        std::snprintf(steal_s, sizeof(steal_s), "steal %.2f%%", steal);
    }
    std::printf("  %-10s %5d it  %8.3f s  %8.2f GB (model)  %6.2f GB/s  %s  %s  relres %.1e\n",
                name, r.iterations, seconds, gb, gb / seconds, pred, steal_s, relres);
}

void run_solve(spume::index_t nx, spume::index_t ny, spume::index_t nz, double tol, double peak) {
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
    std::printf("  (traffic = documented per-iteration model x iterations; single-shot runs, "
                "no cv)\n");
    print_llc_guard("fp64 spmv", a.spmv_bytes());

    // Plain FP64 CG (reference solver).
    std::vector<double> x_cg(n, 0.0);
    {
        const StealGauge gauge;
        const double t0 = omp_get_wtime();
        const auto r_cg = spume::cg(a, b, x_cg, opt);
        const double t_cg = omp_get_wtime() - t0;
        solve_report("cg64", r_cg, t_cg, a.spmv_bytes() + 12.0 * vn, peak, gauge, a, b, x_cg);
    }

    const spume::ChebyshevOptions copt{5, 30.0};
    const int m = copt.steps;

    // FCG + FP64 Chebyshev (same algorithm, full-precision preconditioner).
    {
        const auto op64 = spume::make_eq_operator<double>(csr);
        const double bytes = a.spmv_bytes() + 16.0 * vn +
                             cheb_apply_bytes(op64.a.spmv_bytes(), static_cast<double>(n), 8.0, m);
        spume::ChebyshevPrecond<double> pc(op64, copt, opt.dispatch);
        std::vector<double> x(n, 0.0);
        const StealGauge gauge;
        const double t0 = omp_get_wtime();
        const auto r = spume::fcg(a, pc, b, x, opt);
        solve_report("fcg-cheb64", r, omp_get_wtime() - t0, bytes, peak, gauge, a, b, x);
    }

    // FCG + FP32 Chebyshev (the Milestone 0 claim).
    {
        const auto op32 = spume::make_eq_operator<float>(csr);
        const double bytes = a.spmv_bytes() + 16.0 * vn +
                             cheb_apply_bytes(op32.a.spmv_bytes(), static_cast<double>(n), 4.0, m);
        spume::ChebyshevPrecond<float> pc(op32, copt, opt.dispatch);
        std::vector<double> x(n, 0.0);
        const StealGauge gauge;
        const double t0 = omp_get_wtime();
        const auto r = spume::fcg(a, pc, b, x, opt);
        solve_report("fcg-cheb32", r, omp_get_wtime() - t0, bytes, peak, gauge, a, b, x);
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
        if (peak == 0.0) {
            peak = run_stream(args.mb, 5); // self-calibrate the roofline reference
        }
        const spume::index_t d = args.nx > 0 ? args.nx : 128;
        run_spmv(d, args.ny > 0 ? args.ny : d, args.nz > 0 ? args.nz : d, args.reps, peak);
    }
    if (args.mode == "solve" || args.mode == "all") {
        if (peak == 0.0) {
            peak = run_stream(args.mb, 5);
        }
        const spume::index_t d = args.nx > 0 ? args.nx : 96;
        run_solve(d, args.ny > 0 ? args.ny : d, args.nz > 0 ? args.nz : d, args.tol, peak);
    }
    if (args.mode == "checkasm" || args.mode == "all") {
        const spume::index_t d = args.nx > 0 ? args.nx : 128;
        run_checkasm(d, args.ny > 0 ? args.ny : d, args.nz > 0 ? args.nz : d, args.reps);
    }

    if (g_sink == 12345.6789) { // never true; forces materialization
        std::printf("%f\n", g_sink);
    }
    return 0;
}
