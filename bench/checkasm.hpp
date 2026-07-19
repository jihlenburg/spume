// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

// checkasm-style verify-then-bench harness for SPUME kernel variants, modelled
// on FFmpeg's checkasm (ADR-0016). Every optimized dispatch path — OpenMP,
// hand AVX-512, an NT-store SpMV — must first PROVE it agrees with the portable
// reference on randomized inputs, and only then is it timed. A variant that
// fails verification is never benchmarked: a fast wrong kernel is worthless
// (invariant #4, ADR-0004 — the reference is always the source of truth).
//
// Agreement is checked in the correctness class the numerics policy defines
// (AGENTS.md): a same-precision reordering variant (e.g. OpenMP SELL with
// fixed per-chunk accumulation order) must match BITWISE; a reduced-precision
// variant (FP32 preconditioner kernel) must match within an explicit
// absolute+relative tolerance (atol + rtol*|ref|, elementwise) the caller
// states. The harness never invents a tolerance — the caller declares the
// class, matching how the regression comparator works.
//
// Usage:
//   checkasm::Case c{"spmv", n};                 // label + element count
//   c.reference([&]{ spmv_ref(a, x, y_ref); });  // fills y_ref
//   c.variant("openmp", [&]{ spmv(a, x, y, omp); },  // fills y
//             checkasm::exact(y_ref, y));            // bitwise class
//   c.variant("nt-store", [&]{ spmv_nt(a, x, y); },
//             checkasm::within(y_ref, y, 1e-6, 1e-5)); // atol/rtol reduced class
//   c.report();
//
// Verification runs first (once); timing follows only for variants that pass.
// Timing uses median-of-reps with a CV flag, consistent with bench/main.cpp
// and the performance policy (warm-up excluded, I/O outside the timed region).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include <time.h>

namespace spume::checkasm {

// A verifier: called after the variant has filled its output, returns true iff
// the variant's result is inside the declared correctness class vs the
// reference. Returns a human-readable reason on failure via the out-param.
using Verifier = std::function<bool(std::string& reason)>;

// Bitwise-exact class: same-precision reordering variants must reproduce the
// reference to the last bit (the deterministic-reduction contract, and the
// only honest bar for a variant that claims to be "the same math, threaded").
inline Verifier exact(std::span<const double> ref, std::span<const double> got) {
    return [ref, got](std::string& reason) {
        if (ref.size() != got.size()) {
            reason = "size mismatch";
            return false;
        }
        for (std::size_t i = 0; i < ref.size(); ++i) {
            // Bit-compare so a signed zero or NaN mismatch is not masked.
            std::uint64_t a, b;
            std::memcpy(&a, &ref[i], sizeof(a));
            std::memcpy(&b, &got[i], sizeof(b));
            if (a != b) {
                char buf[128];
                std::snprintf(buf, sizeof(buf),
                              "bitwise diff at [%zu]: ref=%.17g got=%.17g", i,
                              ref[i], got[i]);
                reason = buf;
                return false;
            }
        }
        return true;
    };
}

// Reduced-precision class: |got-ref| <= atol + rtol*|ref| elementwise. For a
// preconditioner-internal FP32 kernel under a flexible outer Krylov the result
// need only be *close* (it accelerates, not defines), so the caller states how
// close is legitimate (both tolerances) — never the harness.
inline Verifier within(std::span<const double> ref, std::span<const double> got,
                       double atol, double rtol) {
    return [ref, got, atol, rtol](std::string& reason) {
        if (ref.size() != got.size()) {
            reason = "size mismatch";
            return false;
        }
        double worst = 0.0;
        std::size_t worst_i = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const double tol = atol + rtol * std::fabs(ref[i]);
            const double d = std::fabs(got[i] - ref[i]);
            if (d > tol && d - tol > worst) {
                worst = d - tol;
                worst_i = i;
            }
        }
        if (worst > 0.0) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "out of class at [%zu]: ref=%.17g got=%.17g "
                          "(exceeds atol=%.3g rtol=%.3g by %.3g)",
                          worst_i, ref[worst_i], got[worst_i], atol, rtol,
                          worst);
            reason = buf;
            return false;
        }
        return true;
    };
}

// One benchmarked variant: label, work thunk, verifier, and the timing result.
struct Variant {
    std::string name;
    std::function<void()> run;
    Verifier verify;
    bool verified = false;
    std::string fail_reason = {};
    double median_ns = 0.0;
    double cv_pct = 0.0;
};

inline double now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) * 1e9 +
           static_cast<double>(ts.tv_nsec);
}

// A test case: a reference kernel plus N candidate variants over the same
// input, each verified then (if it passes) timed. `elems` is the element
// count used only to print a per-element figure.
class Case {
public:
    Case(std::string label, std::size_t elems, int reps = 50, int warmup = 5)
        : label_(std::move(label)), elems_(elems), reps_(reps),
          warmup_(warmup) {}

    // Register the reference kernel. Run once, up front, before any variant is
    // verified — the variants compare against whatever state it produces.
    void reference(std::function<void()> run) { ref_ = std::move(run); }

    void variant(std::string name, std::function<void()> run, Verifier v) {
        variants_.push_back(
            Variant{std::move(name), std::move(run), std::move(v)});
    }

    // Verify every variant, then time the ones that passed. Returns true iff
    // ALL variants verified — a false return is a correctness failure the
    // caller must surface (and never a benchmark to publish).
    bool report() {
        if (ref_) {
            ref_(); // establish the reference output
        }
        bool all_ok = true;
        std::printf("checkasm[%s] n=%zu reps=%d\n", label_.c_str(), elems_,
                    reps_);
        for (Variant& v : variants_) {
            // 1) verify: run once, check the class.
            v.run();
            v.verified = v.verify(v.fail_reason);
            if (!v.verified) {
                all_ok = false;
                std::printf("  %-14s FAIL  %s\n", v.name.c_str(),
                            v.fail_reason.c_str());
                continue; // never benchmark a wrong kernel
            }
            // 2) time: warm-up excluded, median of reps, CV flagged.
            for (int i = 0; i < warmup_; ++i) {
                v.run();
            }
            std::vector<double> t;
            t.reserve(static_cast<std::size_t>(reps_));
            for (int i = 0; i < reps_; ++i) {
                const double a = now_ns();
                v.run();
                t.push_back(now_ns() - a);
            }
            std::sort(t.begin(), t.end());
            v.median_ns = t[t.size() / 2];
            double mean = 0.0;
            for (double x : t) {
                mean += x;
            }
            mean /= static_cast<double>(t.size());
            double ss = 0.0;
            for (double x : t) {
                ss += (x - mean) * (x - mean);
            }
            const double sd =
                t.size() > 1 ? std::sqrt(ss / static_cast<double>(t.size() - 1))
                             : 0.0;
            v.cv_pct = mean > 0 ? 100.0 * sd / mean : 0.0;
            const double per_elem =
                elems_ ? v.median_ns / static_cast<double>(elems_) : 0.0;
            std::printf("  %-14s ok    %10.1f ns  %6.3f ns/elem  CV %.2f%%%s\n",
                        v.name.c_str(), v.median_ns, per_elem, v.cv_pct,
                        v.cv_pct > 5.0 ? "  [UNSTABLE]" : "");
        }
        // Speedup summary vs the first passing variant (usually "reference").
        if (variants_.size() > 1 && variants_[0].verified &&
            variants_[0].median_ns > 0) {
            for (std::size_t i = 1; i < variants_.size(); ++i) {
                if (variants_[i].verified) {
                    const double sp =
                        variants_[0].median_ns / variants_[i].median_ns;
                    std::printf("  %-14s %.3fx vs %s\n", variants_[i].name.c_str(),
                                sp, variants_[0].name.c_str());
                }
            }
        }
        return all_ok;
    }

private:
    std::string label_;
    std::size_t elems_;
    int reps_;
    int warmup_;
    std::function<void()> ref_;
    std::vector<Variant> variants_;
};

} // namespace spume::checkasm
