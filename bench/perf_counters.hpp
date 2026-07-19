// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

// Hardware performance-counter sampling via perf_event_open(2) — the
// counter-evidence substrate the performance policy (AGENTS.md, ADR-0013)
// requires: achieved GB/s, DRAM fills, dTLB misses and LLC behaviour per
// timed region, not a model prediction.
//
// This is a DIAGNOSTIC utility, never a solver path (ADR-0004): it only
// observes. It degrades honestly — when the kernel denies counter access
// (perf_event_paranoid > 2, no CAP_PERFMON) the group reports `available()
// == false` and the caller falls back to the triad-calibrated timing model
// already in bench/main.cpp. Nothing here fabricates a number it could not
// measure.
//
// Portability: the named events use the architectural PERF_TYPE_HARDWARE /
// PERF_TYPE_HW_CACHE encodings, which the kernel maps to whatever the PMU
// provides (so `dram_fills` on this Strix Halo resolves through the generic
// LLC-miss proxy). Vendor-specific raw events — e.g. AMD Zen 5
// ls_any_fills_from_sys.dram_io_all — are passed through raw(type, config)
// so the exact DRAM-fill counter can be named at the call site without
// baking a fragile raw code into a shared header.

#include <cstdint>
#include <string>
#include <vector>

#if defined(__linux__)
#include <asm/unistd.h>
#include <cerrno>
#include <cstring>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace spume::bench {

// One counter request: a symbolic label plus the perf_event (type, config).
struct CounterSpec {
    std::string name;
    std::uint32_t type;
    std::uint64_t config;
};

// A named result: the label and the raw count over the measured region.
struct CounterResult {
    std::string name;
    std::uint64_t value = 0;
    bool valid = false; // false if this event could not be scheduled/read
};

// Portable event constructors — the architectural encodings the kernel maps
// onto the local PMU. Prefer these; reach for raw() only for a vendor event
// with no architectural equivalent.
inline CounterSpec cpu_cycles() {
#if defined(__linux__)
    return {"cpu_cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES};
#else
    return {"cpu_cycles", 0, 0};
#endif
}

inline CounterSpec instructions() {
#if defined(__linux__)
    return {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS};
#else
    return {"instructions", 0, 0};
#endif
}

// LLC (last-level cache) read misses — the generic DRAM-fill proxy when a
// vendor-specific fill counter is not named explicitly. Each miss is one
// cache line (64 B) fetched from memory on this platform.
inline CounterSpec llc_read_misses() {
#if defined(__linux__)
    return {"llc_read_misses", PERF_TYPE_HW_CACHE,
            (PERF_COUNT_HW_CACHE_LL) |
                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)};
#else
    return {"llc_read_misses", 0, 0};
#endif
}

// dTLB load misses — the latency-bound signal for scattered access
// (see the scattered-access-is-latency-bound finding). A spike here at
// unchanged byte traffic is the fingerprint of a poor access pattern.
inline CounterSpec dtlb_load_misses() {
#if defined(__linux__)
    return {"dtlb_load_misses", PERF_TYPE_HW_CACHE,
            (PERF_COUNT_HW_CACHE_DTLB) |
                (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)};
#else
    return {"dtlb_load_misses", 0, 0};
#endif
}

// Vendor-specific raw event — e.g. AMD Zen 5 ls_any_fills_from_sys with the
// dram_io_all unit mask, so the true DRAM-fill count (not the LLC-miss proxy)
// can be sampled without hardcoding the code into this shared header.
inline CounterSpec raw(const std::string& name, std::uint64_t config) {
#if defined(__linux__)
    return {name, PERF_TYPE_RAW, config};
#else
    return {name, 0, config};
#endif
}

// A group of counters measured over one region. Open once, then wrap the
// timed region in start()/stop() and read(). All events share the leader's
// scheduling so their counts are mutually consistent (no multiplexing skew
// within the group's hardware-slot budget).
//
// Lifetime: RAII — the fds close on destruction. Non-copyable; move is not
// needed for the bench's single-region use.
class CounterGroup {
public:
    explicit CounterGroup(const std::vector<CounterSpec>& specs) {
        specs_.reserve(specs.size());
        for (const auto& s : specs) {
            specs_.push_back(s);
        }
#if defined(__linux__)
        open_all();
#endif
    }

    CounterGroup(const CounterGroup&) = delete;
    CounterGroup& operator=(const CounterGroup&) = delete;

    ~CounterGroup() {
#if defined(__linux__)
        for (int fd : fds_) {
            if (fd >= 0) {
                ::close(fd);
            }
        }
#endif
    }

    // True if at least the group leader opened — i.e. counters are usable.
    // False means the kernel denied access; the caller must fall back to the
    // timing model rather than report a zeroed count as if it were measured.
    bool available() const { return available_; }

    // The reason the group is unavailable (errno string), for a one-line
    // diagnostic the caller can print instead of silently degrading.
    const std::string& why() const { return why_; }

    void start() {
#if defined(__linux__)
        if (!available_) {
            return;
        }
        ::ioctl(fds_[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ::ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
#endif
    }

    void stop() {
#if defined(__linux__)
        if (!available_) {
            return;
        }
        ::ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
#endif
    }

    // Read the counts accumulated between start() and stop(). One result per
    // requested spec, in order; results[i].valid is false for any event the
    // kernel could not schedule (leaving the timing model to cover it).
    std::vector<CounterResult> read() const {
        std::vector<CounterResult> out;
        out.reserve(specs_.size());
#if defined(__linux__)
        for (std::size_t i = 0; i < specs_.size(); ++i) {
            CounterResult r;
            r.name = specs_[i].name;
            if (i < fds_.size() && fds_[i] >= 0) {
                std::uint64_t v = 0;
                const ssize_t n = ::read(fds_[i], &v, sizeof(v));
                if (n == static_cast<ssize_t>(sizeof(v))) {
                    r.value = v;
                    r.valid = true;
                }
            }
            out.push_back(std::move(r));
        }
#else
        for (const auto& s : specs_) {
            out.push_back(CounterResult{s.name, 0, false});
        }
#endif
        return out;
    }

private:
#if defined(__linux__)
    static long perf_open(perf_event_attr* attr, int group_fd) {
        return ::syscall(__NR_perf_event_open, attr, /*pid=*/0, /*cpu=*/-1,
                         group_fd, /*flags=*/0UL);
    }

    void open_all() {
        int leader = -1;
        for (std::size_t i = 0; i < specs_.size(); ++i) {
            perf_event_attr attr{};
            attr.size = sizeof(attr);
            attr.type = specs_[i].type;
            attr.config = specs_[i].config;
            attr.disabled = (i == 0) ? 1 : 0; // leader starts disabled
            attr.exclude_kernel = 1;          // user-space work only
            attr.exclude_hv = 1;
            const long fd = perf_open(&attr, leader);
            if (fd < 0) {
                if (i == 0) {
                    // Leader failed: whole group is unusable. Record why so
                    // the caller can print an honest one-liner.
                    available_ = false;
                    why_ = std::strerror(errno);
                    return;
                }
                // A non-leader event is unsupported on this PMU: keep the
                // group, mark this slot absent (read() reports it invalid).
                fds_.push_back(-1);
                continue;
            }
            if (i == 0) {
                leader = static_cast<int>(fd);
            }
            fds_.push_back(static_cast<int>(fd));
        }
        available_ = !fds_.empty() && fds_[0] >= 0;
    }
#endif

    std::vector<CounterSpec> specs_;
#if defined(__linux__)
    std::vector<int> fds_;
#endif
    bool available_ = false;
    std::string why_ = "perf_event_open unavailable on this platform";
};

} // namespace spume::bench
