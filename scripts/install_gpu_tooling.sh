#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Joern Ihlenburg
#
# Install the ROCm GPU profiling / instrumentation layer used for M3 work
# (ADR-0017, docs/roadmap.md). Assumes a working ROCm base + HIP toolchain is
# already present (hipcc, rocminfo); this adds the profilers, the measured-
# roofline tool, the debugger, and the SMI monitors on top.
#
# Idempotent: apt-get install is safe to re-run. Requires sudo (apt).
#
# Usage:
#   scripts/install_gpu_tooling.sh            # install + verify
#   scripts/install_gpu_tooling.sh --verify   # verify only (no install)
#
# Pin a ROCm version by exporting ROCM_PATH (default /opt/rocm).

set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="${ROCM_PATH}/bin:${PATH}"

# Packages, grouped by purpose. Names are the unversioned meta-packages so the
# script tracks whatever ROCm release is installed.
PKGS=(
    # Profilers: rocprofv3/v2/rocprof CLIs + the counter/trace SDK.
    rocprofiler-sdk
    rocprofiler
    # The AQL profiler runtime lib (libhsa-amd-aqlprofile64.so) -- rocprofv3
    # fails to load without it; a frequent missing dependency.
    hsa-amd-aqlprofile
    roctracer
    # Measured memory-bandwidth roofline (the denominator for GB/s claims).
    rocm-bandwidth-test
    # High-level roofline / bottleneck analysis (ex-Omniperf) and CPU+GPU
    # timeline tracing (ex-Omnitrace) for the heterogeneous CPU/GPU split.
    rocprofiler-compute
    rocprofiler-systems
    # GPU kernel debugger, and the SMI monitors for power/clock/thermal so
    # benchmarks can run in a known, fixed power state (performance policy).
    rocm-gdb
    amd-smi-lib
    rocm-smi-lib
)

install() {
    if ! command -v hipcc >/dev/null 2>&1; then
        echo "install_gpu_tooling: hipcc not found under ${ROCM_PATH} -- install the" >&2
        echo "install_gpu_tooling: ROCm base + HIP toolchain first, then re-run." >&2
        exit 1
    fi
    echo "install_gpu_tooling: installing ${#PKGS[@]} packages via apt ..."
    sudo apt-get install -y "${PKGS[@]}"
}

verify() {
    echo "install_gpu_tooling: verifying tool binaries ..."
    local ok=1
    # (binary, providing package) -- amd-smi/rocm-smi are optional monitors.
    for bin in rocprofv3 rocprof rocm-bandwidth-test rocprof-compute rocgdb amd-smi; do
        if command -v "$bin" >/dev/null 2>&1; then
            printf "  ok   %-20s %s\n" "$bin" "$(command -v "$bin")"
        else
            printf "  MISS %-20s (not on PATH)\n" "$bin"
            ok=0
        fi
    done
    # The library rocprofv3 needs at runtime.
    if ls "${ROCM_PATH}"/lib/libhsa-amd-aqlprofile64.so* >/dev/null 2>&1; then
        echo "  ok   libhsa-amd-aqlprofile64.so present"
    else
        echo "  MISS libhsa-amd-aqlprofile64.so (rocprofv3 will fail to load)"
        ok=0
    fi
    [ "$ok" = 1 ] && echo "install_gpu_tooling: all core tools present." \
                  || { echo "install_gpu_tooling: some tools missing (see above)." >&2; exit 1; }
}

case "${1:-}" in
    --verify) verify ;;
    "")       install; verify ;;
    *)        echo "usage: $0 [--verify]" >&2; exit 2 ;;
esac
