# src/backends — IR generator backends (placeholder)

Empty until the SPUME IR lands (roadmap M2/M3). From then on, ALL
ISA-specific work — AVX-512 scheduling and NT stores (CPU/LLVM ORC), RDNA
waitcnt and cache-policy bits (hipRTC), Metal simdgroup ops (runtime MSL) —
lives in generator backends here, never as hand-maintained kernels in
solver code (ADR-0003).

Milestone 0 is portable C++20 + OpenMP only, so nothing lives here yet.
