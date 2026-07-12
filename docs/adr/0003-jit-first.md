# ADR 0003: JIT-first — no hand-maintained ISA kernels

- Status: Accepted
- Date: 2026-07-12

## Context

Hand-written AVX-512 and raw RDNA 3.5 ISA kernels were evaluated and
prototyped conceptually. They deliver real gains (NT stores, software
prefetch on gathers, waitcnt scheduling, cache-policy bits, VOPD packing)
but are single-target, unreviewable by anyone else, and would die on the
first architecture migration (e.g. RDNA4). The mesh is static per case, so
the case itself is a legitimate compilation target.

## Decision

- Kernels are generated at case startup from a common SPUME IR, with
  emission backends per target: CPU (LLVM ORC, Zen 5/AVX-512), HIP
  (hipRTC, RDNA/CDNA), Metal (runtime MSL). Mesh sizes, SELL block widths,
  patch layout, scheme constants are baked in as immediates.
- All ISA-specific tricks live in generator backends under `src/backends/`,
  never as hand-maintained kernels in solver code.
- Every generated path has a portable reference implementation behind
  runtime dispatch; default behavior is the reference.
- A specialized kernel that fails to beat a reference already at >=90% of
  the measured roofline is deleted.
- Optimization admission rule: a change must either (a) bring a kernel to
  the memory-bandwidth roofline or (b) lower the roofline by moving fewer
  bytes. Instruction-level work that does neither is rejected.

## Consequences

Machine-specific tuning scales as generator code, not kernel inventory.
The generator is the project's crown jewel and its bus-factor risk; the
mitigations are the stable IR, the reference dispatch, and contract tests.

## Rejected alternatives

- Hand-written asm kernel library: unmaintainable cul-de-sac; rejected
  after explicit evaluation.
- Compiler-only (trust -O3): leaves per-stream NT stores, gather
  pipelining, and cache-policy control on the table; LLVM extracts RDNA
  dual-issue poorly.
- FPGA overlays as the specialization mechanism: see ADR-0005.
