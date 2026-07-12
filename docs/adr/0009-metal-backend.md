# ADR 0009: Metal backend — scope and sequencing

- Status: Accepted
- Date: 2026-07-12

## Context

Apple GPUs (M-series) have no FP64 in the ISA at all — disqualifying for
stock OpenFOAM, nearly irrelevant for SPUME because ADR-0002 already
confines GPU work to the FP32 preconditioner interior while the CPU
(fast scalar/NEON FP64) runs the outer Krylov. Apple's unified memory and
bandwidth (M4 Max ~546 GB/s, M3 Ultra ~819 GB/s, largely CPU-reachable,
unlike Strix Halo) make it the strongest desk-side roofline available.
~70% of the project ports unchanged (all mathematics, formats,
compression, JIT concept, fork infrastructure, tests).

## Decision

- Metal is the third emission backend of the common IR (runtime MSL
  compilation; 32-wide simdgroups map to wave32 designs; simd_shuffle
  replaces DPP; threadgroup memory replaces LDS; command-buffer chaining
  replaces persistent mega-kernels).
- Scope is one bounded artifact: the FP32 preconditioner engine plus the
  LBM engine on Metal, benchmarked on one M4 Max. Time box: 2-3 months
  once started. Ongoing tax budget: 10-15% per release cycle.
- Sequencing: starts only after the Milestone 3 APU demo is delivered
  (see docs/roadmap.md). The
  reason is focus: one flagship demo before backend breadth.

## Consequences

Proves the IR's portability claim ("the mathematics, not the silicon, is
the product") on the only other UMA silicon that matters. Opens the
Apple-based engineering market, which currently has no serious CFD option.

## Rejected alternatives

- Metal before the APU demo: splits the critical two-month demo effort.
- Skipping Apple entirely: forfeits the best available roofline and the
  portability proof.
- MoltenVK/Vulkan compute instead of native Metal: extra layer, worse
  counter access, no gain for generated kernels.
