# ADR 0017: GPU backend — HIP on gfx1151, unified-memory resident, verified in-class

- Status: Accepted
- Date: 2026-07-19
- Refines: ADR-0004 (GPU execution model); builds on ADR-0002 (precision
  firewall) and ADR-0016 (checkasm-verified pluggable backends)

## Context

ADR-0004 fixed the GPU *execution model* (runtime dispatch, a portable
reference always selectable, reference is the default). It did not fix the M3
backend specifics, and M3 (the flagship, docs/roadmap.md) now needs them nailed
down before any GPU kernel merges. Three specifics forced a decision, and Phase-1
measurement on the target hardware settled each:

1. **Which allocation model on an APU.** The Strix Halo iGPU (Radeon 8060S,
   gfx1151) shares the LPDDR5X with the CPU. `hipMalloc` (a device VRAM carve-out)
   **segfaults for allocations above ~512 MB** here — the default carve-out is
   tiny — and even when it fits it forces explicit host↔device copies that defeat
   the whole point of a unified-memory part.
2. **How to verify a GPU kernel** when host (g++) and device (clang/RDNA) emit
   different FMA fusion. Measured: a correct GPU SELL SpMV differs from the CPU
   reference in ~1/3 of rows *at the last bit* purely from `-ffp-contract=fast`
   on the host vs the device's fused multiply-adds. A bitwise GPU-vs-CPU oracle
   is therefore impractical without forcing a contraction policy on both sides.
3. **Whether the bandwidth thesis even holds on our kernels.** The CPU is
   fabric-walled at ~150 GB/s; the iGPU measured ~237 GB/s raw. Does that
   translate to the actual SpMV, or does the scattered gather kill it?

Phase-1 evidence (2026-07-19, gfx1151, poisson7 128³, 2.1M rows): the GPU SELL
SpMV ran at **208.8 GB/s = 1.67× the 16-core CPU** and ~93% of the GPU triad
roofline; a full FP32 Chebyshev apply verified against the CPU reference at
max-rel 2.7e-7. Both used `hipMallocManaged`.

## Decision

1. **Target ROCm/HIP on gfx1151** (RDNA3.5) as the M3 backend. Kernels are
   emitted through the IR/JIT path (ADR-0003) with hand-tuned HIP permitted as a
   pluggable backend under the ADR-0016 discipline.
2. **Unified memory is the residency model.** GPU buffers use `hipMallocManaged`
   (zero-copy GTT), prefetched with `hipMemPrefetchAsync` to the device before
   timed regions. Never `hipMalloc` a large device carve-out. The AMG hierarchy
   and solver vectors live resident in managed memory; only coefficients update
   per solve (matches the amortisation of `spumePCG`).
3. **The whole solve stays GPU-resident.** SpMV, vector axpy/axpby, and
   dot-product reductions across all multigrid levels run on-device; no
   per-iteration host↔device round-trips. Falls back to the CPU path below a
   measured cell-count threshold (small problems: launch overhead dominates).
4. **GPU kernels live in `src/backends/gpu/`** behind runtime dispatch; the
   portable reference stays the default (ADR-0004, invariant 4).
5. **Verification is in-class, not bitwise.** A GPU kernel is correct when its
   output matches the CPU reference within the reorder-tolerance equivalence
   class (ADR-0002 numerics policy) — the same bar as changing MPI rank count.
   The deterministic-reduction *bitwise* oracle, when needed, compiles both host
   and device with a matched contraction policy (`-ffp-contract=off` or explicit
   `fma()` both sides). Every GPU kernel carries a checkasm-style CPU-reference
   verify (ADR-0016).
6. **Performance claims need rocprof.** Per the performance policy and ADR-0013,
   a merge-grade GPU perf claim attaches rocprof counter evidence (achieved GB/s
   vs measured roofline), not just wall-clock.

## Consequences

- **Easier:** the unified-memory model makes "build the hierarchy once, update
  coefficients, never copy" natural — the resident-execution goal of the whole
  project. The firewall (ADR-0002) means the entire GPU preconditioner may run
  FP32/reduced precision without touching the FP64 answer, so GPU work is bounded
  by convergence speed, never correctness.
- **Harder / committed to:** we own HIP kernels for gfx1151 and must keep the CPU
  reference exactly in step for verification; the in-class (non-bitwise) verify
  means the strict bitwise oracle is a deliberate, separately-compiled mode, not
  the default GPU test. Portability to other GPUs (CDNA, later Metal per ADR-0009)
  rides on the same IR, not on these hand HIP kernels.

## Rejected alternatives

- **`hipMalloc` device VRAM.** Segfaults above the small default carve-out on
  this APU, and forces host↔device copies that defeat zero-copy on a
  unified-memory part. Managed/GTT is strictly better here.
- **Bitwise GPU-vs-CPU as the correctness bar.** Host/device FMA-contraction
  differences make ~1/3 of rows differ at the last bit for a *correct* kernel;
  requiring bitwise equality would either reject correct kernels or force
  `-ffp-contract=off` everywhere (slower, and unnecessary — the equivalence class
  already admits reorder-level differences, ADR-0002).
- **OpenCL / SYCL.** ROCm/HIP is the native, first-class path for gfx1151 with
  working hipRTC for JIT (ADR-0003); a portability layer would add abstraction
  cost for no benefit on the one target that matters for M3.
- **Copy-in/copy-out per solve (discrete-GPU model).** Wastes the APU's shared
  memory and adds per-iteration PCIe-class latency that does not exist here;
  the entire point of Strix Halo is that CPU and GPU see the same DRAM.
