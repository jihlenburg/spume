# ADR 0004: GPU execution model

- Status: Accepted
- Date: 2026-07-12

## Context

On Strix Halo-class APUs the CPU cores cannot reach the memory system
(~124 GB/s aggregate reads vs ~212 GB/s GPU-reachable; see
docs/hardware.md). On split systems, discrete GPU bandwidth dwarfs host
bandwidth but PCIe transfers tax naive offload.

## Decision

APU (unified memory):
- Solver-resident execution: matrices, Krylov vectors, and (target state)
  the whole timestep live in GPU-accessible unified memory (GTT /
  hipMallocManaged). Zero-copy; coefficients update in place.
- CPU and GPU are scheduled staggered, not concurrent, on bandwidth-bound
  phases — they share one memory pool, so overlap adds contention, not
  bandwidth. CPU owns assembly, boundaries, and GAMG coarse levels
  (latency-bound); GPU owns fine-level solves.
- On GPU, benchmark flat Chebyshev/Jacobi-preconditioned Krylov against
  GAMG on time-to-solution; flat methods map better to the iGPU even at
  higher iteration counts.
- Below a measured cell-count threshold, fall back to CPU automatically.

Split systems (n CPU + m GPU):
- Everything resident in VRAM; only halos cross PCIe. One rank per GPU.
- Interior/halo row split so boundary-row SpMV fuses with pack kernels and
  communication hides behind interior work.
- Halo compression (block-scaled FP16) in the fused pack kernels.
- CPU ranks serve GAMG coarse levels and I/O.

Whole-timestep target: tape the PISO/SIMPLE iteration into a dataflow graph
(HIP graphs / HSA signals), launch once per timestep; speculative batched
convergence checks (predict iterations from residual history, check
asynchronously) to remove global syncs.

## Consequences

Unlocks the bandwidth the CPU structurally cannot reach; removes the
transfer tax that kills iGPU-class offload elsewhere. The same residency
model scales to MI300A-class APUs unchanged.

## Rejected alternatives

- Per-solve offload with host<->device copies: transfer tax exceeds gains
  below ~10M cells; the pattern SPUME exists to avoid.
- Concurrent CPU+GPU sparse kernels on the APU: shared-pool contention;
  measured aggregate does not exceed the GPU-alone figure.
- Persistent grid-wide mega-kernels on Metal: no sanctioned grid-wide
  sync; use command-buffer chaining there instead (ADR-0009).
