# ADR 0005: No FPGA target

- Status: Accepted
- Date: 2026-07-12

## Context

AMD Versal HBM-class cards (e.g. Alveo V80: ~32 GB HBM2e, ~800 GB/s) were
evaluated as a third backend, assuming a PCIe card in an AMD host.

## Decision

FPGA is not a SPUME target. Not mainline, not showcase.

Two ideas from the evaluation are salvaged in software:
1. Deterministic reductions via fixed-order tree sums / compensated
   arithmetic as a CPU debug mode (replaces the fabric Kulisch
   accumulator; ~2-3x reduction cost, irrelevant in validation runs).
2. Halo compression moves into the GPU pack kernels (replaces the
   "smart NIC" role; see ADR-0004).

## Consequences

The portfolio stays at three emission targets (CPU, HIP, Metal), all
demoable on purchasable hardware. No P&R toolchain in CI.

## Rejected alternatives (why FPGA lost)

- Bandwidth per dollar: ~$10K card at 800 GB/s vs ~$1K RDNA4 dGPU at
  ~640 GB/s — off by nearly an order of magnitude.
- No native FP64 in DSP58; only viable inside the FP32-interior
  architecture, same as the iGPU, minus the mature toolchain.
- Unstructured-mesh gather is the worst case for dataflow; HBM gather
  engines cost fabric and design-months.
- Hours-long place-and-route is philosophically incompatible with
  JIT-per-case (ADR-0003); the only reconciliation is a soft overlay,
  i.e. a year spent building a worse GPU.
- PCIe transfer tax returns (Gen5 x16 ~50-60 GB/s real); CXL coherence
  is an EPYC story, not a desktop Ryzen one.
- LBM dataflow showcase and exact-arithmetic co-processor were the only
  defensible artifacts; both are achievable in software or on GPU.
