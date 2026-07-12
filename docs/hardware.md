# Hardware facts and roofline numbers

All figures are measurements or vendor specs as known on **2026-07-12**.
Re-verify before citing in benchmarks or design arguments; driver and
kernel versions move these numbers (a 6.14 -> 6.15 kernel change alone
showed ~15% HIP compute swings on gfx1151). Sources at the bottom.

## Why these numbers govern everything

FVM CFD kernels run at ~0.1-0.2 FLOP/byte. Performance = achievable
bandwidth x arithmetic intensity. Every design decision in SPUME
(ADR-0002/0003/0004) is downstream of the table below. Rule: an
optimization must reach this roofline or lower it (fewer bytes).

## AMD Strix Halo (Ryzen AI Max+ 395, primary dev target)

| Quantity | Value | Note |
|---|---|---|
| LPDDR5X-8000, 256-bit | 256 GB/s theoretical | soldered, shared CPU+GPU |
| GPU-reachable (measured) | ~212 GB/s | rocm_bandwidth_test |
| CPU aggregate reads (both CCDs) | ~124 GB/s | die-to-die limited |
| CPU read-modify-add (both CCDs) | ~175 GB/s | |
| Per-CCD read link | ~64 GB/s | 32 B/cycle @ ~2 GHz fabric |
| Per-CCD observed total | ~103 GB/s | read+write combined |
| CPU->GPU copy | ~84 GB/s | unified pool, still a copy path |
| L3 | 32 MB per CCD, 64 MB total | cases <~1-2M cells partly cache-resident |
| Infinity Cache (MALL, GPU-side) | 32 MB | ~73% traffic capture observed in graphics loads |
| CPU | 16x Zen 5, 2 CCDs, full 512-bit FP datapath | desktop-derived CCDs |
| GPU | 40 CU RDNA 3.5, gfx1151 | FP64 at 1:16 of FP32 |
| Package power | 45-120 W configurable | fix the mode before benchmarking |

**The central asymmetry:** the CPU cores cannot reach ~half the memory
system; the iGPU can. GPU-resident solves gain ~1.6-1.7x on bandwidth
alone before precision effects (ADR-0004).

## ROCm on gfx1151 (status 2026-07)

Works in practice (ROCm 7.2, gfx11-generic ISA target; TheRock nightlies
carry native gfx1151), but **not on AMD's official support matrix** as of
early 2026. Treat as functional-unsupported: pin kernel+driver versions
per benchmark, expect regressions across updates.

## Other targets

| Platform | Bandwidth | Notes |
|---|---|---|
| RDNA4 dGPU class | ~640 GB/s GDDR6/7 | ~$1K; split-system building block |
| MI300A APU | ~5.3 TB/s HBM3 | same residency model as Strix Halo, scaled |
| Apple M4 Max | ~546 GB/s | **no FP64 on Apple GPUs at all**; CPU FP64 fast |
| Apple M3 Ultra | ~819 GB/s | largely CPU-reachable, unlike Strix Halo |
| PCIe Gen5 x16 | ~50-60 GB/s real | the transfer tax; halos only (ADR-0004) |

## Derived expectations (bands, not promises)

Vs. a tuned znver5 FP64 CPU baseline, large pressure-dominated case:
- CPU-only path (formats + fusion + SPDP): 1.5-2.0x
- + GPU-resident FP32 preconditioner solve: 2.0-2.6x
- Full program (matrix-free, compression, temporal blocking): 2.5-4x
  (60% confidence; upper end >10M cells)
- Explicit engines on eligible classes (ADR-0010): 10-50x
- Hard ceiling: ~212 GB/s divided by bytes/cell/iteration. Nothing beats
  this arithmetic; claims that ignore it are wrong.

## Sources (retrieved 2026-07)

- chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix (CPU bandwidth, CCD links)
- chipsandcheese.com/p/strix-halos-memory-subsystem-tackling (die-to-die analysis)
- chipsandcheese.com/p/evaluating-the-infinity-cache-in (MALL behavior)
- llm-tracker.info/_TOORG/Strix-Halo (rocm_bandwidth_test 212/84 GB/s, kernel sensitivity)
- tinycomputers.io/posts/upgrading-rocm-7.0-to-7.2-on-amd-strix-halo-gfx1151.html (support-matrix status)
