# ADR 0015: Local Strix Halo is the performance reference

- Status: Accepted
- Date: 2026-07-19
- Amends: ADR-0013 (its "until the hardware arrives" clause is now fulfilled)

## Context

ADR-0013 measured Milestone 0 on a shared cloud VM as an explicit stopgap
"until the primary hardware (Strix Halo) is available," naming a self-hosted
bench runner on that box as the plan of record. The interactive dev host
(`halobox`) IS that box — an AMD Ryzen AI Max+ 395 (Strix Halo), 16 Zen 5 cores
/ 32 threads. The stopgap is over.

## Decision

- The local Strix Halo box is the authoritative performance reference.
  `docs/perf-strix-halo.md` supersedes the container numbers in
  `docs/milestone0.md`.
- Integrated SPUME-vs-stock comparisons run locally, **always against the tuned
  `linux64GccDPInt32OptZnver5` stock build** — the roadmap's "tuned znver5 FP64
  CPU baseline" — never the portable `linux64GccDPInt32Opt` build. Rationale:
  the portable build is ~7-9% slower on its own (per the pitzDaily measurement
  in the dev-environment notes), so benchmarking against it would inflate
  SPUME's apparent speedup. That is the one non-negotiable of this ADR.
- Bench discipline: pin the 16 physical cores (`OMP_NUM_THREADS=16
  OMP_PROC_BIND=close OMP_PLACES=cores`); use bandwidth-bound working sets;
  record governor, power-mode status, steal-time and CV per run. FP32/FP64
  interleaved ratios are the primary, drift-immune evidence; absolute GB/s are
  secondary and valid only against a same-invocation STREAM triad.
- Package power-mode pinning (ryzenadj/BIOS) is out of scope: the ratios do not
  need it, so absolute GB/s simply carry an honest "power mode not fixed"
  caveat until a lab-grade run is warranted.
- CI split is unchanged: fast per-push CI on GitHub `ubuntu-latest` (portable M0
  core, no OpenFOAM); OpenFOAM + perf on the self-hosted `halobox` runner.

## Consequences

Performance claims now rest on real target hardware with honest error bars, and
on a fair (tuned-znver5) baseline. The container-measurement era is closed. Some
runs remain "non-lab" (no fixed power mode, no hardware counters); that is
stated per run, not hidden.

## Rejected alternatives

- Keep quoting cloud-VM numbers: wrong hardware, unstable topology (ADR-0013);
  rejected now that the target is in hand.
- Compare against the portable stock build: self-flattering by ~7-9%; rejected.
- Block on lab-grade power pinning before any local number: the drift-immune
  ratios are already trustworthy without it; rejected as premature.
