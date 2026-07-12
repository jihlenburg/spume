# ADR 0013: Benchmark evidence on non-lab machines

- Status: Accepted
- Date: 2026-07-12

## Context

The AGENTS.md performance policy demands counter evidence and fixed power
mode. The current development environment is a shared cloud VM: no
uProf/perf uncore access, no power-mode control, and the exposed topology
and bandwidth changed under us mid-project (LLC reported as 260 MiB one
day and 33 MiB the next; STREAM triad moved several percent between
invocations minutes apart). Milestone 0 needed measured numbers before the
primary hardware (Strix Halo, docs/hardware.md) is available.

## Decision

- Lab-grade performance numbers come only from dedicated hardware with a
  fixed power mode and hardware counters; the plan of record is a
  self-hosted bench runner on the Strix Halo box, whose results replace
  all container numbers in docs/ when it lands.
- Until then, benchmarks on non-lab machines must **quantify their own
  noise instead of disclaiming it**. The harness is required to report,
  per timed section:
  - steal-time share from /proc/stat bracketing (neighbor interference),
  - median and coefficient of variation across reps, with the section
    flagged UNSTABLE above 5% CV,
  - an LLC working-set guard (warn below 2x LLC) — and readers must also
    watch per-substream sizes: a single vector near LLC size causes
    marginal-residency jitter even when the total is far above it,
  - a triad-calibrated prediction error (model bytes / measured triad vs
    measured time) as the counter substitute: it turns the bytes model
    into a falsifiable prediction,
  - interleaved A/B reps for any precision comparison, making the A/B
    ratio immune to slow drift.
- On non-lab machines, interleaved ratios (FP32/FP64) are the primary
  evidence; absolute GB/s are secondary and only valid against a
  same-invocation STREAM measurement.

## Consequences

Container results carry honest error bars and remain useful for model
validation and relative claims. Doc tables grow noise columns. The
harness carries ~150 lines of instrumentation that also benefits the
eventual lab runs (steal ~0 and tight CV become checkable preconditions).

## Rejected alternatives

- Installing perf in the container: VMs do not expose the uncore/DRAM
  events that would constitute counter evidence; adds a dependency for no
  evidentiary value.
- Best-of-N timing alone: hides contention and marginal cache residency;
  exactly the failure modes observed here.
- Deferring all measurement until the hardware arrives: loses the
  model-validation loop that Milestone 0 exists to close.
