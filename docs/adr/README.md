# Architecture Decision Records

Settled decisions for SPUME. Agents and contributors: do not revisit a
decision recorded here without explicit maintainer instruction. The
"Rejected alternatives" sections exist to prevent re-litigation; read them
before proposing anything similar.

Format: see `0000-template.md`. New decisions get the next number and a
one-line entry in the index below.

| ADR  | Title                                        | Status   |
|------|----------------------------------------------|----------|
| 0001 | Fork topology: own the leaves, vendor trunk  | Accepted |
| 0002 | Mixed-precision numerics architecture        | Accepted |
| 0003 | JIT-first: no hand-maintained ISA kernels    | Accepted |
| 0004 | GPU execution model                          | Accepted |
| 0005 | No FPGA target                               | Accepted |
| 0006 | License: GPL-3.0-or-later, uniform           | Accepted |
| 0007 | Naming: SPUME                                | Accepted |
| 0008 | Upstream contribution strategy               | Accepted |
| 0009 | Metal backend: scope and sequencing          | Accepted |
| 0010 | Explicit engines for eligible case classes   | Accepted |
| 0011 | Human-only commit authorship                 | Accepted |
| 0012 | Test framework: vendored doctest             | Accepted |
| 0013 | Benchmark evidence on non-lab machines       | Accepted |
| 0014 | Upstream baseline: OpenFOAM v2606, pruned    | Accepted |
| 0015 | Local Strix Halo is the performance reference| Accepted |
| 0016 | Inline asm permitted as pluggable backends   | Accepted |
