# ADR 0016: Inline assembly permitted as pluggable specialized backends

- Status: Accepted
- Date: 2026-07-19
- Amends: ADR-0003 (JIT-first)

## Context

ADR-0003 chose JIT-first and rejected a hand-written ISA kernel library as an
"unmaintainable cul-de-sac" — single-target, unreviewable, dying on the first
architecture migration. That rejection assumed *undisciplined* hand-assembly.

Two things change the calculus:

1. **The precision firewall (ADR-0002).** The final result's accuracy is
   defined entirely by the flexible FP64 outer Krylov iterating on the true
   FP64 operator. Everything inside the preconditioner — coefficients,
   smoothers, hand-tuned kernels — only affects convergence *speed*, never the
   answer. So an aggressively hand-tuned assembly kernel in a preconditioner
   cannot damage the final result: the equivalence gate
   (`tests/regression/check_equivalence.py`, reorder-tolerance) is the proof on
   every case. The maintainer's stated single objective is **maximum speed
   without sacrificing precision on the final results**; the architecture makes
   that guarantee structural.

2. **FFmpeg refutes the "unmaintainable" premise.** FFmpeg maintains
   hand-written SIMD assembly across a large ISA matrix precisely *because* of a
   discipline that ADR-0003's rejected alternative lacked: runtime CPU-feature
   dispatch, an always-present C reference, macro-abstracted assembly
   (`x86inc`), and `checkasm` — a harness that verifies each asm routine
   against the C reference **and** benchmarks it. With that discipline,
   hand-assembly is reviewable, testable, and portable enough to live for
   decades.

## Decision

Hand-written / inline ISA assembly is **permitted**, subject to *all* of:

1. **Backend-isolated.** Assembly lives only under `src/backends/` (architecture
   invariant #3). Never in solver, IR, compat, or leaf code.
2. **Pluggable behind runtime dispatch (ADR-0004).** Every asm kernel is
   selected at runtime; a portable reference implementation always exists and
   is the default. No kernel is the *only* implementation of its function.
3. **`checkasm`-grade verification (the FFmpeg model, made binding).** Each asm
   kernel ships with (a) a correctness test asserting its output equals the
   reference within the applicable class (bitwise for deterministic kernels;
   the ADR-0002 rounding-reorder class otherwise), and (b) a bench reporting
   achieved GB/s vs the measured roofline. Both are required in the PR
   (correctness + counter evidence, ADR-0013).
4. **Roofline admission unchanged (ADR-0003, retained).** An asm kernel must
   reach the memory-bandwidth roofline or lower it (fewer bytes). It is deleted
   if the portable reference reaches >=90% of the measured roofline.
5. **Precision confined to preconditioners (ADR-0002, unchanged).** Asm may use
   any reduced or approximate precision **only** inside preconditioners under
   the flexible FP64 outer Krylov; the equivalence gate must confirm the final
   result stays in the FP64 class. Speed is never bought with the final answer.
6. **Portability discipline (FFmpeg model).** Assembly is written through a
   target-abstraction macro layer where practical (in the spirit of `x86inc`),
   documents its target ISA and the reference routine it mirrors, and neither
   blocks nor replaces the JIT path.

What does **not** change: JIT-first remains the default generation strategy and
the answer for architecture portability (RDNA waitcnt/cache-policy, Metal
simdgroup, arch migration); the reference-default invariant; the numerics
policy; the roofline admission rule. Hand-assembly is an **accelerant and a
golden reference for the JIT generator**, not a replacement for it: the fastest
route to the roofline on the primary target now, and the target the JIT backend
must later match.

## Consequences

- A faster path to the roofline on the primary target (Zen 5 / AVX-512):
  hand-asm now, JIT generalizes it across targets later.
- A bounded amount of hand-maintained ISA code — backend-isolated,
  reference-backed, `checkasm`-verified — i.e. the maintainability cost ADR-0003
  feared, contained by the FFmpeg discipline rather than avoided.
- A new prerequisite: a `checkasm`-style harness (per-kernel correctness vs
  reference + bench vs roofline) must exist before the first asm kernel lands.
  It reuses what already exists — the equivalence gate for correctness, the
  `bench/` harness for roofline.

## Rejected alternatives

- **Unrestricted hand-asm anywhere in the tree** — retains ADR-0003's
  cul-de-sac risk in solver/IR code; rejected. Assembly stays in `src/backends/`.
- **Asm without a reference fallback or without `checkasm`** — breaks
  verifiability and the precision guarantee; rejected. Reference + correctness
  test + bench are mandatory, non-negotiable.
- **Keep ADR-0003 as JIT-only** — loses the fastest path to the roofline on the
  primary target and the golden-reference role hand-asm plays for the JIT
  generator; rejected by maintainer decision, with the discipline above as the
  guardrail that made the original rejection unnecessary.
