# SPUME patch stack

`vendor/openfoam/` (arriving in Milestone 1) is unmodified upstream source.
The **only** mechanism for changing vendored code is this ordered patch
stack. Never edit files under `vendor/` directly.

## Layout

```
patches/
  0001-short-slug.patch
  0002-short-slug.patch
  ...
```

Patches are unified diffs relative to the repository root, numbered in
application order, each with a one-line header comment stating what it does
and why it cannot live in `src/` instead.

## Hard budget

The entire stack is limited to **2,000 lines total** (sum of all `.patch`
files, counted with `wc -l`). If a change does not fit, it belongs in
SPUME-owned code behind the compat shim (`src/compat/`), or upstream.

## Procedure

- **Apply** (done by the build/setup tooling once vendoring lands):
  `git apply patches/*.patch` in numeric order.
- **Add**: make the change in a scratch checkout of `vendor/`, generate the
  diff (`git diff > patches/NNNN-slug.patch`), restore `vendor/` to
  pristine, and commit only the patch file.
- **Refresh** after an upstream bump: re-apply the stack in order, fix
  rejects inside the patch files themselves, and verify
  `tests/contract/` still passes — contract tests are the canary for
  upstream drift.
- **Remove**: prefer deleting patches over growing them; renumber only when
  ordering actually changes.

Rules of the road: one logical change per patch, never fold unrelated fixes
together, and never remove or alter upstream copyright headers.
