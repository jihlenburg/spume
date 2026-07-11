# Contributing to SPUME

SPUME is GPL-3.0-or-later. By contributing you certify the Developer
Certificate of Origin (DCO, <https://developercertificate.org/>) for every
commit.

## Required for every commit

1. **DCO sign-off.** Commit with `git commit -s`, which appends
   `Signed-off-by: Your Name <you@example.com>`. The sign-off is the
   committing human's certification of origin and is mandatory — unsigned
   commits are rejected.
2. **Human-only authorship.** No `Co-Authored-By:` trailers, "Generated
   with" bylines, emoji signatures, or any tool/AI attribution in commit
   messages, PR descriptions, or changelogs. Accountability for every line
   rests with the human who commits it. `.githooks/commit-msg` strips
   `Co-Authored-By:` trailers as hard enforcement.
3. **Conventional commits.** Imperative mood, one concern per commit, e.g.
   `perf(spmv): fuse Amul with dot product`.

## One-time repository setup

```sh
git config core.hooksPath .githooks
```

## Source files

- Every new file starts with an SPDX header:

  ```
  // SPDX-License-Identifier: GPL-3.0-or-later
  // Copyright (C) 2026 Joern Ihlenburg
  ```

- SPUME code is C++20, formatted with the repo `.clang-format`, built with
  warnings-as-errors. No raw `new`/`delete`; prefer `std::span` over raw
  pointers.
- Vendored code under `vendor/` keeps upstream style and upstream copyright
  headers, and is modified only through the patch stack in `patches/`.

## Build and test

```sh
cmake --preset cpu-release
cmake --build --preset cpu-release
ctest --preset unit        # unit tests
ctest --preset all         # full suite
```

No merge with failing tests. Never weaken a test or tolerance to make it
pass.

## Dependencies

None may be added without explicit maintainer approval. No AGPL or
proprietary code. Never copy code from other projects — reimplement from
ideas.

See `AGENTS.md` for the full binding policy (numerics, performance evidence,
architecture invariants); it applies to human contributors as much as to
agents.
