# ADR 0012: Test framework: vendored doctest single header

- Status: Accepted
- Date: 2026-07-12

## Context

Milestone 0 needs a unit-test framework. Constraints: no build-time network
access (AGENTS.md), no dependencies beyond C++20 + OpenMP + one header-only
test framework (maintainer-authorized as "doctest or Catch2"),
GPL-compatible license, and fast compiles — the suite builds
warnings-as-errors in CI on every push.

## Decision

Vendor doctest v2.4.12 (single header, MIT) at
`tests/thirdparty/doctest/`, upstream license and embedded copyright
retained. Include it as a SYSTEM directory so its internals are exempt
from warnings-as-errors; exclude `tests/thirdparty/` from the SPDX and
clang-format checks. Upgrade only by whole-file replacement of a tagged
upstream release, recording the version in the accompanying README.

## Consequences

Zero build-time network, one file to audit, the fastest compile times of
the authorized options. Upgrades are manual and deliberate. All test code
is written against doctest's macro API; switching frameworks later would
be a mechanical but wide diff.

## Rejected alternatives

- Catch2 v3: no longer single-header; needs FetchContent (build-time
  network, disallowed) or vendoring a full source tree.
- Catch2 v2.13 single header: maintenance-mode branch; no reason to prefer
  it over an actively maintained single header.
- GoogleTest: not header-only; adds a built dependency.
- Hand-rolled assertion micro-framework: reinvention with weaker
  reporting, and the maintainer explicitly authorized doctest/Catch2.
