# Vendored third-party: doctest

- **What**: `doctest.h`, single-header C++ testing framework.
- **Version**: 2.4.12
- **Source**: <https://github.com/doctest/doctest> (tag `v2.4.12`,
  `doctest/doctest.h`)
- **License**: MIT (see `LICENSE.txt` here; GPL-compatible). Upstream
  copyright header retained inside `doctest.h`.
- **Why doctest**: the mission allows one header-only test framework
  (doctest or Catch2). doctest was chosen because it is a true single
  header with the fastest compile times of the two — Catch2 v3 is no longer
  single-header and would need either build-time downloads (network in the
  build, disallowed) or vendoring a full source tree.

Files under `tests/thirdparty/` are third-party: the repo SPDX-header rule
does not apply to them, and they are excluded from clang-format and SPDX CI
checks. Update only by replacing with a newer upstream release, recording
the version here.
