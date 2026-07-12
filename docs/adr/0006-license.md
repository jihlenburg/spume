# ADR 0006: License — GPL-3.0-or-later, uniform

- Status: Accepted
- Date: 2026-07-12

## Context

SPUME is a derivative work of OpenFOAM, whose headers grant GPLv3
"or (at your option) any later version". License choice is therefore
constrained, not free.

## Decision

- The whole repository is GPL-3.0-or-later. SPDX identifier in every file.
- New files: `Copyright (C) 2026 Joern Ihlenburg` (or successor entity).
  Vendored files keep upstream copyright notices untouched.
- Contributions accepted under GPL-3.0-or-later with DCO sign-off; no CLA.
- Kernels emitted by the JIT at runtime are program output and carry no
  license obligation for users. The generator itself is GPL.
- Documented consequence: no proprietary linked add-ons are possible.

## Rejected alternatives

- AGPL-3.0 for original components (GPLv3 s.13 combination would permit
  it): rejected because (a) blanket corporate AGPL bans create adoption
  friction exactly where the project needs pull, (b) the usual AGPL
  payoff — dual licensing — is unavailable anyway since the
  OpenFOAM-derived parts stay GPL, (c) mixed licensing complicates every
  upstream MR and every user's compliance story. Revisit only if a
  hosted-CFD free-rider materializes; copyright retention keeps the
  option open.
- Relicensing to AGPL outright: legally impossible; "or-later" means
  later GPL versions, and AGPL is not one.
