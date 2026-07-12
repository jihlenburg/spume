# ADR 0011: Human-only commit authorship

- Status: Accepted
- Date: 2026-07-12

## Context

AI coding agents are used heavily in this repository. Claude Code adds
Co-Authored-By trailers and PR bylines by default. Accountability for
every committed line rests with the human who commits it; the tool is not
an author.

## Decision

- No Co-Authored-By trailers, "Generated with" bylines, emoji signatures,
  or any tool/AI attribution in commits, PRs, or changelogs.
- Enforcement, three layers: `.claude/settings.json` (committed) sets
  `"attribution": {"commit": "", "pr": ""}`; `.githooks/commit-msg`
  strips Co-Authored-By trailers (repo setup: `git config core.hooksPath
  .githooks`); AGENTS.md prohibits it for all agents.
- DCO `Signed-off-by:` remains mandatory on every commit — it is the
  human's certification of origin and is unrelated to tool attribution.
- Exception: upstream projects that require AI-involvement disclosure get
  it, per their policy, in Series A/B/C contributions (ADR-0008).
