# tests/contract — upstream invariant assertions (placeholder)

Contract tests assert the upstream invariants SPUME depends on (LDU
addressing, GAMG agglomeration, tutorial iteration counts) and run nightly
against upstream dev as a merge canary.

They activate in Milestone 1 when `vendor/openfoam/` lands; Milestone 0 has
no upstream to contract against.
