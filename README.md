# SPUME

**Bandwidth-first CFD engine derived from OpenFOAM.** Mixed-precision FP64/FP32
solvers, JIT-specialized kernels, and resident execution for unified-memory
hardware (AMD APU/Instinct, RDNA, Apple silicon).

SPUME is a hard fork of OpenFOAM's solver layer, rebuilt around a single
premise: finite-volume CFD is memory-bandwidth-bound, so every byte moved is
the unit of cost. It keeps OpenFOAM's case structure, meshing, and
boundary-condition ecosystem, and replaces the linear-algebra core with a
mixed-precision engine — FP64 outer Krylov, compressed FP32/FP16
preconditioning — executed through per-case JIT-compiled kernels from a common
IR with CPU (Zen 5/AVX-512), HIP (RDNA/CDNA), and Metal backends.

Designed for unified-memory systems where CPU and GPU share one pool:
zero-copy, solver-resident, no transfer tax. Tracks upstream OpenFOAM releases
through a vendored-core, owned-leaves architecture.

## Status

**Pre-alpha.** Nothing here is benchmarked, stable, or fit for production use
yet.

## Design principles

- **Bytes are the budget.** Every optimization must either reach the memory
  roofline or lower it. Instruction-level work that does neither is rejected.
- **FP64 truth, compressed interior.** Solutions converge to FP64 accuracy at
  the requested tolerance; reduced precision lives only inside preconditioners,
  guarded by flexible outer iterations.
- **The mesh is a compilation target.** Static topology, geometry, and scheme
  constants are baked into per-case generated kernels at startup.
- **Own the leaves, vendor the trunk.** Upstream OpenFOAM is tracked as a
  vendored core with a minimal patch stack; all aggressive code lives in
  SPUME-owned solvers, libraries, and backends.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

New files: `Copyright (C) 2026 Joern Ihlenburg`. Vendored OpenFOAM files
retain their original copyright notices. Contributions are accepted under
GPL-3.0-or-later with a DCO sign-off (`git commit -s`).

Kernels emitted by the JIT at runtime are program output and carry no license
obligation for users.

## Attribution

SPUME is based on OpenFOAM technology. OPENFOAM® is a registered trademark of
OpenCFD Ltd. This project is not approved or endorsed by OpenCFD Ltd, producer
and distributor of the OpenFOAM software via www.openfoam.com.
