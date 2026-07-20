// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg

#pragma once

#include "core/amg.hpp"
#include "core/types.hpp"

// GPU aggregation transfer operators (ADR-0017) -- the piecewise-constant
// prolongation P (P[i][agg[i]] = 1) and its transpose, the two grid-transfer
// operators a multigrid V-cycle needs between levels:
//
//   restrict:  rc[J] = sum_{agg[i]=J} res[i]      (P^T, fine -> coarse)
//   prolong:   z[i] += ec[agg[i]]                 (P,   coarse -> fine)
//
// Both are FP64 to match the CPU AmgPrecond scaffolding (only the smoother
// interior is FP32, ADR-0002). The device-to-device methods take resident
// pointers so the cycle chains without host round-trips; the aggregation lives
// resident in unified memory. This header is HIP-free.

namespace spume::gpu {

class AggTransferResident {
public:
    explicit AggTransferResident(const Aggregation& agg);
    ~AggTransferResident();
    AggTransferResident(const AggTransferResident&) = delete;
    AggTransferResident& operator=(const AggTransferResident&) = delete;

    // rc = P^T res (restrict). Zeros rc[0..ncoarse) first, then scatter-adds each
    // fine value into its aggregate. res has nfine() entries, rc has ncoarse().
    void restrict_device(const double* res_dev, double* rc_dev) const;

    // z += P ec (prolong-add). Each fine row gathers its aggregate's coarse value
    // and adds it. ec has ncoarse() entries, z has nfine().
    void prolong_add_device(const double* ec_dev, double* z_dev) const;

    index_t nfine() const { return nfine_; }
    index_t ncoarse() const { return ncoarse_; }

private:
    index_t nfine_ = 0;
    index_t ncoarse_ = 0;
    index_t* d_agg_ = nullptr; // agg[i] in [0, ncoarse), resident
};

} // namespace spume::gpu
