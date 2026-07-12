// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// Contract test: LDU addressing invariants SPUME depends on (ADR-0001).
//
// SPUME's SELL-C-8 conversion and cell-row kernels (Milestone 2) assume,
// for every fvMesh:
//   1. lowerAddr[f] < upperAddr[f] for every internal face
//      (owner strictly below neighbour);
//   2. faces are stored in upper-triangular order: lowerAddr
//      non-decreasing, and upperAddr strictly increasing within a run of
//      equal lowerAddr;
//   3. losortAddr orders faces by non-decreasing upperAddr;
//   4. ownerStartAddr/losortStartAddr are consistent prefix tables for 1-3.
//
// Any failure here means upstream changed a load-bearing invariant and the
// merge canary must go red. Exit code 0 and the SPUME-CONTRACT-LDU: OK
// line are the pass signal.

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"

using namespace Foam;

int main(int argc, char* argv[])
{
    argList::addNote("SPUME contract test: LDU addressing invariants");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    const lduAddressing& addr = mesh.lduAddr();
    const labelUList& lower = addr.lowerAddr();
    const labelUList& upper = addr.upperAddr();
    const label nFaces = lower.size();
    const label nCells = addr.size();

    if (nFaces == 0)
    {
        FatalErrorInFunction
            << "contract mesh has no internal faces" << exit(FatalError);
    }

    // 1 + 2: owner < neighbour, upper-triangular face ordering.
    for (label f = 0; f < nFaces; ++f)
    {
        if (lower[f] >= upper[f])
        {
            FatalErrorInFunction
                << "face " << f << ": lower " << lower[f]
                << " >= upper " << upper[f] << exit(FatalError);
        }
        if (f > 0)
        {
            if (lower[f] < lower[f - 1])
            {
                FatalErrorInFunction
                    << "face " << f << ": lowerAddr decreases" << exit(FatalError);
            }
            if (lower[f] == lower[f - 1] && upper[f] <= upper[f - 1])
            {
                FatalErrorInFunction
                    << "face " << f << ": upperAddr not increasing within owner run"
                    << exit(FatalError);
            }
        }
    }

    // 3: losort visits faces in non-decreasing upperAddr order.
    const labelUList& losort = addr.losortAddr();
    if (losort.size() != nFaces)
    {
        FatalErrorInFunction
            << "losortAddr size " << losort.size() << " != nFaces " << nFaces
            << exit(FatalError);
    }
    for (label i = 1; i < nFaces; ++i)
    {
        if (upper[losort[i]] < upper[losort[i - 1]])
        {
            FatalErrorInFunction
                << "losort order broken at position " << i << exit(FatalError);
        }
    }

    // 4: prefix tables bracket exactly the runs asserted above.
    const labelUList& ownerStart = addr.ownerStartAddr();
    const labelUList& losortStart = addr.losortStartAddr();
    if (ownerStart.size() != nCells + 1 || losortStart.size() != nCells + 1)
    {
        FatalErrorInFunction
            << "start-table sizes wrong: ownerStart " << ownerStart.size()
            << ", losortStart " << losortStart.size() << ", nCells " << nCells
            << exit(FatalError);
    }
    if (ownerStart[0] != 0 || ownerStart[nCells] != nFaces
        || losortStart[0] != 0 || losortStart[nCells] != nFaces)
    {
        FatalErrorInFunction
            << "start tables do not span [0, nFaces]" << exit(FatalError);
    }
    for (label celli = 0; celli < nCells; ++celli)
    {
        for (label f = ownerStart[celli]; f < ownerStart[celli + 1]; ++f)
        {
            if (lower[f] != celli)
            {
                FatalErrorInFunction
                    << "ownerStart bracket wrong for cell " << celli
                    << exit(FatalError);
            }
        }
        for (label i = losortStart[celli]; i < losortStart[celli + 1]; ++i)
        {
            if (upper[losort[i]] != celli)
            {
                FatalErrorInFunction
                    << "losortStart bracket wrong for cell " << celli
                    << exit(FatalError);
            }
        }
    }

    Info<< "SPUME-CONTRACT-LDU: OK (nCells " << nCells
        << ", nInternalFaces " << nFaces << ")" << nl << endl;

    return 0;
}
