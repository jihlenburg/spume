// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Joern Ihlenburg
//
// spumePimpleFoam — first SPUME leaf application (Milestone 1).
//
// A transient incompressible PIMPLE driver that, in M1, runs the *reference*
// (stock) OpenFOAM solvers unchanged: its purpose is to prove the leaf-app +
// compat-shim plumbing and to establish the bit-class-identical baseline
// against stock pimpleFoam before any SPUME numerics are introduced.
//
// Design (architecture invariants):
//   - The only upstream library API reaches this file through the compat
//     shim (compat/pimple.hpp), never via a direct upstream include.
//   - The PIMPLE assembly fragments (createFields.H, UEqn.H, pEqn.H,
//     correctPhi.H, setRDeltaT.H) are the *vendored* upstream fragments,
//     included from vendor/openfoam via -I (see Make/options). Nothing
//     upstream is copied into SPUME-owned space.
//   - Runtime-selectable SPUME solver classes are integrated in M2; until
//     then behaviour is identical to stock pimpleFoam by construction.
//
// The driver loop below mirrors upstream applications/solvers/incompressible/
// pimpleFoam/pimpleFoam.C (OpenFOAM v2606, GPL-3.0-or-later); the SPUME leaf
// owns only this glue, not the algorithm fragments it assembles.

#include "compat/pimple.hpp"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "SPUME leaf: transient solver for incompressible, turbulent flow"
        " of Newtonian fluids on a moving mesh (reference solvers, M1)."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createUfIfPresent.H"
    #include "CourantNo.H"
    #include "setInitialDeltaT.H"

    turbulence->validate();

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                // Do any mesh changes
                mesh.controlledUpdate();

                if (mesh.changing())
                {
                    MRF.update();

                    if (correctPhi)
                    {
                        // Calculate absolute flux
                        // from the mapped surface velocity
                        phi = mesh.Sf() & Uf();

                        #include "correctPhi.H"

                        // Make the flux relative to the mesh motion
                        fvc::makeRelative(phi, U);
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }
            }

            #include "UEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
        }

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
