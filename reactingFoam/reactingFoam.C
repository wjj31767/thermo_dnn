/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    reactingFoam

Group
    grpCombustionSolvers

Description
    Solver for combustion with chemical reactions.

\*---------------------------------------------------------------------------*/
#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include "fvCFD.H"
#include "turbulentFluidThermoModel.H"
#include "psiReactionThermo.H"
#include "CombustionModel.H"
#include "multivariateScheme.H"
#include "pimpleControl.H"
#include "pressureControl.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
torch::jit::script::Module module;
//TODO: o2 0.220142 n2 0.724673 T 293 others 0 CH4 0.0551846
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    std::map<std::string,int> indict = {
        {"CH",0},
        {"CH2",1},
        {"CH2O",2},
        {"CH3",3},
        {"CH4",4},
        {"CO",5},
        {"CO2",6},
        {"H",7},
        {"H2",8},
        {"H2O",9},
        {"H2O2",10},
        {"HCO",11},
        {"HO2",12},
        {"N2",13},
        {"O",14},
        {"O2",15},
        {"OH",16},
        {"T",17},
    };
    std::map<std::string,int> outdict = {
        {"CH",0},
        {"CH2",1},
        {"CH2O",2},
        {"CH3",3},
        {"CH4",4},
        {"CO",5},
        {"CO2",6},
        {"H",7},
        {"H2",8},
        {"H2O",9},
        {"H2O2",10},
        {"HCO",11},
        {"HO2",12},
        {"O",13},
        {"O2",14},
        {"OH",15},

    };
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load("model.pt");
        std::cout << "sucessful load model\n";
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "ok\n";
    argList::addNote
    (
        "Solver for combustion with chemical reactions"
    );
    argList::addNote
    (
        "Solver for combustion with chemical reactions"
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createTimeControls.H"
    #include "initContinuityErrs.H"
    #include "createFields.H"
    #include "createFieldRefs.H"

    turbulence->validate();

    if (!LTS)
    {
        #include "compressibleCourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (runTime.run())
    {
        #include "readTimeControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "compressibleCourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        #include "rhoEqn.H"

        while (pimple.loop())
        {
            #include "UEqn.H"
            #include "YEqn.H"
            #include "EEqn.H"

            // --- Pressure corrector loop
            while (pimple.correct())
            {
                if (pimple.consistent())
                {
                    #include "pcEqn.H"
                }
                else
                {
                    #include "pEqn.H"
                }
            }

            if (pimple.turbCorr())
            {
                turbulence->correct();
            }
        }

        rho = thermo.rho();

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
