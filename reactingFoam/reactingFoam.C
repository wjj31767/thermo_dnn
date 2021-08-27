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
    std::map<std::string,int> dict_in = {
        {"CH4",0},
        {"CO2",1},
        {"H2O",2},
        {"N2",3},
        {"O2",4},
        {"T",5},
    };
    std::map<std::string,float> min_in = {
        {"CH4",7.62566e-20},
        {"CO2",4.39207e-15},
        {"H2O",3.59575e-15},
        {"N2",2.16310e-08},
        {"O2",6.83314e-13},
        {"T",2.93000e+02},
    };
    std::map<std::string,float> max_in = {
        {"CH4",1.00000e+00},
        {"CO2",1.43191e-01},
        {"H2O",1.17229e-01},
        {"N2",7.70000e-01},
        {"O2",2.30000e-01},
        {"T",2.01586e+03},
    };
    std::map<std::string,float> max_out = {
        {"CH4",0.00000e+00},
        {"CO2",9.01711e+00},
        {"H2O",7.38223e+00},
        {"O2",0.00000e+00},
    };
    std::map<std::string,float> min_out = {
        {"CH4",-3.28697e+00},
        {"CO2",-2.13528e-15},
        {"H2O",-1.74813e-15},
        {"O2",-1.31123e+01},
    };
    std::map<std::string,int> dict_out = {
        {"CH4",0},
        {"CO2",1},
        {"H2O",2},
        {"O2",3},
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
