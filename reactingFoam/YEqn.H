tmp<fv::convectionScheme<scalar>> mvConvection
(
    fv::convectionScheme<scalar>::New
    (
        mesh,
        fields,
        phi,
        mesh.divScheme("div(phi,Yi_h)")
    )
);

{
    // reaction->correct();
    torch::Tensor featureTensor = torch::zeros({T.size(),18, 1});
    
    // Execute the model and turn its output into a tensor.
    // std::vector<scalar> output(Y.size());

    forAll(T, celli)
    {
       forAll(Y, i)
       {
           featureTensor[celli][indict[Y[i].name()]][0] = Y[i][celli];
        
       }
                
       featureTensor[celli][indict[T.name()]][0] = T[celli];
    
        
               

    }
    std::vector<torch::jit::IValue> inputs{featureTensor};
        
    at::Tensor output = module.forward(inputs).toTensor();
            
    auto outputa = output.accessor<float,2>();
    forAll(T,celli)
    {
        forAll(Y,i)
        {
            volScalarField::Internal& RRi = mesh.lookupObjectRef<volScalarField::Internal>("RR."+Y[i].name());
           
            RRi[celli] = outputa[celli][outdict[Y[i].name()]];
        }
    }
        
    Qdot = reaction->Qdot();
    volScalarField Yt(0.0*Y[0]);

    forAll(Y, i)
    {
        if (i != inertIndex && composition.active(i))
        {
            volScalarField& Yi = Y[i];

            fvScalarMatrix YiEqn
            (
                fvm::ddt(rho, Yi)
              + mvConvection->fvmDiv(phi, Yi)
              - fvm::laplacian(turbulence->muEff(), Yi)
             ==
                reaction->R(Yi)
              + fvOptions(rho, Yi)
            );

            YiEqn.relax();

            fvOptions.constrain(YiEqn);

            YiEqn.solve(mesh.solver("Yi"));

            fvOptions.correct(Yi);

            Yi.max(0.0);
            Yt += Yi;
        }
    }

    Y[inertIndex] = scalar(1) - Yt;
    Y[inertIndex].max(0.0);
}
