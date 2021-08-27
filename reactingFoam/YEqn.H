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
    torch::Tensor featureTensor = torch::zeros({T.size(),6, 1});
    
    // Execute the model and turn its output into a tensor.
    // std::vector<scalar> output(Y.size());

    forAll(T, celli)
    {
       forAll(Y, i)
       {
           featureTensor[celli][dict_in[Y[i].name()]][0] = (Y[i][celli]-min_in[Y[i].name()])/(max_in[Y[i].name()]-min_in[Y[i].name()]);
        
       }
                
       featureTensor[celli][dict_in[T.name()]][0] = (T[celli]-min_in[T.name()])/(max_in[T.name()]-min_in[T.name()]);
    
        
               

    }
    std::vector<torch::jit::IValue> inputs{featureTensor};
        
    at::Tensor output = module.forward(inputs).toTensor();
            
    auto outputa = output.accessor<float,2>();
    forAll(T,celli)
    {
        forAll(Y,i)
        {
            volScalarField::Internal& RRi = mesh.lookupObjectRef<volScalarField::Internal>("RR."+Y[i].name());
           
            RRi[celli] = outputa[celli][dict_out[Y[i].name()]]*(max_in[Y[i].name()]-min_in[Y[i].name()])+min_in[Y[i].name()];
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
