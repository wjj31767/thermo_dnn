# use C++ pytorch api

### C++ demo for pytorch

1. run evaluate.py generate "model.pt"

2. download torch https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcpu.zip

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
./example-app
```

### OpenFOAM changes for fit pytorch
[OpenFOAM Pytorch Github](https://github.com/AndreWeiner/of_pytorch_docker)

This repo shows how to merge pytorch and OpenFOAM.

my repo based on this.

[install OpenFOAM 2012 with sudo](https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/debian)

```
sudo su # to super user
foam # goto folder under Openfoam2012
vi wmake/rules/General/Gcc/c++
#add -D_GLIBCXX_USE_CXX11_ABI=0 at end of third line
# you will see CC = g++ -std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0
# compiler openfoam again
```

reactingfoam folder and counterFlowFlame2D folder derived from openfoam2012

did some changes for merge openfoam and pytorch

### Pytorch training based on results of counterFlowFlame2D

The dataset for pytorch is getting from normal counterFlowFlame2D's results.

1. Model part consists of linear neurons now. Plan to increase number of neurons to see if it's fitter or not.
2. rectification of dataset or not: Current results shows rectification of dataset make the fitting work worse.
3. Why model is 18 inputs and 16 outputs? Future plan is train based on 18 inputs and 16 outputs dataset.