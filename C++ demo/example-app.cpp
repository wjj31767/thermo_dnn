#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
int main() {

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("model.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

 // Create a vector of inputs.
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::randn({2, 18, 1}));

// Execute the model and turn its output into a tensor.
at::Tensor output = module.forward(inputs).toTensor();
std::cout << output << '\n';

}