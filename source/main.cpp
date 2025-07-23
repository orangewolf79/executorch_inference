#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <iostream>
#include <fstream>
#include <torch/torch.h>

using namespace ::executorch::extension;

int main() {
  // load all imagenet labels into vector using input file stream
  std::ifstream input("imagenet_classes.txt");
  std::string line;
  std::vector<std::string> labels;
  while (getline(input, line)) {
    labels.push_back(line);
  }

  // load the model and the preprocessed image tensor
  Module module("model.pte");
  torch::Tensor image_tensor;
  try {
    torch::load(image_tensor, "image_tensor.pt");
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load tensor" << std::endl;
    return -1;
  }

  // convert torch::Tensor to extension::Tensor using from_blob and perform inference
  float* data_ptr = image_tensor.data_ptr<float>();
  auto et_image_tensor = from_blob(data_ptr, {1, 3, 224, 224}); // required shape (1, 3, 224, 224)
  const auto result = module.forward(et_image_tensor); // inference step
  if (!result.ok()) {
    std::cerr << "Error running inference" << std::endl;
  }
  const auto output = result->at(0).toTensor();

  // manual argmax - executorch tensors do not have it by default
  size_t n = output.numel();
  const float* ptr_output = output.const_data_ptr<float>();
  float max_value = 0;
  size_t max_idx = 0;
  for (size_t i=0; i<n; ++i) {
    if (ptr_output[i] > max_value) {
        max_value = ptr_output[i];
        max_idx = i;
    }
  }
  
  std::cout << "Prediction: " << labels[max_idx] << std::endl;


}