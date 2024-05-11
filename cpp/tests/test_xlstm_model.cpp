#include <iostream>
#include "xlstm_model.h"
#include "utils.h"

int main() {
    int input_size = 10;
    int hidden_size = 20;
    int proj_size = 15;
    std::vector<bool> use_mlstm_vec = {true, false, true};
    int num_layers = use_mlstm_vec.size();

    XLSTMModel<float> xlstm_model(input_size, hidden_size, proj_size, use_mlstm_vec, num_layers);

    // Initialize input
    std::vector<float> input(input_size, 1.0f);
    std::vector<float> output(hidden_size);

    // Copy input data to device
    float* d_input;
    cudaMalloc(&d_input, input_size * sizeof(float));
    copy_host_to_device(input, d_input);

    // Perform forward pass
    xlstm_model.forward(d_input, d_input);

    // Copy output data to host
    copy_device_to_host(d_input, output);

    // Print output
    std::cout << "Output: ";
    for (int i = 0; i < hidden_size; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_input);

    return 0;
}