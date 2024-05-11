#include <iostream>
#include "xlstm_model.h"
#include "utils.h"

int main() {
    int input_size = 10;
    int hidden_size = 20;
    int proj_size = 15;
    std::vector<bool> use_mlstm_vec = {true, false, true};
    int num_layers = use_mlstm_vec.size();
    int seq_length = 5;

    XLSTMModel<float> xlstm_model(input_size, hidden_size, proj_size, use_mlstm_vec, num_layers);

    // Initialize input
    std::vector<float> input(seq_length * input_size);
    std::vector<float> output(hidden_size);

    // Generate random input sequence
    for (int i = 0; i < seq_length * input_size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Copy input data to device
    float* d_input;
    cudaMalloc(&d_input, seq_length * input_size * sizeof(float));
    copy_host_to_device(input, d_input);

    // Process input sequence
    for (int i = 0; i < seq_length; ++i) {
        xlstm_model.forward(d_input + i * input_size, d_input + i * input_size);
    }

    // Copy output data to host
    copy_device_to_host(d_input + (seq_length - 1) * input_size, output);

    // Print final output
    std::cout << "Final output: ";
    for (int i = 0; i < hidden_size; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_input);

    return 0;
}