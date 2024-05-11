#include <iostream>
#include "mlstm_layer.h"
#include "utils.h"

int main() {
    int input_size = 10;
    int hidden_size = 20;

    MLSTMLayer<float> mlstm_layer(input_size, hidden_size);

    // Initialize input and previous states
    std::vector<float> input(input_size, 1.0f);
    std::vector<float> h_prev(hidden_size, 0.0f);
    std::vector<float> C_prev(hidden_size * hidden_size, 0.0f);
    std::vector<float> n_prev(hidden_size, 0.0f);

    // Allocate memory for output states
    std::vector<float> h(hidden_size);
    std::vector<float> C(hidden_size * hidden_size);
    std::vector<float> n(hidden_size);

    // Copy input data to device
    float* d_input;
    float* d_h_prev;
    float* d_C_prev;
    float* d_n_prev;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_h_prev, hidden_size * sizeof(float));
    cudaMalloc(&d_C_prev, hidden_size * hidden_size * sizeof(float));
    cudaMalloc(&d_n_prev, hidden_size * sizeof(float));
    copy_host_to_device(input, d_input);
    copy_host_to_device(h_prev, d_h_prev);
    copy_host_to_device(C_prev, d_C_prev);
    copy_host_to_device(n_prev, d_n_prev);

    // Perform forward pass
    mlstm_layer.forward(d_input, d_h_prev, d_C_prev, d_n_prev,
                        d_h_prev, d_C_prev, d_n_prev);

    // Copy output data to host
    copy_device_to_host(d_h_prev, h);
    copy_device_to_host(d_C_prev, C);
    copy_device_to_host(d_n_prev, n);

    // Print output
    std::cout << "Output (h): ";
    for (int i = 0; i < hidden_size; ++i) {
        std::cout << h[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_h_prev);
    cudaFree(d_C_prev);
    cudaFree(d_n_prev);

    return 0;
}