#include <iostream>
#include "xlstm_block.h"
#include "utils.h"

int main() {
    int input_size = 10;
    int hidden_size = 20;
    int proj_size = 15;
    bool use_mlstm = true;
    int seq_length = 5;

    XLSTMBlock<float> xlstm_block(input_size, hidden_size, proj_size, use_mlstm);

    // Initialize input and previous states
    std::vector<float> input(seq_length * input_size);
    std::vector<float> h_prev(hidden_size, 0.0f);
    std::vector<float> c_prev(hidden_size, 0.0f);
    std::vector<float> C_prev(hidden_size * hidden_size, 0.0f);
    std::vector<float> n_prev(hidden_size, 0.0f);

    // Generate random input sequence
    for (int i = 0; i < seq_length * input_size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory for output states
    std::vector<float> h(hidden_size);
    std::vector<float> c(hidden_size);
    std::vector<float> C(hidden_size * hidden_size);
    std::vector<float> n(hidden_size);

    // Copy input data to device
    float* d_input;
    float* d_h_prev;
    float* d_c_prev;
    float* d_C_prev;
    float* d_n_prev;
    cudaMalloc(&d_input, seq_length * input_size * sizeof(float));
    cudaMalloc(&d_h_prev, hidden_size * sizeof(float));
    cudaMalloc(&d_c_prev, hidden_size * sizeof(float));
    cudaMalloc(&d_C_prev, hidden_size * hidden_size * sizeof(float));
    cudaMalloc(&d_n_prev, hidden_size * sizeof(float));
    copy_host_to_device(input, d_input);
    copy_host_to_device(h_prev, d_h_prev);
    copy_host_to_device(c_prev, d_c_prev);
    copy_host_to_device(C_prev, d_C_prev);
    copy_host_to_device(n_prev, d_n_prev);

    // Process input sequence
    for (int i = 0; i < seq_length; ++i) {
        xlstm_block.forward(d_input + i * input_size, d_h_prev, d_c_prev, d_C_prev, d_n_prev,
                            d_h_prev, d_c_prev, d_C_prev, d_n_prev);
    }

    // Copy output data to host
    copy_device_to_host(d_h_prev, h);

    // Print final hidden state
    std::cout << "Final hidden state: ";
    for (int i = 0; i < hidden_size; ++i) {
        std::cout << h[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_h_prev);
    cudaFree(d_c_prev);
    cudaFree(d_C_prev);
    cudaFree(d_n_prev);

    return 0;
}