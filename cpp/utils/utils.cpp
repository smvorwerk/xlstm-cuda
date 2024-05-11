#include "utils.h"
#include <random>
#include <cuda_runtime.h>

template <typename T>
void xavier_init(T* weights, int input_size, int output_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    T scale = std::sqrt(2.0 / (input_size + output_size));
    std::normal_distribution<T> dist(0.0, scale);

    for (int i = 0; i < input_size * output_size; ++i) {
        weights[i] = dist(gen);
    }
}

template <typename T>
void zero_init(T* biases, int size) {
    for (int i = 0; i < size; ++i) {
        biases[i] = static_cast<T>(0);
    }
}

template <typename T>
void copy_host_to_device(const std::vector<T>& host_data, T* device_data) {
    cudaMemcpy(device_data, host_data.data(), host_data.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void copy_device_to_host(const T* device_data, std::vector<T>& host_data) {
    cudaMemcpy(host_data.data(), device_data, host_data.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

// Explicit instantiation for float and double types
template void xavier_init<float>(float* weights, int input_size, int output_size);
template void xavier_init<double>(double* weights, int input_size, int output_size);
template void zero_init<float>(float* biases, int size);
template void zero_init<double>(double* biases, int size);
template void copy_host_to_device<float>(const std::vector<float>& host_data, float* device_data);
template void copy_host_to_device<double>(const std::vector<double>& host_data, double* device_data);
template void copy_device_to_host<float>(const float* device_data, std::vector<float>& host_data);
template void copy_device_to_host<double>(const double* device_data, std::vector<double>& host_data);