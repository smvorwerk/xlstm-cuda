#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <vector>

// Initialize weights with Xavier initialization
template <typename T>
void xavier_init(T* weights, int input_size, int output_size);

// Initialize biases with zeros
template <typename T>
void zero_init(T* biases, int size);

// Copy data from host to device
template <typename T>
void copy_host_to_device(const std::vector<T>& host_data, T* device_data);

// Copy data from device to host
template <typename T>
void copy_device_to_host(const T* device_data, std::vector<T>& host_data);

#endif // UTILS_H