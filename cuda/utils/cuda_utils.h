#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define CUDA_CHECK(err)                                                                                          \
    do                                                                                                           \
    {                                                                                                            \
        cudaError_t err_ = (err);                                                                                \
        if (err_ != cudaSuccess)                                                                                 \
        {                                                                                                        \
            fprintf(stderr, "CUDA error %d at %s:%d: %s\n", err_, __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(1);                                                                                             \
        }                                                                                                        \
    } while (0)

// Sigmoid activation function
template <typename T>
__device__ inline T sigmoid(T x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Tanh activation function
template <typename T>
__device__ inline T tanh(T x)
{
    return 2.0 / (1.0 + exp(-2.0 * x)) - 1.0;
}

// ReLU activation function
template <typename T>
__device__ inline T relu(T x)
{
    return max(static_cast<T>(0), x);
}

// Exponential linear unit (ELU) activation function
template <typename T>
__device__ inline T elu(T x, T alpha = 1.0)
{
    return x >= 0 ? x : alpha * (exp(x) - 1.0);
}

#endif // CUDA_UTILS_H