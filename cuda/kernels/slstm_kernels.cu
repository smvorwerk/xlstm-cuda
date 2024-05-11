#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#include "cuda_utils.h"

// Define a small epsilon to prevent division by zero
#define epsilon 1e-7

// sLSTM forward pass kernel
template <typename T>
__global__ void slstm_forward_kernel(const T *__restrict__ x,
                                     const T *__restrict__ h_prev,
                                     const T *__restrict__ c_prev,
                                     const T *__restrict__ n_prev,
                                     T *__restrict__ c,
                                     T *__restrict__ n,
                                     T *__restrict__ h,
                                     const T *__restrict__ w_i,
                                     const T *__restrict__ w_f,
                                     const T *__restrict__ w_z,
                                     const T *__restrict__ w_o,
                                     const T *__restrict__ r_i,
                                     const T *__restrict__ r_f,
                                     const T *__restrict__ r_z,
                                     const T *__restrict__ r_o,
                                     const T *__restrict__ b_i,
                                     const T *__restrict__ b_f,
                                     const T *__restrict__ b_z,
                                     const T *__restrict__ b_o,
                                     int batch_size,
                                     int input_size,
                                     int hidden_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < batch_size * hidden_size; i += stride)
    {
        int batch = i / hidden_size;
        int hidx = i % hidden_size;

        T i_gate = sigmoid(w_i[hidx] * x[batch * input_size + hidx] +
                           r_i[hidx] * h_prev[batch * hidden_size + hidx] +
                           b_i[hidx]);
        T f_gate = sigmoid(w_f[hidx] * x[batch * input_size + hidx] +
                           r_f[hidx] * h_prev[batch * hidden_size + hidx] +
                           b_f[hidx]);
        T z_gate = tanh(w_z[hidx] * x[batch * input_size + hidx] +
                        r_z[hidx] * h_prev[batch * hidden_size + hidx] +
                        b_z[hidx]);
        T o_gate = sigmoid(w_o[hidx] * x[batch * input_size + hidx] +
                           r_o[hidx] * h_prev[batch * hidden_size + hidx] +
                           b_o[hidx]);

        c[i] = f_gate * c_prev[i] + i_gate * z_gate;
        n[i] = f_gate * n_prev[i] + i_gate;
        h[i] = o_gate * (c[i] / (n[i] + epsilon));
    }
}

// sLSTM backward pass kernel
template <typename T>
__global__ void slstm_backward_kernel(const T *__restrict__ grad_h,
                                      const T *__restrict__ grad_c,
                                      const T *__restrict__ c,
                                      const T *__restrict__ n,
                                      const T *__restrict__ c_prev,
                                      const T *__restrict__ n_prev,
                                      const T *__restrict__ x,
                                      const T *__restrict__ h_prev,
                                      const T *__restrict__ w_i,
                                      const T *__restrict__ w_f,
                                      const T *__restrict__ w_z,
                                      const T *__restrict__ w_o,
                                      const T *__restrict__ r_i,
                                      const T *__restrict__ r_f,
                                      const T *__restrict__ r_z,
                                      const T *__restrict__ r_o,
                                      T *__restrict__ grad_x,
                                      T *__restrict__ grad_h_prev,
                                      T *__restrict__ grad_c_prev,
                                      T *__restrict__ grad_n_prev,
                                      T *__restrict__ grad_w_i,
                                      T *__restrict__ grad_w_f,
                                      T *__restrict__ grad_w_z,
                                      T *__restrict__ grad_w_o,
                                      T *__restrict__ grad_r_i,
                                      T *__restrict__ grad_r_f,
                                      T *__restrict__ grad_r_z,
                                      T *__restrict__ grad_r_o,
                                      T *__restrict__ grad_b_i,
                                      T *__restrict__ grad_b_f,
                                      T *__restrict__ grad_b_z,
                                      T *__restrict__ grad_b_o,
                                      int batch_size,
                                      int input_size,
                                      int hidden_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < batch_size * hidden_size; i += stride)
    {
        int batch = i / hidden_size;
        int hidx = i % hidden_size;

        // Forward gate computations (need to be recalculated for backward pass)
        T i_gate = sigmoid(w_i[hidx] * x[batch * input_size + hidx] +
                           r_i[hidx] * h_prev[batch * hidden_size + hidx] +
                           b_i[hidx]);
        T f_gate = sigmoid(w_f[hidx] * x[batch * input_size + hidx] +
                           r_f[hidx] * h_prev[batch * hidden_size + hidx] +
                           b_f[hidx]);
        T z_gate = tanh(w_z[hidx] * x[batch * input_size + hidx] +
                        r_z[hidx] * h_prev[batch * hidden_size + hidx] +
                        b_z[hidx]);
        T o_gate = sigmoid(w_o[hidx] * x[batch * input_size + hidx] +
                           r_o[hidx] * h_prev[batch * hidden_size + hidx] +
                           b_o[hidx]);

        // Backpropagation through time (BPTT)
        T grad_o = grad_h[i] * (c[i] / (n[i] + epsilon));
        T grad_c_local = grad_h[i] * o_gate / (n[i] + epsilon) + grad_c[i];
        T grad_n_local = -grad_h[i] * o_gate * c[i] / ((n[i] + epsilon) * (n[i] + epsilon));

        T grad_z = grad_c_local * i_gate * (1 - z_gate * z_gate);
        T grad_i = grad_c_local * z_gate * i_gate * (1 - i_gate) + grad_n_local * i_gate * (1 - i_gate);
        T grad_f = grad_c_local * c_prev[i] * f_gate * (1 - f_gate) + grad_n_local * n_prev[i] * f_gate * (1 - f_gate);

        // Gradients w.r.t inputs and previous hidden states
        grad_x[batch * input_size + hidx] = grad_i * w_i[hidx] + grad_f * w_f[hidx] + grad_z * w_z[hidx] + grad_o * w_o[hidx];
        grad_h_prev[batch * hidden_size + hidx] = grad_i * r_i[hidx] + grad_f * r_f[hidx] + grad_z * r_z[hidx] + grad_o * r_o[hidx];
        grad_c_prev[i] = grad_c_local * f_gate;
        grad_n_prev[i] = grad_n_local * f_gate;

        // Accumulate gradients for weights and biases
        atomicAdd(&grad_w_i[hidx], grad_i * x[batch * input_size + hidx]);
        atomicAdd(&grad_w_f[hidx], grad_f * x[batch * input_size + hidx]);
        atomicAdd(&grad_w_z[hidx], grad_z * x[batch * input_size + hidx]);
        atomicAdd(&grad_w_o[hidx], grad_o * x[batch * input_size + hidx]);

        atomicAdd(&grad_r_i[hidx], grad_i * h_prev[batch * hidden_size + hidx]);
        atomicAdd(&grad_r_f[hidx], grad_f * h_prev[batch * hidden_size + hidx]);
        atomicAdd(&grad_r_z[hidx], grad_z * h_prev[batch * hidden_size + hidx]);
        atomicAdd(&grad_r_o[hidx], grad_o * h_prev[batch * hidden_size + hidx]);

        atomicAdd(&grad_b_i[hidx], grad_i);
        atomicAdd(&grad_b_f[hidx], grad_f);
        atomicAdd(&grad_b_z[hidx], grad_z);
        atomicAdd(&grad_b_o[hidx], grad_o);
    }
}

// Launch the sLSTM forward pass kernel
template <typename T>
void launch_slstm_forward(const T *x,
                          const T *h_prev,
                          const T *c_prev,
                          const T *n_prev,
                          T *c,
                          T *n,
                          T *h,
                          const T *w_i,
                          const T *w_f,
                          const T *w_z,
                          const T *w_o,
                          const T *r_i,
                          const T *r_f,
                          const T *r_z,
                          const T *r_o,
                          const T *b_i,
                          const T *b_f,
                          const T *b_z,
                          const T *b_o,
                          int batch_size,
                          int input_size,
                          int hidden_size)
{
    dim3 block(256);
    dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);

    slstm_forward_kernel<T><<<grid, block>>>(x, h_prev, c_prev, n_prev,
                                             c, n, h,
                                             w_i, w_f, w_z, w_o,
                                             r_i, r_f, r_z, r_o,
                                             b_i, b_f, b_z, b_o,
                                             batch_size, input_size, hidden_size);
}

// Launch the sLSTM backward pass kernel
template <typename T>
void launch_slstm_backward(const T *grad_h,
                           const T *grad_c,
                           const T *c,
                           const T *n,
                           const T *c_prev,
                           const T *n_prev,
                           const T *x,
                           const T *h_prev,
                           const T *w_i,
                           const T *w_f,
                           const T *w_z,
                           const T *w_o,
                           const T *r_i,
                           const T *r_f,
                           const T *r_z,
                           const T *r_o,
                           T *grad_x,
                           T *grad_h_prev,
                           T *grad_c_prev,
                           T *grad_n_prev,
                           T *grad_w_i,
                           T *grad_w_f,
                           T *grad_w_z,
                           T *grad_w_o,
                           T *grad_r_i,
                           T *grad_r_f,
                           T *grad_r_z,
                           T *grad_r_o,
                           T *grad_b_i,
                           T *grad_b_f,
                           T *grad_b_z,
                           T *grad_b_o,
                           int batch_size,
                           int input_size,
                           int hidden_size)
{
    dim3 block(256);
    dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);

    slstm_backward_kernel<T><<<grid, block>>>(grad_h, grad_c,
                                              c, n,
                                              c_prev, n_prev,
                                              x, h_prev,
                                              w_i, w_f, w_z, w_o,
                                              r_i, r_f, r_z, r_o,
                                              grad_x, grad_h_prev,
                                              grad_c_prev, grad_n_prev,
                                              grad_w_i, grad_w_f, grad_w_z, grad_w_o,
                                              grad_r_i, grad_r_f, grad_r_z, grad_r_o,
                                              grad_b_i, grad_b_f, grad_b_z, grad_b_o,
                                              batch_size, input_size, hidden_size);
}