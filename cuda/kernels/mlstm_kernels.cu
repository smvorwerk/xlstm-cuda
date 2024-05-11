#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#include "cuda_utils.h"

// mLSTM forward pass kernel
template <typename T>
__global__ void mlstm_forward_kernel(const T *__restrict__ x,
                                     const T *__restrict__ h_prev,
                                     const T *__restrict__ C_prev,
                                     const T *__restrict__ n_prev,
                                     T *__restrict__ C,
                                     T *__restrict__ n,
                                     T *__restrict__ h,
                                     const T *__restrict__ w_k,
                                     const T *__restrict__ w_v,
                                     const T *__restrict__ w_q,
                                     const T *__restrict__ w_i,
                                     const T *__restrict__ w_f,
                                     const T *__restrict__ w_o,
                                     const T *__restrict__ b_k,
                                     const T *__restrict__ b_v,
                                     const T *__restrict__ b_q,
                                     const T *__restrict__ b_i,
                                     const T *__restrict__ b_f,
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

        T k = w_k[hidx] * x[batch * input_size + hidx] + b_k[hidx];
        T v = w_v[hidx] * x[batch * input_size + hidx] + b_v[hidx];
        T q = w_q[hidx] * x[batch * input_size + hidx] + b_q[hidx];

        T i_gate = sigmoid(w_i[hidx] * x[batch * input_size + hidx] + b_i[hidx]);
        T f_gate = sigmoid(w_f[hidx] * x[batch * input_size + hidx] + b_f[hidx]);
        T o_gate = sigmoid(w_o[hidx] * x[batch * input_size + hidx] + b_o[hidx]);

        for (int j = 0; j < hidden_size; j++)
        {
            C[i * hidden_size + j] = f_gate * C_prev[i * hidden_size + j] + i_gate * v * k;
        }

        n[i] = f_gate * n_prev[i] + i_gate * k;

        T sum = 0;
        for (int j = 0; j < hidden_size; j++)
        {
            sum += C[i * hidden_size + j] * q;
        }

        h[i] = o_gate * sum / max(abs(n[i]), 1.0);
    }
}

// mLSTM backward pass kernel
template <typename T>
__global__ void mlstm_backward_kernel(const T *__restrict__ grad_h,
                                      const T *__restrict__ C,
                                      const T *__restrict__ n,
                                      const T *__restrict__ x,
                                      const T *__restrict__ w_k,
                                      const T *__restrict__ w_v,
                                      const T *__restrict__ w_q,
                                      const T *__restrict__ w_i,
                                      const T *__restrict__ w_f,
                                      const T *__restrict__ w_o,
                                      T *__restrict__ grad_x,
                                      T *__restrict__ grad_C_prev,
                                      T *__restrict__ grad_n_prev,
                                      T *__restrict__ grad_w_k,
                                      T *__restrict__ grad_w_v,
                                      T *__restrict__ grad_w_q,
                                      T *__restrict__ grad_w_i,
                                      T *__restrict__ grad_w_f,
                                      T *__restrict__ grad_w_o,
                                      T *__restrict__ grad_b_k,
                                      T *__restrict__ grad_b_v,
                                      T *__restrict__ grad_b_q,
                                      T *__restrict__ grad_b_i,
                                      T *__restrict__ grad_b_f,
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

        T k = w_k[hidx] * x[batch * input_size + hidx] + b_k[hidx];
        T v = w_v[hidx] * x[batch * input_size + hidx] + b_v[hidx];
        T q = w_q[hidx] * x[batch * input_size + hidx] + b_q[hidx];

        T i_gate = sigmoid(w_i[hidx] * x[batch * input_size + hidx] + b_i[hidx]);
        T f_gate = sigmoid(w_f[hidx] * x[batch * input_size + hidx] + b_f[hidx]);
        T o_gate = sigmoid(w_o[hidx] * x[batch * input_size + hidx] + b_o[hidx]);

        T sum = 0;
        for (int j = 0; j < hidden_size; j++)
        {
            sum += C[i * hidden_size + j] * q;
        }

        T grad_o = grad_h[i] * sum / max(abs(n[i]), 1.0);
        T grad_sum = grad_h[i] * o_gate / max(abs(n[i]), 1.0);

        for (int j = 0; j < hidden_size; j++)
        {
            grad_C_prev[i * hidden_size + j] = grad_sum * q;
        }

        T grad_v = grad_sum * i_gate * k;
        T grad_k = grad_sum * i_gate * v;
        T grad_q = grad_sum * sum / q;

        T grad_i = grad_sum * v * k * i_gate * (1 - i_gate);
        T grad_f = grad_sum * (C[i * hidden_size] - v * k) * f_gate * (1 - f_gate);

        grad_n_prev[i] = grad_sum * f_gate;

        grad_x[batch * input_size + hidx] = grad_k * w_k[hidx] + grad_v * w_v[hidx] + grad_q * w_q[hidx] +
                                            grad_i * w_i[hidx] + grad_f * w_f[hidx] + grad_o * w_o[hidx];

        atomicAdd(&grad_w_k[hidx], grad_k * x[batch * input_size + hidx]);
        atomicAdd(&grad_w_v[hidx], grad_v * x[batch * input_size + hidx]);
        atomicAdd(&grad_w_q[hidx], grad_q * x[batch * input_size + hidx]);
        atomicAdd(&grad_w_i[hidx], grad_i * x[batch * input_size + hidx]);
        atomicAdd(&grad_w_f[hidx], grad_f * x[batch * input_size + hidx]);
        atomicAdd(&grad_w_o[hidx], grad_o * x[batch * input_size + hidx]);

        atomicAdd(&grad_b_k[hidx], grad_k);
        atomicAdd(&grad_b_v[hidx], grad_v);
        atomicAdd(&grad_b_q[hidx], grad_q);
        atomicAdd(&grad_b_i[hidx], grad_i);
        atomicAdd(&grad_b_f[hidx], grad_f);
        atomicAdd(&grad_b_o[hidx], grad_o);
    }
}

// Launch the mLSTM forward pass kernel
template <typename T>
void launch_mlstm_forward(const T *x,
                          const T *h_prev,
                          const T *C_prev,
                          const T *n_prev,
                          T *C,
                          T *n,
                          T *h,
                          const T *w_k,
                          const T *w_v,
                          const T *w_q,
                          const T *w_i,
                          const T *w_f,
                          const T *w_o,
                          const T *b_k,
                          const T *b_v,
                          const T *b_q,
                          const T *b_i,
                          const T *b_f,
                          const T *b_o,
                          int batch_size,
                          int input_size,
                          int hidden_size)
{
    dim3 block(256);
    dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);

    mlstm_forward_kernel<T><<<grid, block>>>(x, h_prev, C_prev, n_prev,
                                             C, n, h,
                                             w_k, w_v, w_q,
                                             w_i, w_f, w_o,
                                             b_k, b_v, b_q,
                                             b_i, b_f, b_o,
                                             batch_size, input_size, hidden_size);
}

// Launch the mLSTM backward pass kernel
template <typename T>
void launch_mlstm_backward(const T *grad_h,
                           const T *C,
                           const T *n,
                           const T *x,
                           const T *w_k,
                           const T *w_v,
                           const T *w_q,
                           const T *w_i,
                           const T *w_f,
                           const T *w_o,
                           T *grad_x,
                           T *grad_C_prev,
                           T *grad_n_prev,
                           T *grad_w_k,
                           T *grad_w_v,
                           T *grad_w_q,
                           T *grad_w_i,
                           T *grad_w_f,
                           T *grad_w_o,
                           T *grad_b_k,
                           T *grad_b_v,
                           T *grad_b_q,
                           T *grad_b_i,
                           T *grad_b_f,
                           T *grad_b_o,
                           int batch_size,
                           int input_size,
                           int hidden_size)
{
    dim3 block(256);
    dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);

    mlstm_backward_kernel<T><<<grid, block>>>(grad_h,
                                              C, n,
                                              x,
                                              w_k, w_v, w_q,
                                              w_i, w_f, w_o,
                                              grad_x,
                                              grad_C_prev, grad_n_prev,
                                              grad_w_k, grad_w_v, grad_w_q,
                                              grad_w_i, grad_w_f, grad_w_o,
                                              grad_b_k, grad_b_v, grad_b_q,
                                              grad_b_i, grad_b_f, grad_b_o,
                                              batch_size, input_size, hidden_size);
}