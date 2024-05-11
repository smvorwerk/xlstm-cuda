// block_kernels.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>

#include "cuda_utils.h"

// xLSTM block forward pass kernel
template <typename T>
__global__ void xlstm_block_forward_kernel(const T *__restrict__ x,
                                           const T *__restrict__ h_prev,
                                           const T *__restrict__ c_prev,
                                           const T *__restrict__ C_prev,
                                           const T *__restrict__ n_prev,
                                           T *__restrict__ h,
                                           T *__restrict__ c,
                                           T *__restrict__ C,
                                           T *__restrict__ n,
                                           const T *__restrict__ w_proj,
                                           const T *__restrict__ w_gate,
                                           const T *__restrict__ b_proj,
                                           const T *__restrict__ b_gate,
                                           const T *__restrict__ w_slstm,
                                           const T *__restrict__ w_mlstm,
                                           const T *__restrict__ b_slstm,
                                           const T *__restrict__ b_mlstm,
                                           int batch_size,
                                           int input_size,
                                           int hidden_size,
                                           int proj_size,
                                           bool use_mlstm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < batch_size * hidden_size; i += stride)
    {
        int batch = i / hidden_size;
        int hidx = i % hidden_size;

        // Projection
        T proj = 0;
        for (int j = 0; j < input_size; j++)
        {
            proj += w_proj[hidx * input_size + j] * x[batch * input_size + j];
        }
        proj += b_proj[hidx];

        // Gate
        T gate = sigmoid(w_gate[hidx] * proj + b_gate[hidx]);

        // sLSTM or mLSTM
        if (use_mlstm)
        {
            // mLSTM forward pass
            T *w_k = w_mlstm + hidx * proj_size * 3;
            T *w_v = w_k + proj_size;
            T *w_q = w_v + proj_size;
            T *w_i = w_mlstm + hidx * proj_size * 6 + proj_size * 3;
            T *w_f = w_i + proj_size;
            T *w_o = w_f + proj_size;
            T *b_k = b_mlstm + hidx * 6;
            T *b_v = b_k + 1;
            T *b_q = b_v + 1;
            T *b_i = b_q + 1;
            T *b_f = b_i + 1;
            T *b_o = b_f + 1;

            mlstm_forward_kernel<T><<<1, 1>>>(x + batch * input_size,
                                              h_prev + batch * hidden_size,
                                              C_prev + i * hidden_size,
                                              n_prev + i,
                                              C + i * hidden_size,
                                              n + i,
                                              h + i,
                                              w_k, w_v, w_q,
                                              w_i, w_f, w_o,
                                              b_k, b_v, b_q,
                                              b_i, b_f, b_o,
                                              1, input_size, hidden_size);
        }
        else
        {
            // sLSTM forward pass
            T *w_i = w_slstm + hidx * proj_size * 4;
            T *w_f = w_i + proj_size;
            T *w_z = w_f + proj_size;
            T *w_o = w_z + proj_size;
            T *b_i = b_slstm + hidx * 4;
            T *b_f = b_i + 1;
            T *b_z = b_f + 1;
            T *b_o = b_z + 1;

            slstm_forward_kernel<T><<<1, 1>>>(x + batch * input_size,
                                              h_prev + batch * hidden_size,
                                              c_prev + i,
                                              n_prev + i,
                                              c + i,
                                              n + i,
                                              h + i,
                                              w_i, w_f, w_z, w_o,
                                              NULL, NULL, NULL, NULL,
                                              b_i, b_f, b_z, b_o,
                                              1, input_size, hidden_size);
        }

        // Apply gate
        h[i] = gate * h[i] + (1 - gate) * proj;
    }
}

// xLSTM block backward pass kernel
template <typename T>
__global__ void xlstm_block_backward_kernel(const T *__restrict__ grad_h,
                                            const T *__restrict__ h,
                                            const T *__restrict__ c,
                                            const T *__restrict__ C,
                                            const T *__restrict__ n,
                                            const T *__restrict__ x,
                                            const T *__restrict__ w_proj,
                                            const T *__restrict__ w_gate,
                                            const T *__restrict__ w_slstm,
                                            const T *__restrict__ w_mlstm,
                                            T *__restrict__ grad_x,
                                            T *__restrict__ grad_h_prev,
                                            T *__restrict__ grad_c_prev,
                                            T *__restrict__ grad_C_prev,
                                            T *__restrict__ grad_n_prev,
                                            T *__restrict__ grad_w_proj,
                                            T *__restrict__ grad_w_gate,
                                            T *__restrict__ grad_b_proj,
                                            T *__restrict__ grad_b_gate,
                                            T *__restrict__ grad_w_slstm,
                                            T *__restrict__ grad_w_mlstm,
                                            T *__restrict__ grad_b_slstm,
                                            T *__restrict__ grad_b_mlstm,
                                            int batch_size,
                                            int input_size,
                                            int hidden_size,
                                            int proj_size,
                                            bool use_mlstm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < batch_size * hidden_size; i += stride)
    {
        int batch = i / hidden_size;
        int hidx = i % hidden_size;

        // Gradient of gate
        T gate = sigmoid(w_gate[hidx] * h[i] + b_gate[hidx]);
        T grad_gate = grad_h[i] * (h[i] - x[batch * input_size + hidx]) * gate * (1 - gate);

        // Gradient of projection
        T grad_proj = grad_h[i] * (1 - gate);

        // sLSTM or mLSTM backward pass
        if (use_mlstm)
        {
            // mLSTM backward pass
            T *w_k = w_mlstm + hidx * proj_size * 3;
            T *w_v = w_k + proj_size;
            T *w_q = w_v + proj_size;
            T *w_i = w_mlstm + hidx * proj_size * 6 + proj_size * 3;
            T *w_f = w_i + proj_size;
            T *w_o = w_f + proj_size;
            T *grad_w_k = grad_w_mlstm + hidx * proj_size * 6;
            T *grad_w_v = grad_w_k + proj_size;
            T *grad_w_q = grad_w_v + proj_size;
            T *grad_w_i = grad_w_q + proj_size;
            T *grad_w_f = grad_w_i + proj_size;
            T *grad_w_o = grad_w_f + proj_size;
            T *grad_b_k = grad_b_mlstm + hidx * 6;
            T *grad_b_v = grad_b_k + 1;
            T *grad_b_q = grad_b_v + 1;
            T *grad_b_i = grad_b_q + 1;
            T *grad_b_f = grad_b_i + 1;
            T *grad_b_o = grad_b_f + 1;

            mlstm_backward_kernel<T><<<1, 1>>>(grad_h + i,
                                               C + i * hidden_size,
                                               n + i,
                                               x + batch * input_size,
                                               w_k, w_v, w_q,
                                               w_i, w_f, w_o,
                                               grad_x + batch * input_size,
                                               grad_C_prev + i * hidden_size,
                                               grad_n_prev + i,
                                               grad_w_k, grad_w_v, grad_w_q,
                                               grad_w_i, grad_w_f, grad_w_o,
                                               grad_b_k, grad_b_v, grad_b_q,
                                               grad_b_i, grad_b_f, grad_b_o,
                                               1, input_size, hidden_size);
        }
        else
        {
            // sLSTM backward pass
            T *w_i = w_slstm + hidx * proj_size * 4;
            T *w_f = w_i + proj_size;
            T *w_z = w_f + proj_size;
            T *w_o = w_z + proj_size;
            T *grad_w_i = grad_w_slstm + hidx * proj_size * 4;
            T *grad_w_f = grad_w_i + proj_size;
            T *grad_w_z = grad_w_f + proj_size;
            T *grad_w_o = grad_w_z + proj_size;
            T *grad_b_i = grad_b_slstm + hidx * 4;
            T *grad_b_f = grad_b_i + 1;
            T *grad_b_z = grad_b_f + 1;
            T *grad_b_o = grad_b_z + 1;

            slstm_backward_kernel<T><<<1, 1>>>(grad_h + i,
                                               grad_gate,
                                               c + i,
                                               n + i,
                                               NULL,
                                               NULL,
                                               x + batch * input_size,
                                               NULL,
                                               w_i, w_f, w_z, w_o,
                                               NULL, NULL, NULL, NULL,
                                               grad_x + batch * input_size,
                                               grad_h_prev + batch * hidden_size,
                                               grad_c_prev + i,
                                               grad_n_prev + i,
                                               grad_w_i, grad_w_f, grad_w_z, grad_w_o,
                                               NULL, NULL, NULL, NULL,
                                               grad_b_i, grad_b_f, grad_b_z, grad_b_o,
                                               1, input_size, hidden_size);
        }

        // Gradient of projection weights and bias
        for (int j = 0; j < input_size; j++)
        {
            atomicAdd(&grad_w_proj[hidx * input_size + j], grad_proj * x[batch * input_size + j]);
        }
        atomicAdd(&grad_b_proj[hidx], grad_proj);

        // Gradient of gate weights and bias
        atomicAdd(&grad_w_gate[hidx], grad_gate * h[i]);
        atomicAdd(&grad_b_gate[hidx], grad_gate);
    }
}

// Launch the xLSTM block forward pass kernel
template <typename T>
void launch_xlstm_block_forward(const T *x,
                                const T *h_prev,
                                const T *c_prev,
                                const T *C_prev,
                                const T *n_prev,
                                T *h,
                                T *c,
                                T *C,
                                T *n,
                                const T *w_proj,
                                const T *w_gate,
                                const T *b_proj,
                                const T *b_gate,
                                const T *w_slstm,
                                const T *w_mlstm,
                                const T *b_slstm,
                                const T *b_mlstm,
                                int batch_size,
                                int input_size,
                                int hidden_size,
                                int proj_size,
                                bool use_mlstm)
{
    dim3 block(256);
    dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);

    xlstm_block_forward_kernel<T><<<grid, block>>>(x, h_prev, c_prev, C_prev, n_prev,
                                                   h, c, C, n,
                                                   w_proj, w_gate,
                                                   b_proj, b_gate,
                                                   w_slstm, w_mlstm,
                                                   b_slstm, b_mlstm,
                                                   batch_size, input_size, hidden_size, proj_size,
                                                   use_mlstm);
}

// Launch the xLSTM block backward pass kernel
template <typename T>
void launch_xlstm_block_backward(const T *grad_h,
                                 const T *h,
                                 const T *c,
                                 const T *C,
                                 const T *n,
                                 const T *x,
                                 const T *w_proj,
                                 const T *w_gate,
                                 const T *w_slstm,
                                 const T *w_mlstm,
                                 T *grad_x,
                                 T *grad_h_prev,
                                 T *grad_c_prev,
                                 T *grad_C_prev,
                                 T *grad_n_prev,
                                 T *grad_w_proj,
                                 T *grad_w_gate,
                                 T *grad_b_proj,
                                 T *grad_b_gate,
                                 T *grad_w_slstm,
                                 T *grad_w_mlstm,
                                 T *grad_b_slstm,
                                 T *grad_b_mlstm,
                                 int batch_size,
                                 int input_size,
                                 int hidden_size,
                                 int proj_size,
                                 bool use_mlstm)
{
    dim3 block(256);
    dim3 grid((batch_size * hidden_size + block.x - 1) / block.x);

    xlstm_block_backward_kernel<T><<<grid, block>>>(grad_h,
                                                    h, c, C, n,
                                                    x,
                                                    w_proj, w_gate, w_slstm, w_mlstm,
                                                    grad_x,
                                                    grad_h_prev, grad_c_prev,
                                                    grad_C_prev, grad_n_prev,
                                                    grad_w_proj, grad_w_gate,
                                                    grad_b_proj, grad_b_gate,
                                                    grad_w_slstm, grad_w_mlstm,
                                                    grad_b_slstm, grad_b_mlstm,
                                                    batch_size, input_size, hidden_size, proj_size,
                                                    use_mlstm);
}
