#include "mlstm_layer.h"
#include "cuda_utils.h"

template <typename T>
MLSTMLayer<T>::MLSTMLayer(int input_size, int hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {
    allocate_memory();
}

template <typename T>
MLSTMLayer<T>::~MLSTMLayer() {
    free_memory();
}

template <typename T>
void MLSTMLayer<T>::forward(const T* input, const T* h_prev, const T* C_prev, const T* n_prev,
                             T* h, T* C, T* n) {
    launch_mlstm_forward(input, h_prev, C_prev, n_prev,
                         C, n, h,
                         w_k_, w_v_, w_q_,
                         w_i_, w_f_, w_o_,
                         b_k_, b_v_, b_q_,
                         b_i_, b_f_, b_o_,
                         1, input_size_, hidden_size_);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void MLSTMLayer<T>::backward(const T* grad_h, const T* C, const T* n, const T* input,
                              T* grad_input, T* grad_C_prev, T* grad_n_prev) {
    launch_mlstm_backward(grad_h,
                          C, n,
                          input,
                          w_k_, w_v_, w_q_,
                          w_i_, w_f_, w_o_,
                          grad_input,
                          grad_C_prev, grad_n_prev,
                          grad_w_k_, grad_w_v_, grad_w_q_,
                          grad_w_i_, grad_w_f_, grad_w_o_,
                          grad_b_k_, grad_b_v_, grad_b_q_,
                          grad_b_i_, grad_b_f_, grad_b_o_,
                          1, input_size_, hidden_size_);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void MLSTMLayer<T>::allocate_memory() {
    // Allocate memory for weights and biases
    CUDA_CHECK(cudaMalloc(&w_k_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&w_v_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&w_q_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&w_i_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&w_f_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&w_o_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_k_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_v_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_q_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_i_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_f_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_o_, hidden_size_ * sizeof(T)));

    // Allocate memory for gradient weights and biases
    CUDA_CHECK(cudaMalloc(&grad_w_k_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_w_v_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_w_q_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_w_i_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_w_f_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_w_o_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_k_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_v_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_q_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_i_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_f_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_o_, hidden_size_ * sizeof(T)));
}

template <typename T>
void MLSTMLayer<T>::free_memory() {
    // Free memory for weights and biases
    CUDA_CHECK(cudaFree(w_k_));
    CUDA_CHECK(cudaFree(w_v_));
    CUDA_CHECK(cudaFree(w_q_));
    CUDA_CHECK(cudaFree(w_i_));
    CUDA_CHECK(cudaFree(w_f_));
    CUDA_CHECK(cudaFree(w_o_));
    CUDA_CHECK(cudaFree(b_k_));
    CUDA_CHECK(cudaFree(b_v_));
    CUDA_CHECK(cudaFree(b_q_));
    CUDA_CHECK(cudaFree(b_i_));
    CUDA_CHECK(cudaFree(b_f_));
    CUDA_CHECK(cudaFree(b_o_));

    // Free memory for gradient weights and biases
    CUDA_CHECK(cudaFree(grad_w_k_));
    CUDA_CHECK(cudaFree(grad_w_v_));
    CUDA_CHECK(cudaFree(grad_w_q_));
    CUDA_CHECK(cudaFree(grad_w_i_));
    CUDA_CHECK(cudaFree(grad_w_f_));
    CUDA_CHECK(cudaFree(grad_w_o_));
    CUDA_CHECK(cudaFree(grad_b_k_));
    CUDA_CHECK(cudaFree(grad_b_v_));
    CUDA_CHECK(cudaFree(grad_b_q_));
    CUDA_CHECK(cudaFree(grad_b_i_));
    CUDA_CHECK(cudaFree(grad_b_f_));
    CUDA_CHECK(cudaFree(grad_b_o_));
}

// Explicit instantiation for float and double types
template class MLSTMLayer<float>;
template class MLSTMLayer<double>;