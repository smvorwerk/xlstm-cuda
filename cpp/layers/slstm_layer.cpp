#include "slstm_layer.h"
#include "cuda_utils.h"

template <typename T>
SLSTMLayer<T>::SLSTMLayer(int input_size, int hidden_size)
    : input_size_(input_size), hidden_size_(hidden_size) {
    allocate_memory();
}

template <typename T>
SLSTMLayer<T>::~SLSTMLayer() {
    free_memory();
}

template <typename T>
void SLSTMLayer<T>::forward(const T* input, const T* h_prev, const T* c_prev, const T* n_prev,
                             T* h, T* c, T* n) {
    launch_slstm_forward(input, h_prev, c_prev, n_prev,
                         c, n, h,
                         w_i_, w_f_, w_z_, w_o_,
                         r_i_, r_f_, r_z_, r_o_,
                         b_i_, b_f_, b_z_, b_o_,
                         1, input_size_, hidden_size_);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void SLSTMLayer<T>::backward(const T* grad_h, const T* grad_c, const T* c, const T* n,
                              const T* c_prev, const T* n_prev, const T* input, const T* h_prev,
                              T* grad_input, T* grad_h_prev, T* grad_c_prev, T* grad_n_prev) {
    launch_slstm_backward(grad_h, grad_c,
                          c, n,
                          c_prev, n_prev,
                          input, h_prev,
                          w_i_, w_f_, w_z_, w_o_,
                          r_i_, r_f_, r_z_, r_o_,
                          grad_input, grad_h_prev,
                          grad_c_prev, grad_n_prev,
                          grad_w_i_, grad_w_f_, grad_w_z_, grad_w_o_,
                          grad_r_i_, grad_r_f_, grad_r_z_, grad_r_o_,
                          grad_b_i_, grad_b_f_, grad_b_z_, grad_b_o_,
                          1, input_size_, hidden_size_);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void SLSTMLayer<T>::allocate_memory() {
    // Allocate memory for weights and biases
    CUDA_CHECK(cudaMalloc(&w_i_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&w_f_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&w_z_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&w_o_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&r_i_, hidden_size_ * hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&r_f_, hidden_size_ * hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&r_z_, hidden_size_ * hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&r_o_, hidden_size_ * hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_i_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_f_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_z_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_o_, hidden_size_ * sizeof(T)));

    // Allocate memory for gradient weights and biases
    CUDA_CHECK(cudaMalloc(&grad_w_i_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_w_f_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_w_z_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_w_o_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_r_i_, hidden_size_ * hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_r_f_, hidden_size_ * hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_r_z_, hidden_size_ * hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_r_o_, hidden_size_ * hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_i_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_f_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_z_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_o_, hidden_size_ * sizeof(T)));
}

template <typename T>
void SLSTMLayer<T>::free_memory() {
    // Free memory for weights and biases
    CUDA_CHECK(cudaFree(w_i_));
    CUDA_CHECK(cudaFree(w_f_));
    CUDA_CHECK(cudaFree(w_z_));
    CUDA_CHECK(cudaFree(w_o_));
    CUDA_CHECK(cudaFree(r_i_));
    CUDA_CHECK(cudaFree(r_f_));
    CUDA_CHECK(cudaFree(r_z_));
    CUDA_CHECK(cudaFree(r_o_));
    CUDA_CHECK(cudaFree(b_i_));
    CUDA_CHECK(cudaFree(b_f_));
    CUDA_CHECK(cudaFree(b_z_));
    CUDA_CHECK(cudaFree(b_o_));

    // Free memory for gradient weights and biases
    CUDA_CHECK(cudaFree(grad_w_i_));
    CUDA_CHECK(cudaFree(grad_w_f_));
    CUDA_CHECK(cudaFree(grad_w_z_));
    CUDA_CHECK(cudaFree(grad_w_o_));
    CUDA_CHECK(cudaFree(grad_r_i_));
    CUDA_CHECK(cudaFree(grad_r_f_));
    CUDA_CHECK(cudaFree(grad_r_z_));
    CUDA_CHECK(cudaFree(grad_r_o_));
    CUDA_CHECK(cudaFree(grad_b_i_));
    CUDA_CHECK(cudaFree(grad_b_f_));
    CUDA_CHECK(cudaFree(grad_b_z_));
    CUDA_CHECK(cudaFree(grad_b_o_));
}

// Explicit instantiation for float and double types
template class SLSTMLayer<float>;
template class SLSTMLayer<double>;