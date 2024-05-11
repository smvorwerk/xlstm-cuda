#include "xlstm_block.h"
#include "cuda_utils.h"

template <typename T>
XLSTMBlock<T>::XLSTMBlock(int input_size, int hidden_size, int proj_size, bool use_mlstm)
    : input_size_(input_size), hidden_size_(hidden_size), proj_size_(proj_size), use_mlstm_(use_mlstm) {
    allocate_memory();

    if (use_mlstm) {
        mlstm_layer_ = new MLSTMLayer<T>(proj_size, hidden_size);
    } else {
        slstm_layer_ = new SLSTMLayer<T>(proj_size, hidden_size);
    }
}

template <typename T>
XLSTMBlock<T>::~XLSTMBlock() {
    free_memory();

    if (use_mlstm_) {
        delete mlstm_layer_;
    } else {
        delete slstm_layer_;
    }
}

template <typename T>
void XLSTMBlock<T>::forward(const T* input, const T* h_prev, const T* c_prev, const T* C_prev, const T* n_prev,
                             T* h, T* c, T* C, T* n) {
    launch_xlstm_block_forward(input, h_prev, c_prev, C_prev, n_prev,
                               h, c, C, n,
                               w_proj_, w_gate_,
                               b_proj_, b_gate_,
                               slstm_layer_->get_weights(), mlstm_layer_->get_weights(),
                               slstm_layer_->get_biases(), mlstm_layer_->get_biases(),
                               1, input_size_, hidden_size_, proj_size_,
                               use_mlstm_);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void XLSTMBlock<T>::backward(const T* grad_h, const T* h, const T* c, const T* C, const T* n, const T* input,
                              T* grad_input, T* grad_h_prev, T* grad_c_prev, T* grad_C_prev, T* grad_n_prev) {
    launch_xlstm_block_backward(grad_h,
                                h, c, C, n,
                                input,
                                w_proj_, w_gate_,
                                slstm_layer_->get_weights(), mlstm_layer_->get_weights(),
                                grad_input,
                                grad_h_prev, grad_c_prev,
                                grad_C_prev, grad_n_prev,
                                grad_w_proj_, grad_w_gate_,
                                grad_b_proj_, grad_b_gate_,
                                slstm_layer_->get_grad_weights(), mlstm_layer_->get_grad_weights(),
                                slstm_layer_->get_grad_biases(), mlstm_layer_->get_grad_biases(),
                                1, input_size_, hidden_size_, proj_size_,
                                use_mlstm_);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void XLSTMBlock<T>::allocate_memory() {
    // Allocate memory for weights and biases
    CUDA_CHECK(cudaMalloc(&w_proj_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&w_gate_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_proj_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&b_gate_, hidden_size_ * sizeof(T)));

    // Allocate memory for gradient weights and biases
    CUDA_CHECK(cudaMalloc(&grad_w_proj_, hidden_size_ * input_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_w_gate_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_proj_, hidden_size_ * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&grad_b_gate_, hidden_size_ * sizeof(T)));
}

template <typename T>
void XLSTMBlock<T>::free_memory() {
    // Free memory for weights and biases
    CUDA_CHECK(cudaFree(w_proj_));
    CUDA_CHECK(cudaFree(w_gate_));
    CUDA_CHECK(cudaFree(b_proj_));
    CUDA_CHECK(cudaFree(b_gate_));

    // Free memory for gradient weights and biases
    CUDA_CHECK(cudaFree(grad_w_proj_));
    CUDA_CHECK(cudaFree(grad_w_gate_));
    CUDA_CHECK(cudaFree(grad_b_proj_));
    CUDA_CHECK(cudaFree(grad_b_gate_));
}

// Explicit instantiation for float and double types
template class XLSTMBlock<float>;
template class XLSTMBlock<double>;