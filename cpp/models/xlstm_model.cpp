#include "xlstm_model.h"
#include "cuda_utils.h"

template <typename T>
XLSTMModel<T>::XLSTMModel(int input_size, int hidden_size, int proj_size,
                           const std::vector<bool>& use_mlstm_vec, int num_layers)
    : input_size_(input_size), hidden_size_(hidden_size), proj_size_(proj_size), num_layers_(num_layers) {
    allocate_memory();

    for (int i = 0; i < num_layers; ++i) {
        xlstm_blocks_.push_back(new XLSTMBlock<T>(i == 0 ? input_size : hidden_size,
                                                  hidden_size, proj_size, use_mlstm_vec[i]));
    }
}

template <typename T>
XLSTMModel<T>::~XLSTMModel() {
    free_memory();

    for (int i = 0; i < num_layers; ++i) {
        delete xlstm_blocks_[i];
    }
}

template <typename T>
void XLSTMModel<T>::forward(const T* input, T* output) {
    T* h_prev = h_states_[0];
    T* c_prev = c_states_[0];
    T* C_prev = C_states_[0];
    T* n_prev = n_states_[0];

    for (int i = 0; i < num_layers; ++i) {
        xlstm_blocks_[i]->forward(i == 0 ? input : h_states_[i - 1],
                                  h_prev, c_prev, C_prev, n_prev,
                                  h_states_[i], c_states_[i], C_states_[i], n_states_[i]);

        h_prev = h_states_[i];
        c_prev = c_states_[i];
        C_prev = C_states_[i];
        n_prev = n_states_[i];
    }

    CUDA_CHECK(cudaMemcpy(output, h_states_[num_layers - 1], hidden_size_ * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void XLSTMModel<T>::backward(const T* grad_output, T* grad_input) {
    CUDA_CHECK(cudaMemcpy(grad_h_states_[num_layers - 1], grad_output, hidden_size_ * sizeof(T), cudaMemcpyDeviceToDevice));

    for (int i = num_layers - 1; i >= 0; --i) {
        xlstm_blocks_[i]->backward(grad_h_states_[i],
                                   h_states_[i], c_states_[i], C_states_[i], n_states_[i],
                                   i == 0 ? nullptr : h_states_[i - 1],
                                   i == 0 ? grad_input : grad_h_states_[i - 1],
                                   grad_h_states_[i], grad_c_states_[i],
                                   grad_C_states_[i], grad_n_states_[i]);
    }
}

template <typename T>
void XLSTMModel<T>::allocate_memory() {
    for (int i = 0; i < num_layers; ++i) {
        T* h_state;
        T* c_state;
        T* C_state;
        T* n_state;
        T* grad_h_state;
        T* grad_c_state;
        T* grad_C_state;
        T* grad_n_state;

        CUDA_CHECK(cudaMalloc(&h_state, hidden_size_ * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&c_state, hidden_size_ * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&C_state, hidden_size_ * hidden_size_ * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&n_state, hidden_size_ * sizeof(T)));

        CUDA_CHECK(cudaMalloc(&grad_h_state, hidden_size_ * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&grad_c_state, hidden_size_ * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&grad_C_state, hidden_size_ * hidden_size_ * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&grad_n_state, hidden_size_ * sizeof(T)));

        h_states_.push_back(h_state);
        c_states_.push_back(c_state);
        C_states_.push_back(C_state);
        n_states_.push_back(n_state);

        grad_h_states_.push_back(grad_h_state);
        grad_c_states_.push_back(grad_c_state);
        grad_C_states_.push_back(grad_C_state);
        grad_n_states_.push_back(grad_n_state);
    }
}

template <typename T>
void XLSTMModel<T>::free_memory() {
    for (int i = 0; i < num_layers; ++i) {
        CUDA_CHECK(cudaFree(h_states_[i]));
        CUDA_CHECK(cudaFree(c_states_[i]));
        CUDA_CHECK(cudaFree(C_states_[i]));
        CUDA_CHECK(cudaFree(n_states_[i]));

        CUDA_CHECK(cudaFree(grad_h_states_[i]));
        CUDA_CHECK(cudaFree(grad_c_states_[i]));
        CUDA_CHECK(cudaFree(grad_C_states_[i]));
        CUDA_CHECK(cudaFree(grad_n_states_[i]));
    }
}

// Explicit instantiation for float and double types
template class XLSTMModel<float>;
template class XLSTMModel<double>;