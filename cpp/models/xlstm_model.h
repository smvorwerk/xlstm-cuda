#ifndef XLSTM_MODEL_H
#define XLSTM_MODEL_H

#include <vector>
#include "xlstm_block.h"

template <typename T>
class XLSTMModel {
public:
    XLSTMModel(int input_size, int hidden_size, int proj_size,
               const std::vector<bool>& use_mlstm_vec, int num_layers);
    ~XLSTMModel();

    void forward(const T* input, T* output);
    void backward(const T* grad_output, T* grad_input);

private:
    int input_size_;
    int hidden_size_;
    int proj_size_;
    int num_layers_;

    std::vector<XLSTMBlock<T>*> xlstm_blocks_;

    std::vector<T*> h_states_;
    std::vector<T*> c_states_;
    std::vector<T*> C_states_;
    std::vector<T*> n_states_;

    std::vector<T*> grad_h_states_;
    std::vector<T*> grad_c_states_;
    std::vector<T*> grad_C_states_;
    std::vector<T*> grad_n_states_;

    void allocate_memory();
    void free_memory();
};

#endif // XLSTM_MODEL_H