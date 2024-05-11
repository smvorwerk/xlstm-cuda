#ifndef XLSTM_BLOCK_H
#define XLSTM_BLOCK_H

#include "slstm_layer.h"
#include "mlstm_layer.h"

template <typename T>
class XLSTMBlock {
public:
    XLSTMBlock(int input_size, int hidden_size, int proj_size, bool use_mlstm);
    ~XLSTMBlock();

    void forward(const T* input, const T* h_prev, const T* c_prev, const T* C_prev, const T* n_prev,
                 T* h, T* c, T* C, T* n);
    void backward(const T* grad_h, const T* h, const T* c, const T* C, const T* n, const T* input,
                  T* grad_input, T* grad_h_prev, T* grad_c_prev, T* grad_C_prev, T* grad_n_prev);

private:
    int input_size_;
    int hidden_size_;
    int proj_size_;
    bool use_mlstm_;

    T* w_proj_;
    T* w_gate_;
    T* b_proj_;
    T* b_gate_;
    T* grad_w_proj_;
    T* grad_w_gate_;
    T* grad_b_proj_;
    T* grad_b_gate_;

    SLSTMLayer<T>* slstm_layer_;
    MLSTMLayer<T>* mlstm_layer_;

    void allocate_memory();
    void free_memory();
};

#endif // XLSTM_BLOCK_H