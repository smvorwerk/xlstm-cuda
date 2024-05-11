#ifndef SLSTM_LAYER_H
#define SLSTM_LAYER_H

#include <vector>

template <typename T>
class SLSTMLayer {
public:
    SLSTMLayer(int input_size, int hidden_size);
    ~SLSTMLayer();

    void forward(const T* input, const T* h_prev, const T* c_prev, const T* n_prev,
                 T* h, T* c, T* n);
    void backward(const T* grad_h, const T* grad_c, const T* c, const T* n,
                  const T* c_prev, const T* n_prev, const T* input, const T* h_prev,
                  T* grad_input, T* grad_h_prev, T* grad_c_prev, T* grad_n_prev);

private:
    int input_size_;
    int hidden_size_;

    T* w_i_;
    T* w_f_;
    T* w_z_;
    T* w_o_;
    T* r_i_;
    T* r_f_;
    T* r_z_;
    T* r_o_;
    T* b_i_;
    T* b_f_;
    T* b_z_;
    T* b_o_;

    T* grad_w_i_;
    T* grad_w_f_;
    T* grad_w_z_;
    T* grad_w_o_;
    T* grad_r_i_;
    T* grad_r_f_;
    T* grad_r_z_;
    T* grad_r_o_;
    T* grad_b_i_;
    T* grad_b_f_;
    T* grad_b_z_;
    T* grad_b_o_;

    void allocate_memory();
    void free_memory();
};

#endif // SLSTM_LAYER_H