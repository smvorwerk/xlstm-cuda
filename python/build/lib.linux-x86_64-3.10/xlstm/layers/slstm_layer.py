import torch
import torch.nn as nn
from xlstm_cpp import SLSTMLayer

class PySLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PySLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer = SLSTMLayer(input_size, hidden_size)

    def forward(self, input, h_prev, c_prev, n_prev):
        h, c, n = torch.zeros((self.hidden_size,), device=input.device), torch.zeros((self.hidden_size,), device=input.device), torch.zeros((self.hidden_size,), device=input.device)
        self.layer.forward(input.contiguous().data_ptr(),
                           h_prev.contiguous().data_ptr(),
                           c_prev.contiguous().data_ptr(),
                           n_prev.contiguous().data_ptr(),
                           h.data_ptr(),
                           c.data_ptr(),
                           n.data_ptr())
        return h, c, n

    def backward(self, grad_h, grad_c, c, n, c_prev, n_prev, input, h_prev):
        grad_input, grad_h_prev, grad_c_prev, grad_n_prev = torch.zeros_like(input), torch.zeros_like(h_prev), torch.zeros_like(c_prev), torch.zeros_like(n_prev)
        self.layer.backward(grad_h.contiguous().data_ptr(),
                            grad_c.contiguous().data_ptr(),
                            c.contiguous().data_ptr(),
                            n.contiguous().data_ptr(),
                            c_prev.contiguous().data_ptr(),
                            n_prev.contiguous().data_ptr(),
                            input.contiguous().data_ptr(),
                            h_prev.contiguous().data_ptr(),
                            grad_input.data_ptr(),
                            grad_h_prev.data_ptr(),
                            grad_c_prev.data_ptr(),
                            grad_n_prev.data_ptr())
        return grad_input, grad_h_prev, grad_c_prev, grad_n_prev