import torch
import torch.nn as nn
from xlstm_cpp import MLSTMLayer

class PyMLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PyMLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer = MLSTMLayer(input_size, hidden_size)

    def forward(self, input, h_prev, C_prev, n_prev):
        h, C, n = torch.zeros((self.hidden_size,), device=input.device), torch.zeros((self.hidden_size, self.hidden_size), device=input.device), torch.zeros((self.hidden_size,), device=input.device)
        self.layer.forward(input.contiguous().data_ptr(),
                           h_prev.contiguous().data_ptr(),
                           C_prev.contiguous().data_ptr(),
                           n_prev.contiguous().data_ptr(),
                           h.data_ptr(),
                           C.data_ptr(),
                           n.data_ptr())
        return h, C, n

    def backward(self, grad_h, C, n, input):
        grad_input, grad_C_prev, grad_n_prev = torch.zeros_like(input), torch.zeros_like(C), torch.zeros_like(n)
        self.layer.backward(grad_h.contiguous().data_ptr(),
                            C.contiguous().data_ptr(),
                            n.contiguous().data_ptr(),
                            input.contiguous().data_ptr(),
                            grad_input.data_ptr(),
                            grad_C_prev.data_ptr(),
                            grad_n_prev.data_ptr())
        return grad_input, grad_C_prev, grad_n_prev