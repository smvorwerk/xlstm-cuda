import torch
import torch.nn as nn
from xlstm_cpp import XLSTMBlock
from xlstm.layers import PySLSTMLayer, PyMLSTMLayer

class PyXLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, use_mlstm):
        super(PyXLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.use_mlstm = use_mlstm
        self.block = XLSTMBlock(input_size, hidden_size, proj_size, use_mlstm)
        self.slstm_layer = PySLSTMLayer(proj_size, hidden_size) if not use_mlstm else None
        self.mlstm_layer = PyMLSTMLayer(proj_size, hidden_size) if use_mlstm else None

    def forward(self, input, h_prev, c_prev, C_prev, n_prev):
        h, c, C, n = torch.zeros((self.hidden_size,), device=input.device), torch.zeros((self.hidden_size,), device=input.device), torch.zeros((self.hidden_size, self.hidden_size), device=input.device), torch.zeros((self.hidden_size,), device=input.device)
        self.block.forward(input.contiguous().data_ptr(),
                           h_prev.contiguous().data_ptr(),
                           c_prev.contiguous().data_ptr(),
                           C_prev.contiguous().data_ptr(),
                           n_prev.contiguous().data_ptr(),
                           h.data_ptr(),
                           c.data_ptr(),
                           C.data_ptr(),
                           n.data_ptr())
        return h, c, C, n

    def backward(self, grad_h, h, c, C, n, input):
        grad_input, grad_h_prev, grad_c_prev, grad_C_prev, grad_n_prev = torch.zeros_like(input), torch.zeros_like(h), torch.zeros_like(c), torch.zeros_like(C), torch.zeros_like(n)
        self.block.backward(grad_h.contiguous().data_ptr(),
                            h.contiguous().data_ptr(),
                            c.contiguous().data_ptr(),
                            C.contiguous().data_ptr(),
                            n.contiguous().data_ptr(),
                            input.contiguous().data_ptr(),
                            grad_input.data_ptr(),
                            grad_h_prev.data_ptr(),
                            grad_c_prev.data_ptr(),
                            grad_C_prev.data_ptr(),
                            grad_n_prev.data_ptr())
        return grad_input, grad_h_prev, grad_c_prev, grad_C_prev, grad_n_prev