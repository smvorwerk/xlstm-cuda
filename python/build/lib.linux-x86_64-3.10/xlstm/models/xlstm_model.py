import torch
import torch.nn as nn
from xlstm_cpp import XLSTMModel
from xlstm.blocks import PyXLSTMBlock

class PyXLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, use_mlstm_vec, num_layers):
        super(PyXLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.use_mlstm_vec = use_mlstm_vec
        self.num_layers = num_layers
        self.model = XLSTMModel(input_size, hidden_size, proj_size, use_mlstm_vec, num_layers)
        self.xlstm_blocks = nn.ModuleList([PyXLSTMBlock(input_size if i == 0 else hidden_size, hidden_size, proj_size, use_mlstm_vec[i]) for i in range(num_layers)])

    def forward(self, input):
        output = torch.zeros((self.hidden_size,), device=input.device)
        self.model.forward(input.contiguous().data_ptr(), output.data_ptr())
        return output

    def backward(self, grad_output):
        grad_input = torch.zeros_like(grad_output)
        self.model.backward(grad_output.contiguous().data_ptr(), grad_input.data_ptr())
        return grad_input