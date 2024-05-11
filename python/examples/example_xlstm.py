import torch
from xlstm import PyXLSTMModel

# Define the xLSTM model
input_size = 10
hidden_size = 64
proj_size = 32
use_mlstm_vec = [True, False, True]
num_layers = len(use_mlstm_vec)
model = PyXLSTMModel(input_size, hidden_size, proj_size, use_mlstm_vec, num_layers)

# Prepare the input data
seq_length = 5
batch_size = 3
input_data = torch.randn(batch_size, seq_length, input_size)

# Forward pass
output = model(input_data)

# Print the output
print("Input shape:", input_data.shape)
print("Output shape:", output.shape)