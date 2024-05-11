import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from xlstm import PyXLSTMModel

# Define the RNN+xLSTM model for multivariate time series forecasting
class MultivariatexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, num_layers, use_mlstm_vec, output_size):
        super(MultivariatexLSTM, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.xlstm = PyXLSTMModel(hidden_size, hidden_size, proj_size, use_mlstm_vec, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        rnn_output, _ = self.rnn(input_seq)
        xlstm_output = self.xlstm(rnn_output)
        output = self.fc(xlstm_output)
        return output

# Set hyperparameters
input_size = 10
hidden_size = 128
proj_size = 64
num_layers = 3
use_mlstm_vec = [True, False, True]
output_size = 5
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# TODO: Find, Load, and Preprocess a multivariate time series dataset
train_dataset= ...
val_dataset = ...

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize the model
model = MultivariatexLSTM(input_size, hidden_size, proj_size, num_layers, use_mlstm_vec, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for input_seq, target_seq in val_loader:
            output = model(input_seq)
            loss = criterion(output, target_seq)
            total_loss += loss.item()
        avg_loss = total_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "multivariate_xlstm_model.pth")