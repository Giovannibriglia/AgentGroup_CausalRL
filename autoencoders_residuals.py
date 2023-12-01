import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_pickle('sample_df_causality_0goals.pkl')
df_cols = df.columns.to_list()
n = -1

time_series_data = df[df_cols[0]][:n]
for feat in df_cols[1:]:
    time_series_data = np.vstack((time_series_data, df[feat][:n]))

# Assuming you have your time_series_data as a PyTorch tensor
# If not, you can convert it to a tensor using torch.tensor()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Set input and hidden dimensions
input_size = len(time_series_data[0])
HIDDEN_LAYERS = 128

# Create an instance of the autoencoder
autoencoder = Autoencoder(input_size, HIDDEN_LAYERS).to(device)

# Convert time_series_data to a PyTorch tensor
time_series_tensor = torch.tensor(time_series_data, dtype=torch.float32).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001)

# Training the autoencoder
num_epochs = 5000

pbar = tqdm(range(num_epochs))
for epoch in pbar:
    # Forward pass
    outputs = autoencoder(time_series_tensor)
    loss = criterion(outputs, time_series_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    pbar.set_postfix({'Loss ': loss.item()})

# Evaluate the autoencoder on the data
autoencoder.eval()

# Obtain predictions and compute residuals
with torch.no_grad():
    predictions = autoencoder(time_series_tensor)
    residuals = time_series_tensor - predictions

# Move residuals and predictions back to CPU if necessary
residuals = residuals.cpu().numpy()
predictions = predictions.cpu().numpy()

for i in range(len(residuals)):
    print(df_cols[i], ' --> ', np.mean(residuals[i]))

