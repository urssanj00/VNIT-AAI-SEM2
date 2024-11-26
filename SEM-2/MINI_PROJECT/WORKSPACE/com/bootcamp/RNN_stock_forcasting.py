import numpy as np
import pandas as pd
import numpy as no
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

data = pd.read_csv("../dataset/stock/AMZN.csv")
print(data.head())

data['Date'] = pd.to_datetime(data['Date'])

plt.plot(data['Date'], data['Close'])

#plt.show()
#LSTM is looking at History

from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range (1, n_steps+1):
        df[f'Close(t-{i}'] = df['Close'].shift(i)

    df.dropna(inplace=True)
    return df


lookback = 7
shifted_df = prepare_dataframe_for_lstm(data,lookback)

print(shifted_df)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))
shifted_df_as_np = scaler.fit_transform(shifted_df)

print(shifted_df_as_np)

X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

X = dc(np.flip(X, axis=1))
print(f'X.shape :{X.shape} || Y.shape : {y.shape}')

split_index = int(len(X) * 0.95)
print(f'split_index :{split_index} ')

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print(f'X_train.shape:{X_train.shape}, X_test.shape:{X_test.shape},y_train.shape:{y_train.shape}, y_test.shape:{y_test.shape}')

# wrapping in pytorch tensors
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

print(f'Tensor Wrapped [ X_train.shape:{X_train.shape} X_test.shape:{X_test.shape} \n y_train.shape:{y_train.shape} y_test.shape:{y_test.shape} ] ')

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

from torch.utils.data import DataLoader

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Check and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

# This defines a class LSTM that inherits from torch.nn.Module.
# nn.Module is the base class for all neural networks in PyTorch.
# It provides methods and functionality for defining and working with models.
class LSTM(nn.Module):
    # input_size = Number of Features
    # hidden_size = Hidden Layer_1 size
    # num_stacked_layers: The number of stacked LSTM layers (default is 1).
    def __init__(self, input_size=1, hidden_size=4, num_stacked_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        # Defines an LSTM layer:
        # batch_first=True: Indicates that the input tensor has the shape (batch_size, sequence_length, input_size).
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)

        # Defines a fully connected(linear) layer:
        # hidden_size: The number of input features to the linear layer(output from the LSTM).
        # 1: The number of output features( for regression or binary classification tasks).
        self.fc = nn.Linear(hidden_size, 1)

    # The forward method defines how data flows through the network during the forward pass.
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _=self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_one_epoch():
    model.train(True)
    print(f'Epoch : {epoch+1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss = running_loss + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99: # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1, avg_loss_across_batches))
            running_loss = 0.0

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('*****************************************')
    print()

model = LSTM(1, 4, 1)
model.to(device)
print(f'LSTM Model is : {model}')

learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

