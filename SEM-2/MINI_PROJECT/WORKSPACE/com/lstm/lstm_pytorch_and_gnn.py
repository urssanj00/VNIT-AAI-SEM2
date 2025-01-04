import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_set_path = "C:/Sanjeev/VNIT_CLASSES/mini-proj-dataset/pm2_5/Johannesburg_Westdene_12.csv"
data = pd.read_csv(data_set_path)

# Convert timestamp to datetime and set as index
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
print(data.head())

# Select features (X) and target (y)
features = ['sensor_id', 'temperature', 'humidity', 'longitude', 'latitude']
target_column = 'pm2p5'

# Scale the features and target
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features + [target_column]])

# Prepare the data for LSTM
sequence_length = 10  # Lookback window

def create_sequences(data, target_idx, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length, :-1]
        label = data[i + seq_length, target_idx]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X, y = create_sequences(data_scaled, target_idx=-1, seq_length=sequence_length)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Custom PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)

# DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out






input_dim = len(features)
hidden_dim = 50
output_dim = 1
num_layers = 2

model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = []
    y_actual = []
    for X_batch, y_batch in test_loader:
        preds = model(X_batch)
        print(f"pred: {preds}")
        y_pred.extend(preds.squeeze().tolist())
        y_actual.extend(y_batch.tolist())

# Rescale predictions and actual values
y_pred_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), len(features))), np.array(y_pred).reshape(-1, 1)), axis=1))[:, -1]
y_test_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), len(features))), np.array(y_actual).reshape(-1, 1)), axis=1))[:, -1]

# Add predicted values to the original dataset
data['predicted_pm2p5'] = np.nan  # Initialize with NaN

# Get indices for the test set
test_indices = data.index[-len(y_test_rescaled):]

# Assign predictions to these indices
data.loc[test_indices, 'predicted_pm2p5'] = y_pred_rescaled

# Save the updated dataset to a new CSV file
output_csv_path = "C:/Sanjeev/VNIT_CLASSES/mini-proj-dataset/pm2_5/Johannesburg_Westdene_12_with_predictions.csv"
data.to_csv(output_csv_path)

print(f"Predictions saved to: {output_csv_path}")

# Convert predictions to binary classes for confusion matrix
threshold = 0.5
y_pred_binary = (y_pred_rescaled > threshold).astype(int)
y_test_binary = (y_test_rescaled > threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test_binary, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.png')

# Classification report
print(classification_report(y_test_binary, y_pred_binary))

# Save the model
torch.save(model.state_dict(), 'lstm_model.pth')



