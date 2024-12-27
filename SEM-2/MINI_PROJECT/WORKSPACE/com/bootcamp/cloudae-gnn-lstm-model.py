import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

class GNNLSTM(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_graph_layers=2, num_lstm_layers=2):
        super(GNNLSTM, self).__init__()
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GCNConv(num_node_features, hidden_dim if i == 0 else hidden_dim)
            for i in range(num_graph_layers)
        ])
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index, batch):
        # Apply GNN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Reshape for LSTM
        # Assuming batch size x sequence length x features
        x = x.view(-1, sequence_length, x.size(-1))
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get last output
        last_output = lstm_out[:, -1, :]
        
        # Final prediction
        out = self.fc(last_output)
        return out

def prepare_data(df):
    """
    Prepare data for GNN-LSTM model
    """
    # Convert timestamp to datetime if not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Select features for the model
    features = ['temperature', 'humidity', 'pm1p0', 'pm4p0', 'pm10p0', 'voc', 'nox']
    
    # Create node features
    X = df[features].values
    
    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Create target variable (pm2.5 concentration)
    y = df['pm2p5'].values
    
    # Create edge index for GNN
    # Here we're creating a simple temporal graph where each node is connected
    # to its previous and next measurements
    num_nodes = len(df)
    edge_index = []
    for i in range(num_nodes - 1):
        edge_index.extend([[i, i+1], [i+1, i]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    print(f'edge_index : {edge_index}')
    return torch.FloatTensor(X), torch.FloatTensor(y), edge_index, scaler

def create_sequences(X, y, edge_index, sequence_length=12):
    """
    Create sequences for LSTM processing
    """
    sequences_X = []
    sequences_y = []
    sequences_edge = []
    
    for i in range(len(X) - sequence_length):
        # Get sequence of features
        seq_X = X[i:i+sequence_length]
        # Get target (next PM2.5 value)
        target = y[i+sequence_length]
        
        # Get corresponding edge indices for the sequence
        seq_edge = edge_index[:, (edge_index[0] >= i) & (edge_index[0] < i+sequence_length)]
        seq_edge = seq_edge - i  # Adjust indices to start from 0
        
        sequences_X.append(seq_X)
        sequences_y.append(target)
        sequences_edge.append(seq_edge)

    print(f'sequences_X : {sequences_X}')
    print(f'sequences_Y : {sequences_y}')
    print(f'sequences_edge : {sequences_edge}')

    return sequences_X, sequences_y, sequences_edge

def train_model(model, train_loader, optimizer, criterion, device):
    """
    Train the GNN-LSTM model
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        print(f'batch:{batch}')
        optimizer.zero_grad()
        
        # Move batch to device
        batch = batch.to(device)
        print(f'batch.edge_index:{batch.edge_index}')

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out.squeeze(), batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Example usage
def main():
    # Hyperparameters
    hidden_dim = 64
    num_graph_layers = 2
    num_lstm_layers = 2
    learning_rate = 0.001
    num_epochs = 100
    sequence_length = 12
    df = pd.read_csv(f'C:/Sanjeev/VNIT_CLASSES/mini-proj-dataset/pm2_5/Joburg_Kya_Sand_site_5_16.csv')
    # Prepare data
    X, y, edge_index, scaler = prepare_data(df)
    sequences_X, sequences_y, sequences_edge = create_sequences(X, y, edge_index, sequence_length)
    
    # Create PyTorch Geometric dataset
    dataset = []
    for X_seq, y_seq, edge_seq in zip(sequences_X, sequences_y, sequences_edge):
        data = Data(x=X_seq, edge_index=edge_seq, y=y_seq)
        dataset.append(data)
    
    # Split into train/test
    print(f'dataset: {len(dataset)}')

    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    print(f'train_dataset: {len(train_dataset)}')
    test_dataset = dataset[train_size:]
    print(f'test_dataset: {len(test_dataset)}')

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f'train_loader: {len(train_loader)}')

    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNLSTM(
        num_node_features=X.shape[1],
        hidden_dim=hidden_dim,
        num_graph_layers=num_graph_layers,
        num_lstm_layers=num_lstm_layers
    ).to(device)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')

if __name__ == "__main__":
    main()
