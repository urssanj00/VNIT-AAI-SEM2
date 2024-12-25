import pandas as pd
from Util import Util
from PropertiesConfig import PropertiesConfig as PC
import torch
import numpy as np
from scipy.spatial import distance_matrix
#from GraphLevelGNN import GraphLevelGNN
from GraphLevelGNNWithLSTM import GraphLevelGNNWithLSTM
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import torch.nn as nn
import os
from sklearn.preprocessing import MinMaxScaler
import random
import string
from BayesianOptimization import BayesianOptimization
from sklearn.model_selection import  train_test_split

class LatticeImpl:
    def __init__(self):
        properties_config = PC()
        properties = properties_config.get_properties_config()
        self.plot_path = properties['plot_path']
        self.dataset_path = properties['data_set_path']
        self.edge_index_threshold = float(properties['edge_index_threshold'])
        self.graph_list = []
        self.pd_list = []
        self.coordinate_to_index = {}  # Mapping of coordinates to global indices
        self.features = [
            'temperature', 'humidity', 'year', 'month', 'day', 'hour', 'weekday', 'is_weekend'
        ]
        self.targets = ['pm2p5']
        self.gnn_model = GraphLevelGNNWithLSTM(in_channels=len(self.features))


    def timstamp_to_numeric(self, df_obj):
        df = df_obj
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)  # Remove timezone info
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday  # 0=Monday, 6=Sunday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        return df

    def load_graph_from_individual_dataset(self, file_name, all_sensor_coordinates):
        df_obj = pd.read_csv(f'{self.dataset_path}/{file_name}')
        df_obj = self.timstamp_to_numeric(df_obj)

        df_obj['timestamp'] = pd.to_datetime(df_obj['timestamp'])
        df_obj = df_obj.sort_values(by=['sensor_id', 'year', 'month', 'day', 'hour']).reset_index(drop=True)

        df_obj[self.features] = df_obj[self.features].apply(pd.to_numeric, errors='coerce')

        # Step 2: Feature Scaling
        scaler = MinMaxScaler()
        df_obj[self.features] = scaler.fit_transform(df_obj[self.features])
        df_obj.fillna(0, inplace=True)

        x = torch.tensor(df_obj[self.features].values, dtype=torch.float)
        y = torch.tensor(df_obj[self.targets].values, dtype=torch.float)

        # Collect coordinates from the dataset
        sensor_coordinates = LatticeImpl.load_sensor_coordinates(df_obj)
        all_sensor_coordinates.extend(sensor_coordinates)

        self.pd_list.append(df_obj)
        df_obj, edge_index = self.create_edge_index_for_individual_graph(df_obj)

        df_obj['sensor_index'] = df_obj[['longitude', 'latitude']].apply(
            lambda row: self.coordinate_to_index.get(tuple(row), -1), axis=1
        )

        graph = Data(x=x, edge_index=edge_index, y=y)
        self.graph_list.append(graph)
        return graph, all_sensor_coordinates

    def load_dataset(self):
        list_files = Util.list_files_of_a_dir(self.dataset_path)

        # Collect all unique coordinates across all datasets
        all_sensor_coordinates = []

        for file_name in list_files:
            if 'station' not in file_name:
                graph, all_sensor_coordinates = self.load_graph_from_individual_dataset(file_name, all_sensor_coordinates)

        unique_coordinates = np.unique(all_sensor_coordinates, axis=0)
        self.coordinate_to_index = {tuple(coord): idx for idx, coord in enumerate(unique_coordinates)}

        edge_index_for_graph_of_graphs = self.create_edge_index_for_graph_of_graphs(unique_coordinates)

        self.add_sensor_index()

        # Create a mapping of coordinates to their index
        coordinate_to_index = {tuple(coord): idx for idx, coord in enumerate(unique_coordinates)}
        #print(f'coordinate_to_index : {coordinate_to_index}')

    def do_train(self):
        # Step 5: Train the Model
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.0381587687863735)
        loss_fn = torch.nn.MSELoss()

        best_loss = float('inf')  # Initialize best loss as infinity
        best_model_path = "best_gnn_model.pkl"  # Path to save the best model

        for graph in self.graph_list:
            for epoch in range(99):
                self.gnn_model.train()
                optimizer.zero_grad()

                # Forward pass
                out = self.gnn_model(graph).squeeze()

                # Compute loss
                loss = loss_fn(out, graph.y.squeeze())

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gnn_model.parameters(), max_norm=1.0)
                optimizer.step()

                # Check if this is the best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(self.gnn_model.state_dict(), best_model_path)  # Save model state
                    print(f"Saved best model with loss: {best_loss}")

                # Periodic logging
                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {loss.item()}')

        print(f"Training completed. Best model saved at {best_model_path} with loss {best_loss}.")

    def do_predict(self):
        self.gnn_model.eval()
        predictions = []
        targets = []
        losses = []
        criterion = nn.MSELoss()

        for graph in self.graph_list:
            with torch.no_grad():
                output = self.gnn_model(graph).squeeze()
                target = graph.y.squeeze()
                loss = criterion(output, target)

                losses.append(loss.item())
                predictions.append(output)
                targets.append(target)

                predictions_np = output.numpy()
                unique_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                output_path = f'{self.dataset_path}/station/_predictions_{unique_filename}.csv'
                self.save_pred_to_csv(graph, predictions_np, output_path)

        avg_loss = sum(losses) / len(losses)
        print(f'Average Loss: {avg_loss}')
        return predictions, targets

    def save_pred_to_csv(self, graph, predictions_np, output_path):
       # x_data = graph.x.numpy()
       # y_data = graph.y.numpy()
        # Convert tensors to NumPy arrays
        x_data = graph.x.numpy() if isinstance(graph.x, torch.Tensor) else graph.x
        y_data = graph.y.numpy() if isinstance(graph.y, torch.Tensor) else graph.y

        df = pd.DataFrame(x_data, columns=self.features)
        df['actual_target'] = y_data.squeeze()
        df['predicted_target'] = predictions_np.squeeze()

        file_exists = os.path.exists(output_path)
        df.to_csv(output_path, mode='a', header=not file_exists, index=False)
        print(f"Results saved to {output_path}")

    def add_sensor_index(self):
        for df in self.pd_list:
            df['sensor_index'] = df[['longitude', 'latitude']].apply(
                lambda row: self.coordinate_to_index.get(tuple(np.round(row, decimals=5)), -1), axis=1
            )

    def create_edge_index_for_individual_graph(self, df_obj):
        # Ensure the dataframe has enough rows to create edges
        num_rows = len(df_obj)
        #df_obj = df_obj.sort_values(by=['sensor_id', 'pm2p5']).reset_index(drop=True)
        if num_rows < 2:
            print("Not enough rows to form edges.")
            return torch.tensor([[], []], dtype=torch.long)  # Empty edge_index

        edge_source, edge_target = [], []

        # Connect each row to the next one (excluding the last row)
        for i in range(num_rows - 1):  # Avoid the last row since it has no next row
            edge_source.append(i)
            edge_target.append(i + 1)

        # Convert the edge source and target lists to a tensor
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        #print(f'edge_index : {edge_index}')
        return df_obj, edge_index



    def create_edge_index_for_graph_of_graphs(self, dataset_coordinates):
        edge_source, edge_target = [], []
        dist_matrix = distance_matrix(dataset_coordinates, dataset_coordinates)

        for i in range(len(dataset_coordinates)):
            for j in range(i + 1, len(dataset_coordinates)):
                dist = dist_matrix[i, j]
                if dist < self.edge_index_threshold:
                    edge_source.extend([i, j])
                    edge_target.extend([j, i])

        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        return edge_index

    @staticmethod
    def load_sensor_coordinates(df_obj):
        df_obj['longitude'] = pd.to_numeric(df_obj['longitude'], errors='coerce')
        df_obj['latitude'] = pd.to_numeric(df_obj['latitude'], errors='coerce')
        df_obj.dropna(subset=['longitude', 'latitude'], inplace=True)
        return df_obj[['longitude', 'latitude']].values

    def create_edge_index(self, coordinates, coordinate_to_index):
        # Initialize empty lists for the edge index
        edge_source = []
        edge_target = []

        # Calculate the distance matrix between all sensor coordinates
        dist_matrix = distance_matrix(coordinates, coordinates)

        # Iterate through all pairs of coordinates
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):  # Check only pairs i, j where i < j
                dist = dist_matrix[i, j]

                if dist < self.edge_index_threshold:  # Only add edges if within threshold
                    edge_source.append(coordinate_to_index[tuple(coordinates[i])])
                    edge_target.append(coordinate_to_index[tuple(coordinates[j])])
                    edge_source.append(coordinate_to_index[tuple(coordinates[j])])
                    edge_target.append(coordinate_to_index[tuple(coordinates[i])])

        # Convert the lists to a PyTorch tensor and return the edge index
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        return edge_index


latticeImpl = LatticeImpl()
latticeImpl.load_dataset()

#bayesianOptimization = BayesianOptimization()
#for graph in latticeImpl.graph_list:
#bayesianOptimization.tune_hyperparameters(latticeImpl.graph_list, len(latticeImpl.features))

latticeImpl.do_train()
latticeImpl.do_predict()

