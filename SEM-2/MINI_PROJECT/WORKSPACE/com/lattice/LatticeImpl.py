import pandas as pd
from Util import Util
from PropertiesConfig import PropertiesConfig as PC
import torch
import numpy as np
from scipy.spatial import distance_matrix
from GraphLevelGNN import  GraphLevelGNN
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import torch.nn as nn
import os
from sklearn.preprocessing import MinMaxScaler
import random
import string

# Author Sanjeev Pandey
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
        self.features = ['temperature', 'humidity', 'year', 'month', 'day', 'hour', 'weekday', 'is_weekend'] #['temperature', 'humidity'] #
        self.targets = ['pm2p5']
        self.gnn_model = GraphLevelGNN(in_channels=len(self.features), hidden_channels=64, out_channels=len(self.targets))

    def timstamp_to_numeric(self, df_obj):
        df = df_obj
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)  # Remove timezone info
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday  # 0=Monday, 6=Sunday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
      # print(f"{df['timestamp']}, {df['year']},{df['month']},{df['day']} ,{df['hour']}")
        return df

    def load_dataset(self):
        list_files = Util.list_files_of_a_dir(self.dataset_path)

        # Collect all unique coordinates across all datasets
        all_sensor_coordinates = []

        for file_name in list_files:
            if 'station' not in file_name:
                df_obj = pd.read_csv(f'{self.dataset_path}/{file_name}')

                df_obj = self.timstamp_to_numeric(df_obj)

                # Ensure numeric data and process dataset
                df_obj['timestamp'] = pd.to_datetime(df_obj['timestamp'])
                df_obj = df_obj.sort_values(by=['sensor_id', 'timestamp']).reset_index(drop=True)

                df_obj[self.features] = df_obj[self.features].apply(pd.to_numeric, errors='coerce')
                # Step 2: Feature Scaling
                scaler = MinMaxScaler()
                # ['temperature', 'humidity']
                df_obj[self.features] = scaler.fit_transform(df_obj[self.features])

                df_obj.fillna(0, inplace=True)
                x = df_obj[self.features]
                # Convert it to a tensor if it's a dataframe
                x = torch.tensor(x.values, dtype=torch.float)  # Convert to tensor of shape

                y = df_obj[self.targets]
                y = torch.tensor(y.values, dtype=torch.float)  # Convert to tensor of shape

                # Collect coordinates from the dataset
                sensor_coordinates = LatticeImpl.load_sensor_coordinates(df_obj)
                all_sensor_coordinates.extend(sensor_coordinates)

                self.pd_list.append(df_obj)
                edge_index = self.create_edge_index_for_individual_graph(df_obj)

                # Map local coordinates to global indices
                df_obj['sensor_index'] = df_obj[['longitude', 'latitude']].apply(
                    lambda row: self.coordinate_to_index.get(tuple(row), -1), axis=1)

                graph = Data(x=x, edge_index=edge_index, y=y)
                self.graph_list.append(graph)

        # Remove duplicate coordinates and keep only unique ones
        unique_coordinates = np.unique(all_sensor_coordinates, axis=0)
        #print(f'unique_coordinates : {unique_coordinates}')

        self.coordinate_to_index = {tuple(coord): idx for idx, coord in enumerate(unique_coordinates)}

        edge_index_for_graph_of_graphs = self.create_edge_index_for_graph_of_graphs(unique_coordinates)

        self.add_sensor_index()

        # Create a mapping of coordinates to their index
        coordinate_to_index = {tuple(coord): idx for idx, coord in enumerate(unique_coordinates)}
        #print(f'coordinate_to_index : {coordinate_to_index}')

    def do_train(self):
        # Step 5: Train the Model
        optimizer = torch.optim.Adam(self.gnn_model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        prev_loss = None
        for graph in self.graph_list:
            for epoch in range(100):
                self.gnn_model.train()
                optimizer.zero_grad()
                out = self.gnn_model(graph).squeeze()
                loss = loss_fn(out, graph.y)
                loss.backward()

                # Apply gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.gnn_model.parameters(), max_norm=1.0)
                optimizer.step()

                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

                # Early stopping check (if loss stops improving)
                if epoch > 10 and loss.item() > prev_loss:
                    print(f"Early stopping triggered.{loss.item()}")
                    #break
                prev_loss = loss.item()

    def do_predict(self):
        self.gnn_model.eval()  # Set model to evaluation mode
        predictions = []
        targets = []
        losses = []
        criterion = nn.MSELoss()  # Use an appropriate loss function (e.g., MSE for regression)
        i = 1
        for graph in self.graph_list:
            with torch.no_grad():  # Disable gradient computation for prediction
                output = self.gnn_model(graph)

                # Assuming that y (targets) are available in the graph
                target = graph.y  # Target values
                #print(f"Predicted: {output}, Actual: {target}")

                # Calculate the loss (compare with actual)
                loss = criterion(output, target)
                losses.append(loss.item())

                predictions.append(output)
                targets.append(target)

                # Convert the predictions from tensor to numpy for easy DataFrame integration
                predictions_np = output.numpy()
                # Save updated DataFrame to CSV (append for each graph)
                unique_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))  # 8-character random string '.csv'
                output_path = f'{self.dataset_path}/station_{unique_filename}.csv'
                self.save_pred_to_csv(graph, predictions_np, output_path)
                i = i + 1
                # Update the original DataFrame (if needed)
                #graph.df_updated = df  # Save the updated DataFrame back to the graph object (optional)

        avg_loss = sum(losses) / len(losses)
        print(f'Average Loss: {avg_loss}')

        return predictions, targets

    def save_pred_to_csv(self, graph, predictions_np, output_path):
        # Extract data from the graph
        x_data = graph.x  # Features (temperature, humidity, timestamp, etc.)
        y_data = graph.y  # Targets (PM2.5, etc.)

        # If x_data and y_data are tensors, convert them to numpy arrays for easy DataFrame creation
        x_np = x_data.numpy() if isinstance(x_data, torch.Tensor) else x_data
        y_np = y_data.numpy() if isinstance(y_data, torch.Tensor) else y_data

        # Create a DataFrame with x, y, and predictions
        df = pd.DataFrame(x_np,
                          columns=self.features) #, 'timestamp'])  # Adjust column names as needed
        df['pm2p5'] = y_np  # Actual target values
        df['predicted_pm2p5'] = predictions_np  # Predicted values
        file_exists = os.path.exists(output_path)
        df.to_csv(f'{output_path}', mode='a', header=not file_exists, index=False)


    def add_sensor_index(self):
        for df in self.pd_list:
            #print('adding sensor index')
            df['sensor_index'] =\
                (df[['longitude', 'latitude']].
                 apply(lambda row: self.coordinate_to_index.get(tuple(np.round(row, decimals=5)), -1), axis=1))
            #print(f'{df.head()}')

    def create_edge_index_for_individual_graph(self, df_obj):
        # Extract unique coordinates (longitude, latitude) from the dataset
        coordinates = df_obj[['longitude', 'latitude']].drop_duplicates().values

        # Calculate the distance matrix for the unique coordinates
        dist_matrix = distance_matrix(coordinates, coordinates)
        #print(f'dist_matrix shape: {dist_matrix.shape}')

        edge_source = []
        edge_target = []

        # Create edges based on proximity (threshold)
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):  # Only consider i < j to avoid duplicates
                dist = dist_matrix[i, j]
                if dist < self.edge_index_threshold:
                    edge_source.append(i)
                    edge_target.append(j)
                    edge_source.append(j)
                    edge_target.append(i)

        # Convert the lists to a PyTorch tensor and return the edge index
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        print(f"Created edge_index: {edge_index}, shape: {edge_index.shape}")

        x = df_obj[self.features]
        print(x.shape[0])
        # If edge_index is empty
        if edge_index.size(1) == 0:
            print("Empty edge_index detected. Adding self-loops as fallback.")
        #    edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        print(f"edge_index :{edge_index}")
        return edge_index

    def create_edge_index_for_graph_of_graphs(self, dataset_coordinates):
        edge_source = []
        edge_target = []

        # Calculate the distance matrix between all coordinates (datasets)
        dist_matrix = distance_matrix(dataset_coordinates, dataset_coordinates)

        # Iterate through each pair of datasets and check the distance
        for i in range(len(dataset_coordinates)):
            for j in range(i + 1, len(dataset_coordinates)):  # Only consider i < j to avoid duplicates
                dist = dist_matrix[i, j]
                if dist < self.edge_index_threshold:
                    # Add edge if distance is within threshold
                    edge_source.append(i)
                    edge_target.append(j)
                    edge_source.append(j)
                    edge_target.append(i)

        # Convert the lists to a PyTorch tensor and return the edge index
        edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
        print(f'create_edge_index_for_graph_of_graphs : edge_index : {edge_index}')
        return edge_index

    @staticmethod
    def load_sensor_coordinates(df_obj):
        # Ensure longitude and latitude are numeric
        df_obj['longitude'] = pd.to_numeric(df_obj['longitude'], errors='coerce')
        df_obj['latitude'] = pd.to_numeric(df_obj['latitude'], errors='coerce')

        # Drop rows with invalid coordinates
        df_obj.dropna(subset=['longitude', 'latitude'], inplace=True)

        # Convert to a NumPy array
        coordinates = df_obj[['longitude', 'latitude']].values
        return coordinates

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
latticeImpl.do_train()
latticeImpl.do_predict()
