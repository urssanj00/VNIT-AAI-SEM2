import pandas as pd
from Util import Util
from PropertiesConfig import PropertiesConfig as PC
import torch
import numpy as np
from scipy.spatial import distance_matrix
from GraphLevelGNN import  GraphLevelGNN
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

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
        self.features = ['temperature', 'humidity', 'timestamp']
        self.targets = ['pm2p5']
    def load_dataset(self):
        list_files = Util.list_files_of_a_dir(self.dataset_path)

        # Collect all unique coordinates across all datasets
        all_sensor_coordinates = []

        for file_name in list_files:
            if 'station' not in file_name:
                df_obj = pd.read_csv(f'{self.dataset_path}/{file_name}')

                # Ensure numeric data and process dataset
                df_obj['timestamp'] = pd.to_datetime(df_obj['timestamp'], errors='coerce').astype(
                    'int64') // 1e9  # Convert to Unix timestamp
                df_obj[['temperature', 'humidity']] = df_obj[['temperature', 'humidity']].apply(pd.to_numeric,
                                                                                                errors='coerce')
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

                print(f"df_obj['sensor_index']: {df_obj['sensor_index'].head() }")
                graph = Data(x=x, edge_index=edge_index, y=y)
                self.graph_list.append(graph)

        # Remove duplicate coordinates and keep only unique ones
        unique_coordinates = np.unique(all_sensor_coordinates, axis=0)
        print(f'unique_coordinates : {unique_coordinates}')

        self.coordinate_to_index = {tuple(coord): idx for idx, coord in enumerate(unique_coordinates)}

        edge_index_for_graph_of_graphs = self.create_edge_index_for_graph_of_graphs(unique_coordinates)

        self.add_sensor_index()

        # Create a mapping of coordinates to their index
        coordinate_to_index = {tuple(coord): idx for idx, coord in enumerate(unique_coordinates)}
        print(f'coordinate_to_index : {coordinate_to_index}')

    def do_predict(self):
        model_graph_level_gnn = GraphLevelGNN(in_channels=len(self.features), hidden_channels=64, out_channels=len(self.targets))
        model_graph_level_gnn.eval()  # Set model to evaluation mode
        predictions = []
        for graph in self.graph_list:
            with torch.no_grad():  # Disable gradient computation for prediction
                prediction = model_graph_level_gnn(graph)
                print(f'P R E D I C T I O N : {graph}:{prediction}')
                predictions.append(prediction)

    def add_sensor_index(self):
        for df in self.pd_list:
            print('adding sensor index')
            df['sensor_index'] =\
                (df[['longitude', 'latitude']].
                 apply(lambda row: self.coordinate_to_index.get(tuple(np.round(row, decimals=5)), -1), axis=1))
            #print(f'{df.head()}')

    def create_edge_index_for_individual_graph(self, df_obj):
        # Extract unique coordinates (longitude, latitude) from the dataset
        coordinates = df_obj[['longitude', 'latitude']].drop_duplicates().values

        # Calculate the distance matrix for the unique coordinates
        dist_matrix = distance_matrix(coordinates, coordinates)
        print(f'dist_matrix shape: {dist_matrix.shape}')

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
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
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
latticeImpl.do_predict()
