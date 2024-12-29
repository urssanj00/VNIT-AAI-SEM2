import pandas as pd
from Util import Util
from PropertiesConfig import PropertiesConfig as PC
import torch
import numpy as np
from scipy.spatial import distance_matrix
from GraphLevelGNN import GraphLevelGNN
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import torch.nn as nn
import os
from sklearn.preprocessing import MinMaxScaler
import random
import string
from BayesianOptimization import BayesianOptimization
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LatticeModelInvoker:
    def __init__(self):
        properties_config = PC()
        properties = properties_config.get_properties_config()
        self.plot_path = properties['plot_path']
        self.dataset_path = properties['data_set_path']
        self.file_name = properties['csv_name']
        self.edge_index_threshold = float(properties['edge_index_threshold'])
        self.graph_list = []
        self.pd_list = []
        self.coordinate_to_index = {}  # Mapping of coordinates to global indices
        self.features = [
            'temperature', 'humidity', 'year', 'month', 'day', 'hour', 'weekday', 'is_weekend', 'longitude', 'latitude'
        ]
        self.targets = ['pm2p5']
        self.gnn_model = None
        self.graph = None
        # Load the saved model state from the pickle file
        self.model_saved_state = None
        self.best_model_path = properties['best_model_path']
        self.pm2p5_output = properties['pm2p5_output']

        print("0. Init")

    def load_model(self):
        self.gnn_model = GraphLevelGNN(in_channels=len(self.features))
        print(f"{self.best_model_path}/best_gnn_model.pkl")
        try:
            # Use torch.load instead of pickle.load to load the model state
            self.model_saved_state = torch.load(f"{self.best_model_path}/best_gnn_model.pkl")
            self.gnn_model.load_state_dict(self.model_saved_state)  # Load the state dictionary into the model
            print("Saved Model Loaded successfully ...")
            self.gnn_model.eval()  # Set the model to evaluation mode
            print("Saved Model Evaluated successfully ...")

        except Exception as e:
            print(f"An error occurred while loading the state dictionary: {e}")
        print("1. load_model Done")


    def load_sample_data(self):
        print('load data')
        all_sensor_coordinates = []
        self.graph, all_sensor_coordinates = (
            self.load_graph_from_individual_dataset(self.file_name, all_sensor_coordinates))
        print("2. load_sample_data Done")

    def load_graph_from_individual_dataset(self, file_name, all_sensor_coordinates):
        df_obj = pd.read_csv(f'{self.dataset_path}/{file_name}')
        df_obj = Util.timstamp_to_numeric(df_obj)

        df_obj['timestamp'] = pd.to_datetime(df_obj['timestamp'])
        df_obj = df_obj.sort_values(by=['sensor_id', 'year', 'month', 'day', 'hour']).reset_index(drop=True)

        df_obj[self.features] = df_obj[self.features].apply(pd.to_numeric, errors='coerce')

        # Step 2: Feature Scaling
        scaler = MinMaxScaler()
        #not including latitude and longitude for trnasform becase adding them will zero them in the whole dataset
        #df_obj[self.features] = scaler.fit_transform(df_obj[self.features])
        features = ['temperature', 'humidity']
        df_obj[features] = scaler.fit_transform(df_obj[features])
        df_obj.fillna(0, inplace=True)

        x = torch.tensor(df_obj[self.features].values, dtype=torch.float)
        y = torch.tensor(df_obj[self.targets].values, dtype=torch.float)

        # Collect coordinates from the dataset
        #sensor_coordinates = self.load_sensor_coordinates(df_obj)
        #all_sensor_coordinates.extend(sensor_coordinates)

        self.pd_list.append(df_obj)
        df_obj, edge_index = self.create_edge_index_for_individual_graph(df_obj)

        df_obj['sensor_index'] = df_obj[['longitude', 'latitude']].apply(
            lambda row: self.coordinate_to_index.get(tuple(row), -1), axis=1
        )

        graph = Data(x=x, edge_index=edge_index, y=y)

        return graph, all_sensor_coordinates

    def create_edge_index_for_individual_graph(self, df_obj):
        # Ensure the dataframe has enough rows to create edges
        num_rows = len(df_obj)
        # df_obj = df_obj.sort_values(by=['sensor_id', 'pm2p5']).reset_index(drop=True)
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
        # print(f'edge_index : {edge_index}')
        return df_obj, edge_index

    def do_prediction(self):
        steps = 7
        all_predictions = []  # To store predictions for each step
        targets = []  # Actual target values (if available)
        losses = []  # Loss values for evaluation
        criterion = nn.MSELoss()

        # Make the prediction
        with torch.no_grad():  # No gradient calculation during inference
            current_input = self.graph.x.clone()  # Clone initial input features
            step_predictions = []  # To store step-wise predictions

            #for step in range(steps):
            # Predict for the current step
            outputs = self.gnn_model(self.graph)
            # Save the prediction
            step_predictions.append(outputs.numpy())  # Convert to NumPy for saving
            print(f"outputs {outputs}")
                # Update graph features for next step
                # Assuming that `current_input` can be updated with `output` for rolling prediction
                # Replace appropriate features with predictions (custom logic might be needed here)
                #self.graph.x[:, 0] = outputs.view(-1, 1)  # Example: Updating the first feature column with predictions
                #print(self.graph.x.shape)  # Debugging shape

            # Save all predictions for the graph
            all_predictions.append(np.stack(step_predictions))

            # Compute loss for the graph's original targets (if available)
            target = self.graph.y.squeeze()
            loss = criterion(outputs, target)
            losses.append(loss.item())

            predictions_np = outputs.numpy()
            unique_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            output_path = f'{self.pm2p5_output}/_predictions_{unique_filename}.csv'
            Util.save_pred_to_csv(self.graph, predictions_np, output_path, self.features)



        avg_loss = sum(losses) / len(losses) if losses else None
        print(f'Average Loss: {avg_loss if avg_loss is not None else "N/A"}')
       # print(f'all_predictions : {all_predictions}')
        print(f'targets : {targets}')
        print("3. do_prediction Done")

        return all_predictions, targets


latticeModelInvoker = LatticeModelInvoker()

latticeModelInvoker.load_model()
latticeModelInvoker.load_sample_data()
latticeModelInvoker.do_prediction()