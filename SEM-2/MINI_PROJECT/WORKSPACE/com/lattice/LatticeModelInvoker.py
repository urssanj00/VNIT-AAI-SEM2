from PropertiesConfig import PropertiesConfig as PC
from GraphLevelGNNWithLSTM import GraphLevelGNNWithLSTM
from GraphLevelGNN import GraphLevelGNN
import torch.nn as nn
import pickle
import torch
import numpy as np
from LatticeImpl import LatticeImpl


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
            'temperature', 'humidity', 'year', 'month', 'day', 'hour', 'weekday', 'is_weekend'
        ]
        self.targets = ['pm2p5']
        self.gnn_model = None
        self.graph = None
        # Load the saved model state from the pickle file
        self.model_saved_state = None
        self.best_model_path = properties['best_model_path']
        print("0. Init")

    def load_model(self):
        self.gnn_model = GraphLevelGNN(in_channels=len(self.features))
        with open(f"{self.best_model_path}/best_gnn_model.pkl", "rb") as f:
            self.model_saved_state = pickle.load(f)
        try:
            self.gnn_model.load_state_dict(self.model_saved_state)
            print("Saved Model Loaded successfully ...")
            self.gnn_model.eval()
            print("Saved Model Evaluated successfully ...")
        except Exception as e:
            print(f"An error occurred while loading the state dictionary: {e}")
        print("1. load_model Done")


    def load_sample_data(self):
        print('load data')
        all_sensor_coordinates = []
        lattice_impl = LatticeImpl()

        self.graph, all_sensor_coordinates = lattice_impl.load_graph_from_individual_dataset(all_sensor_coordinates,
                                                                                             self.file_name)
        print("2. load_sample_data Done")


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

            for step in range(steps):
                # Predict for the current step
                outputs = self.gnn_model(self.graph).squeeze()
                # Save the prediction
                step_predictions.append(outputs.numpy())  # Convert to NumPy for saving

                # Update graph features for next step
                # Assuming that `current_input` can be updated with `output` for rolling prediction
                # Replace appropriate features with predictions (custom logic might be needed here)
                self.graph.x[:, 0] = outputs  # Example: Updating the first feature column with predictions

            # Save all predictions for the graph
            all_predictions.append(np.stack(step_predictions))

            # Compute loss for the graph's original targets (if available)
            target = self.graph.y.squeeze()
            loss = criterion(outputs, target)
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses) if losses else None
        print(f'Average Loss: {avg_loss if avg_loss is not None else "N/A"}')
        print(f'all_predictions : {all_predictions}')
        print(f'targets : {targets}')
        print("3. do_prediction Done")

        return all_predictions, targets


latticeModelInvoker = LatticeModelInvoker()

latticeModelInvoker.load_model()
latticeModelInvoker.load_sample_data()
latticeModelInvoker.do_prediction()