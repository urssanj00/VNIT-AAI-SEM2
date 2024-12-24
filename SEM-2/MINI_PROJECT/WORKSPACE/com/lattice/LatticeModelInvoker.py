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

class LatticeModelInvoker:

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

# Load the saved model state from the pickle file
saved_state = None
with open("best_model.pkl", "rb") as f:
    saved_state = pickle.load(f)



try:
    model = GraphLevelGNNWithLSTM()
    model.load_state_dict(saved_state)
    print("Saved Model Loaded successfully ...")
    model.eval()
    print("Saved Model Evaluated successfully ...")
except Exception as e:
    print(f"An error occurred while loading the state dictionary: {e}")



# Example test input (4 features from the Iris dataset)
# Format: [sepal_length, sepal_width, petal_length, petal_width]
test_sample = np.array([[6.1,2.9,4.7,1.4]])  # This corresponds to Iris-Setosa



# Convert to a PyTorch tensor (float type)
test_tensor = torch.FloatTensor(test_sample)

# Make the prediction
with torch.no_grad():  # No gradient calculation during inference
    outputs = model(test_tensor)
    predicted_class = torch.argmax(outputs, dim=1)  # Get the class index with the highest score

# Map the predicted index to class names
class_names = ["Setosa", "Versicolor", "Virginica"]
predicted_label = class_names[predicted_class.item()]

print(f"Predicted Class: {predicted_label}")

