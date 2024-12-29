# import OS module
import pandas as pd
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

class Util:
    @staticmethod
    def list_files_of_a_dir(dir_path):
        # Get the list of all files and directories
        dir_list = os.listdir(dir_path)
        print("Files and directories in '", dir_path, "' :")
        # prints all files
        print(dir_list)
        return dir_list
    @staticmethod
    def save_pred_to_csv(graph, predictions_np, output_path, in_features):
       # x_data = graph.x.numpy()
       # y_data = graph.y.numpy()
        # Convert tensors to NumPy arrays
        x_data = graph.x.numpy() if isinstance(graph.x, torch.Tensor) else graph.x
        y_data = graph.y.numpy() if isinstance(graph.y, torch.Tensor) else graph.y

        df = pd.DataFrame(x_data, columns=in_features)
        df['actual_target'] = y_data.squeeze()
        df['predicted_target'] = predictions_np.squeeze()

        file_exists = os.path.exists(output_path)
        df.to_csv(output_path, mode='a', header=not file_exists, index=False)
        print(f"Results saved to {output_path}")
    @staticmethod
    def timstamp_to_numeric(df_obj):
        df = df_obj
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)  # Remove timezone info
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday  # 0=Monday, 6=Sunday
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        return df
