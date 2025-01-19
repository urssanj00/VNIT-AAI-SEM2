import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

import uuid
from datetime import datetime
import os
from PropertiesConfig import PropertiesConfig as PC
from AdvancedTemporalGraphNetwork import AdvancedTemporalGraphNetwork

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PM2P5ForcastFacilitator:

    # Function to generate a unique filename
    def generate_filename(self):
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Generate a short UUID (first 8 characters)
        unique_id = str(uuid.uuid4())[:8]
        # Create a readable filename
        filename = f"_{timestamp}_{unique_id}"
        return filename

    def list_files_of_a_dir(self):
        # Get the list of all files and directories
        dir_path = self.input_dir
        dir_list = os.listdir(dir_path)
        #print("Files and directories in '", dir_path, "' :")
        # prints all files
        #print(dir_list)
        return dir_list

    def preprocess_timeseries_data(self, df):
     #   print(df.columns.tolist())
        df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'])
        # Drop rows where timestamp conversion failed
        df = df.dropna(subset=['timestamp'])

        # Select specific feature columns
        df_processed = df[["timestamp"] + self.features_columns].copy()

        # Drop rows with missing values
        df_processed = df_processed.dropna()
        #print(f"{df_processed['timestamp'].head()}")
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])

        # Extract time-based features
        try:
            df_processed['hour'] = df_processed['timestamp'].dt.hour
            df_processed['minute'] = df_processed['timestamp'].dt.minute
            df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
        except AttributeError as e:
            print("Error while extracting time-based features. Ensure 'timestamp' is properly converted.")
            print(e)
            raise

        # Prepare features for scaling
        all_features = self.features_columns + ['hour', 'minute', 'day_of_week']
       # print(f'all_features {all_features}')
        features = df_processed[all_features]

        scaled_features = self.scaler.fit_transform(features)
        #print(f'scaled_features:{scaled_features}')
        return scaled_features, df_processed

    def create_sequence_data(self, features, sequence_length=5, test_size=0.2, random_state=42):
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(features[i + sequence_length][-3]) #column p2p5 - Target Column

        X = np.array(X)
        y = np.array(y)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_size,
                                                            random_state=random_state,
                                                           # stratify=y,
                                                            shuffle=True
                                                            )

        # Convert to PyTorch tensors
        return (
            torch.tensor(X_train, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.float).unsqueeze(1),  # Reshape to (N, 1)
            torch.tensor(X_test, dtype=torch.float),
            torch.tensor(y_test, dtype=torch.float).unsqueeze(1)  # Reshape to (N, 1)
        )

    def create_graph_edges(self, num_nodes, k=5):
        edges = []
        for i in range(num_nodes):
            for j in range(max(0, i-k), min(num_nodes, i+k+1)):
                if i != j:
                    edges.append([i, j])
        return torch.tensor(edges).t().contiguous()


    def train_advanced_temporal_graph_model(self, num_plots=9):
        print(f"Step 04: Preprocess the data")

        # Preprocessing
        features, df_processed = self.preprocess_timeseries_data(self.combined_df)
        print(f"Step 05: Split data in Train and Test")
        # Create sequences
        self.X_train, self.y_train, self.X_test, self.y_test = self.create_sequence_data(features)
      #  print(f'y_test : {self.y_test}')
        # Create graph edges
        edge_index = self.create_graph_edges(features.shape[1])
       # print(f'features.shape[1] {features.shape}')
        # Initialize model
        self.model = AdvancedTemporalGraphNetwork(
            num_features=features.shape[1],
            hidden_channels=self.epoch_config['hidden_channels'],
            num_nodes=features.shape[1]
            #num_nodes=features.shape[0]
        )
        print(f"Step 06: Start Training in Epoch")

        self.do_epoch(edge_index, features)

    def do_epoch(self, edge_index, features):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.epoch_config['lr'])

        criterion = nn.MSELoss()
        # Training Loop

        train_predictions = None
        test_predictions = None
        #print(f"|--------------------------------------------------------------|")
        PM2P5ForcastFacilitator.print_dash_line()
        for epoch in range(self.epoch_config['epochs']):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            # Predict for training sequences
            train_predictions = self.model(None, edge_index, self.X_train)
            #print(f'train_predictions : {train_predictions}')
            train_loss = criterion(train_predictions, self.y_train)

            # Backpropagate
            train_loss.backward()
            optimizer.step()

            # Evaluation phase
            self.model.eval()
            with torch.no_grad():
                test_predictions = self.model(None, edge_index, self.X_test)
                test_loss = criterion(test_predictions, self.y_test)

                # Calculate RMSE and R^2 for test data
                test_rmse_value = torch.sqrt(test_loss).item()
                r2_score_value = r2_score(self.y_test.cpu().numpy(), test_predictions.cpu().numpy())

                # Record metrics
                self.test_rmse.append(test_rmse_value)
                self.r2_scores.append(r2_score_value)

            # Store overall losses
            self.train_losses.append(train_loss.item())
            self.test_losses.append(test_loss.item())

            matrices_train = PM2P5ForcastFacilitator.evaluate_model(self.y_train, train_predictions)
            PM2P5ForcastFacilitator.append_matrices_in_list(self,
                                                            matrices_train,
                                                            self.rmse_train_list,
                                                            self.mae_train_list,
                                                            self.r2_train_list,
                                                            self.mape_train_list)
            matrices_test = PM2P5ForcastFacilitator.evaluate_model(self.y_test, test_predictions)
            PM2P5ForcastFacilitator.append_matrices_in_list(self,
                                                            matrices_train,
                                                            self.rmse_test_list,
                                                            self.mae_test_list,
                                                            self.r2_test_list,
                                                            self.mape_test_list)

            # Print progress
            if epoch % 5 == 0:
                print(f"|                                                              |")
                print(f"|                           Epoch [{epoch + 1}/{self.epoch_config['epochs']}]")
                #print(f"|--------------------------------------------------------------|")
                PM2P5ForcastFacilitator.print_dash_line()
                #print(
                 #   f'Epoch [{epoch + 1}/{self.epoch_config["epochs"]}], Train Loss:{train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
                print(f"|    Train Loss:     {train_loss.item():.4f}, "
                      f"    Test Loss:      {test_loss.item():.4f}   "
                      )

                print(f"|                                                              |")
                print("|     Train Eval                                               |")
                PM2P5ForcastFacilitator.print_matrices(self, matrices_train)
                print(f"|                                                              |")
                print("|     Test Eval                                                |")
                PM2P5ForcastFacilitator.print_matrices(self, matrices_test)
                PM2P5ForcastFacilitator.print_dash_line()
                #print(f"|--------------------------------------------------------------|")

        print(f"Step 07: Plot Train Losses vs Test Losses over Epochs")
        self.do_plot()


        print(f"Step 09: Plot Train Matrices over Epochs")
        self.do_plot_matrices(self.rmse_train_list,
                              self.mae_train_list,
                              self.r2_train_list,
                              self.mape_train_list,
                              "Train")
        print(f"Step 10: Plot Test Matrices over Epochs")
        self.do_plot_matrices(self.rmse_test_list,
                              self.mae_test_list,
                              self.r2_test_list,
                              self.mape_test_list,
                              "Test")

    @staticmethod
    def evaluate_model(y_actual, y_pred):

        # Ensure inputs are numpy arrays for consistency
        y_actual = y_actual.detach().numpy()
        y_pred = y_pred.detach().numpy()

        y_actual = np.array(y_actual)
        y_pred = np.array(y_pred)

        # Metrics calculations
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) # * 100
        return {"RMSE": rmse, "MAE": mae, "R^2": r2, "MAPE": mape}

    @staticmethod
    def append_matrices_in_list(self, metrics, rmse_list, mae_list, r2_list, mape_list):
        rmse = metrics["RMSE"]
        mae = metrics["MAE"]
        r2 = metrics["R^2"]
        mape = metrics["MAPE"]

        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        mape_list.append(mape)

    @staticmethod
    def print_dash_line():
        print(f"|--------------------------------------------------------------|")
    @staticmethod
    def print_dot_line():
        print(f"|..............................................................|")
    @staticmethod
    def print_matrices(self, metrics):
        rmse = metrics["RMSE"]
        mae = metrics["MAE"]
        r2 = metrics["R^2"]
        mape = metrics["MAPE"]

        # Print the results
        #print(f"|--------------------------------------------------------------|")
        PM2P5ForcastFacilitator.print_dash_line()
        print(f"|    RMSE:   {rmse:.4f}                                            |")
        print(f"|    MAE:    {mae:.4f}                                            |")
        if r2 < 0:
            print(f"|    R^2:    {r2:.4f}                                           |")
        else:
            print(f"|    R^2:    {r2:.4f}                                            |")
        print(f"|    MAPE:   {mape:.2f}                                              |")


    def do_plot_matrices(self, rmse_values, mae_values, r2_values, mape_values, title):
        epoch_count = self.epoch_config['epochs']
        epochs = list(range(0, epoch_count))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, rmse_values, label="RMSE", marker="o")
        plt.plot(epochs, mae_values, label="MAE", marker="s")
        plt.plot(epochs, r2_values, label="R^2", marker="^")
        plt.plot(epochs, mape_values, label="MAPE (%)", marker="d")

        plt.xlabel("Epochs")
        plt.ylabel("Metric Values")
        plt.title(f"[{title}] Model Metrics Over Epochs")
        plt.legend()
        plt.grid()
        u_f_name = self.generate_filename()
        plt.savefig(f'{self.plot_path}/plot-{title}-matrices-{u_f_name}.png', format="png", dpi=300, bbox_inches="tight")

    def do_plot(self):
        # Visualization of training process
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.test_losses, label='Testing Loss', color='orange')
        plt.title('Training and Testing Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        u_f_name = self.generate_filename()
        plt.savefig(f'{self.plot_path}/plot-train-vs-test-{u_f_name}.png', format="png", dpi=300, bbox_inches="tight")
        plt.close()

    def do_plot1(self, train_preds, test_preds):
        train_actual = self.scaler.inverse_transform(self.y_train.numpy())
        test_actual = self.scaler.inverse_transform(self.y_test.numpy())
        train_size = self.X_train.shape[0]
        #print(f' features.shape[1] {len( self.features_columns)}')
        # Actual vs Predicted Plots
        for i in range( len(self.features_columns)):
           # print(f"{i}. Plotting Features")
            plt.figure(figsize=(14, 6))
            plt.plot(range(train_size), train_actual[:, i],
                     label=f"Train Actual {self.features_columns[i]}", color="blue", marker='o', alpha=0.7)
            plt.plot(range(train_size), train_preds[:, i],
                     label=f"Train Predicted {self.features_columns[i]}", color="red", marker='x', alpha=0.7)
            plt.plot(range(train_size, train_size + len(test_actual)), test_actual[:, i],
                     label=f"Test Actual {self.features_columns[i]}", color="green", marker='o', alpha=0.7)
            plt.plot(range(train_size, train_size + len(test_preds)), test_preds[:,i],
                     label=f"Test Predicted {self.features_columns[i]}", color="orange",marker='x', alpha=0.7)
            plt.title(f"{self.features_columns[i]}: Actual vs Predicted")
            plt.xlabel("Sequence Index")
            plt.ylabel("Values")
            plt.legend()
            plt.grid(True)
            u_f_name = self.generate_filename()

            plt.savefig(f'{self.plot_path}/plot_features_{u_f_name}_{i}.png', format="png", dpi=300, bbox_inches="tight")

           # print(f"{i}. Plotting Residuals")

            train_residuals = train_actual[:, i] - train_preds[:, i]
            test_residuals = test_actual[:, i] - test_preds[:, i]
            plt.figure(figsize=(14, 6))
            plt.plot(range(train_size), train_residuals,
                     label=f"Train Residuals ({self.features_columns[i]})", color="orange")
            plt.plot(range(train_size, train_size + len(test_residuals)), test_residuals,
                     label=f"Test Residuals ({self.features_columns[i]})", color="green")
            plt.axhline(0, color='black', linestyle='--')
            plt.xlabel("Sequence Index")
            plt.ylabel("Residuals")
            plt.title(f"Residuals for {self.features_columns[i]} (Actual - Predicted)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.plot_path}/plot_residuals_{u_f_name}_{i}.png', format="png", dpi=300, bbox_inches="tight")
            plt.close()


    def load_all_csv(self):
        list_file = pm2p5ForcastFacilitator.list_files_of_a_dir()
        dataframes = []
        print("Step 02: Loading all csvs in DF")
        i = 0
        for f_name in list_file:
            if f_name.endswith(".csv") and "station" not in f_name.lower():
                # Build the full file path
                csv_path = os.path.join(self.input_dir, f_name)

                try:
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(csv_path)
                    dataframes.append(df)
                    print(f"    {i}.    Loaded: {csv_path}")
                    i = i + 1
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        # Combine all DataFrames (optional, if they have the same structure)
        if dataframes:
            self.combined_df = pd.concat(dataframes, ignore_index=True)

            print(self.combined_df.columns.tolist())
        else:
            print("No valid CSV files found.")

        print(f"Step 03: Clubbed all csv in a dataset. Dataset Shape : [{self.combined_df.shape}]")
    # Loss and Optimizer
    def __init__(self):
        print(f"Step 01: Initializing the processing")
        properties_config = PC()
        properties = properties_config.get_properties_config()
        self.plot_path = properties['plot_path']
        self.input_dir = properties['training_data']
        self.output_dir = properties['output_path']
        self.plot_path = properties['plot_path']
        self.epoch_count = properties['epoch_count']
        self.combined_df = None
        self.train_losses, self.test_losses = [], []
        self.train_rmse = []
        self.test_rmse = []
        self.r2_scores = []
        self.matrices_train_list = []
        self.matrices_test_list = []

        self.rmse_test_list = []
        self.mae_test_list = []
        self.r2_test_list = []
        self.mape_test_list = []

        self.rmse_train_list = []
        self.mae_train_list = []
        self.r2_train_list = []
        self.mape_train_list = []

        self.epoch_config = {
            'hidden_channels': 32,
            'dropout': 0.2,
            'lr': 0.001,
            'epochs': int(f"{self.epoch_count}")
        }
        # Scale features
        self.scaler = StandardScaler()
        self.features_columns = ["sensor_id", "temperature", "humidity", "longitude", "latitude", "pm2p5"]
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.model = None


pm2p5ForcastFacilitator = PM2P5ForcastFacilitator()

pm2p5ForcastFacilitator.load_all_csv()
pm2p5ForcastFacilitator.train_advanced_temporal_graph_model()
print("Step 11: End of Execution")
