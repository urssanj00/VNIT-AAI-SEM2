import optuna
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch
from GraphLevelGNN import GraphLevelGNN
class BayesianOptimization:
# Define the objective function for Bayesian optimization
    def objective(self, trial, dataset, num_features):
        # Sample hyperparameters using the `trial` object
        lr = trial.suggest_float('lr', 1e-5, 1e-1)  # Learning rate (log uniform distribution)
        h1 = trial.suggest_int('h1', 32, 128)  # First hidden layer size
        h2 = trial.suggest_int('h2', 64, 256)  # Second hidden layer size
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])  # Batch size
        epochs = trial.suggest_int('epochs', 50, 200)  # Number of epochs

        print(f"Training with lr={lr}, h1={h1}, h2={h2}, batch_size={batch_size}, epochs={epochs}")

        # Initialize the model
        model = GraphLevelGNN(in_channels=num_features, h1=h1, h2=h2, out_channels=1)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train the model and get the final loss
        train_loss = self.train_model(model, dataset, optimizer, epochs, batch_size)

        return train_loss  # Return the final loss to Optuna for optimization


    def train_model(self, model, dataset, optimizer, epochs, batch_size):
        # Create DataLoader for batching
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                out = model(batch)  # Forward pass
                loss = criterion(out, batch.y)  # Calculate loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                total_loss += loss.item()

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader)}')

        return total_loss / len(loader)

    # Run the Optuna optimization process
    def tune_hyperparameters(self, dataset, num_features):
        print(f'tune_hyperparameters : {dataset}')
        # Create an Optuna study to minimize the loss
        study = optuna.create_study(direction="minimize")

        # Start the optimization process
        study.optimize(lambda trial: self.objective(trial, dataset, num_features), n_trials=10)  # Number of trials

        # Print the best hyperparameters and the corresponding loss
        print("Best hyperparameters:", study.best_params)
        print("Best validation loss:", study.best_value)

        return study.best_params


#bayesianOptimization = BayesianOptimization()
# Assuming `dataset` is your dataset object
#best_params = bayesianOptimization.tune_hyperparameters(GraphLevelGNN, dataset, )
