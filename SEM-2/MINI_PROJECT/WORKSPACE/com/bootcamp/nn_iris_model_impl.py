import pickle
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# Load the saved model state from the pickle file
saved_state = None
with open("best_model.pkl", "rb") as f:
    saved_state = pickle.load(f)

class IrisModel(nn.Module):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.fc1 = nn.Linear(4, 8)  # 4 input features, 8 neurons in the first hidden layer
        self.fc2 = nn.Linear(8, 9)  # 9 neurons in the second hidden layer
        self.out = nn.Linear(9, 3)  # 3 output classes (Setosa, Versicolor, Virginica)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.out(x)  # Use `out` as the final layer
        return x

try:
    model = IrisModel()
    # Load the state dictionary into the model
    model.load_state_dict(saved_state)
    print("Saved Model Loaded successfully ...")

    # Set the model to evaluation mode
    model.eval()
    print("Saved Model Evaluated successfully ...")


    # Example batch of test inputs (4 features from the Iris dataset)
    # Format: [sepal_length, sepal_width, petal_length, petal_width]
    # Example batch of test inputs
    test_sample = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Iris-Setosa
        [6.7, 3.1, 4.4, 1.4],  # Iris-Versicolor
        [7.2, 3.6, 6.1, 2.5]   # Iris-Virginica
    ])

    # Convert to a PyTorch tensor (float type)
    test_tensor = torch.FloatTensor(test_sample)

    # Make the prediction
    with torch.no_grad():  # No gradient calculation during inference
        outputs = model(test_tensor)
        predicted_class = torch.argmax(outputs, dim=1)  # Get the class index with the highest score

    print(f'Predicted_class: { predicted_class}')

    # Map the predicted index to class names
    class_names = ["Setosa", "Versicolor", "Virginica"]

    # Map and print predictions
    for sample, pred in zip(test_sample, predicted_class):
        print(f"Sample: {sample}, Predicted Class: {class_names[pred.item()]}")

except Exception as e:
    print(f"An error occurred while loading the state dictionary: {e}")