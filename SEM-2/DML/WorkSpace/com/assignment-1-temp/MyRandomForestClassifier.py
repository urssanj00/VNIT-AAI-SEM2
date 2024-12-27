from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class MyRandomForestClassifier(RandomForestClassifier):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
        print(f"MyRandomForestClassifier: n_estimators:{n_estimators} max_depth:{max_depth} min_samples_split:{min_samples_split}")
        self.is_trained = False

    def train(self, X_train, y_train):
        self.fit(X_train, y_train)
        self.is_trained = True
        print("Training Completed")

    def evaluate(self, X_test, y_test):
        if not self.is_trained:
            raise ValueError("evaluate() Still untrained, not fit for prediction")
        output = self.predict(X_test)
        accuracy = accuracy_score(y_test, output)
        report = classification_report(y_test, output)
        return accuracy, report

    def feature_importances(self):
        if not self.is_trained:
            raise ValueError("feature_importances() The model has not been trained yet!")
        return self.feature_importances_

# Example Usage
if __name__ == "__main__":

    # Load MNIST dataset
    # Load the .npz file with allow_pickle=True
    mnist_data = np.load('mnist_784.npz', allow_pickle=True)

    # Access the data and target
    X = mnist_data['data']
    y = mnist_data['target']

    # Scale features to [0, 1] range
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate the extended RandomForestClassifier
    my_rf = MyRandomForestClassifier(n_estimators=100, max_depth=20)

    # Train the model
    my_rf.train(X_train, y_train)

    # Evaluate the model
    accuracy, report = my_rf.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    print("############          Classification Report          ##############")
    print(report)
    print("###################################################################")

    # Visualize some sample images and predictions
    sample_images = X_test[:16].reshape(-1, 28, 28)  # Reshape first 16 samples to 28x28 images
    sample_labels = y_test[:16]
    sample_predictions = my_rf.predict(X_test[:16])

    plt.figure(figsize=(12, 8))
    print(f"sample_labels   : {sample_predictions}\nPredicted: {sample_predictions}")
    i = 0
    for index in range(len(sample_predictions)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(sample_images[i], cmap='gray')
        plt.title(f"Actual   : {sample_predictions[i]}\nPredicted: {sample_predictions[i]}")
        plt.axis('off')
        i = i+1
    plt.tight_layout()
    plt.savefig("actual-vs-prediction.png", format='png')

    # Plot feature importances if available
    try:
        importances = my_rf.feature_importances()
        indices = np.argsort(importances)[-20:]  # Top 20 important features
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), indices)
        plt.xlabel("Feature Importance")
        plt.title("Top 20 Feature Importances")
        plt.savefig("Top_20_Feature_Importances.png", format='png')
    except AttributeError:
        print("Feature importances are not available for this dataset.")

