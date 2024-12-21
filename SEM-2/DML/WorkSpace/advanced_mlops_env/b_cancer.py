# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels

# Split the dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train an SVM classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# evL the model on the testing data
y_predi = svm_model.predict(X_test)

# Print evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_predi))

print("Accuracy Score:", accuracy_score(y_test, y_predi))
