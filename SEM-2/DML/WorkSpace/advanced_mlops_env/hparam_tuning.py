import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

try:
    mlflow.set_tracking_uri('http://localhost:5000')
    # Load the Breast Cancer Wisconsin dataset
    data = load_breast_cancer()
    X = data.data  # Features
    y = data.target  # Labels

    # Split the dataset into training and testing sets (75-25 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Define the hyperparameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10],         # Regularization parameter
        'kernel': ['linear', 'rbf'],  # SVM kernel types
        'gamma': ['scale', 'auto']   # Kernel coefficient for 'rbf'
    }

    # Initialize the SVM classifier
    svm_model = SVC(random_state=42)

    # Set up GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

    # Start MLflow experiment
    mlflow.start_run()

    # Log hyperparameters
    mlflow.log_params(param_grid)

    # Fit the grid search model
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Infer the signature of the input and output
    signature = infer_signature(X_train, best_model.predict(X_train))

    # Create an input example
    input_example = pd.DataFrame(X_train[:5], columns=data.feature_names)

    # Log the model with signature and input example
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="svm_model",
        signature=signature,
        input_example=input_example
    )

    # Predict on the test set using the best model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)

    # Print the evaluation results
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)

    # End MLflow run
    mlflow.end_run()
except MlflowException as e:
    print(f'{e}')

