from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Create a synthetic dataset
X, y = make_classification(
    n_samples=1000,    # Number of samples
    n_features=20,     # Number of features
    n_informative=15,  # Number of informative features
    n_redundant=5,     # Number of redundant features
    random_state=42    # Ensures reproducibility
)

# Step 2: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Define the model
model = RandomForestClassifier(random_state=42)

# Step 4: Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150],        # Number of trees in the forest
    'max_depth': [None, 10, 20],           # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples to split
    'min_samples_leaf': [1, 2, 4]          # Minimum number of samples per leaf
}

# Step 5: Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                  # 5-fold cross-validation
    scoring='accuracy',    # Performance metric
    verbose=1,             # Print progress
    n_jobs=-1              # Use all available CPU cores
)

# Step 6: Perform the grid search
grid_search.fit(X_train, y_train)

# Step 7: Print the best hyperparameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Step 8: Evaluate the model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", test_accuracy)
