import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("Iris.csv")

print(df.sample(4))
print(df.size)

# Target Variable
# Species (y) : ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']

# Classification Problem
print(f"Species : {df['Species'].unique()}")

# features (x)
# 1. Sepal Length
# 2. Sepal Width
# 3. Petal Length
# 4. Petal Width
features = df[["SepalLengthCm", "SepalWidthCm",  "PetalLengthCm", "PetalWidthCm"]]
print (f"Features : {features}")
target = df["Species"]
print (f"Target : {target}")

# step:1
# hyperparameters = max_depth and min_samples_leaf
# to avoid overfitting limit the max_depth and min_samples_leaf
model = DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=4)

# Gini Impurity (gini)
# Log-Loss / Cross-Entropy (log_loss)
# Mean Squared Error (squared_error) (Regression Trees)

# steps:2 - Train on data
model.fit(features, target)

# step:3 - Predict on Unknown data
unknown_data = pd.DataFrame({
                "SepalLengthCm": [5.5],
                "SepalWidthCm": [4.6],
                "PetalLengthCm": [1],
                "PetalWidthCm": [2.4]
                })
predict_flower = model.predict(unknown_data)

print (f"Predicted Species : {predict_flower}")   # Predicted Species : ['Iris-versicolor']


## Plot te decision tree
plt.figure(figsize=(12, 12))
tree.plot_tree(model, fontsize=10, feature_names=["SepalLengthCm", "SepalWidthCm",  "PetalLengthCm", "PetalWidthCm"])
plt.savefig("plot_decision_tree.png")

# splitting data into training and testing
# split 80% in test and 20% in train. both have features(x) and target data(y)
# getting data after the split
# train_x, train_y     and    test_x, test_y

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=True)

y_pred = model.predict(x_test)
print(f"y_pred : {y_pred}")
print(f"y_test : {y_test}")

final_pred = pd.DataFrame({"actual":y_test, "predicted": y_pred })

print(f"final_pred : \n{final_pred}")

print (f" accuracy_score : {accuracy_score(y_test, y_pred)}")
#  accuracy_score : 1.0  - Model Performance is good