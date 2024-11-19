import pandas as pd

df = pd.read_csv("data/Iris.csv")

print(df.sample (4))

# features (x)
# 1. Sepal Length
# 2. Sepal Width
# 3. Petal Length
# 4. Petal Width

# Target Variable
# Species (y)

# Classification Problem

print(f"Species : {df['Species'].unique()}")
#Species : ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']

features = df[["SepalLengthCm", "SepalWidthCm",  "PetalLengthCm", "PetalWidthCm"]]
print (f"Features : {features}")
target = df["Species"]
print (f"Target : {target}")

from sklearn.tree import  DecisionTreeClassifier

# step:1
# hyperparameters = max_depth and min_samples_leaf
model = DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=4)  # to avoid overfitting limit the max_depth and min_samples_leaf

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

print (f"Predicted Species : {predict_flower}")
# Predicted Species : ['Iris-versicolor']

## Plot te decision tree

from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))
tree.plot_tree(model, fontsize=10, feature_names=["SepalLengthCm", "SepalWidthCm",  "PetalLengthCm", "PetalWidthCm"])
plt.show()


# splitting data into training and testing

from sklearn.model_selection import train_test_split
# split 80% in test and 20% in train. both have features(x) and target data(y)
# getting data after the split
# train_x, train_y     and    test_x, test_y

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=True)



from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
print(f"y_pred : {y_pred}")
print(f"y_test : {y_test}")

final_pred = pd.DataFrame({"actual":y_test, "predicted": y_pred })

print(f"final_pred : \n{final_pred}")

print (f" accuracy_score : {accuracy_score(y_test, y_pred)}")
#  accuracy_score : 1.0  - Model Performance is good