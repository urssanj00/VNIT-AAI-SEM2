# Lecture DML - 19-Oct-2024
# Gender Prediction using Height and Shoe size
# Goal : to create a classifier for gender predictions using Logistic Regression

import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

df = pd.DataFrame({
    "height": [5.1, 5.7, 5.5, 6, 6.3, 5.4, 5.2, 5.4, 5.8, 5.7, 5.3],
    "shoe_size": [6, 8, 7, 10, 9, 5, 6, 7, 8, 8, 7],
    "gender": [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
})

print(df)
# Input features and target variable
X = df[["height", "shoe_size"]]   # input Features
Y = df["gender"]                  # target variable (category)

# Step:1 # Initialize the Logistic Regression model
model1 = LogisticRegression()

# Step:2 Train the model with historical training data
model1.fit(X, Y)

# Step:3 Predicting unknown value
x_h = [5.8]
x_ss = [11]
x_known = pd.DataFrame({"height": x_h, "shoe_size": x_ss})

gender_prediction = model1.predict(x_known)

print(f"Prediction : {gender_prediction}")

fig, ax = plt.subplots()
ax.scatter(df[df["gender"] == 1]["height"], df[df["gender"] == 1]["shoe_size"], label="Female")
ax.scatter(df[df["gender"] == 0]["height"], df[df["gender"] == 0]["shoe_size"], label="Male")
ax.legend()
ax.set_xlabel = "Height"
ax.set_ylabel = "Shoe Size"
plt.show()

ar_coeff = model1.coef_
print(f"Coefficient : {ar_coeff}")

intercept = model1.intercept_
print(f"Intercept : {intercept}")

# Feature values

# Creating the DataFrame correctly by passing an index
features = pd.DataFrame({"height": x_h, "shoe_size": x_ss}, index=[0])

result = np.dot(ar_coeff.T, features)
denominator = 1 + np.exp(-(result + intercept))

probability = 1/denominator

print(f"probability : {probability}")
