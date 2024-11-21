########################  eda_10_bike_sharing.py   ########################

from eda_09_bike_sharing import *

# Sum Square Error
newMatrix_test = x_test
for i in range(2, 4):
    subMatrix_test = np.power(x_test, i)
    print(f"Iteration {i}, shape of matrix is: {subMatrix_test.shape}")
    newMatrix_test = np.append(newMatrix_test, subMatrix_test, axis=1)

rows, cols = newMatrix_test.shape

print(f"Rows: {rows}, Columns: {cols}")
arrayAllOnes_test = np.ones((rows, 1))
theta_test = np.append(arrayAllOnes_test, newMatrix_test, axis=1)

# print(theta_test,theta_test.hape)

y_predicted_test = np.matmul(theta_test, wopt)
print(y_predicted_test, y_predicted_test.shape)

sse = sum((y_test - y_predicted_test) ** 2) / 2
print(f"Sum Square Error: {sse}")

# ## Linear Regression using the API

from sklearn import linear_model

# Create the Model
model = linear_model.LinearRegression()

# Train the model
model.fit(x_train, y_train)

# Predict using the trained model
y_pred = model.predict(x_test)

# Calculate mean square error
from sklearn.metrics import mean_squared_error

# Calculate the Mean Square Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Square Error: {mse}")

# Calculate the Sum Squared Error
ssep = sum((y_test - y_pred) ** 2) / 2

print(f"Sum Squared Error: {ssep}")


