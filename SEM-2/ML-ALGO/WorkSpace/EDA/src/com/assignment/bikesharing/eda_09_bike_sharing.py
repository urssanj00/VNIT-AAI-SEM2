########################  eda_09_bike_sharing.py   ########################

from eda_08_bike_sharing import *

# I am taking my model question till 3th order
newMatrix = x_train
for i in range(2, 4):
    subMatrix = np.power(x_train, i)
    print(f"Iteration {i}, shape of matrix is: {subMatrix.shape}")
    newMatrix = np.append(newMatrix, subMatrix, axis=1)

rows, cols = newMatrix.shape

print(f"Rows: {rows}, Columns: {cols}")

arrayAllOnes = np.ones((rows, 1))
theta = np.append(arrayAllOnes, newMatrix, axis=1)
print(f"Theta: theta={theta}, theta.shape={theta.shape}")

# creating theta transpose
theta_t = np.transpose(theta)
print(f"Theta transpose: theta_t={theta_t}, theta_t.shape={theta_t.shape}")

# multiplying theta transpose with theta
theta_mul = np.matmul(theta_t, theta)
print(f"[Theta * Theta transpose]: theta_mul={theta_mul}, theta_mul.shape={theta_mul.shape}")

# Inverse the multiplication result of the above step
theta_inv = np.linalg.inv(theta_mul)
print(f"[Inverse above Theta Multiplication]: theta_inv={theta_inv}, theta_inv.shape={theta_inv.shape}")

# multiplying the inverse of the multiplication with theta transpose
theta_inv_theta_t = np.matmul(theta_inv, theta_t)
print(f"[Multiplying the inverse of the multiplication with theta transpose]: theta_inv_theta_t={theta_inv_theta_t}, theta_inv_theta_t.shape={theta_inv_theta_t.shape}")

# multiply theta expression with training data set to find WOPT
wopt = np.matmul(theta_inv_theta_t, y_train)
print(f"[WOPT:multiply theta expression with training data set to find WOPT]: wopt={wopt}, wopt.shape={wopt.shape}")


