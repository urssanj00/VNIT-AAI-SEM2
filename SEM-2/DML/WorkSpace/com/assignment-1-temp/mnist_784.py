from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

# Step 1: Download the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Extract data and target
X = mnist.data  # Pixel values (features)
y = mnist.target  # Labels (target)

# Step 2: Save to HDD
# Save as CSV
df = pd.DataFrame(X)
df['label'] = y  # Add the labels as a new column
df.to_csv('mnist_784.csv', index=False)
print("Saved MNIST as mnist_784.csv")

# Save as .npz for efficient storage
np.savez_compressed('mnist_784.npz', data=X, target=y)
print("Saved MNIST as mnist_784.npz")

# Load MNIST dataset
'''  
mnist_data = np.load('mnist_784.npz', allow_pickle=True)
X = mnist_data['data']
y = mnist_data['target']
'''