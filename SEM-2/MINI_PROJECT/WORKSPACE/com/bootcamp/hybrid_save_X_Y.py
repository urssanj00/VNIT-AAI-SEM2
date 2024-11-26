import numpy as np  # For saving arrays as .npy files
import pandas as pd  # For working with the dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For preprocessing

# Load the dataset (update the file path to your dataset)
data = pd.read_csv('../dataset/weather/pollution.csv')

# Preprocess the data
# Handle missing values in 'pm2.5' by forward filling
data['pm2.5'] = data['pm2.5'].fillna(method='ffill')

# Encode the categorical column 'cbwd'
label_encoder = LabelEncoder()
data['cbwd'] = label_encoder.fit_transform(data['cbwd'])

# Normalize numerical features
numerical_features = ['DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Define sequence length (7 days of hourly data)
sequence_length = 24 * 7  # 168 timesteps

# Define feature columns and target column
feature_columns = ['DEWP', 'TEMP', 'PRES', 'cbwd', 'Iws', 'Is', 'Ir']
target_column = 'pm2.5'

# Function to create sequences
def create_sequences(data, features, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 7):  # Leave space for 7-day predictions
        X.append(data[features].iloc[i:i + seq_length].values)
        y.append(data[target].iloc[i + seq_length:i + seq_length + 7].values)
    return np.array(X), np.array(y)

# Generate sequences and targets
X, y = create_sequences(data, feature_columns, target_column, sequence_length)

# Save the processed data as .npy files
np.save('X_data.npy', X)
np.save('y_data.npy', y)

print("Data saved as 'X_data.npy' and 'y_data.npy'")
