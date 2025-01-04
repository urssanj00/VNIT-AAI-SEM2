import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data_set_path = "C:/Sanjeev/VNIT_CLASSES/mini-proj-dataset/pm2_5/Johannesburg_Westdene_14.csv"

# Load the dataset
data = pd.read_csv(f"{data_set_path}")

# Convert timestamp to datetime and set as index
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
print(data.head())
# Select features (X) and target (y)
features = ['sensor_id', 'temperature', 'humidity', 'longitude', 'latitude']
target_column = 'pm2p5'

# Scale the features and target
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[features + [target_column]])

# Prepare the data for LSTM
sequence_length = 10  # Lookback window
def create_sequences(data, target_idx, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length, :-1]
        label = data[i + seq_length, target_idx]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

X, y = create_sequences(data_scaled, target_idx=-1, seq_length=sequence_length)
for a in X:
    print(f"============")
    for b in a:
        print(f"-->{b}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
epochs = 20
batch_size = 16
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
y_pred = model.predict(X_test)

# Rescale predictions and actual values
y_pred_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), len(features))), y_pred), axis=1))[:, -1]
y_test_rescaled = scaler.inverse_transform(np.concatenate((np.zeros((len(y_test), len(features))), y_test.reshape(-1, 1)), axis=1))[:, -1]

# Convert predictions to binary classes for confusion matrix (example threshold)
threshold = 0.1
y_pred_binary = (y_pred_rescaled > threshold).astype(int)
y_test_binary = (y_test_rescaled > threshold).astype(int)

# Confusion matrix
cm = confusion_matrix(y_test_binary, y_pred_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('Confusion_Matrix.png')

# Classification report
print(classification_report(y_test_binary, y_pred_binary))

# Save the model
model.save('lstm_model.h5')
