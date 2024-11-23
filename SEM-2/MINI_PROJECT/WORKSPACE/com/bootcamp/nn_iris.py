import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
#%matplotlib inline
class Model(nn.Module):
    # input layer (4 features of the flower) -->
    #   a) sepal length / width
    #   b) petal length / width
    # Hidden Layer1 (number of neurons) h1
    # H2 (n) -->                        h2
    # O/p (3 classes of Iris flowers)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__() #instantiate nn.Module
      #  self.fc1 = nn.Linear(in_features=4, h1)  # in_features moves to h1
        self.fc1 = nn.Linear(in_features, h1)  # in_features moves to h1
        print(f'self.fc1 : {self.fc1}')
        self.fc2 = nn.Linear(h1, h2)           # h1 (hidden layer1) moves to h1 (hidden layer2)
        print(f'self.fc2 : {self.fc2}')
        self.out = nn.Linear(h2, out_features) # h2 (hidden later2) moves to out_features
        print(f'self.out : {self.out}')


    def forward(self, x):
        print(f'0. forward-> x : {x}')
        x_fc1 = self.fc1(x)
        print(f"Before ReLU: {x_fc1}")
        x_forward = F.relu(x_fc1)
        print(f"After ReLU: {x_forward}")

       # x_forward = F.relu(self.fc1(x))  # relu zeros negative and returns same for positives max(0, x)
        print(f'1. forward-> x_forward shape: {x_forward}')
        x_forward = F.relu(self.fc2(x_forward))
        print(f'2. forward-> x_forward shape: {x_forward}')
        x_forward = self.out(x_forward)
        print(f'3. forward-> x_forward shape: {x_forward}')

        return x_forward

# pick a manual seed for randomization
torch.manual_seed(41)


my_df = pandas.read_csv("../dataset/iris/Iris.csv")

print(f'head {my_df.head()}')

#change last column from Strings to Integers
# my_df['Species']=my_df['Species'].replace('Iris-setosa', 0.0)
# my_df['Species']=my_df['Species'].replace('Iris-versicolor', 1.0)
# my_df['Species']=my_df['Species'].replace('Iris-virginica', 2.0)

# updated code - code above is deprecated
# Replace species names with numeric values and convert explicitly to float
mapping = {'Iris-setosa': 0.0, 'Iris-versicolor': 1.0, 'Iris-virginica': 2.0}
my_df['Species'] = my_df['Species'].map(mapping).astype(float)


print("=================================")

print(f'head {my_df.head()}')

# X is feature so dropping the Y from the dataframe
# X = my_df.drop('Species', axis=1)
# ensure to take only 4 features as model is accepting only 4 features
X = my_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

Y = my_df['Species']

x = X.values
y = Y.values

print(f'-1.a) x shape: {x.shape}')
print(f'-1.b) y shape: {y.shape}')

from sklearn.model_selection import  train_test_split

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=41)
print(f'0.a) X_train shape: {X_train.shape}')
print(f'0.b) X_test shape: {X_test.shape}')
print(f'0.c) Y_train shape: {Y_train.shape}')
print(f'0.d) Y_test shape: {Y_test.shape}')

# convert x label to tensors float
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
print(f'1.a) X_train shape: {X_train.shape}')
print(f'1.b) X_test shape: {X_test.shape}')

# convert y label to tensors long
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)
print(f'1.c) Y_train shape: {Y_train.shape}')
print(f'1.d) Y_test shape: {Y_test.shape}')


model = Model(in_features=4)

print(f'2.a) model: {model}')

# set criteria to measure error
criterion = nn.CrossEntropyLoss()
print(f'2.b) criterion: {criterion}')

# choose Adam Optimiser, lr = learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
print(f'2.c) optimizer: {optimizer}')

# Train our model
# Epochs ? (one run through all the training data in our network)
epochs = 10
losses = []
print("Epoch starting")

for i in range(epochs):
    # get prediction
    y_pred = model.forward(X_train)
    print(f'3.a.{i} y_pred shape: {y_pred.shape}')

    # measure loss / error, high to low
    loss = criterion(y_pred, Y_train) # predicted vs y_train
    print(f'3.b.{i} loss: {loss}')

    # keep track of our losses
    losses.append(loss.item())
    print(f'3.c.{i} losses appended with: {loss}')

    if i % 2 == 0 :
        print(f"3.c.{i} Epoch {i}: Loss = {loss.item()}")

    optimizer.zero_grad()
    print(f'3.d.{i} optimizer.zero_grad(): loss : {loss}')

    loss.backward()
    print(f'3.e.{i} loss.backward(): loss : {loss}')

    optimizer.step()
    print(f'3.f.{i} optimizer.step(): loss : {loss}')

# Visualize Loss
import matplotlib.pyplot as plt

plt.plot(range(epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()



# Put model in evaluation mode
model.eval()

# Get predictions for test set
with torch.no_grad():  # No need to compute gradients
    y_test_pred = model(X_test)
    print(f'y_test_pred : {y_test_pred}')
    y_test_pred_classes = torch.argmax(y_test_pred, dim=1)  # Predicted class
    print(f'y_test_pred_classes : {y_test_pred_classes}')

# Calculate accuracy
accuracy = (y_test_pred_classes == Y_test).float().mean()
print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")
