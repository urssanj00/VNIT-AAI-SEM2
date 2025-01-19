from flask import Flask
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)


@app.route("/hello", methods=["GET"])
def foo():
    return "<H3> Hello User, welcome to ML Ops class</h3>"

@app.route("/get_status", methods=["GET"])
def bar():
    df = pd.read_csv("Iris.csv")

    print(df.sample(4))
    print(df.size)

    # Display the total size of the dataset
    print("Dataset size:", df.size)

    # Split the data into features and target variable
    X = df.drop(columns=["Species"])  # Replace "Species" with your target column name
    y = df["Species"]

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print sizes of train and test sets
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")

    return f"<H3> Iris flower training data shape {X_train.shape}</h3>"


if __name__ == "__main__":
    app.run(port=5000)


# create one classifier model and return the trainnig data shape