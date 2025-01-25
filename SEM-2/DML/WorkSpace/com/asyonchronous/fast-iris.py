from flask import Flask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.metrics import accuracy_score
from pydantic import BaseModel

from fastapi import FastAPI
import asyncio
import datetime
app = FastAPI()

def get_time():
    nw = datetime.datetime.now()
    st_time = nw.strftime("%d-%m-%y %H:%M:%S")
    return st_time
@app.get("/greeting")
async def greeting(data):
    print(f'{data}->start_time {get_time()}')
    await asyncio.sleep(10)
    print(f'{data}->end_time: {get_time()}')
    return {"message": f"{data}Hello World FastAPI"}

# Define a Pydantic model for the request

class InputData(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

class ReqData(BaseModel):
    req_id: str  # User ID or request identifier
    inputs: list[InputData]

@app.get("/iris-shape")
async def iris_shape(request: ReqData):
    print(f'{request}')
    req_id = None
    try:
        req_id = request.req_id
        print(f'{req_id}->iris-shape')
        print(f'{req_id}->start_time {get_time()}')
        await asyncio.sleep(10)
        df = pd.read_csv("Iris.csv")
        print(f'{req_id}{df.sample(4)}')
        # Display the total size of the dataset
        print(f"{req_id}->Dataset size:", df.size)

        # Split the data into features and target variable
        X = df.drop(columns=["Species"])  # Replace "Species" with your target column name
        y = df["Species"]

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Print sizes of train and test sets
        print(f"{req_id}->Training set size: {X_train.shape}")
        print(f"{req_id}->Testing set size: {X_test.shape}")
        print(f'{req_id}->end_time {get_time()}')

        return f"<H3>{req_id}-> Iris flower training data shape {X_train.shape}</h3>"
    except Exception as e:
        return {f"{req_id} error": str(e)}

@app.get("/iris-train")
async def iris_train(request: ReqData):
    req_id = None
    try:
        req_id = request.req_id
        print(f'{req_id} iris-train')
        await asyncio.sleep(1)

        df = pd.read_csv("Iris.csv")

        features = df[["SepalLengthCm", "SepalWidthCm",  "PetalLengthCm", "PetalWidthCm"]]
        print(f"{req_id} : Features : {features}")

        target = df["Species"]
        print(f"{req_id} : Target : {target}")

        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=True)

        model = DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=4)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        final_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        print(f"{req_id} : Final Predictions:\n{final_pred}")

        accuracy = accuracy_score(y_test, y_pred)
        print(f"{req_id} : Accuracy Score: {accuracy}")

        model_filename = "DecisionTreeClassifier_model1.pkl"
        with open(model_filename, "wb") as file_obj:
            pickle.dump(model, file_obj)
            print(f"{req_id} : Model has been saved successfully to {model_filename}!")

        return(f"{req_id} : Model has been saved successfully to {model_filename}!")
    except Exception as e:
        return {f"{req_id} error": str(e)}

# Function to load the model and make predictions
# Define a Pydantic model for the input
# Pydantic model for input validation

@app.get("/iris-predict")
async def iris_predict(request: ReqData):
    print(f"ReqData : {request}")
    try:
        req_id = request.req_id
        print(f'{req_id} iris-predict')
        await asyncio.sleep(1)

        # Convert input data to a DataFrame
        input_dicts = [input_item.dict() for input_item in request.inputs]
        sample_input = pd.DataFrame(input_dicts)

        for input_item in request.inputs:
            input_item_dict = input_item.dict()
            print(f"{req_id} input_item_dict: {input_item_dict}")
            input_dicts.append(input_item_dict)
        print(f"")
        print(f"{req_id} input_dicts: {input_dicts}")
        print(f"{req_id} sample_input: {sample_input}")

        # Load the model
        with open("DecisionTreeClassifier_model1.pkl", "rb") as file_obj:
            model = pickle.load(file_obj)
        print(f"{req_id} Model loaded successfully!")

        # Perform predictions
        predictions = model.predict(sample_input)
        for prediction in predictions:
            print(f"{req_id} prediction: {prediction}")

        return {f"{req_id} predictions": predictions.tolist()}
    except Exception as e:
        return {f"{req_id} error": str(e)}



# Example Usage
# JSON input (as a string)
'''
json_input = ''' '''
[
    {"SepalLengthCm": 5.1, "SepalWidthCm": 3.5, "PetalLengthCm": 1.4, "PetalWidthCm": 0.2},
    {"SepalLengthCm": 6.0, "SepalWidthCm": 3.0, "PetalLengthCm": 4.8, "PetalWidthCm": 1.8}
]
'''
'''
predictions = predict_from_json(json_input)

# Print the final predictions
if predictions:
    print(f"Final Predictions: {predictions}")

'''
