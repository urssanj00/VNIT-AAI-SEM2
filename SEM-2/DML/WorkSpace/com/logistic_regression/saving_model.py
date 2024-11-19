# Lecture DML - 19-Oct-2024
from gender_prediction import *

import pickle

with open("model1.pkl", "wb") as file_obj:
    print(f"file_obj: {file_obj}")
    pickle.dump(model1, file_obj)

print("file created")