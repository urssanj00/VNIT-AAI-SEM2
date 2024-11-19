# Lecture DML - 19-Oct-2024

import pickle
import pandas as pd
with open("model1.pkl", "rb") as fobj:
    model2 = pickle.load(fobj)

x_h = [5.8]
x_ss = [10]
x_known = pd.DataFrame({"height": x_h, "shoe_size": x_ss})

gender = model2.predict(x_known)

print(f"Gender : {gender}")