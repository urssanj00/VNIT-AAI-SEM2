##############################   eda_01_student_performance.py   ################

import pandas as pd
import numpy as np
import seaborn as sb
print ("Output => eda_01_student_performance.py  \n###########################################\n")

data = pd.read_csv("..\\student_performance_data\\StudentsPerformance.csv")

print("First 5 rows of dataset : data.head() ->")
print(data.head())
print()
print()
print("Last 5 rows of dataset  : data.tail() -> ")
print(data.tail())
print()
print()
print(f"Dimension : data.shap -> {data.shape}")

print()
print()
print("Statistics of Data : data.describe() -> ")
print(data.describe())

print()
print()
print("data.nunique() -> The pandas.DataFrame.nunique() method in pandas is used to count the \nnumber of unique values along a specified axis of a DataFrame. \nBy default, it counts unique values for each column (axis=0), \nbut you can also count unique values for each row by setting axis=1.")
print(data.nunique())

print()
print()
print(f"data['gender'].unique() ->{data['gender'].unique() }")


print()
print()
print(f"data['parental level of education'].unique() ->{data['parental level of education'].unique() }")