##############################   eda_02_student_performance.py   ################

from eda_01_student_performance import *
print ("Output => eda_02_student_performance.py  \n###########################################\n")
print(f"Check of columns having null values : data.isnull().sum() ->{data.isnull().sum()} ")

print()
print()
print("Drop colums 'race/ethnicity', 'parental level of education':\n")
print(f"students = data.drop(['race/ethnicity', 'parental level of education'], axis=1)")
students = data.drop(['race/ethnicity', 'parental level of education'], axis=1)
print()
print()
print("New dataset after dropping the columns : students.head() -> ")
print(students.head())
print()
print()
print("Original dataset for comparative study : data.head() -> ")
print(data.head())
