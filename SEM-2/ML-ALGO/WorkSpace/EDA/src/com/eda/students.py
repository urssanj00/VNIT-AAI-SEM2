import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("StudentsPerformance.csv")

# Understanding the data##
print("HEAD")
print(data.head())

print("TAIL")
print(data.tail())

print("SHAPE")
print(data.shape)

print("DESCRIBE")
print(data.describe())

print("COLUMNS")
print(data.columns)

print("NUNIQUE")
print(data.nunique())

print("UNIQUE GENDERS")
print(data['gender'].unique())

print("UNIQUE parental level of education")
print(data['parental level of education'].unique())

# Correcting the data set##
print("data.isnull().sum()")
print(data.isnull().sum())

print("DROPPING 'gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course'")
#Students = data.drop(['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course'], axis=1)
Students = data.drop(['race/ethnicity', 'parental level of education'], axis=1)

print("HEAD of Student After dropping")
print(Students.head())

print("HEAD of Data After dropping")
print(data.head())


# Relationship Analysis ###

numerical_features = Students.select_dtypes(include=np.number)
correlation = numerical_features.corr()
print(f"correlation : {correlation}")

sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True)
sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=False)
sns.pairplot(Students)
sns.relplot(x = 'math score', y = 'reading score', hue= 'gender',data=Students)
sns.relplot(x = 'math score', y = 'reading score', hue= 'lunch', data=Students)
sns.histplot(Students['writing score'])
sns.histplot(Students['writing score'], bins=5)

plt.title('Distribution of Writing Scores')
plt.xlabel('Writing Score')
plt.ylabel('Frequency')
plt.show()
