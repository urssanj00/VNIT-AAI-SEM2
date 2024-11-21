########################  eda_08_bike_sharing.py   ########################


from eda_07_bike_sharing import *

#Target Variable: total count
#Feature Variables: Rest of the other columns

from sklearn.model_selection import train_test_split
import numpy as np

df_bs_new_numerical_features = bikesharing_df_new.select_dtypes(include=np.number)
train, test=train_test_split(df_bs_new_numerical_features, test_size=0.20)

print("First 5 rows for Train Data")
print(train.head())

#Drop The target variable from Train and Test Data
x_train=train.drop(['total_count'], axis=1)
y_train=train['total_count']
x_test=test.drop(['total_count'], axis=1)
y_test=test['total_count']