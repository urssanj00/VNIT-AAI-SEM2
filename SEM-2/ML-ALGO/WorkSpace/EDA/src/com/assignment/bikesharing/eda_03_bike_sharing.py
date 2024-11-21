########################  eda_03_bike_sharing.py   ########################

from eda_02_bike_sharing import *


# Renaming the Data columns

bikesharing_df.rename(columns={'instant':'id',
                        'dteday':'datetime',
                        'holiday':'holiday_ind',
                        'workingday':'workingday_ind',
                        'weathersit':'weather_con',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_count',
                        'hr':'hour',
                        'yr':'year'},inplace=True)

print("Renamed Title of columns:")
print(bikesharing_df.head())

print("\nDatatype of columns:")
print(bikesharing_df.dtypes)


