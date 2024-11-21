########################  eda_07_bike_sharing.py   ########################

from eda_06_bike_sharing import *

# Converting the categorical variables using dummy variable encoding
d_season = pd.get_dummies(bikesharing_df['season'], prefix='season')
d_hol_i = pd.get_dummies(bikesharing_df['holiday_ind'], prefix='hol')
d_wkd = pd.get_dummies(bikesharing_df['weekday'], prefix='weekday')
d_w_con = pd.get_dummies(bikesharing_df['weather_con'], prefix='w_con')
d_wd_i = pd.get_dummies(bikesharing_df['workingday_ind'], prefix='wd_i')
d_mon = pd.get_dummies(bikesharing_df['month'], prefix='mon') 
d_yr = pd.get_dummies(bikesharing_df['year'], prefix='yr')
d_hr = pd.get_dummies(bikesharing_df['hour'], prefix='hour')


# Create the final dataset with all the relevant features - both dependant and predictors
feature_x_cont = ['temp','atemp','humidity','windspeed','casual','registered']
bikesharing_df_cont = bikesharing_df[feature_x_cont]

# Creating the Final data with all the relevant fields and Dep Variable
bikesharing_df_new = pd.concat([d_season,d_hol_i,d_wkd,d_w_con,
                       d_wd_i,d_mon,d_yr,d_hr,
                       bikesharing_df_cont,bikesharing_df['total_count']],axis=1)
print("Final data with all the relevant fields and Dep Variable")
print(bikesharing_df_new.head())

print("Describing Final data ")
print(bikesharing_df_new.describe())

sns.boxplot(x=bikesharing_df["total_count"])
