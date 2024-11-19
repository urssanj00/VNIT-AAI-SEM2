import pandas as pd
from scipy import stats
from scipy.stats import norm, skew
import seaborn as sns
import matplotlib.pyplot as plt

df_bike = pd.read_csv("hour.csv")

print(df_bike.head())
df_bike.info()
print(df_bike.shape)

df_bike.rename(columns={
    'instant': 'id',
    'dteday': 'datetime',
    'holiday': 'holiday_ind',
    'workingday': 'workingday_ind',
    'weathersit': 'weather_con',
    'hum': 'humidity',
    'mnth': 'month',
    'cnt': 'total_count',
    'hr': 'hour',
    'yr': 'year'
}, inplace=True)

print(df_bike.head())
df_bike.info()
print(df_bike.dtypes)

# Date time conversion
df_bike['datetime'] = pd.to_datetime(df_bike.datetime)

# Categorical Variables
df_bike['season'] = df_bike.season.astype('category')
df_bike['holiday_ind'] = df_bike.holiday_ind.astype('category')
df_bike['weekday'] = df_bike.weekday.astype('category')
df_bike['weather_con'] = df_bike.weather_con.astype('category')
df_bike['workingday_ind'] = df_bike.workingday_ind.astype('category')
df_bike['month']= df_bike.month.astype('category')
df_bike['holiday_ind'] = df_bike.holiday_ind.astype("category")
df_bike['weekday'] = df_bike.weekday.astype('category')
df_bike['weather_con'] = df_bike.weather_con.astype('category')
df_bike['workingday_ind'] = df_bike.workingday_ind.astype( 'category')
df_bike['month'] = df_bike.month.astype('category')
df_bike['year'] = df_bike. year.astype('category')
df_bike['hour'] = df_bike.hour.astype('category')


# Converting the categorical variables using dummy variable encoding
d_season = pd.get_dummies(df_bike['season'], prefix='season')
d_hol_i = pd.get_dummies(df_bike['holiday_ind'], prefix='hol')
d_wkd = pd.get_dummies(df_bike['weekday'], prefix='weekday')
d_w_con = pd.get_dummies(df_bike['weather_con'], prefix='w_con')
d_wd_i = pd.get_dummies(df_bike['workingday_ind'], prefix='wd_i')
d_mon = pd.get_dummies(df_bike['month'], prefix='mon')
d_yr = pd.get_dummies(df_bike['year'], prefix='yr')
d_hr = pd.get_dummies(df_bike['hour'], prefix='hour')

# Check for Missing observations
print((df_bike.isnull().sum() / len(df_bike)).sort_values(ascending=False))

# Check for variance


# Test - Distribution of Dep Var
sns.distplot(df_bike['total_count'], fit=norm)
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(df_bike['total_count'])
print('mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))

# Now plot the distribution
plt.legend(['Normal dist. ($\\mu=$ {:.2f} and $\\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('total count distribution')

# Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(df_bike['total_count'], plot=plt)
plt.show()


# Check for correlation with the Numeric features
df_bike_nid = df_bike.drop(['id'], axis=1)
pd.set_option('display.precision', 2)
plt.figure(figsize=(6, 6))
sns.heatmap(df_bike_nid.drop(['total_count'], axis=1).corr(), square=True)
plt.suptitle("Pearson Correlation Heatmap")
plt.show()


