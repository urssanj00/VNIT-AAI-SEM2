########################  eda_06_bike_sharing.py   ########################

from eda_05_bike_sharing import *

# Check for correlation with the Numeric features
numerical_features = bikesharing_df.select_dtypes(include=np.number)
df_bs_nid = numerical_features.drop(['id'],axis=1)
pd.set_option('display.precision',2)
plt.figure(figsize=(6, 6))
sns.heatmap(df_bs_nid.drop(['total_count'],axis=1).corr(), square=True)
plt.suptitle("Pearson Correlation Heatmap")
plt.show()

corr_with_tot_count = df_bs_nid.corr()["total_count"].sort_values(ascending=False)
plt.figure(figsize=(8,6))
corr_with_tot_count.drop("total_count").plot.bar()
plt.show()

sns.pairplot(df_bs_nid[['total_count', 'casual', 'temp', 'windspeed','registered']])
plt.show()
