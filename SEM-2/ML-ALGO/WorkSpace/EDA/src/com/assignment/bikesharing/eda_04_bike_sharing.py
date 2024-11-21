########################  eda_04_bike_sharing.py   ########################


from eda_03_bike_sharing import *


# Check for Missing observations
print("\nCheck for Missing observations:")
(bikesharing_df.isnull().sum() / len(bikesharing_df)).sort_values(ascending=False)



# Test - Distribution of Dep Var
from scipy import stats
from scipy.stats import norm, skew
sns.distplot(bikesharing_df['total_count'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(bikesharing_df['total_count'])
print( 'mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Total count distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(bikesharing_df['total_count'], plot=plt)
plt.show()