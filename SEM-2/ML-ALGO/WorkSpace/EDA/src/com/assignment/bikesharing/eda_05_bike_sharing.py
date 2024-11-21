########################  eda_05_bike_sharing.py   ########################

from eda_04_bike_sharing import *

# An alternative View
bikesharing_df.total_count.hist()
plt.title('Histogram of Total count')
plt.xlabel('Total Count')
plt.ylabel('Frequency')

# Log-transformation of the Dep Variable
sns.distplot(np.log1p(bikesharing_df['total_count']) , fit=norm);
print("Distplot done")
# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(np.log1p(bikesharing_df['total_count']))
print( 'mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('log(total_count+1) distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(np.log1p(bikesharing_df['total_count']), plot=plt)
plt.show()