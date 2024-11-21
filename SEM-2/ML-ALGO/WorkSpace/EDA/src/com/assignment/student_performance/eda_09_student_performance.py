##############################   eda_09_student_performance.py   ################

from scipy import stats
from scipy.stats import norm, skew
import seaborn as sns
import matplotlib.pyplot as plt

from eda_08_student_performance import *

print ( "Output => eda_09_student_performance.py  \n###########################################\n")

sns.distplot (students['writing score'], fit=norm)

#Get the fitted parameters used by the function
(mu, sigma) = norm.fit(students ['writing score'])
print('mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))

#plot the distribution
plt.legend(['Normal dist.($\mu=$ {:.2f} and $\sigma=${:.2f})'.format(mu, sigma)],loc='best')

plt.ylabel('Frequency')
plt.title('Writing score distribution')

#Get also QQ-plot
fig = plt.figure()
res = stats.probplot(students['writing score'], plot=plt)
 
plt.show()  # This line ensures the plot is displayed



