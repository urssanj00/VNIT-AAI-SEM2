
# Importing relevant libraries
#import scorecardpy as sc
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import openpyxl

import matplotlib
print(matplotlib.__version__)

df_bank = pd.read_excel('../bankdataset/bank_LGR.xlsx')

print(df_bank.head())

print(list(df_bank.columns))

# Check the event rate
event_rate = df_bank['y'].value_counts()/len(df_bank)

print(f'0 event_rate : {event_rate}')

# Remove data with any missing information for now
print(f'Before dropping columns having null values: ' )
print(f'{df_bank.describe()} : {df_bank.shape}')
df_bank = df_bank.dropna()
print(f'After Dropping columns having null values: ')
print(f'{df_bank.describe()} : {df_bank.shape}')


# Get summary stats for the categorical features
print('Get summary stats for the categorical features :')
print(f'{df_bank.describe(include = ["O"])}')

# Create the feature/flag for Dep variable - Attrition status
df_bank.y = df_bank.y.apply(lambda x: 1 if x =='yes' else 0)

# Check the event rate
event_rate = df_bank['y'].value_counts()/len(df_bank)

print(f'1 event_rate : {event_rate}')
print()
print('Explore different features for any kind of inconsistent values:')
# Explore different features for any kind of inconsistent values
print(f'age:        {sorted(df_bank.age.unique())}')
print(f'job::       {df_bank.job.unique()}')
print(f'marital::   {df_bank.marital.unique()}')
print(f'education:: {df_bank.education.unique()}')
print(f'default::   {df_bank.default.unique()}')
print(f'housing::   {df_bank.housing.unique()}')
print(f'loan::      {df_bank.loan.unique()}')
print(f'contact::   {df_bank.contact.unique()}')
print(f'day::       {df_bank.day.unique()}')
print(f'month::     {df_bank.month.unique()}')
print(f'duration::  {df_bank.duration.unique()}')
print(f'campaign::  {df_bank.campaign.unique()}')
print(f'poutcome::  {df_bank.poutcome.unique()}')


pd.crosstab(df_bank.marital,df_bank.y).plot(kind='bar')
plt.title('Purchase Frequency based on marital status')
plt.xlabel('Job')
plt.ylabel('Purchase Freq')
plt.show()

df_bank.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Freq')
plt.show()


# Convert the categorical features
# Creating dummies for all these variables
d_job = pd.get_dummies(df_bank['job'], prefix='job',dtype=float)
d_education = pd.get_dummies(df_bank['education'], prefix='edu',dtype=float)
d_default = pd.get_dummies(df_bank['default'], prefix='def',dtype=float)
d_housing = pd.get_dummies(df_bank['housing'], prefix='housing',dtype=float)
d_loan = pd.get_dummies(df_bank['loan'], prefix='loan',dtype=float)
d_contact = pd.get_dummies(df_bank['contact'], prefix='con',dtype=float)
d_poutcome = pd.get_dummies(df_bank['poutcome'], prefix='pout',dtype=float)
d_marital = pd.get_dummies(df_bank['marital'], prefix='marital',dtype=float)
d_month = pd.get_dummies(df_bank['month'], prefix='mon',dtype=float)
print(f'job : {df_bank["job"]}')
print(f'd_job ###: {d_job}')
print(f'd_education ###: {d_education}')


# Create the final dataset with all the relevant features - both dependant and predictors
feature_x_cont = ['age', 'balance', 'duration', 'pdays', 'previous', 'campaign']
df_bank_cont = df_bank[feature_x_cont]

# Creating the Final data with all the relevant fields and Dep Variable
df_bank_new = pd.concat([d_job,d_education,d_default,d_housing, d_loan,d_contact,d_poutcome,d_marital,d_month, df_bank_cont,df_bank['y']],axis=1)
print (f'df_bank_new.columns : {df_bank_new.columns}')
print(f'df_bank_new.shape : {df_bank_new.shape}')

print('df_bank_new.head()')

print()
print('Explore different features for any kind of inconsistent values:')
# Explore different features for any kind of inconsistent values



# Set display options for pandas to improve the readability of the output
#pd.set_option('display.width', 200)
#pd.set_option('precision', 2)

# Calculate Pearson correlation between 'balance' and 'campaign'
correlations = df_bank_new[['balance', 'campaign']].corr(method='pearson')

# Print the correlation matrix
print(f'correlations: ')
print(f'{correlations}')

# Plotting Box Plot of Age by Status
df_bank_new.boxplot(column=['age'], return_type='axes', by='y')
plt.show()

# We can do some further EDA for a pool of features as well
subset_attributes = ['age', 'balance', 'campaign', 'duration','pdays','previous']
err_yes = round(df_bank_new[df_bank_new['y'] == 1][subset_attributes].describe(),2)
err_no = round(df_bank_new[df_bank_new['y'] == 0][subset_attributes].describe(),2)
a = pd.concat([err_yes, err_no], axis=1, keys=['y=1 ', 'y=0'])

print(a)

# Inferential Stats
from scipy import stats
F, p = stats.f_oneway(df_bank_new[df_bank_new['y'] == 1]['balance'],
                      df_bank_new[df_bank_new['y'] == 0]['balance'])
print('ANOVA test for mean balance levels across y status')
print('F Statistic:', F, ' p-value:', p)


# Univariate analysis
df_bank_new.hist(bins=10, color='purple', edgecolor='black', linewidth=1.0,
              xlabelsize=7, ylabelsize=8, grid=False)
plt.tight_layout(rect=(0, 0, 10, 10))
plt.show()
# plt.tight_layout()
rt = plt.suptitle('Bank data', x=0.9, y=2.25, fontsize=20)