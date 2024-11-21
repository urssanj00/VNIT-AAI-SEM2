##############################   eda_03_student_performance.py   ################
import seaborn as sb
import matplotlib.pyplot as plt

from eda_02_student_performance import *
print ("Output => eda_03_student_performance.py  \n###########################################\n")

print ("Selecting only numerical features for correlation analysis:")
print ("numerical_features = students.select_dtypes(include=np.number)")
numerical_features = students.select_dtypes(include=np.number)
print()
print()

print ("corelation = numerical_features.corr()")
corelation = numerical_features.corr()

print("sb.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)")
sb.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns, annot=True)
print("plt.show() ")
plt.show()  # This line ensures the plot is displayed


