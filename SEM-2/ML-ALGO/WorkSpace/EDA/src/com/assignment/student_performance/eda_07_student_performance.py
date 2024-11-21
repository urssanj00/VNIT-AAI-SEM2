##############################   eda_07_student_performance.py   ################

import seaborn as sb
import matplotlib.pyplot as plt

from eda_06_student_performance import *

print ( "Output => eda_07_student_performance.py  \n###########################################\n")

print ( "Distribution plot using writing score ")
print ( "sb.distplot(students['writing score'])")
sb.distplot(students['writing score'])
print("plt.show() ")
plt.show()  # This line ensures the plot is displayed



