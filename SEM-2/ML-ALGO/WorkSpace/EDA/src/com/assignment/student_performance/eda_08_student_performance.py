##############################   eda_08_student_performance.py   ################

import seaborn as sb
import matplotlib.pyplot as plt

from eda_07_student_performance import *

print ( "Output => eda_08_student_performance.py  \n###########################################\n")

print ( "Distribution plot using writing score and bins=5")
print ( "sb.distplot(students['writing score'], bins=5)")
sb.distplot(students['writing score'], bins=5)
print("plt.show() ")
plt.show()  # This line ensures the plot is displayed



