##############################   eda_04_student_performance.py   ################
import seaborn as sb
import matplotlib.pyplot as plt

from eda_03_student_performance import *
print ("Output => eda_04_student_performance.py  \n###########################################\n")

print ("Check the pair plots:-: sb.pairplot(students)")
sb.pairplot(students)
print("plt.show() ")
plt.show()  # This line ensures the plot is displayed


