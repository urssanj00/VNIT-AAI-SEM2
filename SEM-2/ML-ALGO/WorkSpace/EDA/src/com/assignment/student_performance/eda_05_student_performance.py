##############################   eda_05_student_performance.py   ################

import seaborn as sb
import matplotlib.pyplot as plt

from eda_04_student_performance import *

print ( "Output => eda_05_student_performance.py  \n###########################################\n")
print ( "Feature Variable : x = 'Match_score'")
print ( "Target Variable : y = 'reading score'")
print ( "And distribution in terms of Male and Female students ")
print ( "sb.relplot(x = 'math score', y = 'reading score', hue= 'gender', data= students)")
sb.relplot(x = 'math score', y = 'reading score', hue= 'gender', data= students)
print("plt.show() ")
plt.show()  # This line ensures the plot is displayed


