##############################   eda_06_student_performance.py   ################

import seaborn as sb
import matplotlib.pyplot as plt

from eda_05_student_performance import *

print ( "Output => eda_06_student_performance.py  \n###########################################\n")
print ( "Feature Variable : x = 'Match_score'")
print ( "Target Variable : y = 'reading score'")
print ( "And distribution in terms of Lunch Data ")
print ( "sb.relplot(x = 'math score', y = 'reading score', hue= 'lunch', data= students)")
sb.relplot(x = 'math score', y = 'reading score', hue= 'lunch', data= students)
print("plt.show() ")
plt.show()  # This line ensures the plot is displayed


