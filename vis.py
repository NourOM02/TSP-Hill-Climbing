import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

steepest_x = np.array([20, 21, 52, 37, 28])
steepest_y = np.array([0.945192095969434, 0.932918126059017, 0.892366295303266, 0.966305960402411, 0.950770955832419])

stochastic_x = np.array([4, 5, 7, 7, 5])
stochastic_y = np.array([0.9089534121482, 0.925654782429861, 0.809517004817702, 0.921760124914404, 0.904433874925138])

simple_x = np.array([20, 20, 50, 39, 29])
simple_y = np.array([0.908304614519527, 0.913103932767782, 0.890176379592559, 0.967723253060164, 0.939233841429834])

plt.bar(steepest_x, steepest_y, label="steepest")
plt.bar(stochastic_x, stochastic_y, label="stochastic")
plt.bar(simple_x, simple_y, label="simple")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Accuracy")
plt.title("Tradeoff between accuracy and time")
plt.show()