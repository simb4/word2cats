# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("result.csv", names=["X","Yacc","Ypr","Yrc"])
X = data['X'].as_matrix()
Yacc = data['Yacc'].as_matrix()
Ypr = data['Ypr'].as_matrix()
Yrc = data['Yrc'].as_matrix()

data = pd.read_csv("result_forest.csv", names=["X","Yacc","Ypr","Yrc"])
X1 = data['X'].as_matrix()
Yacc1 = data['Yacc'].as_matrix()
Ypr1 = data['Ypr'].as_matrix()
Yrc1 = data['Yrc'].as_matrix()

S = 0.5
Yacc[0] = Yacc1[0] = S
Ypr[0] = Ypr1[0] = S
Yrc[0] = Yrc1[0] = S


def plot_curve():
	plt.title('author dataset comparison')

	plt.plot(X,Yacc, 'b', label="accuracy")
	plt.plot(X,Ypr, 'g', label="precision")
	plt.plot(X,Yrc, 'r', label="recall")

	plt.plot(X,Yacc1, 'b--', label="accuracy forest")
	plt.plot(X,Ypr1, 'g--', label="precision forest")
	plt.plot(X,Yrc1, 'r--', label="recall forest")

	plt.ylabel('percent')
	plt.xlabel('train size')
	plt.legend(loc="lower right")
	plt.show()


plot_curve()