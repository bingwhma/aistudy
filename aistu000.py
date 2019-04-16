import matplotlib.pyplot as plt
import numpy as np


def test_function1(x):
	if x > 0:
		return 1
	else:
		return 0


def test_function2(x):
	y = x > 0
	return y.astype(np.int)

x = np.linspace(-5,5,100)
y = []

for xi in x:
	y.append(test_function1(xi))

#plt.plot(x, y)
plt.plot(x, test_function2(x))
plt.show()


