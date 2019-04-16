import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
	#误差值设定
	h=1e-4
	return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
	return x**2


#等间值序列
x = np.linspace(-5, 5, 100)
print(x * x)



y = []
for xi in x:
    #y.append(function_1(xi))
	y.append(numerical_diff(function_1,xi))


#plt.plot(x, y)

plt.plot(x, numerical_diff(function_1,x))
plt.show()