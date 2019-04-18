import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
	#误差值设定
	h=1e-4
	return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
	return x**3


#等间值序列
x = np.linspace(0, 10, 100)
#print(x * x)



y = []
for xi in x:
    #y.append(function_1(xi))
	y.append(numerical_diff(function_1,xi))


#plt.plot(x, y)

#plt.plot(x, function_1(x), 'r-')
plt.plot(x, numerical_diff(function_1,x), 'r*')

#numpy.gradient(f)：返回 N 维数组的梯度。 
#plt.plot(x, np.diff(function_1(x)), 'b-.')

plt.plot(x, np.gradient(function_1(x),x), 'b-.')

plt.show()


'''
numpy.gradient(): 计算n维数组的梯度，返回和原始数组同样大小的结果。
对于1维的数组：两个边界的元素直接用后一个减去前一个值，得到梯度，即；对于中间的元素，取相邻两个元素差的一半，即。
如：
In [2]: f = np.array([1, 2, 4, 7, 11, 16], dtype=np.float)

In [3]: np.gradient(f)
Out[3]: array([1. , 1.5, 2.5, 3.5, 4.5, 5. ])

对于2维数组：分别计算每个维度上的梯度，每个维度上的梯度和上面的1维数组梯度求法相同。

In [4]: np.gradient(np.array([
	[1, 2, 6],
	[3, 4, 5]], dtype=np.float))
Out[4]: 
[array([[ 2.,  2., -1.],
        [ 2.,  2., -1.]]), 
array([[1. , 2.5, 4. ],
        [1. , 1. , 1. ]])]

'''