import numpy as np

'''
(o-y) * (o-y) / 2

'''
def mean_squared_error(y, t):
	return 0.5 * np.sum((y-t)**2)

y = np.array([0.15,0.4,0.3,0.1,0.15]);
t = np.array([0,1,0,0,0])

print(mean_squared_error(y,t))



y = np.array([0, 0.8, 0.2, 0.1, 0])
t = np.array([0, 1, 0, 0, 0])
print(mean_squared_error(y, t))


'''
Mean Square Error

该损失函数可以认为是最基础的损失函数，最易理解。在机器学习中可以简写为MSE
回归问题最常用的损失函数是均方误差MSE


均方差损失函数常用在最小二乘法中。它的思想是使得各个训练点到最优拟合线的距离最小（平方和最小）。均方差损失函数也是我们最常见的损失函数了


'''